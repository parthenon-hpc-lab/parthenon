//========================================================================================
// (C) (or copyright) 2021-2024. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <bvals/boundary_conditions_generic.hpp>
#include <coordinates/coordinates.hpp>
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <solvers/bicgstab_solver.hpp>
#include <solvers/cg_solver.hpp>
#include <solvers/solver_utils.hpp>

#include "defs.hpp"
#include "kokkos_abstraction.hpp"
#include "helmholtz_equation.hpp"
#include "helmholtz_package.hpp"

using namespace parthenon::package::prelude;
using parthenon::HostArray1D;
namespace helmholtz_package {

using namespace parthenon;
using namespace parthenon::BoundaryFunction;
// We need to register FixedFace boundary conditions by hand since they can't
// be chosen in the parameter input file. FixedFace boundary conditions assume
// Dirichlet booundary conditions on the face of the domain and linearly extrapolate
// into the ghosts to ensure the linear reconstruction on the block face obeys the
// chosen boundary condition. Just setting the ghost zones of CC variables to a fixed
// value results in poor MG convergence because the effective BC at the face
// changes with MG level.

// Build type that selects only variables within the helmholtz namespace. Internal solver
// variables have the namespace of input variables prepended, so they will also be
// selected by this type.
struct any_helmholtz : public parthenon::variable_names::base_t<true> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION any_helmholtz(Ts &&...args)
      : base_t<true>(std::forward<Ts>(args)...) {}
  static std::string name() { return "helmholtz[.].*"; }
};

template <CoordinateDirection DIR, BCSide SIDE>
auto GetBC() {
  return [](std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) -> void {
    using namespace parthenon;
    using namespace parthenon::BoundaryFunction;
    GenericBC<DIR, SIDE, BCType::FixedFace, any_helmholtz>(rc, coarse, 0.0);
  };
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("helmholtz_package");

  // Set boundary conditions for helmholtz variables
  using BF = parthenon::BoundaryFace;
  pkg->UserBoundaryFunctions[BF::inner_x1].push_back(GetBC<X1DIR, BCSide::Inner>());
  pkg->UserBoundaryFunctions[BF::inner_x2].push_back(GetBC<X2DIR, BCSide::Inner>());
  pkg->UserBoundaryFunctions[BF::inner_x3].push_back(GetBC<X3DIR, BCSide::Inner>());
  pkg->UserBoundaryFunctions[BF::outer_x1].push_back(GetBC<X1DIR, BCSide::Outer>());
  pkg->UserBoundaryFunctions[BF::outer_x2].push_back(GetBC<X2DIR, BCSide::Outer>());
  pkg->UserBoundaryFunctions[BF::outer_x3].push_back(GetBC<X3DIR, BCSide::Outer>());

  Real diagonal_alpha = pin->GetOrAddReal("helmholtz", "diagonal_alpha", 0.0);
  pkg->AddParam<>("diagonal_alpha", diagonal_alpha);

  std::string solver = pin->GetOrAddString("helmholtz", "solver", "MG");
  pkg->AddParam<>("solver", solver);

  bool use_exact_rhs = pin->GetOrAddBoolean("helmholtz", "use_exact_rhs", false);
  pkg->AddParam<>("use_exact_rhs", use_exact_rhs);

  std::string prolong = pin->GetOrAddString("helmholtz", "boundary_prolongation", "Linear");

  using PoissEq = helmholtz_package::HelmholtzEquation<u, F>;
  PoissEq eq(pin, "helmholtz");
  pkg->AddParam<>("helmholtz_equation", eq, parthenon::Params::Mutability::Mutable);

  std::shared_ptr<parthenon::solvers::SolverBase> psolver;
  using prolongator_t = parthenon::solvers::ProlongationBlockInteriorDefault;
  using preconditioner_t = parthenon::solvers::MGSolver<PoissEq, prolongator_t>;
  if (solver == "MG") {
    psolver = std::make_shared<parthenon::solvers::MGSolver<PoissEq, prolongator_t>>(
        "base", "u", "rhs", pin, "helmholtz/solver_params", PoissEq(pin, "helmholtz"));
  } else if (solver == "CG") {
    psolver = std::make_shared<parthenon::solvers::CGSolver<PoissEq, preconditioner_t>>(
        "base", "u", "rhs", pin, "helmholtz/solver_params", PoissEq(pin, "helmholtz"));
  } else if (solver == "BiCGSTAB") {
    psolver =
        std::make_shared<parthenon::solvers::BiCGSTABSolver<PoissEq, preconditioner_t>>(
            "base", "u", "rhs", pin, "helmholtz/solver_params", PoissEq(pin, "helmholtz"));
  } else {
    PARTHENON_FAIL("Unknown solver type.");
  }
  pkg->AddParam<>("solver_pointer", psolver);

  using namespace parthenon::refinement_ops;

  std::vector<MetadataFlag> flags_cc{Metadata::Cell,        Metadata::Independent,
                                     Metadata::FillGhost,
                                     Metadata::GMGRestrict, Metadata::GMGProlongate};
  std::vector<MetadataFlag> flags_fc{Metadata::Face,        Metadata::Independent,
                                     Metadata::FillGhost,
                                     Metadata::GMGRestrict, Metadata::GMGProlongate};
  auto mflux_comm_cc = Metadata(flags_cc);
  auto mflux_comm_fc = Metadata(flags_fc);
  if (prolong == "Linear") {
    mflux_comm_cc.RegisterRefinementOps<ProlongateSharedLinear, RestrictAverage>();
    mflux_comm_fc.RegisterRefinementOps<ProlongateSharedLinear, RestrictAverage>();
  } else if (prolong == "Constant") {
    mflux_comm_cc.RegisterRefinementOps<ProlongatePiecewiseConstant, RestrictAverage>();
    mflux_comm_fc.RegisterRefinementOps<ProlongatePiecewiseConstant, RestrictAverage>();
  } else {
    PARTHENON_FAIL("Unknown prolongation method for Helmholtz boundaries.");
  }
  // u is the solution vector that starts with an initial guess and then gets updated
  // by the solver
  pkg->AddField<u>(mflux_comm_cc);
  pkg->AddField<F>(mflux_comm_fc);

  return pkg;
}
} // namespace helmholtz_package
