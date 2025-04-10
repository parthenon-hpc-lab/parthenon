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
#include "poisson_nodal_equation.hpp"
#include "poisson_nodal_package.hpp"

using namespace parthenon::package::prelude;
using parthenon::HostArray1D;
namespace poisson_nodal_package {

using namespace parthenon;
using namespace parthenon::BoundaryFunction;
// We need to register FixedFace boundary conditions by hand since they can't
// be chosen in the parameter input file. FixedFace boundary conditions assume
// Dirichlet booundary conditions on the face of the domain and linearly extrapolate
// into the ghosts to ensure the linear reconstruction on the block face obeys the
// chosen boundary condition. Just setting the ghost zones of CC variables to a fixed
// value results in poor MG convergence because the effective BC at the face
// changes with MG level.

// Build type that selects only variables within the poisson_nodal namespace. Internal solver
// variables have the namespace of input variables prepended, so they will also be
// selected by this type.
struct any_poisson_nodal : public parthenon::variable_names::base_t<true> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION any_poisson_nodal(Ts &&...args)
      : base_t<true>(std::forward<Ts>(args)...) {}
  static std::string name() { return "poisson_nodal[.].*"; }
};

template <CoordinateDirection DIR, BCSide SIDE>
auto GetBC() {
  return [](std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) -> void {
    using namespace parthenon;
    using namespace parthenon::BoundaryFunction;
    GenericBC<DIR, SIDE, BCType::FixedFace, any_poisson_nodal>(rc, coarse, 0.0);
  };
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("poisson_nodal_package");

  // Set boundary conditions for Poisson variables
  using BF = parthenon::BoundaryFace;
  pkg->UserBoundaryFunctions[BF::inner_x1].push_back(GetBC<X1DIR, BCSide::Inner>());
  pkg->UserBoundaryFunctions[BF::inner_x2].push_back(GetBC<X2DIR, BCSide::Inner>());
  pkg->UserBoundaryFunctions[BF::inner_x3].push_back(GetBC<X3DIR, BCSide::Inner>());
  pkg->UserBoundaryFunctions[BF::outer_x1].push_back(GetBC<X1DIR, BCSide::Outer>());
  pkg->UserBoundaryFunctions[BF::outer_x2].push_back(GetBC<X2DIR, BCSide::Outer>());
  pkg->UserBoundaryFunctions[BF::outer_x3].push_back(GetBC<X3DIR, BCSide::Outer>());

  Real diagonal_alpha = pin->GetOrAddReal("poisson_nodal", "diagonal_alpha", 0.0);
  pkg->AddParam<>("diagonal_alpha", diagonal_alpha);

  std::string solver = pin->GetOrAddString("poisson_nodal", "solver", "MG");
  pkg->AddParam<>("solver", solver);

  bool use_exact_rhs = pin->GetOrAddBoolean("poisson_nodal", "use_exact_rhs", false);
  pkg->AddParam<>("use_exact_rhs", use_exact_rhs);

  std::string prolong = pin->GetOrAddString("poisson_nodal", "boundary_prolongation", "Linear");

  using PoissEq = poisson_nodal_package::PoissonEquation<u>;
  PoissEq eq(pin, "poisson_nodal");
  pkg->AddParam<>("poisson_nodal_equation", eq, parthenon::Params::Mutability::Mutable);

  std::shared_ptr<parthenon::solvers::SolverBase> psolver;
  using prolongator_t = parthenon::solvers::ProlongationBlockInteriorDefault;
  using preconditioner_t = parthenon::solvers::MGSolver<PoissEq, prolongator_t>;
  if (solver == "MG") {
    psolver = std::make_shared<parthenon::solvers::MGSolver<PoissEq, prolongator_t>>(
        "base", "u", "rhs", pin, "poisson_nodal/solver_params", PoissEq(pin, "poisson_nodal"));
  } else if (solver == "CG") {
    psolver = std::make_shared<parthenon::solvers::CGSolver<PoissEq, preconditioner_t>>(
        "base", "u", "rhs", pin, "poisson_nodal/solver_params", PoissEq(pin, "poisson_nodal"));
  } else if (solver == "BiCGSTAB") {
    psolver =
        std::make_shared<parthenon::solvers::BiCGSTABSolver<PoissEq, preconditioner_t>>(
            "base", "u", "rhs", pin, "poisson_nodal/solver_params", PoissEq(pin, "poisson_nodal"));
  } else {
    PARTHENON_FAIL("Unknown solver type.");
  }
  pkg->AddParam<>("solver_pointer", psolver);

  using namespace parthenon::refinement_ops;

  std::vector<MetadataFlag> flags{Metadata::Node,        Metadata::Independent,
                                  Metadata::FillGhost,   Metadata::WithFluxes,
                                  Metadata::GMGRestrict, Metadata::GMGProlongate};
  auto mflux_comm = Metadata(flags);
  if (prolong == "Linear") {
    mflux_comm.RegisterRefinementOps<ProlongateSharedLinear, RestrictAverage>();
  } else if (prolong == "Constant") {
    mflux_comm.RegisterRefinementOps<ProlongatePiecewiseConstant, RestrictAverage>();
  } else {
    PARTHENON_FAIL("Unknown prolongation method for Poisson boundaries.");
  }
  // u is the solution vector that starts with an initial guess and then gets updated
  // by the solver
  pkg->AddField(u::name(), mflux_comm);

  auto m_no_ghost = Metadata({Metadata::Node, Metadata::Derived, Metadata::OneCopy});
  // rhs is the field that contains the desired rhs side
  pkg->AddField(rhs::name(), m_no_ghost);

  // Auxillary field for storing the exact solution when it is known
  pkg->AddField(exact::name(), m_no_ghost);

  return pkg;
}
} // namespace poisson_nodal_package
