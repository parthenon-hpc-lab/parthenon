//========================================================================================
// (C) (or copyright) 2021-2023. Triad National Security, LLC. All rights reserved.
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
#include <vector>

#include <bvals/boundary_conditions_generic.hpp>
#include <coordinates/coordinates.hpp>
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <solvers/bicgstab_solver.hpp>
#include <solvers/mg_solver.hpp>
#include <solvers/solver_utils.hpp>

#include "defs.hpp"
#include "kokkos_abstraction.hpp"
#include "poisson_equation.hpp"
#include "poisson_package.hpp"

using namespace parthenon::package::prelude;
using parthenon::HostArray1D;
namespace poisson_package {

using namespace parthenon;
using namespace parthenon::BoundaryFunction;
// We need to register FixedFace boundary conditions by hand since they can't
// be chosen in the parameter input file. FixedFace boundary conditions assume
// Dirichlet booundary conditions on the face of the domain and linearly extrapolate
// into the ghosts to ensure the linear reconstruction on the block face obeys the
// chosen boundary condition. Just setting the ghost zones of CC variables to a fixed
// value results in poor MG convergence because the effective BC at the face
// changes with MG level.

// Build type that selects only variables within the poisson namespace. Internal solver
// variables have the namespace of input variables prepended, so they will also be
// selected by this type.
struct any_poisson : public parthenon::variable_names::base_t<true> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION any_poisson(Ts &&...args)
      : base_t<true>(std::forward<Ts>(args)...) {}
  static std::string name() { return "poisson[.].*"; }
};

template <CoordinateDirection DIR, BCSide SIDE>
auto GetBC() {
  return [](std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) -> void {
    using namespace parthenon;
    using namespace parthenon::BoundaryFunction;
    GenericBC<DIR, SIDE, BCType::FixedFace, any_poisson>(rc, coarse, 0.0);
  };
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("poisson_package");

  // Set boundary conditions for Poisson variables
  using BF = parthenon::BoundaryFace;
  pkg->UserBoundaryFunctions[BF::inner_x1].push_back(GetBC<X1DIR, BCSide::Inner>());
  pkg->UserBoundaryFunctions[BF::inner_x2].push_back(GetBC<X2DIR, BCSide::Inner>());
  pkg->UserBoundaryFunctions[BF::inner_x3].push_back(GetBC<X3DIR, BCSide::Inner>());
  pkg->UserBoundaryFunctions[BF::outer_x1].push_back(GetBC<X1DIR, BCSide::Outer>());
  pkg->UserBoundaryFunctions[BF::outer_x2].push_back(GetBC<X2DIR, BCSide::Outer>());
  pkg->UserBoundaryFunctions[BF::outer_x3].push_back(GetBC<X3DIR, BCSide::Outer>());

  int max_poisson_iterations = pin->GetOrAddInteger("poisson", "max_iterations", 10000);
  pkg->AddParam<>("max_iterations", max_poisson_iterations);

  int pre_smooth_iterations = pin->GetOrAddInteger("poisson", "pre_smooth_iterations", 2);
  pkg->AddParam<>("pre_smooth_iterations", pre_smooth_iterations);

  int post_smooth_iterations =
      pin->GetOrAddInteger("poisson", "post_smooth_iterations", 2);
  pkg->AddParam<>("post_smooth_iterations", post_smooth_iterations);

  Real diagonal_alpha = pin->GetOrAddReal("poisson", "diagonal_alpha", 0.0);
  pkg->AddParam<>("diagonal_alpha", diagonal_alpha);

  Real jacobi_damping = pin->GetOrAddReal("poisson", "jacobi_damping", 0.5);
  pkg->AddParam<>("jacobi_damping", jacobi_damping);

  std::string smoother_method = pin->GetOrAddString("poisson", "smoother", "SRJ2");
  pkg->AddParam<>("smoother", smoother_method);

  bool do_FAS = pin->GetOrAddBoolean("poisson", "do_FAS", false);
  pkg->AddParam<>("do_FAS", do_FAS);

  std::string solver = pin->GetOrAddString("poisson", "solver", "MG");
  pkg->AddParam<>("solver", solver);

  bool precondition = pin->GetOrAddBoolean("poisson", "precondition", true);
  pkg->AddParam<>("precondition", precondition);

  int precondition_vcycles = pin->GetOrAddInteger("poisson", "precondition_vcycles", 1);
  pkg->AddParam<>("precondition_vcycles", precondition_vcycles);

  bool flux_correct = pin->GetOrAddBoolean("poisson", "flux_correct", false);
  pkg->AddParam<>("flux_correct", flux_correct);

  Real restart_threshold = pin->GetOrAddReal("poisson", "restart_threshold", 0.0);
  pkg->AddParam<>("restart_threshold", restart_threshold);

  int check_interval = pin->GetOrAddInteger("poisson", "check_interval", 100);
  pkg->AddParam<>("check_interval", check_interval);

  Real err_tol = pin->GetOrAddReal("poisson", "error_tolerance", 1.e-8);
  pkg->AddParam<>("error_tolerance", err_tol);

  Real res_tol = pin->GetOrAddReal("poisson", "residual_tolerance", 1.e-8);
  pkg->AddParam<>("residual_tolerance", res_tol);

  bool fail_flag = pin->GetOrAddBoolean("poisson", "fail_without_convergence", false);
  pkg->AddParam<>("fail_without_convergence", fail_flag);

  bool warn_flag = pin->GetOrAddBoolean("poisson", "warn_without_convergence", true);
  pkg->AddParam<>("warn_without_convergence", warn_flag);

  bool use_exact_rhs = pin->GetOrAddBoolean("poisson", "use_exact_rhs", false);
  pkg->AddParam<>("use_exact_rhs", use_exact_rhs);

  PoissonEquation eq;
  eq.do_flux_cor = flux_correct;

  parthenon::solvers::MGParams mg_params;
  mg_params.max_iters = max_poisson_iterations;
  mg_params.residual_tolerance = res_tol;
  mg_params.do_FAS = do_FAS;
  mg_params.smoother = smoother_method;
  parthenon::solvers::MGSolver<u, rhs, PoissonEquation> mg_solver(pkg.get(), mg_params,
                                                                  eq);
  pkg->AddParam<>("MGsolver", mg_solver, parthenon::Params::Mutability::Mutable);

  parthenon::solvers::BiCGSTABParams bicgstab_params;
  bicgstab_params.max_iters = max_poisson_iterations;
  bicgstab_params.residual_tolerance = res_tol;
  bicgstab_params.precondition = precondition;
  parthenon::solvers::BiCGSTABSolver<u, rhs, PoissonEquation> bicg_solver(
      pkg.get(), bicgstab_params, eq);
  pkg->AddParam<>("MGBiCGSTABsolver", bicg_solver,
                  parthenon::Params::Mutability::Mutable);

  using namespace parthenon::refinement_ops;
  auto mD = Metadata(
      {Metadata::Independent, Metadata::OneCopy, Metadata::Face, Metadata::GMGRestrict});
  mD.RegisterRefinementOps<ProlongateSharedLinear, RestrictAverage>();
  // Holds the discretized version of D in \nabla \cdot D(\vec{x}) \nabla u = rhs. D = 1
  // for the standard Poisson equation.
  pkg->AddField(D::name(), mD);

  auto mflux_comm = Metadata({Metadata::Cell, Metadata::Independent, Metadata::FillGhost,
                              Metadata::WithFluxes, Metadata::GMGRestrict});
  mflux_comm.RegisterRefinementOps<ProlongateSharedLinear, RestrictAverage>();
  // u is the solution vector that starts with an initial guess and then gets updated
  // by the solver
  pkg->AddField(u::name(), mflux_comm);

  auto m_no_ghost = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
  // rhs is the field that contains the desired rhs side
  pkg->AddField(rhs::name(), m_no_ghost);

  // Auxillary field for storing the exact solution when it is known
  pkg->AddField(exact::name(), m_no_ghost);

  return pkg;
}
} // namespace poisson_package
