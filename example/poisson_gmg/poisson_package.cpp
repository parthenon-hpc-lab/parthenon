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

#include <coordinates/coordinates.hpp>
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <solvers/solver_utils.hpp>
#include <solvers/mg_solver.hpp>

#include "defs.hpp"
#include "kokkos_abstraction.hpp"
#include "poisson_package.hpp"

using namespace parthenon::package::prelude;
using parthenon::HostArray1D;
namespace poisson_package {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("poisson_package");

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
  
  parthenon::solvers::MGSolver<u, rhs> temp(pkg.get(), parthenon::solvers::MGParams());

  // res_err enters a multigrid level as the residual from the previous level, which
  // is the rhs, and leaves as the solution for that level, which is the error for the
  // next finer level
  auto te_type = Metadata::Cell;
  if (GetTopologicalType(te) == parthenon::TopologicalType::Node) {
    te_type = Metadata::Node;
  } else if (GetTopologicalType(te) == parthenon::TopologicalType::Edge) {
    te_type = Metadata::Edge;
  } else if (GetTopologicalType(te) == parthenon::TopologicalType::Face) {
    te_type = Metadata::Face;
  }
  using namespace parthenon::refinement_ops;
  auto mres_err = Metadata({te_type, Metadata::Independent, Metadata::FillGhost,
                            Metadata::GMGRestrict, Metadata::GMGProlongate});
  mres_err.RegisterRefinementOps<ProlongateSharedLinear, RestrictAverage>();
  pkg->AddField(res_err::name(), mres_err);

  auto mD = Metadata(
      {Metadata::Independent, Metadata::OneCopy, Metadata::Face, Metadata::GMGRestrict});
  mD.RegisterRefinementOps<ProlongateSharedLinear, RestrictAverage>();
  pkg->AddField(D::name(), mD);

  auto mflux_comm = Metadata({te_type, Metadata::Independent, Metadata::FillGhost,
                              Metadata::WithFluxes, Metadata::GMGRestrict});
  mflux_comm.RegisterRefinementOps<ProlongateSharedLinear, RestrictAverage>();
  pkg->AddField(u::name(), mflux_comm);

  auto mflux = Metadata(
      {te_type, Metadata::Independent, Metadata::FillGhost, Metadata::WithFluxes});
  mflux.RegisterRefinementOps<ProlongateSharedLinear, RestrictAverage>();
  pkg->AddField(temp::name(), mflux);

  auto m_no_ghost = Metadata({te_type, Metadata::Derived, Metadata::OneCopy});
  // BiCGSTAB fields
  pkg->AddField(rhat0::name(), m_no_ghost);
  pkg->AddField(v::name(), m_no_ghost);
  pkg->AddField(h::name(), mflux);
  pkg->AddField(s::name(), m_no_ghost);
  pkg->AddField(t::name(), m_no_ghost);
  pkg->AddField(x::name(), m_no_ghost);
  pkg->AddField(r::name(), m_no_ghost);
  pkg->AddField(p::name(), m_no_ghost);

  // Other storage fields
  pkg->AddField(exact::name(), m_no_ghost);
  pkg->AddField(u0::name(), m_no_ghost);
  pkg->AddField(rhs::name(), m_no_ghost);
  pkg->AddField(rhs_base::name(), m_no_ghost);

  return pkg;
}
} // namespace poisson_package
