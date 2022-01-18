//========================================================================================
// (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
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
#include <parthenon/package.hpp>
#include <solvers/cg_solver.hpp>
#include <solvers/newton_krylov.hpp>
#include <solvers/solver_utils.hpp>

#include "defs.hpp"
#include "kokkos_abstraction.hpp"
#include "poisson_nonlinear_package.hpp"

using namespace parthenon::package::prelude;

namespace poisson_package {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  using namespace parthenon::solvers;
  auto pkg = std::make_shared<StateDescriptor>("poisson_package");

  Real lambda = pin->GetOrAddReal("poisson", "lambda", 1.5);
  pkg->AddParam<>("lambda", lambda);

  int newton_max_iterations =
      pin->GetOrAddInteger("poisson", "newton_max_iterations", 10000);
  pkg->AddParam<>("newton_max_iterations", newton_max_iterations);

  int newton_check_interval = pin->GetOrAddInteger("poisson", "newton_check_interval", 1);
  pkg->AddParam<>("newton_check_interval", newton_check_interval);

  Real newt_err_tol = pin->GetOrAddReal("poisson", "newton_error_tolerance", 1.e-8);
  pkg->AddParam<>("newton_error_tolerance", newt_err_tol);

  bool fail_flag =
      pin->GetOrAddBoolean("poisson", "newton_fail_without_convergence", false);
  pkg->AddParam<>("newton_abort_on_fail", fail_flag);

  bool warn_flag =
      pin->GetOrAddBoolean("poisson", "newton_warn_without_convergence", true);
  pkg->AddParam<>("newton_warn_on_fail", warn_flag);

  int lin_max_iterations = pin->GetOrAddInteger("poisson", "lin_max_iterations", 10000);
  pkg->AddParam<>("cg_max_iterations", lin_max_iterations);

  int lin_check_interval = pin->GetOrAddInteger("poisson", "lin_check_interval", 1);
  pkg->AddParam<>("cg_check_interval", lin_check_interval);

  Real lin_err_tol = pin->GetOrAddReal("poisson", "lin_error_tolerance", 1.e-8);
  pkg->AddParam<>("cg_error_tolerance", lin_err_tol);

  fail_flag = pin->GetOrAddBoolean("poisson", "lin_fail_without_convergence", false);
  pkg->AddParam<>("cg_abort_on_fail", fail_flag);

  warn_flag = pin->GetOrAddBoolean("poisson", "lin_warn_without_convergence", true);
  pkg->AddParam<>("cg_warn_on_fail", warn_flag);

  auto mres = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
  pkg->AddField("residual", mres);
  pkg->AddField("delta", mres);

  auto mphi = Metadata({Metadata::Cell, Metadata::Independent, Metadata::FillGhost});
  pkg->AddField("potential", mphi);

  pkg->AddParam<std::string>("spm_name", "poisson_jacobian");
  pkg->AddParam<std::string>("rhs_name", "residual");
  pkg->AddParam<std::string>("sol_name", "potential");

  std::string precon_name = pin->GetOrAddString("poisson", "precon_name", "diag");
  pkg->AddParam<std::string>("precon_name", precon_name);

  // set up the stencil object corresponding to the finite difference
  // discretization we adopt in this pacakge
  int ndim = 1 + (pin->GetInteger("parthenon/mesh", "nx2") > 1) +
             (pin->GetInteger("parthenon/mesh", "nx3") > 1);
  const int nstencil = 1 + 2 * ndim;
  std::vector<std::vector<int>> offsets(
      {{-1, 0, 1, 0, 0, 0, 0}, {0, 0, 0, -1, 1, 0, 0}, {0, 0, 0, 0, 0, -1, 1}});
  auto sp_accessor =
      parthenon::solvers::SparseMatrixAccessor("accessor", nstencil, offsets);

  // setup the sparse matrix
  Metadata msp = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
                          std::vector<int>({nstencil}));
  pkg->AddField("poisson_jacobian", msp);

  if (precon_name == "icc") {
    // setup the sparse matrix
    pkg->AddField("icc_matrix", msp);
    pkg->AddParam<std::string>("pcm_name", "icc_matrix");

    auto spcm_accessor =
        parthenon::solvers::SparseMatrixAccessor("accessor", nstencil, offsets);
  }

  auto cg_sol = std::make_shared<CG_Solver<SparseMatrixAccessor>>(pkg.get(), lin_err_tol,
                                                                  sp_accessor);

  pkg->AddParam<std::function<TaskStatus(MeshData<Real> *, Real *)>>(
      "ResidualFunc", Residual<MeshData<Real>>);
  pkg->AddParam<std::function<TaskStatus(MeshData<Real> *)>>("JacobianFunc",
                                                             Jacobian<MeshData<Real>>);
  auto newton_krylov =
      std::make_shared<NewtonKrylov<CG_Solver<SparseMatrixAccessor>, MeshData<Real>>>(
          pkg.get(), newt_err_tol, cg_sol);
  pkg->AddParam<>("PoissonSolver", newton_krylov);

  return pkg;
}

auto &GetCoords(std::shared_ptr<MeshBlock> &pmb) { return pmb->coords; }
auto &GetCoords(Mesh *pm) { return pm->block_list[0]->coords; }

TaskStatus PrintComplete() {
  if (parthenon::Globals::my_rank == 0) {
    std::cout << "Poisson solver complete!" << std::endl;
  }
  return TaskStatus::complete;
}

/////////////////////////////////////////////////////////////////////////////////////////
// Utility tasks for solver..
/////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
TaskStatus Residual(T *u, Real *res) {
  auto pm = u->GetParentPointer();

  IndexRange ib = u->GetBoundsI(IndexDomain::interior);
  IndexRange jb = u->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = u->GetBoundsK(IndexDomain::interior);

  PackIndexMap imap;
  const std::vector<std::string> vars({"potential", "residual"});
  const auto &v = u->PackVariables(vars, imap);
  const int iphi = imap["potential"].first;
  const int ires = imap["residual"].first;

  auto coords = GetCoords(pm);
  const Real dx = coords.Dx(X1DIR);
  const int ndim = v.GetNdim();

  for (int i = X2DIR; i <= ndim; i++) {
    const Real dy = coords.Dx(i);
    PARTHENON_REQUIRE_THROWS(dx == dy,
                             "Residual requires that DX be equal in all directions.");
  }
  const Real dx2 = dx * dx;
  const Real dV = std::pow(dx, ndim);
  StateDescriptor *pkg = pm->packages.Get("poisson_package").get();
  const Real lam = pkg->Param<Real>("lambda");

  Real local_res(0.0);
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "Residual", DevExecSpace(), 0, v.GetDim(5) - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lres) {
        v(b, ires, k, j, i) =
            dx2 * lam * std::exp(-v(b, iphi, k, j, i)) - 2 * ndim * v(b, iphi, k, j, i);
        v(b, ires, k, j, i) += (v(b, iphi, k, j, i - 1) + v(b, iphi, k, j, i + 1));
        if (ndim > 1)
          v(b, ires, k, j, i) += (v(b, iphi, k, j - 1, i) + v(b, iphi, k, j + 1, i));
        if (ndim == 3)
          v(b, ires, k, j, i) += (v(b, iphi, k - 1, j, i) + v(b, iphi, k + 1, j, i));
        lres += dV * v(b, ires, k, j, i) * v(b, ires, k, j, i);
      },
      Kokkos::Sum<Real>(local_res));
  *res += local_res;

  return TaskStatus::complete;
}

template <typename T>
TaskStatus Jacobian(T *u) {
  auto pm = u->GetParentPointer();

  IndexRange ib = u->GetBoundsI(IndexDomain::interior);
  IndexRange jb = u->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = u->GetBoundsK(IndexDomain::interior);

  PackIndexMap imap;
  const std::vector<std::string> vars({"poisson_jacobian", "potential"});
  const auto &v = u->PackVariables(vars, imap);
  const int isp_lo = imap["poisson_jacobian"].first;
  const int isp_hi = imap["poisson_jacobian"].second;
  const int iphi = imap["potential"].first;

  if (isp_hi < 0) { // must be using the stencil so return
    return TaskStatus::complete;
  }

  auto coords = GetCoords(pm);
  const Real dx = coords.Dx(X1DIR);
  const int ndim = v.GetNdim();
  for (int i = X2DIR; i <= ndim; i++) {
    const Real dy = coords.Dx(i);
    PARTHENON_REQUIRE_THROWS(dx == dy,
                             "Residual requires that DX be equal in all directions.");
  }
  const Real dx2 = dx * dx;
  StateDescriptor *pkg = pm->packages.Get("poisson_package").get();
  const Real lam = pkg->Param<Real>("lambda");

  const Real w0 = 2.0 * ndim;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SetMatElem", DevExecSpace(), 0, v.GetDim(5) - 1, kb.s, kb.e,
      jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        for (int n = isp_lo; n <= isp_hi; n++) {
          v(b, n, k, j, i) = -1;
        }
        v(b, isp_lo + 1, k, j, i) = w0 + dx2 * lam * std::exp(-v(b, iphi, k, j, i));
      });

  return TaskStatus::complete;
}

/////////////////////////////////////////////////////////////////////////////////////////
template TaskStatus Jacobian<MeshData<Real>>(MeshData<Real> *);
template TaskStatus Jacobian<MeshBlockData<Real>>(MeshBlockData<Real> *);

template TaskStatus Residual<MeshData<Real>>(MeshData<Real> *, Real *);
template TaskStatus Residual<MeshBlockData<Real>>(MeshBlockData<Real> *, Real *);
} // namespace poisson_package
