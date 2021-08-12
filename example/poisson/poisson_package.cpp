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
#include <solvers/solver_utils.hpp>

#include "defs.hpp"
#include "kokkos_abstraction.hpp"
#include "poisson_package.hpp"

using namespace parthenon::package::prelude;

namespace poisson_package {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("poisson_package");

  int max_poisson_iterations = pin->GetOrAddInteger("poisson", "max_iterations", 10000);
  pkg->AddParam<>("max_iterations", max_poisson_iterations);

  int check_interval = pin->GetOrAddInteger("poisson", "check_interval", 100);
  pkg->AddParam<>("check_interval", check_interval);

  Real err_tol = pin->GetOrAddReal("poisson", "error_tolerance", 1.e-8);
  pkg->AddParam<>("error_tolerance", err_tol);

  bool fail_flag = pin->GetOrAddBoolean("poisson", "fail_without_convergence", false);
  pkg->AddParam<>("fail_without_convergence", fail_flag);

  bool warn_flag = pin->GetOrAddBoolean("poisson", "warn_without_convergence", true);
  pkg->AddParam<>("warn_without_convergence", warn_flag);

  auto mrho = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
  pkg->AddField("density", mrho);
  pkg->AddField("rhs", mrho);

  auto mphi = Metadata({Metadata::Cell, Metadata::Independent, Metadata::FillGhost});
  pkg->AddField("potential", mphi);

  int ndim = 1
           + (pin->GetInteger("parthenon/mesh", "nx2") > 1)
           + (pin->GetInteger("parthenon/mesh", "nx3") > 1);
  // set up the stencil object corresponding to the finite difference
  // discretization we adopt in this pacakge
  const int nstencil = 1 + 2*ndim;
  const Real w0 = 1.0/(2.0*ndim);
  std::vector<Real> wgts({w0, -1.0, w0, w0, w0, w0, w0});
  std::vector<std::vector<int>> offsets ({
    {-1, 0, 1, 0, 0, 0, 0},
    {0, 0, 0, -1, 1, 0, 0},
    {0, 0, 0, 0, 0, -1, 1}
  });

  auto stencil = parthenon::solvers::Stencil<Real>("stencil", nstencil, wgts, offsets);
  pkg->AddParam<>("stencil", stencil);

  return pkg;
}

auto &GetCoords(std::shared_ptr<MeshBlock> &pmb) { return pmb->coords; }
auto &GetCoords(Mesh *pm) { return pm->block_list[0]->coords; }

template <typename T>
TaskStatus ComputeRHS(T *u) {
  auto pm = u->GetParentPointer();

  IndexRange ib = u->GetBoundsI(IndexDomain::interior);
  IndexRange jb = u->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = u->GetBoundsK(IndexDomain::interior);

  PackIndexMap imap;
  const std::vector<std::string> vars({"density", "rhs"});
  const auto &v = u->PackVariables(vars, imap);
  const int irho = imap["density"].first;
  const int irhs = imap["rhs"].first;

  auto coords = GetCoords(pm);
  const int ndim = v.GetNdim();
  const Real dx = coords.Dx(X1DIR);
  for (int i = X2DIR; i <= ndim; i++) {
    const Real dy = coords.Dx(i);
    PARTHENON_REQUIRE_THROWS(dx == dy,
                             "ComputeRHS requires that DX be equal in all directions.");
  }

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ComputeRHS", DevExecSpace(), 0, v.GetDim(5) - 1, kb.s, kb.e,
      jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        // first add the RHS
        v(b, irhs, k, j, i) = -v(b, irho, k, j, i) * std::pow(dx, ndim);
      });
  return TaskStatus::complete;
}

template <typename T>
TaskStatus UpdatePhi(T *u, T *du) {
  using Stencil_t = parthenon::solvers::Stencil<Real>;
  Kokkos::Profiling::pushRegion("Task_Poisson_UpdatePhi");
  auto pm = u->GetParentPointer();

  IndexRange ib = u->GetBoundsI(IndexDomain::interior);
  IndexRange jb = u->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = u->GetBoundsK(IndexDomain::interior);

  PackIndexMap imap;
  const std::vector<std::string> vars({"rhs", "potential"});
  const auto &v = u->PackVariables(vars, imap);
  const int irhs = imap["rhs"].first;
  const int iphi = imap["potential"].first;
  const std::vector<std::string> phi_var({"potential"});
  PackIndexMap imap2;
  const auto &dv = du->PackVariables(phi_var, imap2);
  const int idphi = imap2["potential"].first;

  StateDescriptor *pkg = pm->packages.Get("poisson_package").get();
  const auto stencil = pkg->Param<Stencil_t>("stencil");

  StencilMatVec(v, iphi, dv, idphi, v, irhs, stencil, ib, jb, kb);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "UpdatePhi", DevExecSpace(), 0, dv.GetDim(5) - 1, kb.s, kb.e,
      jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        v(b, iphi, k, j, i) += dv(b, idphi, k, j, i);
      });

  Kokkos::Profiling::popRegion(); // Task_Poisson_UpdatePhi
  return TaskStatus::complete;
}

template <typename T>
TaskStatus CheckConvergence(T *u, T *du) {
  Kokkos::Profiling::pushRegion("Task_Poisson_UpdatePhi");
  auto pm = u->GetParentPointer();

  IndexRange ib = u->GetBoundsI(IndexDomain::interior);
  IndexRange jb = u->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = u->GetBoundsK(IndexDomain::interior);

  const std::vector<std::string> vars({"potential"});
  PackIndexMap imap;
  const auto &v = u->PackVariables(vars, imap);
  const int iphi = imap["potential"].first;
  PackIndexMap imap2;
  const auto &dv = du->PackVariables(vars, imap2);
  const int idphi = imap2["potential"].first;

  Real max_err;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "CheckConvergence", DevExecSpace(), 0,
      v.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &eps) {
        Real reps = std::abs(dv(b, idphi, k, j, i) / v(b, iphi, k, j, i));
        Real aeps = std::abs(dv(b, idphi, k, j, i));
        eps = std::max(eps, std::min(reps, aeps));
      },
      Kokkos::Max<Real>(max_err));

  StateDescriptor *pkg = pm->packages.Get("poisson_package").get();
  Real err_tol = pkg->Param<Real>("error_tolerance");

  auto status = (max_err < err_tol ? TaskStatus::complete : TaskStatus::iterate);

  Kokkos::Profiling::popRegion(); // Task_Poisson_CheckConvergence
  return status;
}

TaskStatus PrintComplete() {
  if (parthenon::Globals::my_rank == 0) {
    std::cout << "Poisson solver complete!" << std::endl;
  }
  return TaskStatus::complete;
}

template TaskStatus CheckConvergence<MeshData<Real>>(MeshData<Real> *, MeshData<Real> *);
template TaskStatus CheckConvergence<MeshBlockData<Real>>(MeshBlockData<Real> *,
                                                          MeshBlockData<Real> *);
template TaskStatus UpdatePhi<MeshData<Real>>(MeshData<Real> *, MeshData<Real> *);
template TaskStatus UpdatePhi<MeshBlockData<Real>>(MeshBlockData<Real> *,
                                                   MeshBlockData<Real> *);
template TaskStatus ComputeRHS<MeshData<Real>>(MeshData<Real> *);
template TaskStatus ComputeRHS<MeshBlockData<Real>>(MeshBlockData<Real> *);

} // namespace poisson_package
