//========================================================================================
// (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
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

  auto mphi = Metadata(
      {Metadata::Cell, Metadata::Independent, Metadata::FillGhost, Metadata::WithFluxes});
  pkg->AddField("potential", mphi);

  return pkg;
}

auto &GetCoords(std::shared_ptr<MeshBlock> &pmb) { return pmb->coords; }
auto &GetCoords(Mesh *pm) { return pm->block_list[0]->coords; }

template <typename T>
TaskStatus UpdatePhi(T *u, T *du) {
  Kokkos::Profiling::pushRegion("Task_Poisson_UpdatePhi");
  auto pm = u->GetParentPointer();

  IndexRange ib = u->GetBoundsI(IndexDomain::interior);
  IndexRange jb = u->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = u->GetBoundsK(IndexDomain::interior);

  PackIndexMap imap;
  std::vector<std::string> vars({"density", "potential"});
  auto v = u->PackVariables(vars, imap);
  const int irho = imap["density"].first;
  const int iphi = imap["potential"].first;
  std::vector<std::string> phi_var({"potential"});
  auto dv = du->PackVariables(phi_var);

  auto coords = GetCoords(pm);
  const int ndim = v.GetNdim();
  const Real dx = coords.Dx(X1DIR);
  for (int i = X2DIR; i <= ndim; i++) {
    const Real dy = coords.Dx(i);
    PARTHENON_REQUIRE_THROWS(dx == dy,
                             "UpdatePhi requires that DX be equal in all directions.");
  }

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "UpdatePhi", DevExecSpace(), 0, v.GetDim(5) - 1, kb.s, kb.e,
      jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        dv(b, 0, k, j, i) = -v(b, irho, k, j, i) * dx * dx;
        dv(b, 0, k, j, i) += v(b, iphi, k, j, i - 1) + v(b, iphi, k, j, i + 1);
        if (ndim > 1) {
          dv(b, 0, k, j, i) += v(b, iphi, k, j - 1, i) + v(b, iphi, k, j + 1, i);
          if (ndim == 3) {
            dv(b, 0, k, j, i) += v(b, iphi, k - 1, j, i) + v(b, iphi, k - 1, j, i);
          }
        }
        dv(b, 0, k, j, i) /= 2.0 * ndim;
        dv(b, 0, k, j, i) -= v(b, iphi, k, j, i);
      });

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "UpdatePhi", DevExecSpace(), 0, dv.GetDim(5) - 1, kb.s, kb.e,
      jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        v(b, iphi, k, j, i) += dv(b, 0, k, j, i);
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

  std::vector<std::string> vars({"potential"});
  auto v = u->PackVariables(vars);
  auto dv = du->PackVariables(vars);

  Real max_err;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "CheckConvergence", DevExecSpace(), 0,
      v.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &eps) {
        Real reps = std::abs(dv(b, 0, k, j, i) / v(b, 0, k, j, i));
        Real aeps = std::abs(dv(b, 0, k, j, i));
        eps = std::max(eps, std::min(reps, aeps));
      },
      Kokkos::Max<Real>(max_err));

  StateDescriptor *pkg = pm->packages.Get("poisson_package").get();
  Real err_tol = pkg->Param<Real>("error_tolerance");

  auto status = (max_err < err_tol ? TaskStatus::complete : TaskStatus::iterate);

  Kokkos::Profiling::popRegion(); // Task_Poisson_CheckConvergence
  return status;
}

template TaskStatus CheckConvergence<MeshData<Real>>(MeshData<Real> *, MeshData<Real> *);
template TaskStatus CheckConvergence<MeshBlockData<Real>>(MeshBlockData<Real> *,
                                                          MeshBlockData<Real> *);
template TaskStatus UpdatePhi<MeshData<Real>>(MeshData<Real> *, MeshData<Real> *);
template TaskStatus UpdatePhi<MeshBlockData<Real>>(MeshBlockData<Real> *,
                                                   MeshBlockData<Real> *);

} // namespace poisson_package
