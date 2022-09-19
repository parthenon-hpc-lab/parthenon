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
#ifndef INTERFACE_UPDATE_HPP_
#define INTERFACE_UPDATE_HPP_

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

#include "defs.hpp"
#include "interface/metadata.hpp"
#include "interface/params.hpp"
#include "interface/sparse_pack.hpp"
#include "interface/state_descriptor.hpp"

#include "kokkos_abstraction.hpp"
#include "mesh/domain.hpp"

namespace parthenon {

namespace Update {

// Calculate the flux divergence for a specific component l of a variable v
KOKKOS_FORCEINLINE_FUNCTION
Real FluxDivHelper(const int l, const int k, const int j, const int i, const int ndim,
                   const Coordinates_t &coords, const VariableFluxPack<Real> &v) {
  Real du = (coords.Area(X1DIR, k, j, i + 1) * v.flux(X1DIR, l, k, j, i + 1) -
             coords.Area(X1DIR, k, j, i) * v.flux(X1DIR, l, k, j, i));
  if (ndim >= 2) {
    du += (coords.Area(X2DIR, k, j + 1, i) * v.flux(X2DIR, l, k, j + 1, i) -
           coords.Area(X2DIR, k, j, i) * v.flux(X2DIR, l, k, j, i));
  }
  if (ndim == 3) {
    du += (coords.Area(X3DIR, k + 1, j, i) * v.flux(X3DIR, l, k + 1, j, i) -
           coords.Area(X3DIR, k, j, i) * v.flux(X3DIR, l, k, j, i));
  }
  return -du / coords.Volume(k, j, i);
}

template <typename T>
TaskStatus FluxDivergence(T *in, T *dudt_obj);

// Update for low-storage integrators implemented as described in Sec. 3.2.3 of
// Athena++ method paper. Specifically eq (11) at stage s
// u(0) <- gamma_s0 * u(0) + gamma_s1 * u(1) + beta_{s,s-1} * dt * F(u(0))
// Also requires u(1) <- u(0) at the beginning of the first stage.
// Current implementation supports RK1, RK2, RK3, and VL2 with just two registers.
template <typename T>
TaskStatus UpdateWithFluxDivergence(T *data_u0, T *data_u1, const Real gam0,
                                    const Real gam1, const Real beta_dt);

template <typename F, typename T>
TaskStatus WeightedSumData(const std::vector<F> &flags, T *in1, T *in2, const Real w1,
                           const Real w2, T *out) {
  Kokkos::Profiling::pushRegion("Task_WeightedSumData");
  const auto &x = in1->PackVariables(flags);
  const auto &y = in2->PackVariables(flags);
  const auto &z = out->PackVariables(flags);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "WeightedSumData", DevExecSpace(), 0, x.GetDim(5) - 1, 0,
      x.GetDim(4) - 1, 0, x.GetDim(3) - 1, 0, x.GetDim(2) - 1, 0, x.GetDim(1) - 1,
      KOKKOS_LAMBDA(const int b, const int l, const int k, const int j, const int i) {
        // TOOD(someone) This is potentially dangerous and/or not intended behavior
        // as we still may want to update (or populate) z if any of those vars are
        // not allocated yet.
        if (x.IsAllocated(b, l) && y.IsAllocated(b, l) && z.IsAllocated(b, l)) {
          z(b, l, k, j, i) = w1 * x(b, l, k, j, i) + w2 * y(b, l, k, j, i);
        }
      });
  Kokkos::Profiling::popRegion(); // Task_WeightedSumData
  return TaskStatus::complete;
}

template <typename F, typename T>
TaskStatus SumData(const std::vector<F> &flags, T *in1, T *in2, T *out) {
  return WeightedSumData(flags, in1, in2, 1.0, 1.0, out);
}

template <typename F, typename T>
TaskStatus UpdateData(const std::vector<F> &flags, T *in, T *dudt, const Real dt,
                      T *out) {
  return WeightedSumData(flags, in, dudt, 1.0, dt, out);
}

template <typename T>
TaskStatus UpdateIndependentData(T *in, T *dudt, const Real dt, T *out) {
  return WeightedSumData(std::vector<MetadataFlag>({Metadata::Independent}), in, dudt,
                         1.0, dt, out);
}

template <typename F, typename T>
TaskStatus AverageData(const std::vector<F> &flags, T *c1, T *c2, const Real wgt1) {
  return WeightedSumData(flags, c1, c2, wgt1, (1.0 - wgt1), c1);
}

template <typename T>
TaskStatus AverageIndependentData(T *c1, T *c2, const Real wgt1) {
  return WeightedSumData(std::vector<MetadataFlag>({Metadata::Independent}), c1, c2, wgt1,
                         (1.0 - wgt1), c1);
}

template <typename T>
TaskStatus EstimateTimestep(T *rc) {
  Kokkos::Profiling::pushRegion("Task_EstimateTimestep");
  Real dt_min = std::numeric_limits<Real>::max();
  for (const auto &pkg : rc->GetParentPointer()->packages.AllPackages()) {
    Real dt = pkg.second->EstimateTimestep(rc);
    dt_min = std::min(dt_min, dt);
  }
  rc->SetAllowedDt(dt_min);
  Kokkos::Profiling::popRegion(); // Task_EstimateTimestep
  return TaskStatus::complete;
}

template <typename T>
TaskStatus PreCommFillDerived(T *rc) {
  Kokkos::Profiling::pushRegion("Task_PreCommFillDerived");
  auto pm = rc->GetParentPointer();
  for (const auto &pkg : pm->packages.AllPackages()) {
    pkg.second->PreCommFillDerived(rc);
  }
  Kokkos::Profiling::popRegion();
  return TaskStatus::complete;
}

template <typename T>
TaskStatus FillDerived(T *rc) {
  Kokkos::Profiling::pushRegion("Task_FillDerived");
  auto pm = rc->GetParentPointer();
  Kokkos::Profiling::pushRegion("PreFillDerived");
  for (const auto &pkg : pm->packages.AllPackages()) {
    pkg.second->PreFillDerived(rc);
  }
  Kokkos::Profiling::popRegion(); // PreFillDerived
  Kokkos::Profiling::pushRegion("FillDerived");
  for (const auto &pkg : pm->packages.AllPackages()) {
    pkg.second->FillDerived(rc);
  }
  Kokkos::Profiling::popRegion(); // FillDerived
  Kokkos::Profiling::pushRegion("PostFillDerived");
  for (const auto &pkg : pm->packages.AllPackages()) {
    pkg.second->PostFillDerived(rc);
  }
  Kokkos::Profiling::popRegion(); // PostFillDerived
  Kokkos::Profiling::popRegion(); // Task_FillDerived
  return TaskStatus::complete;
}

template <typename T>
TaskStatus InitNewlyAllocatedVars(T *rc) {
  if (!rc->AllVariablesInitialized()) {
    std::cout << "Initializing variable interiors after allocation" << std::endl;

    const IndexDomain interior = IndexDomain::interior;
    const IndexRange ib = rc->GetBoundsI(interior);
    const IndexRange jb = rc->GetBoundsJ(interior);
    const IndexRange kb = rc->GetBoundsK(interior);
    const int Ni = ib.e + 1 - ib.s;
    const int Nj = jb.e + 1 - jb.s;
    const int Nk = kb.e + 1 - kb.s;
    const int NjNi = Nj * Ni;
    const int NkNjNi = Nk * NjNi;

    // This pack will always be freshly built, since we only get here if sparse data
    // was allocated and hasn't been initialized, which in turn implies the cached
    // pack must be stale.
    auto v = parthenon::SparsePack<variable_names::any>::Get(rc, {Metadata::Sparse});

    Kokkos::parallel_for(
        "Set newly allocated interior to default",
        Kokkos::TeamPolicy<>(parthenon::DevExecSpace(), v.GetNBlocks(), Kokkos::AUTO),
        KOKKOS_LAMBDA(parthenon::team_mbr_t team_member) {
          const int b = team_member.league_rank();
          int lo = v.GetLowerBound(b, variable_names::any());
          int hi = v.GetUpperBound(b, variable_names::any());

          for (int vidx = lo; vidx <= hi; ++vidx) {
            if (!v(b, vidx).initialized) {
              Real val = v(b, vidx).sparse_default_val;
              // printf("Initializing block % i variable %i to %e \n", b, vidx, val);
              Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team_member, NkNjNi),
                                   [&](const int idx) {
                                     const int k = kb.s + idx / NjNi;
                                     const int j = jb.s + (idx % NjNi) / Ni;
                                     const int i = ib.s + idx % Ni;
                                     v(b, vidx, k, j, i) = val;
                                   });
            }
          }
        });

    // Set initialized here since everything has been filled with default values,
    // user defined functions may overwrite these in the next step but that doesn't
    // change initialization status of the interior
    rc->SetAllVariablesToInitialized();
  }

  // Do user defined initializations if present
  // This has to be done even in the case where no blocks have been allocated
  // since the boundaries of allocated blocks could have received default data
  // in any case
  Kokkos::Profiling::pushRegion("Task_InitNewlyAllocatedVars");
  auto pm = rc->GetParentPointer();
  for (const auto &pkg : pm->packages.AllPackages()) {
    pkg.second->InitNewlyAllocatedVars(rc);
  }
  Kokkos::Profiling::popRegion();

  // Don't worry about flagging variables as initialized
  // since they will be flagged at the beginning of the
  // next step in the evolution driver

  return TaskStatus::complete;
}

TaskStatus SparseDealloc(MeshData<Real> *md);

} // namespace Update

} // namespace parthenon

#endif // INTERFACE_UPDATE_HPP_
