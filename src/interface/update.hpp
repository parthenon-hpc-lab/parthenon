//========================================================================================
// (C) (or copyright) 2020-2023. Triad National Security, LLC. All rights reserved.
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
#include "time_integration/staged_integrator.hpp"

#include "kokkos_abstraction.hpp"
#include "mesh/domain.hpp"

namespace parthenon {

namespace Update {

// Calculate the flux divergence for a specific component l of a variable v
KOKKOS_FORCEINLINE_FUNCTION
Real FluxDivHelper(const int l, const int k, const int j, const int i, const int ndim,
                   const Coordinates_t &coords, const VariableFluxPack<Real> &v) {
  Real du = (coords.FaceArea<X1DIR>(k, j, i + 1) * v.flux(X1DIR, l, k, j, i + 1) -
             coords.FaceArea<X1DIR>(k, j, i) * v.flux(X1DIR, l, k, j, i));
  if (ndim >= 2) {
    du += (coords.FaceArea<X2DIR>(k, j + 1, i) * v.flux(X2DIR, l, k, j + 1, i) -
           coords.FaceArea<X2DIR>(k, j, i) * v.flux(X2DIR, l, k, j, i));
  }
  if (ndim == 3) {
    du += (coords.FaceArea<X3DIR>(k + 1, j, i) * v.flux(X3DIR, l, k + 1, j, i) -
           coords.FaceArea<X3DIR>(k, j, i) * v.flux(X3DIR, l, k, j, i));
  }
  return -du / coords.CellVolume(k, j, i);
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
TaskStatus WeightedSumData(const F &flags, T *in1, T *in2, const Real w1, const Real w2,
                           T *out) {
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
TaskStatus CopyData(const F &flags, T *in, T *out) {
  return WeightedSumData(flags, in, in, 1, 0, out);
}

template <typename F, typename T>
TaskStatus SetDataToConstant(const F &flags, T *data, const Real val) {
  Kokkos::Profiling::pushRegion("Task_SetDataToConstant");
  const auto &x = data->PackVariables(flags);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SetDataToConstant", DevExecSpace(), 0, x.GetDim(5) - 1, 0,
      x.GetDim(4) - 1, 0, x.GetDim(3) - 1, 0, x.GetDim(2) - 1, 0, x.GetDim(1) - 1,
      KOKKOS_LAMBDA(const int b, const int l, const int k, const int j, const int i) {
        if (x.IsAllocated(b, l)) {
          x(b, l, k, j, i) = val;
        }
      });
  Kokkos::Profiling::popRegion(); // Task_SetDataToConstant
  return TaskStatus::complete;
}

template <typename F, typename T>
TaskStatus SumData(const F &flags, T *in1, T *in2, T *out) {
  return WeightedSumData(flags, in1, in2, 1.0, 1.0, out);
}

template <typename F, typename T>
TaskStatus UpdateData(const F &flags, T *in, T *dudt, const Real dt, T *out) {
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

// See equation 14 in Ketcheson, Jcomp 229 (2010) 1763-1773
// In Parthenon language, s0 is the variable we are updating
// and rhs should be computed with respect to s0.
// if update_s1, s1 should be set at the beginning of the cycle to 0
// otherwise, s1 should be set at the beginning of the RK update to be
// a copy of base. in the final stage, base for the next cycle should
// be set to s0.
template <typename F, typename T>
TaskStatus Update2S(const F &flags, T *s0_data, T *s1_data, T *rhs_data,
                    const LowStorageIntegrator *pint, Real dt, int stage,
                    bool update_s1) {
  Kokkos::Profiling::pushRegion("Task_2S_Update");
  const auto &s0 = s0_data->PackVariables(flags);
  const auto &s1 = s1_data->PackVariables(flags);
  const auto &rhs = rhs_data->PackVariables(flags);

  const IndexDomain interior = IndexDomain::interior;
  const IndexRange ib = s0_data->GetBoundsI(interior);
  const IndexRange jb = s0_data->GetBoundsJ(interior);
  const IndexRange kb = s0_data->GetBoundsK(interior);

  Real delta = pint->delta[stage - 1];
  Real beta = pint->beta[stage - 1];
  Real gam0 = pint->gam0[stage - 1];
  Real gam1 = pint->gam1[stage - 1];
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "2S_Update", DevExecSpace(), 0, s0.GetDim(5) - 1, 0,
      s0.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int l, const int k, const int j, const int i) {
        if (s0.IsAllocated(b, l) && s1.IsAllocated(b, l) && rhs.IsAllocated(b, l)) {
          if (update_s1) {
            s1(b, l, k, j, i) = s1(b, l, k, j, i) + delta * s0(b, l, k, j, i);
          }
          s0(b, l, k, j, i) = gam0 * s0(b, l, k, j, i) + gam1 * s1(b, l, k, j, i) +
                              beta * dt * rhs(b, l, k, j, i);
        }
      });
  Kokkos::Profiling::popRegion(); // Task_2S_Update
  return TaskStatus::complete;
}
template <typename T>
TaskStatus Update2SIndependent(T *s0_data, T *s1_data, T *rhs_data,
                               const LowStorageIntegrator *pint, Real dt, int stage,
                               bool update_s1) {
  return Update2S(std::vector<MetadataFlag>({Metadata::Independent}), s0_data, s1_data,
                  rhs_data, pint, dt, stage, update_s1);
}

// For integration with Butcher tableaus
// returns base + dt * sum_{j=0}^{k-1} a_{kj} S_j
// for stages S_j
// This can then be used to compute right-hand sides.
template <typename F, typename T>
TaskStatus SumButcher(const F &flags, std::shared_ptr<T> base_data,
                      std::vector<std::shared_ptr<T>> stage_data,
                      std::shared_ptr<T> out_data, const ButcherIntegrator *pint, Real dt,
                      int stage) {
  Kokkos::Profiling::pushRegion("Task_Butcher_Sum");
  const auto &out = out_data->PackVariables(flags);
  const auto &in = base_data->PackVariables(flags);
  const IndexDomain interior = IndexDomain::interior;
  const IndexRange ib = out_data->GetBoundsI(interior);
  const IndexRange jb = out_data->GetBoundsJ(interior);
  const IndexRange kb = out_data->GetBoundsK(interior);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ButcherSumInit", DevExecSpace(), 0, out.GetDim(5) - 1, 0,
      out.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int l, const int k, const int j, const int i) {
        if (out.IsAllocated(b, l) && in.IsAllocated(b, l)) {
          out(b, l, k, j, i) = in(b, l, k, j, i);
        }
      });
  for (int prev = 0; prev < stage; ++prev) {
    Real a = pint->a[stage - 1][prev];
    const auto &in = stage_data[stage]->PackVariables(flags);
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "ButcherSum", DevExecSpace(), 0, out.GetDim(5) - 1, 0,
        out.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int l, const int k, const int j, const int i) {
          if (out.IsAllocated(b, l) && in.IsAllocated(b, l)) {
            out(b, l, k, j, i) += dt * a * in(b, l, k, j, i);
          }
        });
  }
  Kokkos::Profiling::popRegion(); // Task_Butcher_Sum
  return TaskStatus::complete;
}
template <typename T>
TaskStatus SumButcherIndependent(std::shared_ptr<T> base_data,
                                 std::vector<std::shared_ptr<T>> stage_data,
                                 std::shared_ptr<T> out_data,
                                 const ButcherIntegrator *pint, Real dt, int stage) {
  return SumButcher(std::vector<MetadataFlag>({Metadata::Independent}), base_data,
                    stage_data, out_data, pint, dt, stage);
}

// The actual butcher update at the final stage of a cycle
template <typename F, typename T>
TaskStatus UpdateButcher(const F &flags, std::vector<std::shared_ptr<T>> stage_data,
                         std::shared_ptr<T> out_data, const ButcherIntegrator *pint,
                         Real dt) {
  Kokkos::Profiling::pushRegion("Task_Butcher_Update");

  const auto &out = out_data->PackVariables(flags);
  const IndexDomain interior = IndexDomain::interior;
  const IndexRange ib = out_data->GetBoundsI(interior);
  const IndexRange jb = out_data->GetBoundsJ(interior);
  const IndexRange kb = out_data->GetBoundsK(interior);

  const int nstages = pint->nstages;
  for (int stage = 0; stage < nstages; ++stage) {
    const Real butcher_b = pint->b[stage];
    const auto &in = stage_data[stage]->PackVariables(flags);
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "ButcherUpdate", DevExecSpace(), 0, out.GetDim(5) - 1, 0,
        out.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int l, const int k, const int j, const int i) {
          if (out.IsAllocated(b, l) && in.IsAllocated(b, l)) {
            out(b, l, k, j, i) += dt * b * in(b, l, k, j, i);
          }
        });
  }
  Kokkos::Profiling::popRegion(); // Task_Butcher_Update
  return TaskStatus::complete;
}
template <typename F, typename T>
TaskStatus UpdateButcherIndependent(std::vector<std::shared_ptr<T>> stage_data,
                                    std::shared_ptr<T> out_data,
                                    const ButcherIntegrator *pint, Real dt) {
  return UpdateButcherIndependent(std::vector<MetadataFlag>({Metadata::Independent}),
                                  stage_data, out_data, pint, dt);
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
    auto desc = parthenon::MakePackDescriptor<variable_names::any>(
        rc->GetMeshPointer()->resolved_packages.get(), {Metadata::Sparse});
    auto v = desc.GetPack(rc);
    // auto v = parthenon::SparsePack<variable_names::any>::Get(rc, {Metadata::Sparse});

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
