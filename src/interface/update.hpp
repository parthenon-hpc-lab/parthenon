//========================================================================================
// (C) (or copyright) 2020-2024. Triad National Security, LLC. All rights reserved.
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
#include <set>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

#include "defs.hpp"
#include "interface/metadata.hpp"
#include "interface/params.hpp"
#include "interface/state_descriptor.hpp"
#include "pack/sparse_pack/make_pack_descriptor.hpp"
#include "pack/sparse_pack/sparse_pack.hpp"
#include "sparse/sparse_management.hpp"
#include "time_integration/staged_integrator.hpp"
#include "utils/indexer.hpp"

#include "kokkos_abstraction.hpp"
#include "mesh/domain.hpp"

namespace parthenon {

namespace Update {

namespace impl {
template <typename F, typename T>
auto MakeUpdatePackDescriptor(const F &flags, T *data,
                              const std::set<PDOpt> &options = {}) {
  auto *block_data = data->GetBlockDataRawPointer(0);
  const auto uids = [&] {
    if constexpr (std::is_same_v<F, std::vector<MetadataFlag>>) {
      return block_data->GetVariableUIDs(Metadata::FlagCollection(flags));
    } else {
      return block_data->GetVariableUIDs(flags);
    }
  }();
  return MakePackDescriptor(block_data->GetBlockPointer()->resolved_packages.get(), uids,
                            {}, options);
}

// A common descriptor gives fields a common order, but sparse allocation must also
// match before component indices can be reused between operands.
template <typename... Packs>
void CheckPackLayouts(const SparsePack<>::Descriptor &desc, const SparsePack<> &first,
                      const Packs &...others) {
  auto check = [&](const auto &other) {
    PARTHENON_REQUIRE_THROWS(first.GetNBlocks() == other.GetNBlocks(),
                             "Update operands must contain the same blocks.");
    for (int b = 0; b < first.GetNBlocks(); ++b) {
      for (int v = 0; v < desc.nvar_groups; ++v) {
        const PackIdx idx(v);
        PARTHENON_REQUIRE_THROWS(
            first.GetLowerBoundHost(b, idx) == other.GetLowerBoundHost(b, idx) &&
                first.GetUpperBoundHost(b, idx) == other.GetUpperBoundHost(b, idx),
            "Update operands must have matching allocated fields.");
      }
    }
  };
  (check(others), ...);
}

template <typename Function>
void ForEachCell(const SparsePack<> &pack, const IndexRange &ib, const IndexRange &jb,
                 const IndexRange &kb, const Function &function) {
  const Indexer3D idxer(kb, jb, ib);
  parthenon::par_for_outer(
      PARTHENON_AUTO_LABEL, 0, 0, 0, pack.GetNBlocks() - 1,
      KOKKOS_LAMBDA(team_mbr_t member, const int b) {
        for (int l = pack.GetLowerBound(b); l <= pack.GetUpperBound(b); ++l) {
          parthenon::par_for_inner(member, 0, idxer.size() - 1, [&](const int idx) {
            const auto [k, j, i] = idxer(idx);
            function(b, l, k, j, i);
          });
        }
      });
}

template <typename Function>
void ForEachCell(const SparsePack<> &pack, const Function &function) {
  parthenon::par_for_outer(
      PARTHENON_AUTO_LABEL, 0, 0, 0, pack.GetNBlocks() - 1,
      KOKKOS_LAMBDA(team_mbr_t member, const int b) {
        for (int l = pack.GetLowerBound(b); l <= pack.GetUpperBound(b); ++l) {
          const auto &var = pack(b, l);
          const Indexer3D idxer({0, var.GetDim(3) - 1}, {0, var.GetDim(2) - 1},
                                {0, var.GetDim(1) - 1});
          parthenon::par_for_inner(member, 0, idxer.size() - 1, [&](const int idx) {
            const auto [k, j, i] = idxer(idx);
            function(b, l, k, j, i);
          });
        }
      });
}
} // namespace impl

// Calculate the flux divergence for a specific component l of a variable v
template <typename... Ts>
KOKKOS_FORCEINLINE_FUNCTION Real FluxDivHelper(const int b, const int l, const int k,
                                               const int j, const int i, const int ndim,
                                               const Coordinates_t &coords,
                                               const SparsePack<Ts...> &v) {
  Real du = (coords.FaceArea<X1DIR>(k, j, i + 1) * v.flux(b, X1DIR, l, k, j, i + 1) -
             coords.FaceArea<X1DIR>(k, j, i) * v.flux(b, X1DIR, l, k, j, i));
  if (ndim >= 2) {
    du += (coords.FaceArea<X2DIR>(k, j + 1, i) * v.flux(b, X2DIR, l, k, j + 1, i) -
           coords.FaceArea<X2DIR>(k, j, i) * v.flux(b, X2DIR, l, k, j, i));
  }
  if (ndim == 3) {
    du += (coords.FaceArea<X3DIR>(k + 1, j, i) * v.flux(b, X3DIR, l, k + 1, j, i) -
           coords.FaceArea<X3DIR>(k, j, i) * v.flux(b, X3DIR, l, k, j, i));
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
  PARTHENON_INSTRUMENT
  if (in1->NumBlocks() == 0) return TaskStatus::complete;
  const auto desc = impl::MakeUpdatePackDescriptor(flags, in1);
  const auto x = desc.GetPack(in1, false);
  const auto y = desc.GetPack(in2, false);
  const auto z = desc.GetPack(out, false);
  impl::CheckPackLayouts(desc, x, y, z);
  impl::ForEachCell(
      x, KOKKOS_LAMBDA(const int b, const int l, const int k, const int j, const int i) {
        z(b, l, k, j, i) = w1 * x(b, l, k, j, i) + w2 * y(b, l, k, j, i);
      });
  return TaskStatus::complete;
}

template <typename F, typename T>
TaskStatus CopyData(const F &flags, T *in, T *out) {
  return WeightedSumData(flags, in, in, 1, 0, out);
}

template <typename F, typename T>
TaskStatus SetDataToConstant(const F &flags, T *data, const Real val) {
  PARTHENON_INSTRUMENT
  if (data->NumBlocks() == 0) return TaskStatus::complete;
  const auto desc = impl::MakeUpdatePackDescriptor(flags, data);
  const auto x = desc.GetPack(data, false);
  impl::ForEachCell(
      x, KOKKOS_LAMBDA(const int b, const int l, const int k, const int j, const int i) {
        x(b, l, k, j, i) = val;
      });
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
  PARTHENON_INSTRUMENT
  if (s0_data->NumBlocks() == 0) return TaskStatus::complete;
  const auto desc = impl::MakeUpdatePackDescriptor(flags, s0_data);
  const auto s0 = desc.GetPack(s0_data, false);
  const auto s1 = desc.GetPack(s1_data, false);
  const auto rhs = desc.GetPack(rhs_data, false);
  impl::CheckPackLayouts(desc, s0, s1, rhs);

  const IndexDomain interior = IndexDomain::interior;
  const IndexRange ib = s0_data->GetBoundsI(interior);
  const IndexRange jb = s0_data->GetBoundsJ(interior);
  const IndexRange kb = s0_data->GetBoundsK(interior);

  Real delta = pint->delta[stage - 1];
  Real beta = pint->beta[stage - 1];
  Real gam0 = pint->gam0[stage - 1];
  Real gam1 = pint->gam1[stage - 1];
  impl::ForEachCell(
      s0, ib, jb, kb,
      KOKKOS_LAMBDA(const int b, const int l, const int k, const int j, const int i) {
        if (update_s1) {
          s1(b, l, k, j, i) = s1(b, l, k, j, i) + delta * s0(b, l, k, j, i);
        }
        s0(b, l, k, j, i) = gam0 * s0(b, l, k, j, i) + gam1 * s1(b, l, k, j, i) +
                            beta * dt * rhs(b, l, k, j, i);
      });
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
  PARTHENON_INSTRUMENT
  if (out_data->NumBlocks() == 0) return TaskStatus::complete;
  const auto desc = impl::MakeUpdatePackDescriptor(flags, base_data.get());
  const auto out = desc.GetPack(out_data.get(), false);
  const auto in = desc.GetPack(base_data.get(), false);
  impl::CheckPackLayouts(desc, out, in);
  const IndexDomain interior = IndexDomain::interior;
  const IndexRange ib = out_data->GetBoundsI(interior);
  const IndexRange jb = out_data->GetBoundsJ(interior);
  const IndexRange kb = out_data->GetBoundsK(interior);
  impl::ForEachCell(
      out, ib, jb, kb,
      KOKKOS_LAMBDA(const int b, const int l, const int k, const int j, const int i) {
        out(b, l, k, j, i) = in(b, l, k, j, i);
      });
  for (int prev = 0; prev < stage; ++prev) {
    Real a = pint->a[stage - 1][prev];
    const auto in = desc.GetPack(stage_data[prev].get(), false);
    impl::CheckPackLayouts(desc, out, in);
    impl::ForEachCell(
        out, ib, jb, kb,
        KOKKOS_LAMBDA(const int b, const int l, const int k, const int j, const int i) {
          out(b, l, k, j, i) += dt * a * in(b, l, k, j, i);
        });
  }
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
  PARTHENON_INSTRUMENT

  if (out_data->NumBlocks() == 0) return TaskStatus::complete;
  const auto desc = impl::MakeUpdatePackDescriptor(flags, out_data.get());
  const auto out = desc.GetPack(out_data.get(), false);
  const IndexDomain interior = IndexDomain::interior;
  const IndexRange ib = out_data->GetBoundsI(interior);
  const IndexRange jb = out_data->GetBoundsJ(interior);
  const IndexRange kb = out_data->GetBoundsK(interior);

  const int nstages = pint->nstages;
  for (int stage = 0; stage < nstages; ++stage) {
    const Real butcher_b = pint->b[stage];
    const auto in = desc.GetPack(stage_data[stage].get(), false);
    impl::CheckPackLayouts(desc, out, in);
    impl::ForEachCell(
        out, ib, jb, kb,
        KOKKOS_LAMBDA(const int b, const int l, const int k, const int j, const int i) {
          out(b, l, k, j, i) += dt * butcher_b * in(b, l, k, j, i);
        });
  }
  return TaskStatus::complete;
}
template <typename T>
TaskStatus UpdateButcherIndependent(std::vector<std::shared_ptr<T>> stage_data,
                                    std::shared_ptr<T> out_data,
                                    const ButcherIntegrator *pint, Real dt) {
  return UpdateButcher(std::vector<MetadataFlag>({Metadata::Independent}), stage_data,
                       out_data, pint, dt);
}

template <typename T>
TaskStatus EstimateTimestep(T *rc) {
  PARTHENON_INSTRUMENT
  Real dt_min = std::numeric_limits<Real>::max();
  for (const auto &pkg : rc->GetParentPointer()->packages.AllPackages()) {
    Real dt = pkg.second->EstimateTimestep(rc);
    dt_min = std::min(dt_min, dt);
  }
  rc->SetAllowedDt(dt_min);
  return TaskStatus::complete;
}

template <typename T>
TaskStatus PreCommFillDerived(T *rc) {
  PARTHENON_INSTRUMENT
  auto pm = rc->GetParentPointer();
  for (const auto &pkg : pm->packages.AllPackages()) {
    pkg.second->PreCommFillDerived(rc);
  }
  return TaskStatus::complete;
}

template <typename T>
TaskStatus FillDerived(T *rc) {
  PARTHENON_INSTRUMENT
  auto pm = rc->GetParentPointer();
  { // PreFillDerived region
    PARTHENON_INSTRUMENT
    for (const auto &pkg : pm->packages.AllPackages()) {
      pkg.second->PreFillDerived(rc);
    }
  } // PreFillDerived region
  { // FillDerived region
    PARTHENON_INSTRUMENT
    for (const auto &pkg : pm->packages.AllPackages()) {
      pkg.second->FillDerived(rc);
    }
  } // FillDerived region
  { // PostFillDerived region
    PARTHENON_INSTRUMENT
    for (const auto &pkg : pm->packages.AllPackages()) {
      pkg.second->PostFillDerived(rc);
    }
  } // PostFillDerived region
  return TaskStatus::complete;
}

// In sparse/sparse_management.hpp but convenient to include here
using parthenon::InitNewlyAllocatedVars;
using parthenon::SparseDealloc;

} // namespace Update
} // namespace parthenon

#endif // INTERFACE_UPDATE_HPP_
