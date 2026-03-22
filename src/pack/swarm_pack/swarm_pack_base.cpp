//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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

// This file was made in part with generative AI

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "interface/mesh_data.hpp"
#include "interface/meshblock_data.hpp"
#include "pack/pack_utils.hpp"
#include "pack/swarm_pack/swarm_pack_base.hpp"
#include "pack/swarm_pack/swarm_pack_cache.hpp"

namespace parthenon {

template <typename TYPE>
template <class T>
void SwarmPackBase<TYPE>::BuildSupplemental(T *pmd,
                                            const impl::SwarmPackDescriptor<TYPE> &desc,
                                            SwarmPackBase<TYPE> &pack) {
  auto flat_index_map_h = Kokkos::create_mirror_view(pack.flat_index_map_);
  auto max_active_indices_h = Kokkos::create_mirror_view(pack.max_active_indices_);
  ForEachBlock(pmd, std::vector<bool>{}, [&](int b, auto *pmbd) {
    auto swarm = pmbd->GetSwarm(desc.swarm_name);
    pack.contexts_h_(b) = swarm->GetDeviceContext();
    max_active_indices_h(b) = swarm->GetMaxActiveIndex();
    flat_index_map_h(b) =
        (b == 0 ? 0 : flat_index_map_h(b - 1) + max_active_indices_h(b - 1) + 1);
  });
  flat_index_map_h(pack.nblocks_) =
      flat_index_map_h(pack.nblocks_ - 1) + max_active_indices_h(pack.nblocks_ - 1) + 1;
  pack.max_flat_index_ = flat_index_map_h(pack.nblocks_) - 1;

  Kokkos::deep_copy(pack.contexts_, pack.contexts_h_);
  Kokkos::deep_copy(pack.max_active_indices_, max_active_indices_h);
  Kokkos::deep_copy(pack.flat_index_map_, flat_index_map_h);
}

template <typename TYPE>
template <class T>
SwarmPackBase<TYPE>
SwarmPackBase<TYPE>::Build(T *pmd, const impl::SwarmPackDescriptor<TYPE> &desc) {
  const int nvar = desc.vars.size();

  SwarmPackBase<TYPE> pack;
  pack.nvar_ = nvar;

  int max_size = 0;
  int nblocks = 0;
  ForEachBlock(pmd, std::vector<bool>{}, [&](int b, auto *pmbd) {
    auto swarm = pmbd->GetSwarm(desc.swarm_name);
    int size = 0;
    nblocks++;
    for (auto &pv : swarm->template GetVariableVector<TYPE>()) {
      for (int i = 0; i < nvar; ++i) {
        if (desc.IncludeVariable(i, pv)) {
          size += pv->GetDim(6) * pv->GetDim(5) * pv->GetDim(4) * pv->GetDim(3) *
                  pv->GetDim(2);
        }
      }
    }
    max_size = std::max(size, max_size);
  });
  pack.nblocks_ = nblocks;

  constexpr int leading_dim = 1;
  pack.pack_ = pack_t(ViewOfViewAlloc("data_ptr"), leading_dim, nblocks, max_size);
  auto pack_h = create_view_of_view_mirror(pack.pack_);

  pack.bounds_ = bounds_t("bounds", 2, nblocks, nvar);
  auto bounds_h = Kokkos::create_mirror_view(pack.bounds_);

  ForEachBlock(pmd, std::vector<bool>{}, [&](int b, auto *pmbd) {
    int idx = 0;
    auto swarm = pmbd->GetSwarm(desc.swarm_name);
    for (int i = 0; i < nvar; ++i) {
      bounds_h(0, b, i) = idx;
      for (auto &pv : swarm->template GetVariableVector<TYPE>()) {
        if (desc.IncludeVariable(i, pv)) {
          for (int t = 0; t < pv->GetDim(6); ++t) {
            for (int u = 0; u < pv->GetDim(5); ++u) {
              for (int v = 0; v < pv->GetDim(4); ++v) {
                for (int l = 0; l < pv->GetDim(3); ++l) {
                  for (int m = 0; m < pv->GetDim(2); ++m) {
                    pack_h(0, b, idx) = pv->data.Get(0, t, u, v, l, m);
                    PARTHENON_REQUIRE(pack_h(0, b, idx).size() > 0,
                                      "Seems like this variable might not actually be "
                                      "allocated.");
                    idx++;
                  }
                }
              }
            }
          }
        }
      }
      bounds_h(1, b, i) = idx - 1;

      if (bounds_h(1, b, i) < bounds_h(0, b, i)) {
        bounds_h(0, b, i) = -1;
        bounds_h(1, b, i) = -2;
      }
    }
  });

  Kokkos::deep_copy(pack.pack_, pack_h);
  Kokkos::deep_copy(pack.bounds_, bounds_h);

  pack.contexts_ = contexts_t(ViewOfViewAlloc("contexts"), nblocks);
  pack.contexts_h_ = create_view_of_view_mirror(pack.contexts_);
  pack.max_active_indices_ = max_active_indices_t("max_active_indices", nblocks);
  pack.flat_index_map_ = max_active_indices_t("flat_index_map", nblocks + 1);
  BuildSupplemental(pmd, desc, pack);

  return pack;
}

template <typename TYPE>
template <class T>
SwarmPackBase<TYPE>
SwarmPackBase<TYPE>::GetPack(T *pmd, const impl::SwarmPackDescriptor<TYPE> &desc) {
  auto &cache = pmd->template GetSwarmPackCache<TYPE>();
  return cache.Get(pmd, desc);
}

template <typename TYPE>
typename SwarmPackBase<TYPE>::idx_map_t
SwarmPackBase<TYPE>::GetIdxMap(const desc_t &desc) {
  idx_map_t map;
  std::size_t idx = 0;
  for (const auto &var : desc.vars) {
    map[var] = idx;
    ++idx;
  }
  return map;
}

template void SwarmPackBase<Real>::BuildSupplemental<MeshData<Real>>(
    MeshData<Real> *, const impl::SwarmPackDescriptor<Real> &, SwarmPackBase<Real> &);
template void SwarmPackBase<Real>::BuildSupplemental<MeshBlockData<Real>>(
    MeshBlockData<Real> *, const impl::SwarmPackDescriptor<Real> &,
    SwarmPackBase<Real> &);
template void SwarmPackBase<int>::BuildSupplemental<MeshData<Real>>(
    MeshData<Real> *, const impl::SwarmPackDescriptor<int> &, SwarmPackBase<int> &);
template void SwarmPackBase<int>::BuildSupplemental<MeshBlockData<Real>>(
    MeshBlockData<Real> *, const impl::SwarmPackDescriptor<int> &, SwarmPackBase<int> &);
template void SwarmPackBase<std::uint64_t>::BuildSupplemental<MeshData<Real>>(
    MeshData<Real> *, const impl::SwarmPackDescriptor<std::uint64_t> &,
    SwarmPackBase<std::uint64_t> &);
template void SwarmPackBase<std::uint64_t>::BuildSupplemental<MeshBlockData<Real>>(
    MeshBlockData<Real> *, const impl::SwarmPackDescriptor<std::uint64_t> &,
    SwarmPackBase<std::uint64_t> &);

template SwarmPackBase<Real>
SwarmPackBase<Real>::Build<MeshData<Real>>(MeshData<Real> *,
                                           const impl::SwarmPackDescriptor<Real> &);
template SwarmPackBase<Real>
SwarmPackBase<Real>::Build<MeshBlockData<Real>>(MeshBlockData<Real> *,
                                                const impl::SwarmPackDescriptor<Real> &);
template SwarmPackBase<int>
SwarmPackBase<int>::Build<MeshData<Real>>(MeshData<Real> *,
                                          const impl::SwarmPackDescriptor<int> &);
template SwarmPackBase<int>
SwarmPackBase<int>::Build<MeshBlockData<Real>>(MeshBlockData<Real> *,
                                               const impl::SwarmPackDescriptor<int> &);
template SwarmPackBase<std::uint64_t> SwarmPackBase<std::uint64_t>::Build<MeshData<Real>>(
    MeshData<Real> *, const impl::SwarmPackDescriptor<std::uint64_t> &);
template SwarmPackBase<std::uint64_t>
SwarmPackBase<std::uint64_t>::Build<MeshBlockData<Real>>(
    MeshBlockData<Real> *, const impl::SwarmPackDescriptor<std::uint64_t> &);

template SwarmPackBase<Real>
SwarmPackBase<Real>::GetPack<MeshData<Real>>(MeshData<Real> *,
                                             const impl::SwarmPackDescriptor<Real> &);
template SwarmPackBase<Real> SwarmPackBase<Real>::GetPack<MeshBlockData<Real>>(
    MeshBlockData<Real> *, const impl::SwarmPackDescriptor<Real> &);
template SwarmPackBase<int>
SwarmPackBase<int>::GetPack<MeshData<Real>>(MeshData<Real> *,
                                            const impl::SwarmPackDescriptor<int> &);
template SwarmPackBase<int>
SwarmPackBase<int>::GetPack<MeshBlockData<Real>>(MeshBlockData<Real> *,
                                                 const impl::SwarmPackDescriptor<int> &);
template SwarmPackBase<std::uint64_t>
SwarmPackBase<std::uint64_t>::GetPack<MeshData<Real>>(
    MeshData<Real> *, const impl::SwarmPackDescriptor<std::uint64_t> &);
template SwarmPackBase<std::uint64_t>
SwarmPackBase<std::uint64_t>::GetPack<MeshBlockData<Real>>(
    MeshBlockData<Real> *, const impl::SwarmPackDescriptor<std::uint64_t> &);

template SwarmPackBase<Real>::idx_map_t
SwarmPackBase<Real>::GetIdxMap(const SwarmPackBase<Real>::desc_t &);
template SwarmPackBase<int>::idx_map_t
SwarmPackBase<int>::GetIdxMap(const SwarmPackBase<int>::desc_t &);
template SwarmPackBase<std::uint64_t>::idx_map_t
SwarmPackBase<std::uint64_t>::GetIdxMap(const SwarmPackBase<std::uint64_t>::desc_t &);

} // namespace parthenon
