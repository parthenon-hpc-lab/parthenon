//========================================================================================
// (C) (or copyright) 2020-2025. Triad National Security, LLC. All rights reserved.
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

#include <vector>

#include "interface/mesh_data.hpp"
#include "interface/meshblock_data.hpp"
#include "pack/sparse_pack_cache.hpp"

namespace parthenon {

using namespace impl;

template <class T>
SparsePackBase &SparsePackCache::Get(T *pmd, const PackDescriptor &desc,
                                     const std::vector<bool> &include_block) {
  if (pack_map.count(desc.identifier) > 0) {
    auto &desc_pack_map = pack_map[desc.identifier];
    if (desc_pack_map.count(include_block)) {
      auto &cache_tuple = desc_pack_map[include_block];
      auto &pack = std::get<0>(cache_tuple);
      auto alloc_status_in = SparsePackBase::GetAllocStatus(pmd, desc, include_block);
      auto &alloc_status = std::get<1>(cache_tuple);
      if (alloc_status.size() != alloc_status_in.size())
        return BuildAndAdd(pmd, desc, include_block);
      for (int i = 0; i < alloc_status_in.size(); ++i) {
        if (alloc_status[i] != alloc_status_in[i])
          return BuildAndAdd(pmd, desc, include_block);
      }
      // Cached version is not stale, so just return a reference to it
      return std::get<0>(cache_tuple);
    }
  }
  return BuildAndAdd(pmd, desc, include_block);
}
template SparsePackBase &SparsePackCache::Get<MeshData<Real>>(MeshData<Real> *,
                                                              const PackDescriptor &,
                                                              const std::vector<bool> &);
template SparsePackBase &
SparsePackCache::Get<MeshBlockData<Real>>(MeshBlockData<Real> *, const PackDescriptor &,
                                          const std::vector<bool> &);

template <class T>
SparsePackBase &SparsePackCache::BuildAndAdd(T *pmd, const PackDescriptor &desc,
                                             const std::vector<bool> &include_block) {
  if (pack_map.count(desc.identifier) == 0) pack_map[desc.identifier] = desc_pack_map_t();

  desc_pack_map_t &desc_pack_map = pack_map[desc.identifier];
  desc_pack_map[include_block] = {
      SparsePackBase::Build(pmd, desc, include_block),
      SparsePackBase::GetAllocStatus(pmd, desc, include_block)};

  return std::get<0>(desc_pack_map[include_block]);
}
template SparsePackBase &
SparsePackCache::BuildAndAdd<MeshData<Real>>(MeshData<Real> *, const PackDescriptor &,
                                             const std::vector<bool> &);
template SparsePackBase &SparsePackCache::BuildAndAdd<MeshBlockData<Real>>(
    MeshBlockData<Real> *, const PackDescriptor &, const std::vector<bool> &);

} // namespace parthenon
