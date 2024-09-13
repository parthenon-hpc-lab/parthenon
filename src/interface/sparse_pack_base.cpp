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

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <regex>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "coordinates/coordinates.hpp"
#include "interface/mesh_data.hpp"
#include "interface/meshblock_data.hpp"
#include "interface/pack_utils.hpp"
#include "interface/sparse_pack_base.hpp"
#include "interface/state_descriptor.hpp"
#include "interface/variable.hpp"
#include "utils/utils.hpp"
namespace parthenon {
namespace impl {

void PackDescriptor::Print() const {
  printf("--------------------\n");
  for (int i = 0; i < var_group_names.size(); ++i) {
    printf("group name: %s\n", var_group_names[i].c_str());
    printf("--------------------\n");
    for (const auto &[var_name, uid] : var_groups[i]) {
      printf("%s\n", var_name.label().c_str());
    }
  }
  printf("--------------------\n");
}
} // namespace impl
} // namespace parthenon

namespace parthenon {

using namespace impl;

SparsePackIdxMap SparsePackBase::GetIdxMap(const impl::PackDescriptor &desc) {
  SparsePackIdxMap map;
  std::size_t idx = 0;
  for (const auto &var : desc.var_group_names) {
    map[var] = idx;
    ++idx;
  }
  return map;
}

template <class T>
SparsePackBase::alloc_t
SparsePackBase::GetAllocStatus(T *pmd, const PackDescriptor &desc,
                               const std::vector<bool> &include_block) {
  using mbd_t = MeshBlockData<Real>;

  const int nvar = desc.nvar_groups;
  const int nblock = pmd->NumBlocks();
  std::vector<int> astat(nblock * desc.nvar_tot);
  int idx = 0;
  ForEachBlock(pmd, include_block, [&](int b, mbd_t *pmbd) {
    const auto &uid_map = pmbd->GetUidMap();
    for (int i = 0; i < nvar; ++i) {
      for (const auto &[var_name, uid] : desc.var_groups[i]) {
        astat[idx++] =
            uid_map.count(uid) > 0 ? (uid_map.at(uid))->GetAllocationStatus() : -1;
      }
    }
  });
  return astat;
}

// Specialize for the only two types this should work for
template SparsePackBase::alloc_t SparsePackBase::GetAllocStatus<MeshBlockData<Real>>(
    MeshBlockData<Real> *, const PackDescriptor &, const std::vector<bool> &);
template SparsePackBase::alloc_t
SparsePackBase::GetAllocStatus<MeshData<Real>>(MeshData<Real> *, const PackDescriptor &,
                                               const std::vector<bool> &);

template <class T>
SparsePackBase SparsePackBase::Build(T *pmd, const PackDescriptor &desc,
                                     const std::vector<bool> &include_block) {
  using mbd_t = MeshBlockData<Real>;
  int nvar = desc.nvar_groups;

  SparsePackBase pack;
  pack.with_fluxes_ = desc.with_fluxes;
  pack.coarse_ = desc.coarse;
  pack.nvar_ = desc.nvar_groups;
  pack.flat_ = desc.flat;
  pack.size_ = 0;

  // Count up the size of the array that is required
  int max_size = 0;
  int nblocks = 0;
  bool contains_face_or_edge = false;
  bool contains_face_with_fluxes = false;
  int size = 0; // local var used to compute size/block
  ForEachBlock(pmd, include_block, [&](int b, mbd_t *pmbd) {
    if (!desc.flat) {
      size = 0;
    }
    nblocks++;
    const auto &uid_map = pmbd->GetUidMap();
    for (int i = 0; i < nvar; ++i) {
      for (const auto &[var_name, uid] : desc.var_groups[i]) {
        if (uid_map.count(uid) > 0) {
          const auto pv = uid_map.at(uid);
          if (pv->IsAllocated()) {
            if (pv->IsSet(Metadata::Edge)) contains_face_or_edge = true;
            if (pv->IsSet(Metadata::Face)) {
              if (pv->IsSet(Metadata::WithFluxes) && desc.with_fluxes) {
                contains_face_with_fluxes = true;
              }
              contains_face_or_edge = true;
            }
            int prod = pv->GetDim(6) * pv->GetDim(5) * pv->GetDim(4);
            size += prod;       // max size/block (or total size for flat)
            pack.size_ += prod; // total ragged size
          }
        }
      }
    }

    max_size = std::max(size, max_size);
  });
  pack.nblocks_ = desc.flat ? 1 : nblocks;

  // Allocate the views
  int leading_dim = 1;
  pack.flx_idx_ = 1;
  if (contains_face_with_fluxes) {
    leading_dim += 5;
    pack.flx_idx_ = 3;
  } else if (desc.with_fluxes) {
    leading_dim += 3;
  } else if (contains_face_or_edge) {
    leading_dim += 2;
  }
  pack.pack_ = pack_t("data_ptr", leading_dim, pack.nblocks_, max_size);
  pack.pack_h_ = Kokkos::create_mirror_view(pack.pack_);

  // For non-flat packs, shape of pack is type x block x var x k x j x i
  // where type here might be a flux.
  // For flat packs, shape is type x (some var on some block)  x k x j x 1
  // in the latter case, coords indexes into the some var on some
  // block. Bounds provides the start and end index of a var in a block in the flat array.
  // Size is nvar + 1 to store the maximum idx for easy access
  pack.bounds_ = bounds_t("bounds", 2, nblocks, nvar + 1);
  pack.bounds_h_ = Kokkos::create_mirror_view(pack.bounds_);

  // This array stores refinement levels of current block and all neighboring blocks.
  pack.block_props_ = block_props_t("block_props", nblocks, 27 + 1);
  pack.block_props_h_ = Kokkos::create_mirror_view(pack.block_props_);

  pack.coords_ = coords_t("coords", desc.flat ? max_size : nblocks);
  auto coords_h = Kokkos::create_mirror_view(pack.coords_);

  // Fill the views
  int idx = 0;
  int blidx = 0;
  ForEachBlock(pmd, include_block, [&](int block, mbd_t *pmbd) {
    int b = 0;
    const auto &uid_map = pmbd->GetUidMap();
    if (!desc.flat) {
      idx = 0;
      b = blidx;
      // JMM: This line could be unified with the coords_h line below,
      // but it would imply unnecessary copies in the case of non-flat
      // packs.
      coords_h(b) = pmbd->GetBlockPointer()->coords_device;
    }

    // Initialize block refinement levels to current block level to provide default if
    // neighbors not present
    for (int n = 0; n < 27; n++) {
      pack.block_props_h_(blidx, (1 + 3 * (1 + 3 * 1))) =
          pmbd->GetBlockPointer()->loc.level();
    }
    // This block's gid stored in central (1, 1, 1, 1) element
    pack.block_props_h_(blidx, 27) = pmbd->GetBlockPointer()->gid;
    for (auto &neighbor : pmbd->GetBlockPointer()->neighbors) {
      // Multiple refined neighbors may write to the same index but they will always have
      // the same refinement level.
      pack.block_props_h_(
          blidx, (neighbor.offsets[2] + 1) +
                     3 * ((neighbor.offsets[1] + 1) + 3 * (neighbor.offsets[0] + 1))) =
          neighbor.loc.level();
      // Currently not storing neighbor gids
    }

    for (int i = 0; i < nvar; ++i) {
      pack.bounds_h_(0, blidx, i) = idx;
      for (const auto &[var_name, uid] : desc.var_groups[i]) {
        if (uid_map.count(uid) > 0) {
          const auto pv = uid_map.at(uid);
          if (pv->IsAllocated()) {
            Variable<Real> *pvf;
            if (desc.with_fluxes && pv->IsSet(Metadata::WithFluxes)) {
              std::string flux_name = pv->metadata().GetFluxName();
              if (flux_name != "") pvf = &pmbd->Get(flux_name);
            }
            for (int t = 0; t < pv->GetDim(6); ++t) {
              for (int u = 0; u < pv->GetDim(5); ++u) {
                for (int v = 0; v < pv->GetDim(4); ++v) {
                  if (pv->IsSet(Metadata::Face) || pv->IsSet(Metadata::Edge)) {
                    if (pack.coarse_) {
                      pack.pack_h_(0, b, idx) = pv->coarse_s.Get(0, t, u, v);
                      pack.pack_h_(1, b, idx) = pv->coarse_s.Get(1, t, u, v);
                      pack.pack_h_(2, b, idx) = pv->coarse_s.Get(2, t, u, v);
                    } else {
                      pack.pack_h_(0, b, idx) = pv->data.Get(0, t, u, v);
                      pack.pack_h_(1, b, idx) = pv->data.Get(1, t, u, v);
                      pack.pack_h_(2, b, idx) = pv->data.Get(2, t, u, v);
                    }
                    if (pv->IsSet(Metadata::Vector)) {
                      pack.pack_h_(0, b, idx).vector_component = X1DIR;
                      pack.pack_h_(1, b, idx).vector_component = X2DIR;
                      pack.pack_h_(2, b, idx).vector_component = X3DIR;
                    }

                    if (pv->IsSet(Metadata::Face)) {
                      pack.pack_h_(0, b, idx).topological_element =
                          TopologicalElement::F1;
                      pack.pack_h_(1, b, idx).topological_element =
                          TopologicalElement::F2;
                      pack.pack_h_(2, b, idx).topological_element =
                          TopologicalElement::F3;
                    }

                  } else { // This is a cell, node, or a variable that doesn't have
                           // topology information
                    if (pack.coarse_) {
                      pack.pack_h_(0, b, idx) = pv->coarse_s.Get(0, t, u, v);
                    } else {
                      pack.pack_h_(0, b, idx) = pv->data.Get(0, t, u, v);
                    }
                    if (pv->IsSet(Metadata::Vector))
                      pack.pack_h_(0, b, idx).vector_component = v + 1;
                  }

                  if (desc.with_fluxes && pv->IsSet(Metadata::WithFluxes)) {
                    pack.pack_h_(0 + pack.flx_idx_, b, idx) = pvf->data.Get(0, t, u, v);
                    if (!pv->IsSet(Metadata::Edge)) {
                      pack.pack_h_(1 + pack.flx_idx_, b, idx) = pvf->data.Get(1, t, u, v);
                      pack.pack_h_(2 + pack.flx_idx_, b, idx) = pvf->data.Get(2, t, u, v);
                    }
                  }

                  for (auto el :
                       GetTopologicalElements(pack.pack_h_(0, b, idx).topological_type)) {
                    pack.pack_h_(static_cast<int>(el) % 3, b, idx).topological_element =
                        el;
                  }
                  PARTHENON_REQUIRE(
                      pack.pack_h_(0, b, idx).size() > 0,
                      "Seems like this variable might not actually be allocated.");

                  // Save the tensor indices for easy access
                  for (int l = 0; l < leading_dim; ++l) {
                    pack.pack_h_(l, b, idx).t = t;
                    pack.pack_h_(l, b, idx).u = u;
                    pack.pack_h_(l, b, idx).v = v;
                  }

                  if (desc.flat) {
                    coords_h(idx) = pmbd->GetBlockPointer()->coords_device;
                  }
                  idx++;
                }
              }
            }
          }
        }
      }
      pack.bounds_h_(1, blidx, i) = idx - 1;
      if (pack.bounds_h_(1, blidx, i) < pack.bounds_h_(0, blidx, i)) {
        // Did not find any allocated variables meeting our criteria
        pack.bounds_h_(0, blidx, i) = -1;
        // Make the upper bound more negative so a for loop won't iterate once
        pack.bounds_h_(1, blidx, i) = -2;
      }
    }
    // Record the maximum for easy access
    pack.bounds_h_(1, blidx, nvar) = idx - 1;
    blidx++;
  });
  Kokkos::deep_copy(pack.pack_, pack.pack_h_);
  Kokkos::deep_copy(pack.bounds_, pack.bounds_h_);
  Kokkos::deep_copy(pack.block_props_, pack.block_props_h_);
  Kokkos::deep_copy(pack.coords_, coords_h);

  return pack;
}

// Specialize for the only two types this should work for
template SparsePackBase
SparsePackBase::Build<MeshBlockData<Real>>(MeshBlockData<Real> *, const PackDescriptor &,
                                           const std::vector<bool> &);
template SparsePackBase SparsePackBase::Build<MeshData<Real>>(MeshData<Real> *,
                                                              const PackDescriptor &,
                                                              const std::vector<bool> &);

template <class T>
SparsePackBase &SparsePackCache::Get(T *pmd, const PackDescriptor &desc,
                                     const std::vector<bool> &include_block) {
  if (pack_map.count(desc.identifier) > 0) {
    auto &cache_tuple = pack_map[desc.identifier];
    auto &pack = std::get<0>(cache_tuple);
    auto alloc_status_in = SparsePackBase::GetAllocStatus(pmd, desc, include_block);
    auto &alloc_status = std::get<1>(cache_tuple);
    if (alloc_status.size() != alloc_status_in.size())
      return BuildAndAdd(pmd, desc, include_block);
    for (int i = 0; i < alloc_status_in.size(); ++i) {
      if (alloc_status[i] != alloc_status_in[i])
        return BuildAndAdd(pmd, desc, include_block);
    }
    auto &include_status = std::get<2>(cache_tuple);
    if (include_status.size() != include_block.size())
      return BuildAndAdd(pmd, desc, include_block);
    for (int i = 0; i < include_block.size(); ++i) {
      if (include_status[i] != include_block[i])
        return BuildAndAdd(pmd, desc, include_block);
    }
    // Cached version is not stale, so just return a reference to it
    return std::get<0>(cache_tuple);
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
  if (pack_map.count(desc.identifier) > 0) pack_map.erase(desc.identifier);
  pack_map[desc.identifier] = {SparsePackBase::Build(pmd, desc, include_block),
                               SparsePackBase::GetAllocStatus(pmd, desc, include_block),
                               include_block};
  return std::get<0>(pack_map[desc.identifier]);
}
template SparsePackBase &
SparsePackCache::BuildAndAdd<MeshData<Real>>(MeshData<Real> *, const PackDescriptor &,
                                             const std::vector<bool> &);
template SparsePackBase &SparsePackCache::BuildAndAdd<MeshBlockData<Real>>(
    MeshBlockData<Real> *, const PackDescriptor &, const std::vector<bool> &);

} // namespace parthenon
