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
#ifndef INTERFACE_MESH_DATA_HPP_
#define INTERFACE_MESH_DATA_HPP_

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "bvals/cc/bvals_cc_in_one.hpp"
#include "interface/variable_pack.hpp"
#include "mesh/domain.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/meshblock_pack.hpp"
#include "utils/error_checking.hpp"
#include "utils/utils.hpp"

namespace parthenon {

class Mesh;
template <typename T>
class MeshBlockData;

template <typename T>
using BlockDataList_t = std::vector<std::shared_ptr<MeshBlockData<T>>>;

namespace pack_on_mesh_impl {

// This function template takes a new key and adds it to a key collection
template <typename K>
inline void AppendKey(K *key_collection, const K *new_key);

// Specialization for variable packs where key is a std::vector<std::string>
template <>
inline void AppendKey<std::vector<std::string>>(std::vector<std::string> *key_collection,
                                                const std::vector<std::string> *new_key) {
  for (const auto &k : *new_key) {
    key_collection->push_back(k);
  }
}

// Specialization for flux-variable packs where key is a vpack_types::StringPair (which is
// a pair of std::vector<std::string>)
template <>
inline void AppendKey<vpack_types::StringPair>(vpack_types::StringPair *key_collection,
                                               const vpack_types::StringPair *new_key) {
  for (const auto &k : new_key->first) {
    key_collection->first.push_back(k);
  }
  for (const auto &k : new_key->second) {
    key_collection->second.push_back(k);
  }
}

// This functor template takes a pack (VariablePack or VariableFluxPack) and appends
// all the allocation statuses to the given collection of allocation statuses. We have to
// use a functor instead of a template function because template function cannot be
// partially specialized
template <typename P>
struct AllocationStatusCollector {
  static inline void Append(std::vector<bool> *alloc_status_collection, const P &pack);
};

// Specialization for VariablePack<T>
template <typename T>
struct AllocationStatusCollector<VariablePack<T>> {
  static inline void Append(std::vector<bool> *alloc_status_collection,
                            const VariablePack<T> &var_pack) {
    alloc_status_collection->insert(alloc_status_collection->end(),
                                    var_pack.alloc_status()->begin(),
                                    var_pack.alloc_status()->end());
  }
};

// Specialization for VariableFluxPack<T>
template <typename T>
struct AllocationStatusCollector<VariableFluxPack<T>> {
  static inline void Append(std::vector<bool> *alloc_status_collection,
                            const VariableFluxPack<T> &var_flux_pack) {
    alloc_status_collection->insert(alloc_status_collection->end(),
                                    var_flux_pack.alloc_status()->cbegin(),
                                    var_flux_pack.alloc_status()->cend());
    alloc_status_collection->insert(alloc_status_collection->end(),
                                    var_flux_pack.flux_alloc_status()->cbegin(),
                                    var_flux_pack.flux_alloc_status()->cend());
  }
};

// TODO(JMM): pass the coarse/fine option through the meshblockpack machinery
template <typename P, typename K, typename M, typename F>
const MeshBlockPack<P> &PackOnMesh(M &map, BlockDataList_t<Real> &block_data_,
                                   F &packing_function, PackIndexMap *map_out,
                                   std::vector<std::string> &ordered_list,
                                   const int vsize) {
  const auto nblocks = block_data_.size();

  // since the pack keys used by MeshBlockData includes the allocation status of each
  // variable, we cannot simply use the key from the first MeshBlockData, but we need to
  // get the keys from all MeshBlockData instances and concatenate them
  K total_key;
  K this_key;

  PackIndexMap pack_idx_map;
  PackIndexMap this_map;

  std::vector<bool> alloc_status_collection;

  for (size_t i = 0; i < nblocks; i++) {
    const auto &pack = packing_function(block_data_[i], this_map, this_key);
    AppendKey(&total_key, &this_key);
    AllocationStatusCollector<P>::Append(&alloc_status_collection, pack);

    if (i == 0) {
      pack_idx_map = this_map;
    } else {
      assert(this_map == pack_idx_map);
    }
  }

  auto itr = map.find(total_key);
  bool make_new_pack = false;
  if (itr == map.end()) {
    // we don't have a cached pack, need to make a new one
    make_new_pack = true;
  } else {
    // we have a cached pack, check allocation status
    if (alloc_status_collection != itr->second.alloc_status) {
      // allocation statuses differ, need to make a new pack and remove outdated one
      make_new_pack = true;
      map.erase(itr);
    }
  }

  if (make_new_pack) {
    ParArray1D<P> packs("MeshData::PackVariables::packs", nblocks);
    auto packs_host = Kokkos::create_mirror_view(packs);
    // does this cost something even when the size is zero?
    ParArray2D<int> start("MeshData::PackVariables::start", nblocks, ordered_list.size());
    auto start_host = Kokkos::create_mirror_view(start);
    ParArray2D<int> stop("MeshData::PackVariables::stop", nblocks, ordered_list.size());
    auto stop_host = Kokkos::create_mirror_view(stop);
    ParArray1D<Coordinates_t> coords("MeshData::PackVariables::coords", nblocks);
    auto coords_host = Kokkos::create_mirror_view(coords);

    for (size_t i = 0; i < nblocks; i++) {
      const auto &pack = packing_function(block_data_[i], this_map, this_key);
      for (int j = 0; j < vsize; j++) {
        start_host(i, j) = this_map[ordered_list[j]].first;
        stop_host(i, j) = this_map[ordered_list[j]].second;
      }
      for (int j = vsize; j < ordered_list.size(); j++) {
        start_host(i, j) = this_map["flux::" + ordered_list[j]].first;
        stop_host(i, j) = this_map["flux::" + ordered_list[j]].second;
      }
      packs_host(i) = pack;
    }

    std::array<int, 5> dims;
    for (int i = 0; i < 3; i++) {
      dims[i] = packs_host(0).GetDim(i + 1);
    }
    // for dims[3], set it to the max of all VariablePacks
    dims[3] = 0;
    for (int i = 0; i < nblocks; i++) {
      dims[3] = std::max(dims[3], packs_host(i).GetDim(4));
    }
    dims[4] = nblocks;

    // just in case zero size deep_copy calls still incur a penalty
    if (ordered_list.size()) {
      Kokkos::deep_copy(start, start_host);
      Kokkos::deep_copy(stop, stop_host);
    }
    Kokkos::deep_copy(packs, packs_host);

    typename M::mapped_type new_item;
    new_item.alloc_status = alloc_status_collection;
    new_item.map = pack_idx_map;
    new_item.pack = MeshBlockPack<P>(
        packs, start, stop, dims);
    itr = map.insert({total_key, new_item}).first;
  }

  if (map_out != nullptr) {
    *map_out = itr->second.map;
  }

  return itr->second.pack;
}

} // namespace pack_on_mesh_impl

/// The MeshData class is a container for cached MeshBlockPacks, i.e., it
/// contains both the pointers to the MeshBlockData of the MeshBlocks contained
/// in the object as well as maps to the cached MeshBlockPacks of VariablePacks or
/// VariableFluxPacks.

template <typename T>
class MeshData {
 public:
  MeshData() = default;

  Mesh *GetMeshPointer() const { return pmy_mesh_; }
  auto GetParentPointer() const { return GetMeshPointer(); }

  void SetMeshPointer(Mesh *pmesh) { pmy_mesh_ = pmesh; }
  void SetMeshPointer(const std::shared_ptr<MeshData<T>> &other) {
    pmy_mesh_ = other->GetMeshPointer();
  }

  void SetAllowedDt(const Real dt) const {
    for (const auto &pbd : block_data_) {
      pbd->SetAllowedDt(std::min(dt, pbd->GetBlockPointer()->NewDt()));
    }
  }

  void SetSendBuffers(const cell_centered_bvars::BufferCache_t &send_buffers) {
    send_buffers_ = send_buffers;
  }

  auto &GetSendBuffers() const { return send_buffers_; }

  void SetSetBuffers(const cell_centered_bvars::BufferCache_t &set_buffers) {
    set_buffers_ = set_buffers;
  }

  auto &GetSetBuffers() const { return set_buffers_; }

  void SetRestrictBuffers(const cell_centered_bvars::BufferCache_t &restrict_buffers) {
    restrict_buffers_ = restrict_buffers;
  }

  auto &GetRestrictBuffers() const { return restrict_buffers_; }

  TaskStatus StartReceiving(BoundaryCommSubset phase) {
    for (const auto &pbd : block_data_) {
      auto status = pbd->StartReceiving(phase);
      if (status != TaskStatus::complete) {
        PARTHENON_THROW("StartReceiving failed!");
      }
    }
    return TaskStatus::complete;
  }

  TaskStatus ClearBoundary(BoundaryCommSubset phase) {
    for (const auto &pbd : block_data_) {
      auto status = pbd->ClearBoundary(phase);
      if (status != TaskStatus::complete) {
        PARTHENON_THROW("ClearBoundary failed!");
      }
    }
    return TaskStatus::complete;
  }

  IndexRange GetBoundsI(const IndexDomain &domain) const {
    return block_data_[0]->GetBoundsI(domain);
  }
  IndexRange GetBoundsJ(const IndexDomain &domain) const {
    return block_data_[0]->GetBoundsJ(domain);
  }
  IndexRange GetBoundsK(const IndexDomain &domain) const {
    return block_data_[0]->GetBoundsK(domain);
  }

  std::vector<std::string> GetVariablesByFlag(const std::vector<MetadataFlag> &flags,
                                              bool match_all,
                                              const std::vector<int> &sparse_ids = {}) {
    std::set<std::string> unique_names;
    for (int b = 0; b < NumBlocks(); b++) {
      auto list = block_data_[b]->GetVariablesByFlag(flags, match_all, sparse_ids);
      for (auto &v : list.vars()) {
        unique_names.insert(v->label());
      }
    }
    std::vector<std::string> total_list;
    for (auto &name : unique_names) {
      total_list.push_back(name);
    }
    return total_list;
  }

  template <class... Args>
  void Add(Args &&... args) {
    for (const auto &pbd : block_data_) {
      pbd->Add(std::forward<Args>(args)...);
    }
  }

  void Set(BlockList_t blocks, const std::string &name) {
    const int nblocks = blocks.size();
    block_data_.resize(nblocks);
    SetMeshPointer(blocks[0]->pmy_mesh);
    for (int i = 0; i < nblocks; i++) {
      block_data_[i] = blocks[i]->meshblock_data.Get(name);
    }
  }

  template <typename... Args>
  void Copy(const std::shared_ptr<MeshData<T>> src, Args &&... args) {
    if (src.get() == nullptr) {
      PARTHENON_THROW("src points at null");
    }
    const int nblocks = src->NumBlocks();
    block_data_.resize(nblocks);
    for (int i = 0; i < nblocks; i++) {
      block_data_[i]->Copy(src->GetBlockData(i), std::forward<Args>(args)...);
    }
  }

  const std::shared_ptr<MeshBlockData<T>> &GetBlockData(int n) const {
    assert(n >= 0 && n < block_data_.size());
    return block_data_[n];
  }

 private:
  template <typename... Args>
  const auto &PackVariablesAndFluxesImpl(std::vector<std::string> ordered_list, const int vsize,
                                         PackIndexMap *map_out, Args &&... args) {
    auto pack_function = [&](std::shared_ptr<MeshBlockData<T>> meshblock_data,
                             PackIndexMap &map, vpack_types::StringPair &key) {
      return meshblock_data->PackVariablesAndFluxes(std::forward<Args>(args)..., map,
                                                    key);
    };

    return pack_on_mesh_impl::PackOnMesh<VariableFluxPack<T>, vpack_types::StringPair>(
        varFluxPackMap_, block_data_, pack_function, map_out, ordered_list, vsize);
  }

  template <typename... Args>
  const auto &PackVariablesImpl(std::vector<std::string> ordered_list, const int vsize,
                                PackIndexMap *map_out, bool coarse, Args &&... args) {
    auto pack_function = [&](std::shared_ptr<MeshBlockData<T>> meshblock_data,
                             PackIndexMap &map, std::vector<std::string> &key) {
      return meshblock_data->PackVariables(std::forward<Args>(args)..., map, key, coarse);
    };
    return pack_on_mesh_impl::PackOnMesh<VariablePack<T>, vpack_types::VPackKey_t>(
        varPackMap_, block_data_, pack_function, map_out, ordered_list, vsize);
  }

 public:
  // DO NOT use variatic templates here. They shadow each other

  // Pack by separate variable and flux names
  const auto &PackVariablesAndFluxes(const std::vector<std::string> &var_names,
                                     const std::vector<std::string> &flx_names,
                                     const std::vector<int> &sparse_ids,
                                     PackIndexMap &map) {
    std::vector<std::string> ord_list;
    pack_on_mesh_impl::AppendKey(&ord_list, &var_names);
    const int var_size = ord_list.size();
    pack_on_mesh_impl::AppendKey(&ord_list, &flx_names);
    return PackVariablesAndFluxesImpl(ord_list, var_size, &map, var_names, flx_names, sparse_ids);
  }
  const auto &PackVariablesAndFluxes(const std::vector<std::string> &var_names,
                                     const std::vector<std::string> &flx_names,
                                     const std::vector<int> &sparse_ids) {
    std::vector<std::string> ord_list;
    pack_on_mesh_impl::AppendKey(&ord_list, &var_names);
    const int var_size = ord_list.size();
    pack_on_mesh_impl::AppendKey(&ord_list, &flx_names);
    return PackVariablesAndFluxesImpl(ord_list, var_size, nullptr, var_names, flx_names, sparse_ids);
  }
  // no sparse ids
  const auto &PackVariablesAndFluxes(const std::vector<std::string> &var_names,
                                     const std::vector<std::string> &flx_names,
                                     PackIndexMap &map) {
    std::vector<std::string> ord_list;
    pack_on_mesh_impl::AppendKey(&ord_list, &var_names);
    const int var_size = ord_list.size();
    pack_on_mesh_impl::AppendKey(&ord_list, &flx_names);
    return PackVariablesAndFluxesImpl(ord_list, var_size, &map, var_names, flx_names);
  }
  const auto &PackVariablesAndFluxes(const std::vector<std::string> &var_names,
                                     const std::vector<std::string> &flx_names) {
    std::vector<std::string> ord_list;
    pack_on_mesh_impl::AppendKey(&ord_list, &var_names);
    const int var_size = ord_list.size();
    pack_on_mesh_impl::AppendKey(&ord_list, &flx_names);
    return PackVariablesAndFluxesImpl(ord_list, var_size, nullptr, var_names, flx_names);
  }
  // Pack by either the same variable and flux names, or by metadata flags
  const auto &PackVariablesAndFluxes(const std::vector<std::string> &names,
                                     const std::vector<int> &sparse_ids,
                                     PackIndexMap &map) {
    return PackVariablesAndFluxesImpl(names, names.size(), &map, names, sparse_ids);
  }
  const auto &PackVariablesAndFluxes(const std::vector<std::string> &names,
                                     const std::vector<int> &sparse_ids) {
    return PackVariablesAndFluxesImpl(names, names.size(), nullptr, names, sparse_ids);
  }
  const auto &PackVariablesAndFluxes(const std::vector<MetadataFlag> &flags,
                                     const std::vector<int> &sparse_ids,
                                     PackIndexMap &map) {
    std::vector<std::string> names;
    return PackVariablesAndFluxesImpl(names, 0, &map, flags, sparse_ids);
  }
  const auto &PackVariablesAndFluxes(const std::vector<MetadataFlag> &flags,
                                     const std::vector<int> &sparse_ids) {
    std::vector<std::string> names;
    return PackVariablesAndFluxesImpl(names, 0, nullptr, flags, sparse_ids);
  }
  // no sparse ids
  const auto &PackVariablesAndFluxes(const std::vector<std::string> &names,
                                     PackIndexMap &map) {
    return PackVariablesAndFluxesImpl(names, names.size(), &map, names);
  }
  const auto &PackVariablesAndFluxes(const std::vector<std::string> &names) {
    return PackVariablesAndFluxesImpl(names, names.size(), nullptr, names);
  }
  const auto &PackVariablesAndFluxes(const std::vector<MetadataFlag> &flags,
                                     PackIndexMap &map) {
    std::vector<std::string> names;
    return PackVariablesAndFluxesImpl(names, 0, &map, flags);
  }
  const auto &PackVariablesAndFluxes(const std::vector<MetadataFlag> &flags) {
    std::vector<std::string> names;
    return PackVariablesAndFluxesImpl(names, 0, nullptr, flags);
  }
  // only sparse ids
  const auto &PackVariablesAndFluxes(const std::vector<int> &sparse_ids,
                                     PackIndexMap &map) {
    std::vector<std::string> names;
    return PackVariablesAndFluxesImpl(names, 0, &map, sparse_ids);
  }
  const auto &PackVariablesAndFluxes(const std::vector<int> &sparse_ids) {
    std::vector<std::string> names;
    return PackVariablesAndFluxesImpl(names, 0, nullptr, sparse_ids);
  }
  // No nothing
  const auto &PackVariablesAndFluxes(PackIndexMap &map) {
    std::vector<std::string> names;
    return PackVariablesAndFluxesImpl(names, 0, &map);
  }
  const auto &PackVariablesAndFluxes() {
    std::vector<std::string> names;
    return PackVariablesAndFluxesImpl(names, 0, nullptr);
  }

  // As above, DO NOT use variatic templates here. They shadow each other.
  // covers names and metadata flags
  const auto &PackVariables(const std::vector<std::string> names,
                            const std::vector<int> &sparse_ids, PackIndexMap &map,
                            bool coarse = false) {
    return PackVariablesImpl(names, names.size(), &map, coarse, names, sparse_ids);
  }
  const auto &PackVariables(const std::vector<std::string> names,
                            const std::vector<int> &sparse_ids, bool coarse = false) {
    return PackVariablesImpl(names, names.size(), nullptr, coarse, names, sparse_ids);
  }
  const auto &PackVariables(const std::vector<MetadataFlag> flags,
                            const std::vector<int> &sparse_ids, PackIndexMap &map,
                            bool coarse = false) {
    std::vector<std::string> names;
    return PackVariablesImpl(names, 0, &map, coarse, flags, sparse_ids);
  }
  const auto &PackVariables(const std::vector<MetadataFlag> flags,
                            const std::vector<int> &sparse_ids, bool coarse = false) {
    std::vector<std::string> names;
    return PackVariablesImpl(names, 0, nullptr, coarse, flags, sparse_ids);
  }
  // no sparse ids
  const auto &PackVariables(const std::vector<std::string> names, PackIndexMap &map,
                            bool coarse = false) {
    return PackVariablesImpl(names, names.size(), &map, coarse, names);
  }
  const auto &PackVariables(const std::vector<std::string> names, bool coarse = false) {
    return PackVariablesImpl(names, names.size(), nullptr, coarse, names);
  }
  const auto &PackVariables(const std::vector<MetadataFlag> flags, PackIndexMap &map,
                            bool coarse = false) {
    std::vector<std::string> names;
    return PackVariablesImpl(names, 0, &map, coarse, flags);
  }
  const auto &PackVariables(const std::vector<MetadataFlag> flags, bool coarse = false) {
    std::vector<std::string> names;
    return PackVariablesImpl(names, 0, nullptr, coarse, flags);
  }
  // No names or flags
  const auto &PackVariables(const std::vector<int> &sparse_ids, PackIndexMap &map,
                            bool coarse = false) {
    std::vector<std::string> names;
    return PackVariablesImpl(names, 0, &map, coarse, sparse_ids);
  }
  const auto &PackVariables(const std::vector<int> &sparse_ids, bool coarse = false) {
    std::vector<std::string> names;
    return PackVariablesImpl(names, 0, nullptr, coarse, sparse_ids);
  }
  // no nothing
  const auto &PackVariables(PackIndexMap &map, bool coarse = false) {
    std::vector<std::string> names;
    return PackVariablesImpl(names, 0, &map, coarse);
  }
  const auto &PackVariables(bool coarse = false) {
    std::vector<std::string> names;
    return PackVariablesImpl(names, 0, nullptr, coarse);
  }

  void ClearCaches() {
    block_data_.clear();
    varPackMap_.clear();
    varFluxPackMap_.clear();
    send_buffers_ = cell_centered_bvars::BufferCache_t{};
    set_buffers_ = cell_centered_bvars::BufferCache_t{};
    restrict_buffers_ = cell_centered_bvars::BufferCache_t{};
  }

  int NumBlocks() const { return block_data_.size(); }

  bool operator==(MeshData<T> &cmp) const {
    const int nblocks = block_data_.size();
    const int nblocks_cmp = cmp.NumBlocks();
    if (nblocks != nblocks_cmp) return false;

    for (int i = 0; i < nblocks; i++) {
      if (!(*block_data_[i] == *(cmp.GetBlockData(i)))) return false;
    }
    return true;
  }

  bool Contains(const std::vector<std::string> &names) const {
    for (const auto &b : block_data_) {
      if (!b->Contains(names)) return false;
    }
    return true;
  }

 private:
  Mesh *pmy_mesh_;
  BlockDataList_t<T> block_data_;
  // caches for packs
  MapToMeshBlockVarPack<T> varPackMap_;
  MapToMeshBlockVarFluxPack<T> varFluxPackMap_;
  // caches for boundary information
  cell_centered_bvars::BufferCache_t send_buffers_{};
  cell_centered_bvars::BufferCache_t set_buffers_{};
  cell_centered_bvars::BufferCache_t restrict_buffers_{};
};

} // namespace parthenon

#endif // INTERFACE_MESH_DATA_HPP_
