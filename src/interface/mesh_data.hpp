//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
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
#include <string>
#include <utility>
#include <vector>

#include "bvals/cc/bvals_cc_in_one.hpp"
#include "mesh/domain.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/meshblock_pack.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

class Mesh;
template <typename T>
class MeshBlockData;

template <typename T>
using BlockDataList_t = std::vector<std::shared_ptr<MeshBlockData<T>>>;

namespace pack_on_mesh_impl {
// TODO(JMM): pass the coarse/fine option through the meshblockpack machinery
template <typename T, typename K, typename M, typename F>
MeshBlockPack<T> PackOnMesh(K &key, M &map, BlockDataList_t<Real> &block_data_,
                            F &packing_function) {
  auto kvpair = map.find(key);
  if (kvpair == map.end()) {
    int nblocks = block_data_.size();
    ParArray1D<T> packs("MeshData::PackVariables::packs", nblocks);
    auto packs_host = Kokkos::create_mirror_view(packs);
    ParArray1D<Coordinates_t> coords("MeshData::PackVariables::coords", nblocks);
    auto coords_host = Kokkos::create_mirror_view(coords);
    for (int i = 0; i < nblocks; i++) {
      packs_host(i) = packing_function(block_data_[i]);
      coords_host(i) = block_data_[i]->GetBlockPointer()->coords;
    }
    std::array<int, 5> dims;
    for (int i = 0; i < 4; i++) {
      dims[i] = packs_host(0).GetDim(i + 1);
    }
    dims[4] = nblocks;

    Kokkos::deep_copy(packs, packs_host);
    Kokkos::deep_copy(coords, coords_host);

    auto cellbounds = block_data_[0]->GetBlockPointer()->cellbounds;
    auto mbp = MeshBlockPack<T>(packs, cellbounds, coords, dims);
    map[key] = mbp;
    return mbp;
  }
  return kvpair->second;
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

  IndexRange GetBoundsI(const IndexDomain &domain) const {
    return block_data_[0]->GetBoundsI(domain);
  }
  IndexRange GetBoundsJ(const IndexDomain &domain) const {
    return block_data_[0]->GetBoundsJ(domain);
  }
  IndexRange GetBoundsK(const IndexDomain &domain) const {
    return block_data_[0]->GetBoundsK(domain);
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

  template <typename... Args>
  auto PackVariables(Args &&... args) {
    auto pack_function = [&](std::shared_ptr<MeshBlockData<T>> meshblock_data) {
      return meshblock_data->PackVariables(std::forward<Args>(args)...);
    };
    std::vector<std::string> key;
    auto vpack = block_data_[0]->PackVariables(std::forward<Args>(args)..., key);
    return pack_on_mesh_impl::PackOnMesh<VariablePack<T>>(key, varPackMap_, block_data_,
                                                          pack_function);
  }

  template <typename... Args>
  auto PackVariablesAndFluxes(Args &&... args) {
    auto pack_function = [&](std::shared_ptr<MeshBlockData<T>> meshblock_data) {
      return meshblock_data->PackVariablesAndFluxes(std::forward<Args>(args)...);
    };
    vpack_types::StringPair key;
    auto vpack = block_data_[0]->PackVariablesAndFluxes(std::forward<Args>(args)..., key);
    return pack_on_mesh_impl::PackOnMesh<VariableFluxPack<T>>(key, varFluxPackMap_,
                                                              block_data_, pack_function);
  }

  void ClearCaches() {
    block_data_.clear();
    varPackMap_.clear();
    varFluxPackMap_.clear();
    send_buffers_ = cell_centered_bvars::BufferCache_t{};
    set_buffers_ = cell_centered_bvars::BufferCache_t{};
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
};

using MeshDataCollection = DataCollection<MeshData<Real>>;

} // namespace parthenon

#endif // INTERFACE_MESH_DATA_HPP_
