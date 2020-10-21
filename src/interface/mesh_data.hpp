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

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mesh/meshblock.hpp"
#include "mesh/meshblock_pack.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

class Mesh;
template <typename T>
class MeshBlockData;

template <typename T>
class MeshData {
 public:
  MeshData() = default;
  // MeshData(const Mesh *pmesh, const std::string &name);
  // explicit MeshData(const Mesh *pmesh) : MeshData(pmesh, "base") { }

  Mesh *GetMeshPointer() const { return pmy_mesh_; }

  void SetMeshPointer(Mesh *pmesh) { pmy_mesh_ = pmesh; }
  void SetMeshPointer(const std::shared_ptr<MeshData<T>> &other) {
    pmy_mesh_ = other->GetMeshPointer();
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

  void Copy(const std::shared_ptr<MeshData<T>> src) {
    const int nblocks = src->NumBlocks();
    block_data_.resize(nblocks);
    for (int i = 0; i < nblocks; i++) {
      block_data_[i]->Copy(src->GetBlockData(i));
    }
  }

  std::shared_ptr<MeshBlockData<T>> &GetBlockData(int n) {
    assert(n >= 0 && n < block_data_.size());
    return block_data_[n];
  }

  template <class... Args>
  MeshBlockVarPack<T> PackVariables(Args &&... args) {
    std::vector<std::string> key;
    auto vpack = block_data_[0]->PackVariables(std::forward<Args>(args)..., key);
    auto kvpair = varPackMap_.find(key);
    if (kvpair == varPackMap_.end()) {
      int nblocks = block_data_.size();
      ViewOfPacks<T> packs("MeshData::PackVariables::packs", nblocks);
      auto packs_host = Kokkos::create_mirror_view(packs);
      ParArray1D<Coordinates_t> coords("MeshData::PackVariables::coords", nblocks);
      auto coords_host = Kokkos::create_mirror_view(coords);
      for (int i = 0; i < nblocks; i++) {
        packs_host(i) = block_data_[i]->PackVariables(std::forward<Args>(args)..., key);
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
      auto mbp = MeshBlockVarPack<T>(packs, cellbounds, coords, dims);
      varPackMap_[key] = mbp;
      return mbp;
    }
    return kvpair->second;
  }
  template <class... Args>
  MeshBlockVarFluxPack<T> PackVariablesAndFluxes(Args &&... args) {
    vpack_types::StringPair key;
    auto vpack = block_data_[0]->PackVariablesAndFluxes(std::forward<Args>(args)..., key);
    auto kvpair = varFluxPackMap_.find(key);
    if (kvpair == varFluxPackMap_.end()) {
      int nblocks = block_data_.size();
      ViewOfFluxPacks<T> packs("MeshData::PackVariables::packs", nblocks);
      auto packs_host = Kokkos::create_mirror_view(packs);
      ParArray1D<Coordinates_t> coords("MeshData::PackVariables::coords", nblocks);
      auto coords_host = Kokkos::create_mirror_view(coords);
      for (int i = 0; i < nblocks; i++) {
        packs_host(i) =
            block_data_[i]->PackVariablesAndFluxes(std::forward<Args>(args)..., key);
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
      auto mbp = MeshBlockVarFluxPack<T>(packs, cellbounds, coords, dims);
      varFluxPackMap_[key] = mbp;
      return mbp;
    }
    return kvpair->second;
  }

  void ClearCaches() {
    varPackMap_.clear();
    varFluxPackMap_.clear();
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

 private:
  Mesh *pmy_mesh_;
  std::vector<std::shared_ptr<MeshBlockData<T>>> block_data_;
  // caches for packs
  MapToMeshBlockPack<T> varPackMap_;
  MapToMeshBlockFluxPack<T> varFluxPackMap_;
};

} // namespace parthenon

#endif // INTERFACE_MESH_DATA_HPP_
