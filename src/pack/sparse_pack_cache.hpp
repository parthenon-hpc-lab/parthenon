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
#ifndef PACK_SPARSE_PACK_CACHE_HPP_
#define PACK_SPARSE_PACK_CACHE_HPP_

#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "pack/pack_descriptor.hpp"
#include "pack/sparse_pack_base.hpp"

namespace parthenon {
// Object for cacheing sparse packs in MeshData and MeshBlockData objects. This
// handles checking for a pre-existing pack and creating a new SparsePackBase if
// a cached pack is unavailable. Essentially, this operates as a map from
// `PackDescriptor` to `SparsePackBase`
class SparsePackCache {
 public:
  std::size_t size() const {
    std::size_t size{0};
    for (const auto &[key, desc_pack_map] : pack_map) {
      size += desc_pack_map.size();
    }
    return size;
  }

  void clear() { pack_map.clear(); }

 protected:
  template <class T>
  SparsePackBase &Get(T *pmd, const impl::PackDescriptor &desc,
                      const std::vector<bool> &include_block);

  template <class T>
  SparsePackBase &BuildAndAdd(T *pmd, const impl::PackDescriptor &desc,
                              const std::vector<bool> &include_block);
  // For a given pack descriptor, this contains a map from a vector containing the
  // requested included blocks to a cached pack, along with the allocation status of
  // varirables in the pack
  using desc_pack_map_t =
      std::map<std::vector<bool>, std::tuple<SparsePackBase, SparsePackBase::alloc_t>>;
  // Map from pack descriptor id to map from included blocks
  using pack_map_t = std::unordered_map<std::string, desc_pack_map_t>;
  pack_map_t pack_map;

  friend class SparsePackBase;
};

} // namespace parthenon

#endif // PACK_SPARSE_PACK_CACHE_HPP_
