//========================================================================================
// (C) (or copyright) 2020-2026. Triad National Security, LLC. All rights reserved.
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
#ifndef PACK_SWARM_PACK_SWARM_PACK_CACHE_HPP_
#define PACK_SWARM_PACK_SWARM_PACK_CACHE_HPP_

// This file was made in part with generative AI

#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "pack/swarm_pack/swarm_pack_base.hpp"

namespace parthenon {
// Object for cacheing swarm packs in MeshData and MeshBlockData objects. This
// handles checking for a pre-existing pack and creating a new SwarmPackBase if
// a cached pack is unavailable. Essentially, this operates as a map from
// `SwarmPackDescriptor` to `SwarmPackBase`
template <typename TYPE>
class SwarmPackCache {
 public:
  std::size_t size() const { return pack_map.size(); }

  void clear() { pack_map.clear(); }

 protected:
  template <class T>
  SwarmPackBase<TYPE> &Get(T *pmd, const impl::SwarmPackDescriptor<TYPE> &desc);

  template <class T>
  SwarmPackBase<TYPE> &BuildAndAdd(T *pmd, const impl::SwarmPackDescriptor<TYPE> &desc);

  using pack_map_t = std::unordered_map<std::string, SwarmPackBase<TYPE>>;
  pack_map_t pack_map;

  friend class SwarmPackBase<TYPE>;
};

} // namespace parthenon

#endif // PACK_SWARM_PACK_SWARM_PACK_CACHE_HPP_
