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

// This file was made in part with generative AI

#include "pack/swarm_pack/swarm_pack_cache.hpp"
#include "interface/mesh_data.hpp"
#include "interface/meshblock_data.hpp"
#include "pack/swarm_pack/swarm_pack_types.hpp"

namespace parthenon {

using namespace impl;

//----------------------------------------------------------------------------------------
template <typename TYPE>
template <class T>
SwarmPackBase<TYPE> &
SwarmPackCache<TYPE>::Get(T *pmd, const impl::SwarmPackDescriptor<TYPE> &desc) {
  if (pack_map.count(desc.identifier) > 0) {
    // Cached version is not stale, so just return a reference to it
    auto &pack = pack_map[desc.identifier];
    SwarmPackBase<TYPE>::BuildSupplemental(pmd, desc, pack);
    return pack;
  }
  return BuildAndAdd(pmd, desc);
}

#define INSTANTIATE_GET(TYPE)                                                            \
  template SwarmPackBase<TYPE> &SwarmPackCache<TYPE>::Get<MeshData<Real>>(               \
      MeshData<Real> *, const impl::SwarmPackDescriptor<TYPE> &);                        \
  template SwarmPackBase<TYPE> &SwarmPackCache<TYPE>::Get<MeshBlockData<Real>>(          \
      MeshBlockData<Real> *, const impl::SwarmPackDescriptor<TYPE> &);
PARTHENON_SWARM_PACK_TYPES(INSTANTIATE_GET)
#undef INSTANTIATE_GET

//----------------------------------------------------------------------------------------
template <typename TYPE>
template <class T>
SwarmPackBase<TYPE> &
SwarmPackCache<TYPE>::BuildAndAdd(T *pmd, const impl::SwarmPackDescriptor<TYPE> &desc) {
  pack_map[desc.identifier] = SwarmPackBase<TYPE>::Build(pmd, desc);
  return pack_map[desc.identifier];
}

#define INSTANTIATE_GET(TYPE)                                                            \
  template SwarmPackBase<TYPE> &SwarmPackCache<TYPE>::BuildAndAdd<MeshData<Real>>(       \
      MeshData<Real> *, const impl::SwarmPackDescriptor<TYPE> &);                        \
  template SwarmPackBase<TYPE> &SwarmPackCache<TYPE>::BuildAndAdd<MeshBlockData<Real>>(  \
      MeshBlockData<Real> *, const impl::SwarmPackDescriptor<TYPE> &);
PARTHENON_SWARM_PACK_TYPES(INSTANTIATE_GET)
#undef INSTANTIATE_GET

} // namespace parthenon
