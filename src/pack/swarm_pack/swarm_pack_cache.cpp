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

#include <cstdint>

#include "interface/mesh_data.hpp"
#include "interface/meshblock_data.hpp"
#include "pack/swarm_pack/swarm_pack_cache.hpp"

namespace parthenon {

template <typename TYPE>
template <class T>
SwarmPackBase<TYPE> &
SwarmPackCache<TYPE>::Get(T *pmd, const impl::SwarmPackDescriptor<TYPE> &desc) {
  if (pack_map.count(desc.identifier) > 0) {
    auto &pack = pack_map[desc.identifier];
    SwarmPackBase<TYPE>::BuildSupplemental(pmd, desc, pack);
    return pack;
  }
  return BuildAndAdd(pmd, desc);
}

template <typename TYPE>
template <class T>
SwarmPackBase<TYPE> &
SwarmPackCache<TYPE>::BuildAndAdd(T *pmd, const impl::SwarmPackDescriptor<TYPE> &desc) {
  pack_map[desc.identifier] = SwarmPackBase<TYPE>::Build(pmd, desc);
  return pack_map[desc.identifier];
}

template SwarmPackBase<Real> &
SwarmPackCache<Real>::Get<MeshData<Real>>(MeshData<Real> *,
                                          const impl::SwarmPackDescriptor<Real> &);
template SwarmPackBase<Real> &
SwarmPackCache<Real>::Get<MeshBlockData<Real>>(MeshBlockData<Real> *,
                                               const impl::SwarmPackDescriptor<Real> &);
template SwarmPackBase<int> &
SwarmPackCache<int>::Get<MeshData<Real>>(MeshData<Real> *,
                                         const impl::SwarmPackDescriptor<int> &);
template SwarmPackBase<int> &
SwarmPackCache<int>::Get<MeshBlockData<Real>>(MeshBlockData<Real> *,
                                              const impl::SwarmPackDescriptor<int> &);
template SwarmPackBase<std::uint64_t> &SwarmPackCache<std::uint64_t>::Get<MeshData<Real>>(
    MeshData<Real> *, const impl::SwarmPackDescriptor<std::uint64_t> &);
template SwarmPackBase<std::uint64_t> &
SwarmPackCache<std::uint64_t>::Get<MeshBlockData<Real>>(
    MeshBlockData<Real> *, const impl::SwarmPackDescriptor<std::uint64_t> &);

template SwarmPackBase<Real> &SwarmPackCache<Real>::BuildAndAdd<MeshData<Real>>(
    MeshData<Real> *, const impl::SwarmPackDescriptor<Real> &);
template SwarmPackBase<Real> &SwarmPackCache<Real>::BuildAndAdd<MeshBlockData<Real>>(
    MeshBlockData<Real> *, const impl::SwarmPackDescriptor<Real> &);
template SwarmPackBase<int> &
SwarmPackCache<int>::BuildAndAdd<MeshData<Real>>(MeshData<Real> *,
                                                 const impl::SwarmPackDescriptor<int> &);
template SwarmPackBase<int> &SwarmPackCache<int>::BuildAndAdd<MeshBlockData<Real>>(
    MeshBlockData<Real> *, const impl::SwarmPackDescriptor<int> &);
template SwarmPackBase<std::uint64_t> &
SwarmPackCache<std::uint64_t>::BuildAndAdd<MeshData<Real>>(
    MeshData<Real> *, const impl::SwarmPackDescriptor<std::uint64_t> &);
template SwarmPackBase<std::uint64_t> &
SwarmPackCache<std::uint64_t>::BuildAndAdd<MeshBlockData<Real>>(
    MeshBlockData<Real> *, const impl::SwarmPackDescriptor<std::uint64_t> &);

} // namespace parthenon
