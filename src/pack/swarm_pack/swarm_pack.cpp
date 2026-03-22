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
#include "pack/swarm_pack/swarm_pack.hpp"

namespace parthenon {
namespace impl {

template <typename TYPE, class T>
SwarmPackBase<TYPE> GetSwarmPack(T *pmd, const SwarmPackDescriptor<TYPE> &desc) {
  return SwarmPackBase<TYPE>::GetPack(pmd, desc);
}

template SwarmPackBase<Real> GetSwarmPack(MeshData<Real> *pmd,
                                          const SwarmPackDescriptor<Real> &desc);
template SwarmPackBase<Real> GetSwarmPack(MeshBlockData<Real> *pmd,
                                          const SwarmPackDescriptor<Real> &desc);
template SwarmPackBase<int> GetSwarmPack(MeshData<Real> *pmd,
                                         const SwarmPackDescriptor<int> &desc);
template SwarmPackBase<int> GetSwarmPack(MeshBlockData<Real> *pmd,
                                         const SwarmPackDescriptor<int> &desc);
template SwarmPackBase<std::uint64_t>
GetSwarmPack(MeshData<Real> *pmd, const SwarmPackDescriptor<std::uint64_t> &desc);
template SwarmPackBase<std::uint64_t>
GetSwarmPack(MeshBlockData<Real> *pmd, const SwarmPackDescriptor<std::uint64_t> &desc);

} // namespace impl
} // namespace parthenon
