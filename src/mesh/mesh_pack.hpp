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
//=======================================================================================
#ifndef MESH_MESH_PACK_HPP_
#define MESH_MESH_PACK_HPP_

#include <array>
#include <utility>

#include "interface/container.hpp"
#include "interface/variable_pack.hpp"
#include "mesh/mesh.hpp" // TODO(JMM): Replace with forward declaration?

namespace parthenon {

// a separate dims array removes a branch case in `GetDim`
template <typename T>
class MeshPack {
 public:
  MeshPack() = default;
  MeshPack(const ParArray1D<T> view,
           const std::array<int, 5> dims)
    : v_(view), dims_(dims) {}
  KOKKOS_FORCEINLINE_FUNCTION
  auto &operator()(const int block) const {
    return v_(block);
  }
  KOKKOS_FORCEINLINE_FUNCTION
  auto &operator()(const int block, const int n) const {
    return v_(block)(n);
  }
  KOKKOS_FORCEINLINE_FUNCTION
  auto &operator()(const int block, const int n,
                   const int k, const int j, const int i) const {
    reutrn v_(block)(n)(k,j,i);
  }
  KOKKOS_FORCEINLINE_FUNCTION
  int GetDim(const int i) const {
    assert(i > 0 && i < 6);
    return dims_[i - 1];
  }
  KOKKOS_FORCEINLINE_FUNCTION
  int GetNdim() const { return v_(0).GetNdim(); }
  KOKKOS_FORCEINLINE_FUNCTION
  int GetSparse(const int n) const {
    return v_(0).GetSparse(n);
  }

 private:
  ParArray1D<T> v_;
  std::array<int, 5> dims_;
};

template <typename T>
using ViewOfPacks = ParArray1D<VariablePack<T>>;
template <typename T>
using ViewOfFluxPacks = ParArray1D<VariableFluxPack<T>>;
template <typename T>
using MeshVariablePack = MeshPack<VariablePack<T>>;
template <typename T>
using MeshVariableFluxPack = MeshPack<VariableFluxPack<T>>;

// Uses Real only because meshblock only owns real containers
template <typename... Args>
auto MakeMeshVariablePack(Mesh* pmesh,
                          const std::string& container_name,
                          Args... args,
                          PackIndexMap &vmap) {
  int nblocks = pmesh->GetNumMeshBlocksThisRank();
  ViewOfPacks<T> packs("MakeMeshVariablePack::cv", nblocks);
  auto packs_host = packs.GetHostMirror();
  
  // TODO(JMM): Update to Andrew's C++ std::list when available
  MeshBlock *pmb = pmesh->pblock;
  int b = 0;
  while (pmb != nullptr) { 
    auto &container = pmb->real_containers.Get(container_name);
    packs_host(b) = container->PackVariables(std::forward<Args>(args)...,vmap);
    pmb = pmb->next;
    b++;
  }
  auto pack = packs_host(0);
  packs.DeepCopy(packs_host);

  std::array<int, 5> dims;
  for (int i = 0; i < 4; i++) {
    dims[i] = pack.GetDim(i+1);
  }
  dims[4] = nblock;

  return MeshVariablePack(packs,pack.GetSparseIDs(),dims);
}



} // namespace parthenon

#endif // MESH_MESH_PACK_HPP_
