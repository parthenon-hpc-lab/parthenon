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
//=======================================================================================
#ifndef MESH_MESHBLOCK_PACK_HPP_
#define MESH_MESHBLOCK_PACK_HPP_

#include <array>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "coordinates/coordinates.hpp"
#include "interface/variable_pack.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/domain.hpp"
#include "mesh/meshblock.hpp" // TODO(JMM): Replace with forward declaration?

namespace parthenon {

class Mesh;
// class MeshBlock;

// a separate dims array removes a branch case in `GetDim`
// TODO(JMM): Using one IndexShape because its the same for all
// meshblocks. This needs careful thought before sticking with it.
template <typename T>
class MeshBlockPack {
 public:
  using pack_type = T;

  MeshBlockPack() = default;
  MeshBlockPack(const ParArray1D<T> view,
                const std::array<int, 5> dims)
      : v_(view), dims_(dims),
        ndim_((dims[2] > 1 ? 3 : (dims[1] > 1 ? 2 : 1))) {}

  KOKKOS_FORCEINLINE_FUNCTION
  auto &operator()(const int block) const { return v_(block); }
  KOKKOS_FORCEINLINE_FUNCTION
  auto &operator()(const int block, const int n) const { return v_(block)(n); }
  KOKKOS_FORCEINLINE_FUNCTION
  auto &operator()(const int block, const int n, const int k, const int j,
                   const int i) const {
    return v_(block)(n)(k, j, i);
  }

  KOKKOS_FORCEINLINE_FUNCTION bool IsSparseIDAllocated(const int block,
                                                       const int var) const {
    return v_(block).GetDim(4) > var && v_(block)(var).is_allocated();
  }

  KOKKOS_FORCEINLINE_FUNCTION
  int GetDim(const int i) const {
    assert(i > 0 && i < 6);
    return dims_[i - 1];
  }
  KOKKOS_FORCEINLINE_FUNCTION
  int GetNdim() const { return ndim_; }
  KOKKOS_FORCEINLINE_FUNCTION
  int GetSparse(const int n) const { return v_(0).GetSparse(n); }

  KOKKKOS_FORCEINLINE_FUNCTION
  Coordinates_t GetCoords(const int i) {
    return v_(i).GetCoords();
  }

 private:
  ParArray1D<T> v_;
  std::array<int, 5> dims_;
  int ndim_;
};

template <typename T>
using MeshBlockVarPack = MeshBlockPack<VariablePack<T>>;
template <typename T>
using MeshBlockVarFluxPack = MeshBlockPack<VariableFluxPack<T>>;

template <typename T>
using MapToMeshBlockVarPack =
    std::map<std::vector<std::string>, PackAndIndexMap<MeshBlockVarPack<T>>>;
template <typename T>
using MapToMeshBlockVarFluxPack =
    std::map<vpack_types::StringPair, PackAndIndexMap<MeshBlockVarFluxPack<T>>>;

} // namespace parthenon

#endif // MESH_MESHBLOCK_PACK_HPP_
