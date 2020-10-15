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
#ifndef MESH_MESHBLOCK_PACK_HPP_
#define MESH_MESHBLOCK_PACK_HPP_

#include <array>
#include <functional>
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

// a separate dims array removes a branch case in `GetDim`
// TODO(JMM): Using one IndexShape because its the same for all
// meshblocks. This needs careful thought before sticking with it.
template <typename T>
class MeshBlockPack {
 public:
  MeshBlockPack() = default;
  MeshBlockPack(const ParArray1D<T> view, const IndexShape shape,
                const ParArray1D<Coordinates_t> coordinates,
                const std::array<int, 5> dims)
      : v_(view), cellbounds(shape), coords(coordinates), dims_(dims) {}
  KOKKOS_FORCEINLINE_FUNCTION
  auto &operator()(const int block) const { return v_(block); }
  KOKKOS_FORCEINLINE_FUNCTION
  auto &operator()(const int block, const int n) const { return v_(block)(n); }
  KOKKOS_FORCEINLINE_FUNCTION
  auto &operator()(const int block, const int n, const int k, const int j,
                   const int i) const {
    return v_(block)(n)(k, j, i);
  }
  KOKKOS_FORCEINLINE_FUNCTION
  int GetDim(const int i) const {
    assert(i > 0 && i < 6);
    return dims_[i - 1];
  }
  KOKKOS_FORCEINLINE_FUNCTION
  int GetNdim() const { return v_(0).GetNdim(); }
  KOKKOS_FORCEINLINE_FUNCTION
  int GetSparse(const int n) const { return v_(0).GetSparse(n); }

  // TODO(JMM): Also include mesh domain object?
  IndexShape cellbounds;
  ParArray1D<Coordinates_t> coords;

 private:
  ParArray1D<T> v_;
  std::array<int, 5> dims_;
};

template <typename T>
using ViewOfPacks = ParArray1D<VariablePack<T>>;
template <typename T>
using ViewOfFluxPacks = ParArray1D<VariableFluxPack<T>>;

template <typename T>
using MeshBlockVarPack = MeshBlockPack<VariablePack<T>>;
template <typename T>
using MeshBlockVarFluxPack = MeshBlockPack<VariableFluxPack<T>>;

template <typename T>
using VarPackingFunc = std::function<std::vector<MeshBlockVarPack<T>>(Mesh *)>;
template <typename T>
using FluxPackingFunc = std::function<std::vector<MeshBlockVarFluxPack<T>>(Mesh *)>;

// TODO(JMM): Should this be cached?
namespace mesh_pack_impl {

// TODO(JMM): blocks data type might change
template <typename T, typename F>
auto PackMesh(BlockList_t &blocks, F &packing_function) {
  int nblocks = blocks.size();
  ParArray1D<T> packs("MakeMeshVariablePack::view", nblocks);
  auto packs_host = Kokkos::create_mirror_view(packs);
  ParArray1D<Coordinates_t> coords("MakeMeshBlockPackVariable::coords", nblocks);
  auto coords_host = Kokkos::create_mirror_view(coords);

  int b = 0;
  for (auto &pmb : blocks) {
    coords_host(b) = pmb->coords;
    packs_host(b) = packing_function(pmb.get());
    b++;
  }

  std::array<int, 5> dims;
  for (int i = 0; i < 4; i++) {
    dims[i] = packs_host(0).GetDim(i + 1);
  }
  dims[4] = nblocks;

  Kokkos::deep_copy(packs, packs_host);
  Kokkos::deep_copy(coords, coords_host);

  auto cellbounds = blocks.front()->cellbounds;

  return MeshBlockPack<T>(packs, cellbounds, coords, dims);
}
} // namespace mesh_pack_impl

// Uses Real only because meshblock only owns real containers
template <typename T, typename... Args>
auto PackVariablesOnMesh(T &blocks, const std::string &container_name, Args &&... args) {
  using namespace mesh_pack_impl;
  auto pack_function = [&](MeshBlock *pmb) {
    auto container = pmb->real_containers.Get(container_name);
    return container->PackVariables(std::forward<Args>(args)...);
  };
  return PackMesh<VariablePack<Real>>(blocks, pack_function);
}
template <typename T, typename... Args>
auto PackVariablesAndFluxesOnMesh(T &blocks, const std::string &container_name,
                                  Args &&... args) {
  using namespace mesh_pack_impl;
  auto pack_function = [&](MeshBlock *pmb) {
    auto container = pmb->real_containers.Get(container_name);
    return container->PackVariablesAndFluxes(std::forward<Args>(args)...);
  };
  return PackMesh<VariableFluxPack<Real>>(blocks, pack_function);
}
} // namespace parthenon

#endif // MESH_MESHBLOCK_PACK_HPP_
