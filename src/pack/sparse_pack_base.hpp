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
#ifndef PACK_SPARSE_PACK_BASE_HPP_
#define PACK_SPARSE_PACK_BASE_HPP_

#include <string>
#include <unordered_map>
#include <vector>

#include "coordinates/coordinates.hpp"
#include "interface/variable.hpp"
#include "interface/variable_state.hpp"
#include "kokkos_abstraction.hpp"
#include "pack/pack_descriptor.hpp"

namespace parthenon {
// Map for going from variable names to sparse pack variable indices
using SparsePackIdxMap = std::unordered_map<std::string, std::size_t>;

class SparsePackBase {
 public:
  SparsePackBase() = default;
  virtual ~SparsePackBase() = default;

 protected:
  friend class SparsePackCache;

  using alloc_t = std::vector<int>;
  using include_t = std::vector<bool>;
  using pack_t = ParArray3DRaw<ParArray3D<Real, VariableState>>;
  using pack_h_t = typename pack_t::HostMirror;
  using bounds_t = ParArray3D<int>;
  using bounds_h_t = typename bounds_t::HostMirror;
  using block_props_t = ParArray2D<int>;
  using block_props_h_t = typename block_props_t::HostMirror;
  using coords_t = ParArray1DRaw<ParArray0D<Coordinates_t>>;

  static constexpr int physical_bnd_flag = -2000;

  // Returns a SparsePackBase object that is either newly created or taken
  // from the cache in pmd. The cache itself handles the all of this logic
  template <class T>
  static SparsePackBase GetPack(T *pmd, const impl::PackDescriptor &desc,
                                const std::vector<bool> &include_block);

  // Return a map from variable names to pack variable indices
  static SparsePackIdxMap GetIdxMap(const impl::PackDescriptor &desc);

  // Get a list of booleans of the allocation status of every variable in pmd matching the
  // PackDescriptor desc
  template <class T>
  static alloc_t GetAllocStatus(T *pmd, const impl::PackDescriptor &desc,
                                const std::vector<bool> &include_block);

  // Actually build a `SparsePackBase` (i.e. create a view of views, fill on host, and
  // deep copy the view of views to device) from the variables specified in desc contained
  // from the blocks contained in pmd (which can either be MeshBlockData/MeshData).
  template <class T>
  static SparsePackBase Build(T *pmd, const impl::PackDescriptor &desc,
                              const std::vector<bool> &include_block);

  pack_t pack_;
  pack_h_t pack_h_;
  bounds_t bounds_;
  bounds_h_t bounds_h_;
  block_props_t block_props_;
  block_props_h_t block_props_h_;
  coords_t coords_;

  int flx_idx_;
  bool with_fluxes_;
  bool coarse_;
  bool flat_;
  int nblocks_;
  int nvar_;
  int size_;
};

} // namespace parthenon

#endif // PACK_SPARSE_PACK_BASE_HPP_
