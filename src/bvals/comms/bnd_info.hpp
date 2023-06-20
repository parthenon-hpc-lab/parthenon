//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
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

#ifndef BVALS_COMMS_BND_INFO_HPP_
#define BVALS_COMMS_BND_INFO_HPP_

#include <memory>
#include <string>
#include <vector>

#include "basic_types.hpp"
#include "bvals/bvals_interfaces.hpp"
#include "coordinates/coordinates.hpp"
#include "interface/variable_state.hpp"
#include "mesh/domain.hpp"
#include "utils/communication_buffer.hpp"
#include "utils/indexer.hpp"
#include "utils/object_pool.hpp"

namespace parthenon {

template <typename T>
class MeshData;
class IndexRange;
class NeighborBlock;
template <typename T>
class Variable;

struct BndInfo {
  int ntopological_elements = 1;
  Indexer6D idxer[3];
  Indexer6D prores_idxer[10]; // Has to be large enough to allow for maximum integer
                              // conversion of TopologicalElements

  CoordinateDirection dir;
  bool allocated = true;
  bool buf_allocated = true;
  RefinementOp_t refinement_op = RefinementOp_t::None;
  Coordinates_t coords, coarse_coords; // coords

  buf_pool_t<Real>::weak_t buf;         // comm buffer from pool
  ParArrayND<Real, VariableState> var;  // data variable used for comms
  ParArrayND<Real, VariableState> fine; // fine data variable for prolongation/restriction
  ParArrayND<Real, VariableState>
      coarse; // coarse data variable for prolongation/restriction

  BndInfo() = default;
  BndInfo(const BndInfo &) = default;

  // These are are used to generate the BndInfo struct for various
  // kinds of boundary types and operations.
  static BndInfo GetSendBndInfo(std::shared_ptr<MeshBlock> pmb, const NeighborBlock &nb,
                                std::shared_ptr<Variable<Real>> v,
                                CommBuffer<buf_pool_t<Real>::owner_t> *buf);
  static BndInfo GetSetBndInfo(std::shared_ptr<MeshBlock> pmb, const NeighborBlock &nb,
                               std::shared_ptr<Variable<Real>> v,
                               CommBuffer<buf_pool_t<Real>::owner_t> *buf);
  static BndInfo GetSendCCFluxCor(std::shared_ptr<MeshBlock> pmb, const NeighborBlock &nb,
                                  std::shared_ptr<Variable<Real>> v,
                                  CommBuffer<buf_pool_t<Real>::owner_t> *buf);
  static BndInfo GetSetCCFluxCor(std::shared_ptr<MeshBlock> pmb, const NeighborBlock &nb,
                                 std::shared_ptr<Variable<Real>> v,
                                 CommBuffer<buf_pool_t<Real>::owner_t> *buf);
};

int GetBufferSize(std::shared_ptr<MeshBlock> pmb, const NeighborBlock &nb,
                  std::shared_ptr<Variable<Real>> v);

using BufferCache_t = ParArray1D<BndInfo>;
using BufferCacheHost_t = typename BufferCache_t::HostMirror;

// This is just a struct to cleanly hold all of the information it is useful to cache
// for the block boundary communication routines. A copy of it is contained in MeshData.
struct BvarsSubCache_t {
  void clear() {
    buf_vec.clear();
    idx_vec.clear();
    if (sending_non_zero_flags.KokkosView().is_allocated())
      sending_non_zero_flags = ParArray1D<bool>{};
    if (sending_non_zero_flags_h.KokkosView().is_allocated())
      sending_non_zero_flags_h = ParArray1D<bool>::host_mirror_type{};
    bnd_info = BufferCache_t{};
    bnd_info_h = BufferCache_t::host_mirror_type{};
    buffer_subset_sizes.clear();
    buffer_subsets = ParArray2D<std::size_t>{};
    buffer_subsets_h = ParArray2D<std::size_t>::host_mirror_type{};
  }

  std::vector<std::size_t> idx_vec;
  std::vector<CommBuffer<buf_pool_t<Real>::owner_t> *> buf_vec;
  ParArray1D<bool> sending_non_zero_flags;
  // Cache both host and device buffer info. Reduces mallocs, and
  // also means the bounds values are available on host if needed.
  ParArray1D<bool>::host_mirror_type sending_non_zero_flags_h;

  BufferCache_t bnd_info{};
  BufferCache_t::host_mirror_type bnd_info_h{};

  // Can be used to inform the infrastructure to loop over only a
  // subset of the bvars cache. Used for prolongation/restriction.
  std::vector<std::size_t> buffer_subset_sizes;
  ParArray2D<std::size_t> buffer_subsets{};
  ParArray2D<std::size_t>::host_mirror_type buffer_subsets_h{};
};

struct BvarsCache_t {
  std::array<BvarsSubCache_t, NUM_BNDRY_TYPES * 2> caches;
  auto &GetSubCache(BoundaryType boundType, bool send) {
    return caches[2 * static_cast<int>(boundType) + send];
  }
  // auto &operator[](BoundaryType boundType) { return
  // caches[static_cast<int>(boundType)]; }
  void clear() {
    for (int i = 0; i < caches.size(); ++i)
      caches[i].clear();
  }
};

} // namespace parthenon

#endif // BVALS_COMMS_BND_INFO_HPP_
