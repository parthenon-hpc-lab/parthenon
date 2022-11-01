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

#ifndef BVALS_CC_BND_INFO_HPP_
#define BVALS_CC_BND_INFO_HPP_

#include <memory>
#include <string>
#include <vector>

#include "basic_types.hpp"
#include "bvals/bvals_interfaces.hpp"
#include "coordinates/coordinates.hpp"
#include "interface/variable_state.hpp"
#include "utils/communication_buffer.hpp"
#include "utils/object_pool.hpp"

namespace parthenon {

template <typename T>
class MeshData;
class IndexRange;
class NeighborBlock;
template <typename T>
class CellVariable;

namespace cell_centered_bvars {

struct BndInfo {
  int si = 0;
  int ei = 0;
  int sj = 0;
  int ej = 0;
  int sk = 0;
  int ek = 0;

  int Nt = 0;
  int Nu = 0;
  int Nv = 0;

  CoordinateDirection dir;
  bool allocated = true;
  RefinementOp_t refinement_op = RefinementOp_t::None;
  Coordinates_t coords, coarse_coords; // coords

  buf_pool_t<Real>::weak_t buf; // comm buffer from pool
  ParArray6D<Real, VariableState> var;         // data variable used for comms
  ParArray6D<Real, VariableState> fine;        // fine data variable for prolongation/restriction
  ParArray6D<Real, VariableState> coarse;      // coarse data variable for prolongation/restriction

  static BndInfo GetSendBndInfo(std::shared_ptr<MeshBlock> pmb, const NeighborBlock &nb,
                                std::shared_ptr<CellVariable<Real>> v,
                                CommBuffer<buf_pool_t<Real>::owner_t> *buf);
  static BndInfo GetSetBndInfo(std::shared_ptr<MeshBlock> pmb, const NeighborBlock &nb,
                               std::shared_ptr<CellVariable<Real>> v,
                               CommBuffer<buf_pool_t<Real>::owner_t> *buf);
  static BndInfo GetSendCCFluxCor(std::shared_ptr<MeshBlock> pmb, const NeighborBlock &nb,
                                  std::shared_ptr<CellVariable<Real>> v,
                                  CommBuffer<buf_pool_t<Real>::owner_t> *buf);
  static BndInfo GetSetCCFluxCor(std::shared_ptr<MeshBlock> pmb, const NeighborBlock &nb,
                                 std::shared_ptr<CellVariable<Real>> v,
                                 CommBuffer<buf_pool_t<Real>::owner_t> *buf);
};

int GetBufferSize(std::shared_ptr<MeshBlock> pmb, const NeighborBlock &nb,
                  std::shared_ptr<CellVariable<Real>> v);

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
  }

  std::vector<std::size_t> idx_vec;
  std::vector<CommBuffer<buf_pool_t<Real>::owner_t> *> buf_vec;
  ParArray1D<bool> sending_non_zero_flags;
  ParArray1D<bool>::host_mirror_type sending_non_zero_flags_h;

  BufferCache_t bnd_info{};
  BufferCache_t::host_mirror_type bnd_info_h{};
};

struct BvarsCache_t {
  // The five here corresponds to the current size of the BoundaryType enum
  std::array<BvarsSubCache_t, 5 * 2> caches;
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

} // namespace cell_centered_bvars
} // namespace parthenon

#endif // BVALS_CC_BND_INFO_HPP_
