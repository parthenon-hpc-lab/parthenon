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

#ifndef TENSORS_TT_BOUNDARY_CACHE_HPP
#define TENSORS_TT_BOUNDARY_CACHE_HPP

#include <cstddef>
#include <vector>

#include "kokkos_abstraction.hpp"
#include "kokkos_types.hpp"
#include "mesh/forest/logical_coordinate_transformation.hpp"
#include "utils/indexer.hpp"

// Data structures for the tensor-train boundary-comm cache. Kept in this light,
// tensor-free header (no mesh.hpp) so MeshTTData can hold the cache without pulling in the
// heavy mesh headers; the builder that populates it lives in tt_boundary_comm.{hpp,cpp}.

namespace parthenon {

class TTCommChannel;

// One boundary's device-resident index info, analogous to a regular-field BndInfo but for
// the whole-block spatial core of a tensor train. Fixed size (two indexers + block
// extents), so an array of these packs into a single flat view for one batched launch.
//
// send/recv enumerate the same number of cells in the same order (congruent boxes, Step
// 6a); the kernel reads send-cell e from the source train and writes recv-cell e into the
// addend train, flattening (k, j, i) to the spatial-core index via the whole-block extents
// held on the cache. Placement is 1-to-1 -- multilevel is handled by restricting/
// prolongating the spatial core (via coarse train buffers) on the send side, as for
// regular fields, not by weights in this map. channel_idx indexes the host-side channel
// vector on the cache.
struct TTBndInfo {
  SpatiallyMaskedIndexer6D send;
  SpatiallyMaskedIndexer6D recv;
  // Sender->neighbor logical coordinate transformation. Identity for same-tree
  // boundaries; for a rotated/flipped cross-tree boundary the set kernel applies its
  // InverseTransform to the recv cell before writing (as regular-field comm does), so the
  // send-cell e -> recv-cell e correspondence holds under the transform.
  parthenon::forest::LogicalCoordinateTransformation lcoord_trans;
  int channel_idx = -1; // index into the cache's host-side channel vector

  KOKKOS_DEFAULTED_FUNCTION TTBndInfo() = default;
};

using TTBndInfoArr_t = ParArray1DRaw<TTBndInfo>;
using TTBndInfoArrHost_t = typename TTBndInfoArr_t::host_mirror_type;

// Per-MeshTTData cache of boundary index info, mirroring the regular BvarsSubCache_t: a
// flat device array (+ host mirror) of TTBndInfo for a single batched launch, a parallel
// host vector of channel pointers into Mesh::tt_comm_map (stable within an epoch, like
// buf_vec), and the channel-map epoch it was built against.
struct TTBoundaryCache {
  TTBndInfoArr_t bnd_info{};
  TTBndInfoArrHost_t bnd_info_h{};
  std::vector<TTCommChannel *> channels; // indexed by TTBndInfo::channel_idx
  // Whole-block (entire, incl. ghosts) spatial extents used to flatten an indexer's
  // (k, j, i) to the spatial-core index: idx = (k * nj + j) * ni + i. These are mesh-wide
  // (every block shares the same cell shape), so they live once on the cache rather than
  // per boundary.
  int ni{0}, nj{0};
  std::size_t epoch{0};

  void clear() {
    bnd_info = TTBndInfoArr_t{};
    bnd_info_h = TTBndInfoArrHost_t{};
    channels.clear();
    ni = 0;
    nj = 0;
    epoch = 0;
  }
};

} // namespace parthenon

#endif // TENSORS_TT_BOUNDARY_CACHE_HPP
