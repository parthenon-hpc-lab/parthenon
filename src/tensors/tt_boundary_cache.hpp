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

enum class BoundaryRelation {same, f2c, c2f};

// Uniform 2:1 coarse<->fine cell map for a boundary's spatial core. A coarse cell covers a
// 2^d block of fine cells; the coarse buffer and the full-resolution train both anchor their
// interior at Globals::nghost, so a coarse logical index c maps to fine base 2*(c - ng) + ng
// in each refined direction (identity in symmetry directions, where the coarse and fine
// extents coincide and the refinement factor is 1). Built once per cross-level boundary in
// BuildTTBoundaryCache and stored on TTBndInfo. Kept as a small device-callable struct --
// ghost anchor plus a per-direction refinement factor, with a functor-driven iterator over
// the fine cells under one coarse (k, j, i) -- so a coordinate-aware operator can be dropped
// in later without touching the kernels. (The per-direction factors are redundant with the
// spatial-core extents today, but will diverge for non-cell-centered fields.)
struct CoarseFineMap {
  int ng{0};                     // interior anchor (ghost count), shared by coarse and fine
  int rfk{1}, rfj{1}, rfi{1};    // per-direction refinement factor (1 in symmetry dirs, 2 else)

  KOKKOS_DEFAULTED_FUNCTION CoarseFineMap() = default;

  // Number of fine cells under one coarse cell (2^d).
  KOKKOS_FORCEINLINE_FUNCTION int NumFine() const { return rfk * rfj * rfi; }

  // Invoke fn(kf, jf, if) for each fine cell under coarse cell (kc, jc, ic).
  template <class Fn>
  KOKKOS_FORCEINLINE_FUNCTION void ForEachFine(int kc, int jc, int ic, Fn &&fn) const {
    const int kf0 = rfk == 2 ? 2 * (kc - ng) + ng : kc;
    const int jf0 = rfj == 2 ? 2 * (jc - ng) + ng : jc;
    const int if0 = rfi == 2 ? 2 * (ic - ng) + ng : ic;
    for (int ok = 0; ok < rfk; ++ok)
      for (int oj = 0; oj < rfj; ++oj)
        for (int oi = 0; oi < rfi; ++oi)
          fn(kf0 + ok, jf0 + oj, if0 + oi);
  }

  // Build from the mesh dimensionality: directions are active (and so 2:1 refined) in a
  // contiguous block from x1, so x1 (i) is refined for ndim >= 1, x2 (j) for ndim >= 2, and
  // x3 (k) for ndim >= 3. Symmetry directions keep factor 1.
  static CoarseFineMap FromNDim(int ndim, int ng) {
    CoarseFineMap m;
    m.ng = ng;
    m.rfi = ndim >= 1 ? 2 : 1;
    m.rfj = ndim >= 2 ? 2 : 1;
    m.rfk = ndim >= 3 ? 2 : 1;
    return m;
  }
};

// One boundary's device-resident index info, analogous to a regular-field BndInfo but for
// the whole-block spatial core of a tensor train. Fixed size (two indexers + block
// extents), so an array of these packs into a single flat view for one batched launch.
//
// send/recv enumerate the same number of cells in the same order (congruent boxes, Step
// 6a); the kernel reads send-cell e from the source train and writes recv-cell e into the
// addend train, flattening (k, j, i) to the spatial-core index via the whole-block extents
// held on the cache. Placement is 1-to-1 -- multilevel is handled by restricting/
// prolongating the spatial core (via coarse train buffers) on the send side, as for
// regular fields, not by weights in this map.
struct TTBndInfo {
  SpatiallyMaskedIndexer6D send;
  SpatiallyMaskedIndexer6D recv;
  SpatiallyMaskedIndexer6D prores;
  BoundaryRelation btype;
  // Coarse<->fine cell map for cross-level boundaries (unused for same-level).
  CoarseFineMap cfmap;
  // Sender->neighbor logical coordinate transformation. Identity for same-tree
  // boundaries; for a rotated/flipped cross-tree boundary the set kernel applies its
  // InverseTransform to the recv cell before writing (as regular-field comm does), so the
  // send-cell e -> recv-cell e correspondence holds under the transform.
  parthenon::forest::LogicalCoordinateTransformation lcoord_trans;

  KOKKOS_DEFAULTED_FUNCTION TTBndInfo() = default;
};

using TTBndInfoArr_t = ParArray1DRaw<TTBndInfo>;
using TTBndInfoArrHost_t = typename TTBndInfoArr_t::host_mirror_type;

// Per-MeshTTData send-side boundary cache for tensor-train comm. Only the send side needs
// a cache: it launches one batched device kernel over all boundaries to build the addend
// trains, which needs the per-boundary index boxes (bnd_info) packed into a flat device
// array. The receive/set side needs no cache -- it walks ForEachBoundary, looks the
// channel up inline by ReceiveKey, sums the addend into the block train, and stales it.
//
// bnd_info: flat device array (+ host mirror) of TTBndInfo for the batched launch.
// channels: send channel per boundary (into Mesh::tt_comm_map), in ForEachBoundary order,
//   stable within an epoch (like the regular buf_vec).
// epoch: the channel-map epoch this cache was built against, to detect (re)mesh.
//
// Flattening (k, j, i) to a spatial-core physical index is not held here: the spatial core
// carries its own logical Indexer6D, exposed per core slot on the device pack
// (pack.indexer(0)), so kernels flatten via that shared indexer rather than cached extents.
struct TTBoundaryCache {
  TTBndInfoArr_t bnd_info{};
  TTBndInfoArrHost_t bnd_info_h{};
  std::vector<TTCommChannel *> channels;
  std::size_t epoch{0};

  void clear() {
    bnd_info = TTBndInfoArr_t{};
    bnd_info_h = TTBndInfoArrHost_t{};
    channels.clear();
    epoch = 0;
  }
};

} // namespace parthenon

#endif // TENSORS_TT_BOUNDARY_CACHE_HPP
