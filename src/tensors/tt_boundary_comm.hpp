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

#ifndef TENSORS_TT_BOUNDARY_COMM_HPP
#define TENSORS_TT_BOUNDARY_COMM_HPP

#include <memory>
#include <vector>

#include "tensors/tt_boundary_cache.hpp"
#include "tensors/tt_container.hpp"
#include "tensors/tt_types.hpp"

namespace parthenon {

// Build (or rebuild) the boundary-index cache for a MeshTTData partition: walk every
// boundary, derive the sender's interior-send box and the receiver's exterior-recv box
// (via CalcIndices + the reverse-neighbor descriptor), store the two indexers plus the
// whole-block extents as a TTBndInfo, and ensure a TTCommChannel exists in
// mesh->tt_comm_map for each boundary (recorded by index in cache.channels). The result is
// a single flat device array for one batched Send launch. Identity-transform, same-level
// boundaries only for now (Step 6 scope); asserts otherwise. Rebuilds when the mesh
// channel-map epoch has advanced (i.e. after (re)mesh).
TaskStatus BuildTTBoundaryCache(std::shared_ptr<MeshTTData> &md);

// Build the boundary "addend" trains: one whole-block train per cached boundary, holding
// the sending block's field on the *receiver's* index space (its interior cells gathered
// into the receiver's ghost layer, all other cells zero). This is the generic,
// mesh-driven analogue of the prototype's MakeNeighborTensors: the block/neighbor loop is
// the boundary cache and the shift is the cached send/recv indexers. The returned trains
// are the addends to be summed into each neighbor. Rounding and channel deposit are done
// by the caller (TTSend). The cache must be current (see BuildTTBoundaryCache).
std::vector<std::shared_ptr<tensor::TensorTrain>>
BuildBoundaryTensors(std::shared_ptr<MeshTTData> &md, const TTBoundaryCache &cache);

// TT boundary communication (single rank, same-level). Send builds the per-boundary addend
// trains (BuildBoundaryTensors), rounds each, and deposits them into the send channels via
// the cache. Only the send side needs the cache. Receive/Set look their channels up inline
// by ReceiveKey: Set is a pure additive combine that sums each received addend into the
// destination block's train, rounds once per block, and stales the channel -- no index
// math on the receive side. The cache must be current (BuildTTBoundaryCache after (re)mesh).
TaskStatus TTSend(std::shared_ptr<MeshTTData> &md, Real round_eps);

// Try to receive every boundary's addend. Returns true once all receive channels have
// deposited (single rank: complete after the matching TTSend). Idempotent.
TaskStatus TTReceive(std::shared_ptr<MeshTTData> &md);

TaskStatus TTSetBounds(std::shared_ptr<MeshTTData> &md, Real round_eps);

} // namespace parthenon

#endif // TENSORS_TT_BOUNDARY_COMM_HPP
