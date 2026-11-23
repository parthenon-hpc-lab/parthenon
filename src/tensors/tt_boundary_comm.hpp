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

// This file was made in part with generative AI.

#ifndef TENSORS_TT_BOUNDARY_COMM_HPP_
#define TENSORS_TT_BOUNDARY_COMM_HPP_

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
// mesh->tt_comm_map for each boundary (recorded by index in cache.channels). The result
// is a single flat device array for one batched Send launch. Same-rank neighbors get a
// local channel shared by both ends; remote neighbors get a remote send channel plus a
// remote receive channel for the reverse direction (see TTCommChannel).
// Identity-transform boundaries only for now; asserts otherwise. Always rebuilds. The
// cache records the channel-map epoch it was built against, and TTSend/TTReceive rebuild
// it when the map has been cleared by a (re)mesh.
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

// Zero the ghost cells of every FillGhost field's spatial core, on every block of the
// partition, leaving the interior untouched. This is exact and keeps the ranks: it only
// scales core 0 along the spatial index. Call it before TTSetBounds whenever the ghosts
// may hold data (e.g. any exchange after the first), since Set adds the received values
// rather than overwriting. Physical-boundary ghosts are zeroed too.
TaskStatus TTZeroBoundaries(std::shared_ptr<MeshTTData> &md);

// TT boundary communication. Send builds the per-boundary addend trains
// (BuildBoundaryTensors) and rounds each. Same-rank addends are deposited directly into
// their local channels; addends for other ranks are serialized in one batch
// (tt_serialize.hpp) and sent over MPI. Returns incomplete, without building anything,
// while any channel is still busy with the previous exchange. Only the send side needs
// the cache. Receive/Set look their channels up inline by ReceiveKey.
TaskStatus TTSend(std::shared_ptr<MeshTTData> &md, Real round_eps);

// Try to receive every boundary's addend. Returns complete once all receive channels have
// arrived (same rank: right after the matching TTSend), incomplete otherwise, so it can
// be polled from a task list. Idempotent.
TaskStatus TTReceive(std::shared_ptr<MeshTTData> &md);

// Deserialize the addends received from other ranks (in one batch), sum each received
// addend into the destination block's train, round every field of every block once, and
// stale the channels. This is a pure additive combine with no index
// math on the receive side: each addend is zero outside the receiver's ghost cells, so
// the result is correct only if those ghost cells are zero beforehand. Call
// TTZeroBoundaries first unless the ghosts are known to be zero.
TaskStatus TTSetBounds(std::shared_ptr<MeshTTData> &md, Real round_eps);

} // namespace parthenon

#endif // TENSORS_TT_BOUNDARY_COMM_HPP_
