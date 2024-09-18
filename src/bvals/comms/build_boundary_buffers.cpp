//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2022 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2022-2024. Triad National Security, LLC. All rights reserved.
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

#include <algorithm>
#include <cstddef>
#include <iostream> // debug
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "bvals_in_one.hpp"
#include "bvals_utils.hpp"
#include "config.hpp"
#include "globals.hpp"
#include "interface/variable.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock.hpp"
#include "utils/error_checking.hpp"
#include "utils/loop_utils.hpp"

namespace parthenon {

using namespace loops;
using namespace loops::shorthands;

namespace {
template <BoundaryType BTYPE>
void BuildBoundaryBufferSubset(std::shared_ptr<MeshData<Real>> &md,
                               Mesh::comm_buf_map_t &buf_map) {
  Mesh *pmesh = md->GetMeshPointer();
  std::unordered_map<int, int>
      nbufs; // total (existing and new) number of buffers for given size

  ForEachBoundary<BTYPE>(md, [&](auto pmb, sp_mbd_t /*rc*/, nb_t &nb, const sp_cv_t v) {
    // Calculate the required size of the buffer for this boundary
    int buf_size = GetBufferSize(pmb, nb, v);
    //  LR: Multigrid logic requires blocks sending messages to themselves (since the same
    //  block can show up on two multigrid levels). This doesn't require any data
    //  transfer, so the message size can be zero. It is essentially just a flag to show
    //  that the block is done being used on one level and can be used on the next level.
    if (pmb->gid == nb.gid && nb.offsets.IsCell()) buf_size = 0;

    nbufs[buf_size] += 1; // relying on value init of int to 0 for initial entry
  });

  ForEachBoundary<BTYPE>(md, [&](auto pmb, sp_mbd_t /*rc*/, nb_t &nb, const sp_cv_t v) {
    // Calculate the required size of the buffer for this boundary
    int buf_size = GetBufferSize(pmb, nb, v);
    // See comment above on the same logic.
    if (pmb->gid == nb.gid && nb.offsets.IsCell()) buf_size = 0;

    // Add a buffer pool if one does not exist for this size
    using buf_t = buf_pool_t<Real>::base_t;
    if (pmesh->pool_map.count(buf_size) == 0) {
      // Might be worth discussing what a good default is.
      // Using the number of packs, assumes that all blocks in a pack have fairly similar
      // buffer configurations, which may or may not be a good approximation.
      // An alternative would be "1", which would reduce the memory footprint, but
      // increase the number of individual memory allocations.
      const int64_t nbuf = pmesh->DefaultNumPartitions();
      pmesh->pool_map.emplace(
          buf_size, buf_pool_t<Real>([buf_size, nbuf](buf_pool_t<Real> *pool) {
            const auto pool_size = nbuf * buf_size;
            buf_t chunk("pool buffer", pool_size);
            for (int i = 1; i < nbuf; ++i) {
              pool->AddFreeObjectToPool(
                  buf_t(chunk, std::make_pair(i * buf_size, (i + 1) * buf_size)));
            }
            return buf_t(chunk, std::make_pair(0, buf_size));
          }));
    }
    // Now that the pool is guaranteed to exist we can add free objects of the required
    // amount.
    auto &pool = pmesh->pool_map.at(buf_size);
    const std::int64_t new_buffers_req = nbufs.at(buf_size) - pool.NumBuffersInPool();
    if (new_buffers_req > 0) {
      const auto pool_size = new_buffers_req * buf_size;
      buf_t chunk("pool buffer", pool_size);
      for (int i = 0; i < new_buffers_req; ++i) {
        pool.AddFreeObjectToPool(
            buf_t(chunk, std::make_pair(i * buf_size, (i + 1) * buf_size)));
      }
    }

    const int receiver_rank = nb.rank;
    const int sender_rank = Globals::my_rank;

    int tag = 0;
#ifdef MPI_PARALLEL
    // Get a bi-directional mpi tag for this pair of blocks
    tag = pmesh->tag_map.GetTag(pmb, nb);
    auto comm_label = v->label();
    mpi_comm_t comm = pmesh->GetMPIComm(comm_label);
#else
      // Setting to zero is fine here since this doesn't actually get used when everything
      // is on the same rank
      mpi_comm_t comm = 0;
#endif

    bool use_sparse_buffers = v->IsSet(Metadata::Sparse);
    auto get_resource_method = [pmesh, buf_size]() {
      return buf_pool_t<Real>::owner_t(pmesh->pool_map.at(buf_size).Get());
    };

    // Build send buffer (unless this is a receiving flux boundary)
    if constexpr (IsSender(BTYPE)) {
      auto s_key = SendKey(pmb, nb, v, BTYPE);
      if (buf_map.count(s_key) == 0)
        buf_map[s_key] = CommBuffer<buf_pool_t<Real>::owner_t>(
            tag, sender_rank, receiver_rank, comm, get_resource_method,
            use_sparse_buffers);
    }

    // Also build the non-local receive buffers here
    if constexpr (IsReceiver(BTYPE)) {
      if (sender_rank != receiver_rank) {
        auto r_key = ReceiveKey(pmb, nb, v, BTYPE);
        if (buf_map.count(r_key) == 0)
          buf_map[r_key] = CommBuffer<buf_pool_t<Real>::owner_t>(
              tag, receiver_rank, sender_rank, comm, get_resource_method,
              use_sparse_buffers);
      }
    }
  });
}
} // namespace

TaskStatus BuildBoundaryBuffers(std::shared_ptr<MeshData<Real>> &md) {
  PARTHENON_INSTRUMENT
  Mesh *pmesh = md->GetMeshPointer();
  auto &all_caches = md->GetBvarsCache();

  // Clear the fast access vectors for this block since they are no longer valid
  // after all MeshData call BuildBoundaryBuffers
  all_caches.clear();

  BuildBoundaryBufferSubset<BoundaryType::any>(md, pmesh->boundary_comm_map);
  BuildBoundaryBufferSubset<BoundaryType::flxcor_send>(md, pmesh->boundary_comm_map);
  BuildBoundaryBufferSubset<BoundaryType::flxcor_recv>(md, pmesh->boundary_comm_map);

  return TaskStatus::complete;
}

TaskStatus BuildGMGBoundaryBuffers(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::Profiling::pushRegion("Task_BuildSendBoundBufs");
  Mesh *pmesh = md->GetMeshPointer();
  auto &all_caches = md->GetBvarsCache();

  // Clear the fast access vectors for this block since they are no longer valid
  // after all MeshData call BuildBoundaryBuffers
  all_caches.clear();
  BuildBoundaryBufferSubset<BoundaryType::gmg_same>(md, pmesh->boundary_comm_map);
  BuildBoundaryBufferSubset<BoundaryType::gmg_prolongate_send>(md,
                                                               pmesh->boundary_comm_map);
  BuildBoundaryBufferSubset<BoundaryType::gmg_prolongate_recv>(md,
                                                               pmesh->boundary_comm_map);
  BuildBoundaryBufferSubset<BoundaryType::gmg_restrict_send>(md,
                                                             pmesh->boundary_comm_map);
  BuildBoundaryBufferSubset<BoundaryType::gmg_restrict_recv>(md,
                                                             pmesh->boundary_comm_map);

  Kokkos::Profiling::popRegion(); // "Task_BuildSendBoundBufs"
  return TaskStatus::complete;
}

} // namespace parthenon
