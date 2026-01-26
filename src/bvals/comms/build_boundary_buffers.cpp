//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2022 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2022-2025. Triad National Security, LLC. All rights reserved.
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
#include "coalesced_buffers.hpp"
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
  std::unordered_map<std::size_t, std::size_t>
      nbufs; // total (existing and new) number of buffers for
             // given size (ignoring sparse)
  std::unordered_map<std::size_t, std::size_t>
      nbufs_allocated; // total that are actually allocated

  ForEachBoundary<BTYPE>(md, [&](auto pmb, sp_mbd_t /*rc*/, nb_t &nb, const sp_cv_t v) {
    // Calculate the required size of the buffer for this boundary
    int buf_size = GetBufferSize(pmb, nb, v);
    //  LR: Multigrid logic requires blocks sending messages to themselves (since the same
    //  block can show up on two multigrid levels). This doesn't require any data
    //  transfer, so the message size can be zero. It is essentially just a flag to show
    //  that the block is done being used on one level and can be used on the next level.
    if (pmb->gid == nb.gid && nb.offsets.IsCell()) buf_size = 0;

    nbufs[buf_size] += 1; // relying on value init of int to 0 for initial entry
    nbufs_allocated[buf_size] += v->IsAllocated();
  });

  ForEachBoundary<BTYPE>(md, [&](auto pmb, sp_mbd_t /*rc*/, nb_t &nb, const sp_cv_t v) {
    // Calculate the required size of the buffer for this boundary
    int buf_size = GetBufferSize(pmb, nb, v);
    // See comment above on the same logic.
    if (pmb->gid == nb.gid && nb.offsets.IsCell()) buf_size = 0;

    // Add a buffer pool if one does not exist for this size
    using buf_t = buf_pool_t<Real>::base_t;
    if (!pmesh->pool_map.Contains(buf_size)) {
      // Minimum number of comm buffers to initialize a pool of a
      // given shape with. As an upper bound, we use the maximum
      // number that might be present.
      const std::size_t nbuf = std::min(pmesh->CommBufferChunkSize(), nbufs.at(buf_size));
      pmesh->pool_map.AddPool(buf_size, nbuf);
    }
    // Now that the pool is guaranteed to exist we can add free objects of the required
    // amount.
    const std::int64_t new_buffers_req =
        nbufs_allocated.at(buf_size) -
        pmesh->pool_map.GetPool(buf_size).NumBuffersInPool();
    if (new_buffers_req > 0) {
      pmesh->pool_map.AddFreeObjectsToPool(buf_size, new_buffers_req);
    }

    const int receiver_rank = nb.rank;
    const int sender_rank = Globals::my_rank;

#ifdef MPI_PARALLEL
    // Get a bi-directional mpi tag for this pair of blocks
    auto comm_label = v->label();
    mpi_comm_t comm = pmesh->GetMPIComm(comm_label);

#else
      // Setting to zero is fine here since this doesn't actually get used when everything
      // is on the same rank
      mpi_comm_t comm = 0;
#endif

    bool use_sparse_buffers = v->IsSet(Metadata::Sparse);
    auto get_resource_method = [pmesh, buf_size](int size) {
      PARTHENON_REQUIRE(size <= buf_size,
                        "Asking for a buffer that is larger than size of pool.");
      return buf_pool_t<Real>::owner_t(pmesh->pool_map.GetPool(buf_size).Get());
    };

    // Build send buffer (unless this is a receiving flux boundary)
    if constexpr (IsSender(BTYPE)) {
      for (int id = 0; id <= (BTYPE == BoundaryType::gmg_restrict_send); ++id) {
        auto s_key = SendKey(pmb, nb, v, BTYPE, id);
        const int tag = pmesh->tag_map.GetTag(pmb, nb, id);
        if (buf_map.count(s_key) == 0)
          buf_map[s_key] = CommBuffer<buf_pool_t<Real>::owner_t>(
              tag, sender_rank, receiver_rank, comm, get_resource_method,
              use_sparse_buffers);
      }
    }

    // Also build the non-local receive buffers here
    if constexpr (IsReceiver(BTYPE)) {
      if (sender_rank != receiver_rank) {
        for (int id = 0; id <= (BTYPE == BoundaryType::gmg_restrict_recv); ++id) {
          auto r_key = ReceiveKey(pmb, nb, v, BTYPE, id);
          const int tag = pmesh->tag_map.GetTag(pmb, nb, id);
          if (buf_map.count(r_key) == 0)
            buf_map[r_key] = CommBuffer<buf_pool_t<Real>::owner_t>(
                tag, receiver_rank, sender_rank, comm, get_resource_method,
                use_sparse_buffers);
        }
      }
    }
  });
}

template <BoundaryType BTYPE>
void RegisterCoalescedCommsSubset(std::shared_ptr<MeshData<Real>> &md) {
  Mesh *pmesh = md->GetMeshPointer();
  ForEachBoundary<BTYPE>(md, [&](auto pmb, sp_mbd_t /*rc*/, nb_t &nb, const sp_cv_t v) {
    const int receiver_rank = nb.rank;
    const int sender_rank = Globals::my_rank;
    if (receiver_rank != sender_rank) {
      if constexpr (IsSender(BTYPE)) {
        pmesh->pcoalesced_comms->AddSendBuffer(md->partition, pmb, nb, v, BTYPE);
      }
      if constexpr (IsReceiver(BTYPE)) {
        pmesh->pcoalesced_comms->AddRecvBuffer(pmb, nb, v, BTYPE);
      }
    }
  });
}

} // namespace

TaskStatus BuildBoundaryBuffers(std::shared_ptr<MeshData<Real>> &md) {
  PARTHENON_INSTRUMENT
  Mesh *pmesh = md->GetMeshPointer();

  BuildBoundaryBufferSubset<BoundaryType::any>(md, pmesh->boundary_comm_map);
  BuildBoundaryBufferSubset<BoundaryType::flxcor_send>(md, pmesh->boundary_comm_map);
  BuildBoundaryBufferSubset<BoundaryType::flxcor_recv>(md, pmesh->boundary_comm_map);

  return TaskStatus::complete;
}

TaskStatus BuildGMGBoundaryBuffers(std::shared_ptr<MeshData<Real>> &md) {
  PARTHENON_INSTRUMENT
  Mesh *pmesh = md->GetMeshPointer();

  BuildBoundaryBufferSubset<BoundaryType::gmg_same>(md, pmesh->boundary_comm_map);
  BuildBoundaryBufferSubset<BoundaryType::gmg_prolongate_send>(md,
                                                               pmesh->boundary_comm_map);
  BuildBoundaryBufferSubset<BoundaryType::gmg_prolongate_recv>(md,
                                                               pmesh->boundary_comm_map);
  BuildBoundaryBufferSubset<BoundaryType::gmg_restrict_send>(md,
                                                             pmesh->boundary_comm_map);
  BuildBoundaryBufferSubset<BoundaryType::gmg_restrict_recv>(md,
                                                             pmesh->boundary_comm_map);

  return TaskStatus::complete;
}

TaskStatus RegisterCoalescedComms(std::shared_ptr<MeshData<Real>> &md) {
  PARTHENON_INSTRUMENT

  RegisterCoalescedCommsSubset<BoundaryType::any>(md);
  RegisterCoalescedCommsSubset<BoundaryType::flxcor_send>(md);
  RegisterCoalescedCommsSubset<BoundaryType::flxcor_recv>(md);

  return TaskStatus::complete;
}

TaskStatus RegisterCoalescedCommsGMG(std::shared_ptr<MeshData<Real>> &md) {
  PARTHENON_INSTRUMENT

  RegisterCoalescedCommsSubset<BoundaryType::gmg_same>(md);
  RegisterCoalescedCommsSubset<BoundaryType::gmg_prolongate_send>(md);
  RegisterCoalescedCommsSubset<BoundaryType::gmg_prolongate_recv>(md);
  RegisterCoalescedCommsSubset<BoundaryType::gmg_restrict_send>(md);
  RegisterCoalescedCommsSubset<BoundaryType::gmg_restrict_recv>(md);

  return TaskStatus::complete;
}

TaskStatus RegisterCoalescedComms(Mesh *pmesh) {
  for (auto &partition : pmesh->GetDefaultBlockPartitions()) {
    auto &md = pmesh->mesh_data.Add("base", partition);
    RegisterCoalescedComms(md);
  }
  return TaskStatus::complete;
}

} // namespace parthenon
