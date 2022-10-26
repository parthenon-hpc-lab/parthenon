//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2022 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2022. Triad National Security, LLC. All rights reserved.
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
#include <iostream> // debug
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "bvals_cc_in_one.hpp"
#include "bvals_utils.hpp"
#include "config.hpp"
#include "globals.hpp"
#include "interface/variable.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/refinement_in_one.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {
namespace cell_centered_bvars {

using namespace impl;

// pmesh->boundary_comm_map.clear() after every remesh
// in InitializeBlockTimeStepsAndBoundaries()
TaskStatus BuildBoundaryBuffers(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::Profiling::pushRegion("Task_BuildSendBoundBufs");
  Mesh *pmesh = md->GetMeshPointer();
  auto &all_caches = md->GetBvarsCache();

  // Clear the fast access vectors for this block since they are no longer valid
  // after all MeshData call BuildBoundaryBuffers
  all_caches.clear();

  // Build buffers for all boundaries, both local and nonlocal
  ForEachBoundary(md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb, const sp_cv_t v,
                          const OffsetIndices &no) {
    // Calculate the required size of the buffer for this boundary
    int buf_size = GetBufferSize(pmb, nb, v);

    // Add a buffer pool if one does not exist for this size
    if (pmesh->pool_map.count(buf_size) == 0) {
      pmesh->pool_map.emplace(std::make_pair(
          buf_size, buf_pool_t<Real>([buf_size](buf_pool_t<Real> *pool) {
            using buf_t = buf_pool_t<Real>::base_t;
            // TODO(LFR): Make nbuf a user settable parameter
            const int nbuf = 200;
            buf_t chunk("pool buffer", buf_size * nbuf);
            for (int i = 1; i < nbuf; ++i) {
              pool->AddFreeObjectToPool(
                  buf_t(chunk, std::make_pair(i * buf_size, (i + 1) * buf_size)));
            }
            return buf_t(chunk, std::make_pair(0, buf_size));
          })));
    }

    const int receiver_rank = nb.snb.rank;
    const int sender_rank = Globals::my_rank;

    int tag = 0;
#ifdef MPI_PARALLEL
    // Get a bi-directional mpi tag for this pair of blocks
    tag = pmesh->tag_map.GetTag(pmb, nb);

    mpi_comm_t comm = pmesh->GetMPIComm(v->label());
    mpi_comm_t comm_flxcor = comm;
    if (nb.snb.level != pmb->loc.level)
      comm_flxcor = pmesh->GetMPIComm(v->label() + "_flcor");
#else
    // Setting to zero is fine here since this doesn't actually get used when everything
    // is on the same rank
    mpi_comm_t comm = 0;
    mpi_comm_t comm_flxcor = 0;
#endif
    // Build send buffers
    auto s_key = SendKey(pmb, nb, v);
    PARTHENON_DEBUG_REQUIRE(pmesh->boundary_comm_map.count(s_key) == 0,
                            "Two communication buffers have the same key.");

    auto get_resource_method = [pmesh, buf_size]() {
      return buf_pool_t<Real>::owner_t(pmesh->pool_map.at(buf_size).Get());
    };

    bool use_sparse_buffers = v->IsSet(Metadata::Sparse);
    pmesh->boundary_comm_map[s_key] = CommBuffer<buf_pool_t<Real>::owner_t>(
        tag, sender_rank, receiver_rank, comm, get_resource_method, use_sparse_buffers);

    // Separate flxcor buffer if needed, first part of if statement checks that this
    // is fine to coarse and the second checks the two blocks share a face
    if ((nb.snb.level == pmb->loc.level - 1) &&
        (std::abs(nb.ni.ox1) + std::abs(nb.ni.ox2) + std::abs(nb.ni.ox3) == 1))
      pmesh->boundary_comm_flxcor_map[s_key] = CommBuffer<buf_pool_t<Real>::owner_t>(
          tag, sender_rank, receiver_rank, comm_flxcor, get_resource_method,
          use_sparse_buffers);

    // Also build the non-local receive buffers here
    if (sender_rank != receiver_rank) {
      auto r_key = ReceiveKey(pmb, nb, v);
      pmesh->boundary_comm_map[r_key] = CommBuffer<buf_pool_t<Real>::owner_t>(
          tag, receiver_rank, sender_rank, comm, get_resource_method, use_sparse_buffers);
      // Separate flxcor buffer if needed
      if ((nb.snb.level - 1 == pmb->loc.level) &&
          (std::abs(nb.ni.ox1) + std::abs(nb.ni.ox2) + std::abs(nb.ni.ox3) == 1))
        pmesh->boundary_comm_flxcor_map[r_key] = CommBuffer<buf_pool_t<Real>::owner_t>(
            tag, receiver_rank, sender_rank, comm_flxcor, get_resource_method,
            use_sparse_buffers);
    }
  });

  Kokkos::Profiling::popRegion(); // "Task_BuildSendBoundBufs"
  return TaskStatus::complete;
}
} // namespace cell_centered_bvars
} // namespace parthenon
