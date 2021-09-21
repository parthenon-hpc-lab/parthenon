//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
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

//! \file bvals_swarm.cpp
//  \brief functions that apply BCs for SWARMs

#include "bvals/bvals_interfaces.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include "basic_types.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "utils/buffer_utils.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

BoundarySwarm::BoundarySwarm(std::weak_ptr<MeshBlock> pmb)
    : bswarm_index(), pmy_block(pmb), pmy_mesh_(pmb.lock()->pmy_mesh) {
#ifdef MPI_PARALLEL
  swarm_id_ = pmb.lock()->pbval->bvars_next_phys_id_;
#endif

  InitBoundaryData(bd_var_);
}

void BoundarySwarm::InitBoundaryData(BoundaryData<> &bd) {
  auto pmb = GetBlockPointer();
  NeighborIndexes *ni = pmb->pbval->ni;

  bd.nbmax = pmb->pbval->maxneighbor_;

  for (int n = 0; n < bd.nbmax; n++) {
    bd.flag[n] = BoundaryStatus::waiting;
#ifdef MPI_PARALLEL
    bd.req_send[n] = MPI_REQUEST_NULL;
    bd.req_recv[n] = MPI_REQUEST_NULL;
#endif
  }
}

void BoundarySwarm::SetupPersistentMPI() {
#ifdef MPI_PARALLEL
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();

  // Initialize neighbor communications to other ranks
  for (int n = 0; n < pmb->pbval->nneighbor; n++) {
    NeighborBlock &nb = pmb->pbval->neighbor[n];

    // Neighbor on different MPI process
    if (nb.snb.rank != Globals::my_rank) {
      send_tag[nb.bufid] =
          pmb->pbval->CreateBvalsMPITag(nb.snb.lid, nb.targetid, swarm_id_);
      recv_tag[nb.bufid] = pmb->pbval->CreateBvalsMPITag(pmb->lid, nb.bufid, swarm_id_);
      if (bd_var_.req_send[nb.bufid] != MPI_REQUEST_NULL) {
        MPI_Request_free(&bd_var_.req_send[nb.bufid]);
      }
      if (bd_var_.req_recv[nb.bufid] != MPI_REQUEST_NULL) {
        MPI_Request_free(&bd_var_.req_recv[nb.bufid]);
      }
    }
  }
#endif
}

// Send particle buffers across meshblocks. If different MPI ranks, use MPI, if same rank,
// do a deep copy on device.
void BoundarySwarm::Send(BoundaryCommSubset phase) {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  // Fence to make sure buffers are loaded before sending
  pmb->exec_space.fence();
  for (int n = 0; n < pmb->pbval->nneighbor; n++) {
    NeighborBlock &nb = pmb->pbval->neighbor[n];
    if (nb.snb.rank != Globals::my_rank) {
#ifdef MPI_PARALLEL
      PARTHENON_REQUIRE(bd_var_.req_send[nb.bufid] == MPI_REQUEST_NULL,
                        "Trying to create a new send before previous send completes!");
      PARTHENON_MPI_CHECK(MPI_Isend(bd_var_.send[nb.bufid].data(), send_size[nb.bufid],
                                    MPI_PARTHENON_REAL, nb.snb.rank, send_tag[nb.bufid],
                                    MPI_COMM_WORLD, &(bd_var_.req_send[nb.bufid])));
#endif // MPI_PARALLEL
    } else {
      MeshBlock &target_block = *pmy_mesh_->FindMeshBlock(nb.snb.gid);
      std::shared_ptr<BoundarySwarm> ptarget_bswarm =
          target_block.pbswarm->bswarms[bswarm_index];
      if (send_size[nb.bufid] > 0) {
        // Ensure target buffer is large enough
        if (bd_var_.send[nb.bufid].extent(0) >
            ptarget_bswarm->bd_var_.recv[nb.targetid].extent(0)) {
          ptarget_bswarm->bd_var_.recv[nb.targetid] =
              BufArray1D<Real>("Buffer", (bd_var_.send[nb.bufid].extent(0)));
        }

        target_block.deep_copy(ptarget_bswarm->bd_var_.recv[nb.targetid],
                               bd_var_.send[nb.bufid]);
        ptarget_bswarm->recv_size[nb.targetid] = send_size[nb.bufid];
        ptarget_bswarm->bd_var_.flag[nb.targetid] = BoundaryStatus::arrived;
      } else {
        ptarget_bswarm->recv_size[nb.targetid] = 0;
        ptarget_bswarm->bd_var_.flag[nb.targetid] = BoundaryStatus::completed;
      }
    }
  }
}

void BoundarySwarm::Receive(BoundaryCommSubset phase) {
#ifdef MPI_PARALLEL
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  int &mylevel = pmb->loc.level;
  for (int n = 0; n < pmb->pbval->nneighbor; n++) {
    NeighborBlock &nb = pmb->pbval->neighbor[n];
    if (nb.snb.rank != Globals::my_rank) {
      // Check to see if we got a message
      int test;
      MPI_Status status;

      if (bd_var_.flag[nb.bufid] != BoundaryStatus::completed) {
        PARTHENON_MPI_CHECK(MPI_Iprobe(MPI_ANY_SOURCE, recv_tag[nb.bufid], MPI_COMM_WORLD,
                                       &test, &status));
        if (!static_cast<bool>(test)) {
          bd_var_.flag[nb.bufid] = BoundaryStatus::waiting;
        } else {
          bd_var_.flag[nb.bufid] = BoundaryStatus::arrived;

          // If message is available, receive it
          PARTHENON_MPI_CHECK(
              MPI_Get_count(&status, MPI_PARTHENON_REAL, &(recv_size[nb.bufid])));
          if (recv_size[nb.bufid] > bd_var_.recv[nb.bufid].extent(0)) {
            bd_var_.recv[nb.bufid] = ParArray1D<Real>("Buffer", recv_size[nb.bufid]);
          }
          PARTHENON_MPI_CHECK(MPI_Recv(bd_var_.recv[nb.bufid].data(), recv_size[nb.bufid],
                                       MPI_PARTHENON_REAL, nb.snb.rank,
                                       recv_tag[nb.bufid], MPI_COMM_WORLD, &status));
        }
      }
    }
  }
#endif
}

} // namespace parthenon
