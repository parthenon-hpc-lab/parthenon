//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
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
  printf("[%i] BoundarySwarm::BoundarySwarm\n", Globals::my_rank);
#ifdef MPI_PARALLEL
  swarm_id_ = pmb.lock()->pbval->bvars_next_phys_id_;
#endif

  InitBoundaryData(bd_var_);
}

void BoundarySwarm::InitBoundaryData(BoundaryData<> &bd) {
  auto pmb = GetBlockPointer();
  NeighborIndexes *ni = pmb->pbval->ni;
  int size = 0;

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
  int &mylevel = pmb->loc.level;

  // Initialize neighbor communications to other ranks
  int tag;
  int ssize = 0;
  int rsize = 0;
  for (int n = 0; n < pmb->pbval->nneighbor; n++) {
    NeighborBlock &nb = pmb->pbval->neighbor[n];

    // Neighbor on different MPI process
    if (nb.snb.rank != Globals::my_rank) {
      send_tag[n] = pmb->pbval->CreateBvalsMPITag(nb.snb.lid, nb.targetid, swarm_id_);
      //recv_tag[n] = pmb->pbval->CreateBvalsMPITag(pmb->lid, nb.targetid, swarm_id_);
      recv_tag[n] = pmb->pbval->CreateBvalsMPITag(pmb->lid, nb.bufid, swarm_id_);
      //printf("[%i] neighbor %i send_tag: %i recv_tag: %i\n", Globals::my_rank, n, send_tag[n], recv_tag[n]);
      // TODO(BRR) these tags need to work with other bvals
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

// TODO(BRR) is this necessary? Reset tags? This is just MPI_Start for cc vars
void BoundarySwarm::StartReceiving(BoundaryCommSubset phase) {}

// Send particle buffers across meshblocks. If different MPI ranks, use MPI, if same rank,
// do a deep copy on device.
void BoundarySwarm::Send(BoundaryCommSubset phase) {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  int &mylevel = pmb->loc.level;
  for (int n = 0; n < pmb->pbval->nneighbor; n++) {
    NeighborBlock &nb = pmb->pbval->neighbor[n];
    if (nb.snb.rank != Globals::my_rank) {
#ifdef MPI_PARALLEL
      // TODO(BRR) Check to see if already sending?

      MPI_Request request;
      MPI_Isend(bd_var_.send[n].data(), send_size[n], MPI_PARTHENON_REAL, nb.snb.rank,
                send_tag[n], MPI_COMM_WORLD, &request);
      //printf("[%i] Sending size: %i to nb %i rank %i (tag %i)\n", Globals::my_rank, send_size[n], n, nb.snb.rank, send_tag[n]);
#endif // MPI_PARALLEL
    } else {
      MeshBlock &target_block = *pmy_mesh_->FindMeshBlock(nb.snb.gid);
      std::shared_ptr<BoundarySwarm> ptarget_bswarm =
          target_block.pbswarm->bswarms[bswarm_index];
      //printf("[%i] COPYING size: %i to nb %i rank %i (tag %i)\n", Globals::my_rank, send_size[n], n, nb.snb.rank, send_tag[n]);
      if (send_size[nb.bufid] > 0) {
        // Ensure target buffer is large enough
        if (bd_var_.send[nb.bufid].extent(0) >
            ptarget_bswarm->bd_var_.recv[nb.targetid].extent(0)) {
          ptarget_bswarm->bd_var_.recv[nb.targetid] =
              ParArray1D<Real>("Buffer", (bd_var_.send[nb.bufid].extent(0)));
        }

        target_block.deep_copy(ptarget_bswarm->bd_var_.recv[nb.targetid],
                               bd_var_.send[nb.bufid]);
        ptarget_bswarm->recv_size[nb.targetid] = send_size[nb.bufid];
        //ptarget_bswarm->bd_var_.flag[nb.targetid] = BoundaryStatus::completed;
        ptarget_bswarm->bd_var_.flag[nb.targetid] = BoundaryStatus::arrived;
        //printf("[%i] nb.targetid: %i flag: %i\n", Globals::my_rank, nb.targetid,
        //       static_cast<int>(BoundaryStatus::completed));
      } else {
        ptarget_bswarm->recv_size[nb.targetid] = 0;
        ptarget_bswarm->bd_var_.flag[nb.targetid] = BoundaryStatus::completed;
        //printf("[%i] nb.targetid: %i flag: %i\n", Globals::my_rank, nb.targetid,
        //       static_cast<int>(BoundaryStatus::completed));
      }
    }
  }
}

void BoundarySwarm::Receive(BoundaryCommSubset phase) {
#ifdef MPI_PARALLEL
  //MPI_Barrier(MPI_COMM_WORLD);
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  int &mylevel = pmb->loc.level;
  for (int n = 0; n < pmb->pbval->nneighbor; n++) {
    NeighborBlock &nb = pmb->pbval->neighbor[n];
    if (nb.snb.rank != Globals::my_rank) {
      pmb->exec_space.fence();
      // Check to see if we got a message
      int test;
      MPI_Status status;

      MPI_Iprobe(MPI_ANY_SOURCE,  recv_tag[nb.bufid], MPI_COMM_WORLD, &test, &status);
      //printf("[%i] Probing for tag %i neighbor %i: test: %i\n", Globals::my_rank, recv_tag[nb.bufid], n, test);

      //MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &test, &status);
      //MPI_Iprobe(nb.snb.rank, MPI_ANY_TAG, MPI_COMM_WORLD, &test, &status);
      //MPI_Test(&(bd_var_.req_recv[nb.bufid]), &test, MPI_STATUS_IGNORE);
      if (!static_cast<bool>(test)) {
        bd_var_.flag[nb.bufid] = BoundaryStatus::waiting;
      } else {
        bd_var_.flag[nb.bufid] = BoundaryStatus::arrived;

        // If message is available, receive it
        int nbytes = 0;
        MPI_Get_count(&status, MPI_CHAR, &nbytes);
        //printf("[%i] nbytes: %i n: %i nb.bufid: %i\n", Globals::my_rank, nbytes, n, nb.bufid);
        if (nbytes / sizeof(Real) > bd_var_.recv[n].extent(0)) {
          bd_var_.recv[n] = ParArray1D<Real>("Buffer", nbytes / sizeof(Real));
        }
        MPI_Recv(bd_var_.recv[n].data(), nbytes, MPI_CHAR, nb.snb.rank, recv_tag[nb.bufid],
                 MPI_COMM_WORLD, &status);
        recv_size[n] = nbytes / sizeof(Real);
        //printf("[%i] nb.bufid: %i boundary status: %i\n", Globals::my_rank, nb.bufid,
        //       static_cast<int>(bd_var_.flag[nb.bufid]));
      }
    }
  }
  //MPI_Barrier(MPI_COMM_WORLD);
#endif
}

void BoundarySwarm::ClearBoundary(BoundaryCommSubset phase) {}

} // namespace parthenon
