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

#include "bvals/swarm/bvals_swarm.hpp"

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

BoundarySwarm::BoundarySwarm(std::weak_ptr<MeshBlock> pmb) : pmy_block(pmb) {
  printf("BoundarySwarm::BoundarySwarm\n");
#ifdef MPI_PARALLEL
  // TODO(BRR) Need to update swarm id counter!
  swarm_id_ = 1;
// swarm_id_ = pmb.lock()->pbval->bvars_next_phys_id_;
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
      recv_tag[n] = pmb->pbval->CreateBvalsMPITag(pmb->lid, nb.targetid, swarm_id_);
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

void BoundarySwarm::Send(BoundaryCommSubset phase) {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  int &mylevel = pmb->loc.level;
  for (int n = 0; n < pmb->pbval->nneighbor; n++) {
    NeighborBlock &nb = pmb->pbval->neighbor[n];
    if (nb.snb.rank != Globals::my_rank) {
#ifdef MPI_PARALLEL
      // TODO(BRR) Check to see if already sending!

      MPI_Request request;
      MPI_Isend(bd_var_.send[n].data(), send_size[n], MPI_PARTHENON_REAL, nb.snb.rank,
                send_tag[n], MPI_COMM_WORLD, &request);
#endif // MPI_PARALLEL
    } else {
      // CopyVariableBufferSameProcess
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
      pmb->exec_space.fence();
      // Check to see if we got a message
      int test;
      MPI_Status status;
      MPI_Iprobe(nb.snb.rank, MPI_ANY_TAG, MPI_COMM_WORLD, &test, &status);
      if (!static_cast<bool>(test)) {
        bd_var_.flag[nb.bufid] = BoundaryStatus::waiting;
      } else {
        bd_var_.flag[nb.bufid] = BoundaryStatus::arrived;

        // If message is available, receive it
        int nbytes;
        MPI_Get_count(&status, MPI_CHAR, &nbytes);
        if (nbytes / sizeof(Real) > bd_var_.recv[n].extent(0)) {
          bd_var_.recv[n] = ParArray1D<Real>("Buffer", nbytes / sizeof(Real));
        }
        MPI_Recv(bd_var_.recv[n].data(), nbytes, MPI_CHAR, nb.snb.rank, MPI_ANY_TAG,
                 MPI_COMM_WORLD, &status);
        recv_size[n] = nbytes / sizeof(Real);
      }
    }
  }
#endif
}

void BoundarySwarm::ClearBoundary(BoundaryCommSubset phase) {}

} // namespace parthenon
