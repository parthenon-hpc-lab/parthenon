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
//  swarm_id_ = pmb.lock()->pbval->bvars_next_phys_id_;
  #endif

  InitBoundaryData(bd_var_);
}

void BoundarySwarm::InitBoundaryData(BoundaryData<> &bd) {
  printf("BoundarySwarm::InitBoundaryData\n");
  auto pmb = GetBlockPointer();
  NeighborIndexes *ni = pmb->pbval->ni;
  int size = 0;

  bd.nbmax = pmb->pbval->maxneighbor_;

  for (int n = 0; n < bd.nbmax; n++) {
    printf("SET REQ [%i] NULL!\n", n);
    bd.flag[n] = BoundaryStatus::waiting;
    #ifdef MPI_PARALLEL
    bd.req_send[n] = MPI_REQUEST_NULL;
    bd.req_recv[n] = MPI_REQUEST_NULL;
    #endif
  }

  // TODO(BRR) More to do here -- see BoundaryVariable!
}

BoundarySwarm::~BoundarySwarm() {
  //DestroyBoundaryData(bd_var_);
}

int BoundarySwarm::ComputeVariableBufferSize(const NeighborIndexes &ni, int cng) {
  return 0;
}

void BoundarySwarm::SetupPersistentMPI() {
  // TODO(BRR) don't actually need this?
#ifdef MPI_PARALLEL
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  int &mylevel = pmb->loc.level;

  // Initialize neighbor communications to other ranks
  int tag;
  int ssize = 0;
  int rsize = 0;
  for (int n = 0; n < pmb->pbval->nneighbor; n++) {
    NeighborBlock &nb = pmb->pbval->neighbor[n];

    printf("rank: %i Neighbor: %i Neighbor rank: %i\n", Globals::my_rank, n, nb.snb.rank);

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
    printf("Done!\n");
  }
#endif
}

// TODO(BRR) is this necessary?
void BoundarySwarm::StartReceiving(BoundaryCommSubset phase) {
  printf("Start receiving!\n");

  // TODO(BRR) Reset tags? This is just MPI_Start for cc vars
  //for (int n = 0; n < pmb->pbval->nneighbor; n++) {
  //  NeighborBlock &nb = pmb->pbval->neighbor[n];
  //}
}

/*void BoundarySwarm::Receive() {
  printf("[%i] BoundarySwarm::Receive\n\n\n", Globals::my_rank);
#ifdef MPI_PARALLEL
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  int &mylevel = pmb->loc.level;

  // Check to see what messages have been received
  int tag;
  int ssize = 0;
  int rsize = 0;
  for (int n = 0; n < pmb->pbval->nneighbor; n++) {
    NeighborBlock &nb = pmb->pbval->neighbor[n];

    tag = pmb->pbval->CreateBvalsMPITag(pmb->lid, nb.bufid, swarm_id_);
    printf("[%i] (%i %i %i) tag: %i\n", Globals::my_rank, pmb->lid, nb.bufid, swarm_id_, tag);

    printf("rank: %i Neighbor: %i Neighbor rank: %i\n", Globals::my_rank, n, nb.snb.rank);
    int test;
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &test, MPI_STATUS_IGNORE);
    MPI_Test(&(bd_var_.req_recv[nb.bufid]), &test, MPI_STATUS_IGNORE);
    if (!static_cast<bool>(test)) {
      bd_var_.flag[nb.bufid] = BoundaryStatus::waiting;
    } else {
      bd_var_.flag[nb.bufid] = BoundaryStatus::arrived;
    }
  }
#endif
}*/

void BoundarySwarm::Send(BoundaryCommSubset phase) {
  printf("BoundarySwarm::Send\n");
#ifdef MPI_PARALLEL
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  int &mylevel = pmb->loc.level;
  for (int n = 0; n < pmb->pbval->nneighbor; n++) {
    NeighborBlock &nb = pmb->pbval->neighbor[n];
    if (nb.snb.rank != Globals::my_rank) {
      // Send a message to different rank neighbor just for fun
      printf("[%i] Sending a message!\n", Globals::my_rank);
      Real buffer[1] = {1.0};
      MPI_Request request;
      MPI_Isend(buffer, 1, MPI_PARTHENON_REAL, nb.snb.rank, 0,
        MPI_COMM_WORLD, &request);
    }
  }
#endif
}

void BoundarySwarm::Receive(BoundaryCommSubset phase) {
  printf("BoundarySwarm::Receive\n");
#ifdef MPI_PARALLEL
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  int &mylevel = pmb->loc.level;
  for (int n = 0; n < pmb->pbval->nneighbor; n++) {
    NeighborBlock &nb = pmb->pbval->neighbor[n];
    if (nb.snb.rank != Globals::my_rank) {
      pmb->exec_space.fence();
      //MPI_Start(&(bd_var_.req_recv[nb.bufid]));
      //MPI_Irecv(
      // Check to see if we got a message
      int test;
      MPI_Status status;
      MPI_Iprobe(nb.snb.rank, MPI_ANY_TAG, MPI_COMM_WORLD, &test, &status);
      printf("[%i] n: %i test; %i\n", Globals::my_rank, n, test);
      if (!static_cast<bool>(test)) {
        bd_var_.flag[nb.bufid] = BoundaryStatus::waiting;
      } else {
        bd_var_.flag[nb.bufid] = BoundaryStatus::arrived;

        // If message is available, receive it
        int nbytes;
        MPI_Get_count(&status, MPI_CHAR, &nbytes);
        printf("Message is this many bytes: %i!", nbytes);
        double *buf = (double*)malloc(nbytes);
        MPI_Recv(buf, nbytes, MPI_CHAR, nb.snb.rank, MPI_ANY_TAG,
          MPI_COMM_WORLD, &status);
        printf("Message received! %e\n", buf[0]);
      }
    }
  }
#endif
}

void BoundarySwarm::ClearBoundary(BoundaryCommSubset phase) {}

int BoundarySwarm::LoadBoundaryBufferSameLevel(ParArray1D<Real> &buf, const NeighborBlock &nb) {
  return 0;}

void BoundarySwarm::SetBoundarySameLevel(ParArray1D<Real> &buf, const NeighborBlock &nb) {}

} // namespace parthenon

