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
      printf("%s:%i\n", __FILE__, __LINE__);
      tag = pmb->pbval->CreateBvalsMPITag(nb.snb.lid, nb.targetid, swarm_id_);
      printf("%s:%i\n", __FILE__, __LINE__);
      if (bd_var_.req_send[nb.bufid] != MPI_REQUEST_NULL) {
      printf("%s:%i\n", __FILE__, __LINE__);
        MPI_Request_free(&bd_var_.req_send[nb.bufid]);
      }
      printf("%s:%i\n", __FILE__, __LINE__);
      MPI_Send_init(bd_var_.send[nb.bufid].data(), ssize, MPI_PARTHENON_REAL, nb.snb.rank,
                    tag, MPI_COMM_WORLD, &(bd_var_.req_recv[nb.bufid]));
      printf("%s:%i\n", __FILE__, __LINE__);
      //tag = pmb->pbval->CreateBvalsMPITag(pmb->lid, nb.bufid, swarm_id_);
      //if (bd_var_.req_recv[nb.bufid] != MPI_REQUEST_NULL) {
      //  MPI_Request_free(&bd_var_.req_recv[nb.bufid]);
     // }
      //MPI_Recv

    }
  }
#endif
}

void BoundarySwarm::StartReceiving(BoundaryCommSubset phase) {
  printf("BoundarySwarm::StartReceiving\n");
#ifdef MPI_PARALLEL
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  int &mylevel = pmb->loc.level;
  for (int n = 0; n < pmb->pbval->nneighbor; n++) {
    NeighborBlock &nb = pmb->pbval->neighbor[n];
    if (nb.snb.rank != Globals::my_rank) {
    }
  }
#endif
}

void BoundarySwarm::ClearBoundary(BoundaryCommSubset phase) {}

int BoundarySwarm::LoadBoundaryBufferSameLevel(ParArray1D<Real> &buf, const NeighborBlock &nb) {
  return 0;}

void BoundarySwarm::SetBoundarySameLevel(ParArray1D<Real> &buf, const NeighborBlock &nb) {}

} // namespace parthenon

