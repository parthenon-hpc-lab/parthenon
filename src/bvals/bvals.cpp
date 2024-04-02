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

#include "bvals/bvals.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "parthenon_mpi.hpp"

#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock.hpp"
#include "parameter_input.hpp"
#include "utils/buffer_utils.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

BoundarySwarm::BoundarySwarm(std::weak_ptr<MeshBlock> pmb, const std::string &label)
    : bswarm_index(), pmy_block(pmb), pmy_mesh_(pmb.lock()->pmy_mesh) {
#ifdef MPI_PARALLEL
  swarm_comm = pmy_mesh_->GetMPIComm(label);
#endif
  InitBoundaryData(bd_var_);
}

void BoundarySwarm::InitBoundaryData(BoundaryData<> &bd) {
  auto pmb = GetBlockPointer();
  BufferID buffer_id(pmb->pmy_mesh->ndim, pmb->pmy_mesh->multilevel);
  bd.nbmax = buffer_id.size();

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
  for (int n = 0; n < pmb->neighbors.size(); n++) {
    NeighborBlock &nb = pmb->neighbors[n];
    // Neighbor on different MPI process
    if (nb.rank != Globals::my_rank) {
      send_tag[nb.bufid] = pmb->pmy_mesh->tag_map.GetTag(pmb.get(), nb);
      recv_tag[nb.bufid] = pmb->pmy_mesh->tag_map.GetTag(pmb.get(), nb);
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
  for (int n = 0; n < pmb->neighbors.size(); n++) {
    NeighborBlock &nb = pmb->neighbors[n];
    if (nb.rank != Globals::my_rank) {
#ifdef MPI_PARALLEL
      PARTHENON_REQUIRE(bd_var_.req_send[nb.bufid] == MPI_REQUEST_NULL,
                        "Trying to create a new send before previous send completes!");
      PARTHENON_MPI_CHECK(MPI_Isend(bd_var_.send[nb.bufid].data(), send_size[nb.bufid],
                                    MPI_PARTHENON_REAL, nb.rank, send_tag[nb.bufid],
                                    swarm_comm, &(bd_var_.req_send[nb.bufid])));
#endif // MPI_PARALLEL
    } else {
      MeshBlock &target_block = *pmy_mesh_->FindMeshBlock(nb.gid);
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
  const int &mylevel = pmb->loc.level();
  for (int n = 0; n < pmb->neighbors.size(); n++) {
    NeighborBlock &nb = pmb->neighbors[n];
    if (nb.rank != Globals::my_rank) {
      // Check to see if we got a message
      int test;
      MPI_Status status;

      if (bd_var_.flag[nb.bufid] != BoundaryStatus::completed) {
        PARTHENON_MPI_CHECK(
            MPI_Iprobe(nb.rank, recv_tag[nb.bufid], swarm_comm, &test, &status));
        if (!static_cast<bool>(test)) {
          bd_var_.flag[nb.bufid] = BoundaryStatus::waiting;
        } else {
          bd_var_.flag[nb.bufid] = BoundaryStatus::arrived;

          // If message is available, receive it
          PARTHENON_MPI_CHECK(
              MPI_Get_count(&status, MPI_PARTHENON_REAL, &(recv_size[nb.bufid])));
          if (recv_size[nb.bufid] > bd_var_.recv[nb.bufid].extent(0)) {
            bd_var_.recv[nb.bufid] = BufArray1D<Real>("Buffer", recv_size[nb.bufid]);
          }
          PARTHENON_MPI_CHECK(MPI_Recv(bd_var_.recv[nb.bufid].data(), recv_size[nb.bufid],
                                       MPI_PARTHENON_REAL, nb.rank, recv_tag[nb.bufid],
                                       swarm_comm, &status));
        }
      }
    }
  }
#endif
}

// BoundarySwarms constructor (the first object constructed inside the MeshBlock()
// constructor): sets functions for the appropriate boundary conditions at each of the 6
// dirs of a MeshBlock
BoundarySwarms::BoundarySwarms(std::weak_ptr<MeshBlock> wpmb, BoundaryFlag *input_bcs,
                               ParameterInput *pin)
    : pmy_block_(wpmb) {
  // Check BC functions for each of the 6 boundaries in turn ---------------------
  // TODO(BRR) Add physical particle boundary conditions, maybe using the below code
  /*for (int i = 0; i < 6; i++) {
    switch (block_bcs[i]) {
    case BoundaryFlag::reflect:
    case BoundaryFlag::outflow:
      apply_bndry_fn_[i] = true;
      break;
    default: // already initialized to false in class
      break;
    }
  }*/
  for (int i = 0; i < 6; ++i)
    block_bcs[i] = input_bcs[i];

  // Inner x1
  nface_ = 2;
  nedge_ = 0;
  CheckBoundaryFlag(block_bcs[BoundaryFace::inner_x1], CoordinateDirection::X1DIR);
  CheckBoundaryFlag(block_bcs[BoundaryFace::outer_x1], CoordinateDirection::X1DIR);

  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  if (!pmb->block_size.symmetry(X2DIR)) {
    nface_ = 4;
    nedge_ = 4;
    CheckBoundaryFlag(block_bcs[BoundaryFace::inner_x2], CoordinateDirection::X2DIR);
    CheckBoundaryFlag(block_bcs[BoundaryFace::outer_x2], CoordinateDirection::X2DIR);
  }

  if (!pmb->block_size.symmetry(X3DIR)) {
    nface_ = 6;
    nedge_ = 12;
    CheckBoundaryFlag(block_bcs[BoundaryFace::inner_x3], CoordinateDirection::X3DIR);
    CheckBoundaryFlag(block_bcs[BoundaryFace::outer_x3], CoordinateDirection::X3DIR);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void BoundarySwarms::SetupPersistentMPI()
//  \brief Setup persistent MPI requests to be reused throughout the entire simulation

void BoundarySwarms::SetupPersistentMPI() {
  for (auto bswarms_it = bswarms.begin(); bswarms_it != bswarms.end(); ++bswarms_it) {
    (*bswarms_it)->SetupPersistentMPI();
  }
}

} // namespace parthenon
