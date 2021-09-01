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
//! \file bvals_var.cpp
//  \brief constructor/destructor and default implementations for some functions in the
//         abstract BoundaryVariable class

#include "bvals/bvals_interfaces.hpp"

#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "bvals/cc/bvals_cc.hpp"
#include "parthenon_mpi.hpp"

#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

BoundaryVariable::BoundaryVariable(std::weak_ptr<MeshBlock> pmb, bool is_sparse)
    : bvar_index(), pmy_block_(pmb), pmy_mesh_(pmb.lock()->pmy_mesh),
      is_sparse_(is_sparse) {
  // if this is a sparse variable, neighbor allocation status will be set later, we
  // initialize it to false here. For dense variable we initialize to true, as all
  // neighbors will always have this variable allocated
  for (int i = 0; i < NMAX_NEIGHBORS; ++i) {
    neighbor_allocated[i] = !is_sparse_;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void BoundaryVariable::InitBoundaryData(BoundaryData<> &bd, BoundaryQuantity type)
//  \brief Initialize BoundaryData structure

void BoundaryVariable::InitBoundaryData(BoundaryData<> &bd, BoundaryQuantity type) {
  auto pmb = GetBlockPointer();
  NeighborIndexes *ni = pmb->pbval->ni;
  int cng = pmb->cnghost;
  int size = 0;

  bd.nbmax = pmb->pbval->maxneighbor_;
  // KGF: what is happening in the next two conditionals??
  // they are preventing the elimination of "BoundaryQuantity type" function parameter in
  // favor of a simpler boolean switch
  if (type == BoundaryQuantity::cc_flcor || type == BoundaryQuantity::fc_flcor) {
    for (bd.nbmax = 0; pmb->pbval->ni[bd.nbmax].type == NeighborConnect::face;
         bd.nbmax++) {
    }
  }
  if (type == BoundaryQuantity::fc_flcor) {
    for (; pmb->pbval->ni[bd.nbmax].type == NeighborConnect::edge; bd.nbmax++) {
    }
  }
  auto total_size = 0;
  std::vector<size_t> offsets;
  offsets.reserve(bd.nbmax + 1);

  for (int n = 0; n < bd.nbmax; n++) {
    // Clear flags and requests
    bd.flag[n] = BoundaryStatus::waiting;
    bd.sflag[n] = BoundaryStatus::waiting;
#ifdef MPI_PARALLEL
    bd.req_send[n] = MPI_REQUEST_NULL;
    bd.req_recv[n] = MPI_REQUEST_NULL;
#endif
    // Allocate buffers, calculating the buffer size (variable vs. flux correction)
    if (type == BoundaryQuantity::cc || type == BoundaryQuantity::fc) {
      size = this->ComputeVariableBufferSize(ni[n], cng);
    } else if (type == BoundaryQuantity::cc_flcor || type == BoundaryQuantity::fc_flcor) {
      size = this->ComputeFluxCorrectionBufferSize(ni[n], cng);
    } else {
      std::stringstream msg;
      msg << "### FATAL ERROR in InitBoundaryData" << std::endl
          << "Invalid boundary type is specified." << std::endl;
      PARTHENON_FAIL(msg);
    }
    offsets.push_back(total_size);
    total_size += size;
  }
  bd.buffers = BufArray1D<Real>("comm buffers", 2 * total_size);
  offsets.push_back(total_size);
  for (int n = 0; n < bd.nbmax; n++) {
    if (offsets.at(n) == offsets.at(n + 1)) {
      continue;
    }
    bd.send[n] =
        BufArray1D<Real>(bd.buffers, std::make_pair(offsets.at(n), offsets.at(n + 1)));
    bd.recv[n] =
        BufArray1D<Real>(bd.buffers, std::make_pair(offsets.at(n) + total_size,
                                                    offsets.at(n + 1) + total_size));
  }
}

//----------------------------------------------------------------------------------------
//! \fn void BoundaryVariable::DestroyBoundaryData(BoundaryData<> &bd)
//  \brief Destroy BoundaryData structure

void BoundaryVariable::DestroyBoundaryData(BoundaryData<> &bd) {
  for (int n = 0; n < bd.nbmax; n++) {
#ifdef MPI_PARALLEL
    if (bd.req_send[n] != MPI_REQUEST_NULL) MPI_Request_free(&bd.req_send[n]);
    if (bd.req_recv[n] != MPI_REQUEST_NULL) MPI_Request_free(&bd.req_recv[n]);
#endif
  }
}

//----------------------------------------------------------------------------------------
//! \fn void BoundaryVariable::CopyVariableBufferSameProcess(NeighborBlock& nb, int ssize)
//  \brief

//  Called in BoundaryVariable::SendBoundaryBuffer(), SendFluxCorrection() calls when the
//  destination neighbor block is on the same MPI rank as the sending MeshBlock. So
//  std::memcpy() call requires pointer to "void *dst" corresponding to
//  bd_var_.recv[nb.targetid] in separate BoundaryVariable object in separate vector in
//  separate BoundaryValues

void BoundaryVariable::CopyVariableBufferSameProcess(NeighborBlock &nb, int ssize) {
  // Locate target buffer
  // 1) which MeshBlock?
  MeshBlock &target_block = *pmy_mesh_->FindMeshBlock(nb.snb.gid);
  // 2) which element in vector of BoundaryVariable *?
  BoundaryData<> *ptarget_bdata = &(target_block.pbval->bvars[bvar_index]->bd_var_);
  target_block.deep_copy(ptarget_bdata->recv[nb.targetid], bd_var_.send[nb.bufid]);
  // finally, set the BoundaryStatus flag on the destination buffer
  ptarget_bdata->flag[nb.targetid] = BoundaryStatus::arrived;
  return;
}

// KGF: change ssize to send_count

void BoundaryVariable::CopyFluxCorrectionBufferSameProcess(NeighborBlock &nb, int ssize) {
  // Locate target buffer
  // 1) which MeshBlock?
  MeshBlock &target_block = *pmy_mesh_->FindMeshBlock(nb.snb.gid);
  // 2) which element in vector of BoundaryVariable *?
  BoundaryData<> *ptarget_bdata = &(target_block.pbval->bvars[bvar_index]->bd_var_flcor_);
  target_block.deep_copy(ptarget_bdata->recv[nb.targetid], bd_var_flcor_.send[nb.bufid]);
  ptarget_bdata->flag[nb.targetid] = BoundaryStatus::arrived;
  return;
}

// Default / shared implementations of 4x BoundaryBuffer public functions

//----------------------------------------------------------------------------------------
//! \fn void BoundaryVariable::SendBoundaryBuffers()
//  \brief Send boundary buffers of variables

void BoundaryVariable::SendBoundaryBuffers() {
  auto pmb = GetBlockPointer();
  int mylevel = pmb->loc.level;
  for (int n = 0; n < pmb->pbval->nneighbor; n++) {
    if (!neighbor_allocated[n]) continue;

    NeighborBlock &nb = pmb->pbval->neighbor[n];
    if (bd_var_.sflag[nb.bufid] == BoundaryStatus::completed) continue;
    int ssize;
    if (nb.snb.level == mylevel)
      ssize = LoadBoundaryBufferSameLevel(bd_var_.send[nb.bufid], nb);
    else if (nb.snb.level < mylevel)
      ssize = LoadBoundaryBufferToCoarser(bd_var_.send[nb.bufid], nb);
    else
      ssize = LoadBoundaryBufferToFiner(bd_var_.send[nb.bufid], nb);
    // fence to make sure buffers are loaded and ready to send
    pmb->exec_space.fence();
    if (nb.snb.rank == Globals::my_rank) {
      // on the same process
      CopyVariableBufferSameProcess(nb, ssize);
    } else {
#ifdef MPI_PARALLEL
      PARTHENON_MPI_CHECK(MPI_Start(&(bd_var_.req_send[nb.bufid])));
#endif
    }

    bd_var_.sflag[nb.bufid] = BoundaryStatus::completed;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn bool BoundaryVariable::ReceiveBoundaryBuffers()
//  \brief receive the boundary data

bool BoundaryVariable::ReceiveBoundaryBuffers() {
  bool bflag = true;

  auto pmb = GetBlockPointer();
  for (int n = 0; n < pmb->pbval->nneighbor; n++) {
    if (!neighbor_allocated[n]) continue;

    NeighborBlock &nb = pmb->pbval->neighbor[n];
    if (bd_var_.flag[nb.bufid] == BoundaryStatus::arrived) continue;
    if (bd_var_.flag[nb.bufid] == BoundaryStatus::waiting) {
      if (nb.snb.rank == Globals::my_rank) { // on the same process
        printf("Block %4i is waiting to get boundary data from block %4i for %s "
               "(nb.bufid = % 2i, nb.targetid = % 2i)\n",
               pmb->gid, nb.snb.gid,
               dynamic_cast<CellCenteredBoundaryVariable *>(this)->label.c_str(),
               nb.bufid, nb.targetid);
        bflag = false;
        continue;
      }
#ifdef MPI_PARALLEL
      else { // NOLINT // MPI boundary
        int test;
        // Comment from original Athena++ code about the MPI_Iprobe call:
        //
        // Although MPI_Iprobe does nothing for us (it checks arrival of any message but
        // we do not use the result), this is ABSOLUTELY NECESSARY for the performance of
        // Athena++. Although non-blocking MPI communications look like multi-tasking
        // running behind our code, actually they are not. The network interface card can
        // run autonomously from the CPU, but to move the data between the memory and the
        // network interface and initiate/complete communications, MPI has to do something
        // using CPU. So to process communications, we have to allow MPI to use CPU.
        // Theoretically MPI can use multi-thread for this (OpenMPI can be configured so)
        // but it is not common because of performance and compatibility issues. Instead,
        // MPI processes communications whenever any MPI function is called. MPI_Iprobe is
        // one of the cheapest function in MPI and by calling this occasionally MPI can
        // process communications "as if it is in the background". Using only MPI_Test,
        // the communications were very slow. I suspect that MPI_Test changes the ordering
        // of the messages internally (I guess it tries to promote the message it is
        // Testing), and if we call MPI_Test for different messages, they are left half
        // done. So if we remove them, I am sure we will see significant performance drop.
        // I could not dig it up right now, Collela or Woodward mentioned this in a paper.
        PARTHENON_MPI_CHECK(MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &test,
                                       MPI_STATUS_IGNORE));
        PARTHENON_MPI_CHECK(
            MPI_Test(&(bd_var_.req_recv[nb.bufid]), &test, MPI_STATUS_IGNORE));
        if (!static_cast<bool>(test)) {
          bflag = false;
          continue;
        }
        bd_var_.flag[nb.bufid] = BoundaryStatus::arrived;
      }
#endif
    }
  }
  return bflag;
}

//----------------------------------------------------------------------------------------
//! \fn void BoundaryVariable::SetBoundaries()
//  \brief set the boundary data

void BoundaryVariable::SetBoundaries() {
  auto pmb = GetBlockPointer();
  int mylevel = pmb->loc.level;
  for (int n = 0; n < pmb->pbval->nneighbor; n++) {
    NeighborBlock &nb = pmb->pbval->neighbor[n];
    if (nb.snb.level == mylevel)
      // TODO(pgrete) FIX interface
      SetBoundarySameLevel(bd_var_.recv[nb.bufid], nb);
    else if (nb.snb.level < mylevel) // only sets the prolongation buffer
      SetBoundaryFromCoarser(bd_var_.recv[nb.bufid], nb);
    else
      SetBoundaryFromFiner(bd_var_.recv[nb.bufid], nb);
    bd_var_.flag[nb.bufid] = BoundaryStatus::completed; // completed
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void BoundaryVariable::ReceiveAndSetBoundariesWithWait()
//  \brief receive and set the boundary data for initialization

void BoundaryVariable::ReceiveAndSetBoundariesWithWait() {
  auto pmb = GetBlockPointer();
  int mylevel = pmb->loc.level;
  pmb->exec_space.fence();
  for (int n = 0; n < pmb->pbval->nneighbor; n++) {
    NeighborBlock &nb = pmb->pbval->neighbor[n];
#ifdef MPI_PARALLEL
    if (nb.snb.rank != Globals::my_rank) {
      PARTHENON_MPI_CHECK(MPI_Wait(&(bd_var_.req_recv[nb.bufid]), MPI_STATUS_IGNORE));
    }
#endif
    if (nb.snb.level == mylevel)
      SetBoundarySameLevel(bd_var_.recv[nb.bufid], nb);
    else if (nb.snb.level < mylevel)
      SetBoundaryFromCoarser(bd_var_.recv[nb.bufid], nb);
    else
      SetBoundaryFromFiner(bd_var_.recv[nb.bufid], nb);
    bd_var_.flag[nb.bufid] = BoundaryStatus::completed; // completed
  }

  return;
}

} // namespace parthenon
