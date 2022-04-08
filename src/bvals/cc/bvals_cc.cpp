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

//! \file bvals_cc.cpp
//  \brief functions that apply BCs for CELL_CENTERED variables

#include "bvals/cc/bvals_cc.hpp"
#include "bvals/cc/bvals_cc_in_one.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mpi.h>
#include <sstream>
#include <stdexcept>
#include <string>

#include "parthenon_mpi.hpp"

#include "basic_types.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock.hpp"
#include "parameter_input.hpp"
#include "utils/buffer_utils.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

CellCenteredBoundaryVariable::CellCenteredBoundaryVariable(std::weak_ptr<MeshBlock> pmb,
                                                           bool is_sparse,
                                                           const std::string &label,
                                                           int dim4)
    : BoundaryVariable(pmb, is_sparse, label), nl_(0), nu_(dim4 - 1) {
  // CellCenteredBoundaryVariable should only be used w/ 4D or 3D (nx4=1) ParArrayND
  // For now, assume that full span of 4th dim of input ParArrayND should be used:
  // ---> get the index limits directly from the input ParArrayND
  // <=nu_ (inclusive), <nx4 (exclusive)
  if (nu_ < 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in CellCenteredBoundaryVariable constructor" << std::endl
        << "An 'ParArrayND<Real> *var' of nx4_ = " << dim4 << " was passed\n"
        << "Should be nx4 >= 1 (likely uninitialized)." << std::endl;
    PARTHENON_FAIL(msg);
  }

  InitBoundaryData(bd_var_, BoundaryQuantity::cc);

#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Comm_dup(MPI_COMM_WORLD, &cc_var_comm));
#endif
  if (pmy_mesh_->multilevel) { // SMR or AMR
    InitBoundaryData(bd_var_flcor_, BoundaryQuantity::cc_flcor);
#ifdef MPI_PARALLEL
    PARTHENON_MPI_CHECK(MPI_Comm_dup(MPI_COMM_WORLD, &cc_flcor_comm));
#endif
  }
}

// destructor

CellCenteredBoundaryVariable::~CellCenteredBoundaryVariable() {
  DestroyBoundaryData(bd_var_);
  if (pmy_mesh_->multilevel) DestroyBoundaryData(bd_var_flcor_);
}

void CellCenteredBoundaryVariable::Reset(ParArrayND<Real> var,
                                         ParArrayND<Real> coarse_var,
                                         ParArrayND<Real> *var_flux) {
  var_cc = var;
  coarse_buf = coarse_var;
  x1flux = var_flux[X1DIR];
  x2flux = var_flux[X2DIR];
  x3flux = var_flux[X3DIR];
}

CellCenteredBoundaryVariable::VariableBufferSizes
CellCenteredBoundaryVariable::ComputeVariableBufferSizes(const NeighborIndexes &ni,
                                                         int cng) {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  int cng1, cng2, cng3;
  cng1 = cng;
  cng2 = cng * (pmb->block_size.nx2 > 1 ? 1 : 0);
  cng3 = cng * (pmb->block_size.nx3 > 1 ? 1 : 0);

  VariableBufferSizes res;
  res.same = ((ni.ox1 == 0) ? pmb->block_size.nx1 : Globals::nghost) *
             ((ni.ox2 == 0) ? pmb->block_size.nx2 : Globals::nghost) *
             ((ni.ox3 == 0) ? pmb->block_size.nx3 : Globals::nghost);

  res.f2c = ((ni.ox1 == 0) ? ((pmb->block_size.nx1 + 1) / 2) : Globals::nghost) *
            ((ni.ox2 == 0) ? ((pmb->block_size.nx2 + 1) / 2) : Globals::nghost) *
            ((ni.ox3 == 0) ? ((pmb->block_size.nx3 + 1) / 2) : Globals::nghost);
  res.c2f = ((ni.ox1 == 0) ? ((pmb->block_size.nx1 + 1) / 2 + cng1) : cng) *
            ((ni.ox2 == 0) ? ((pmb->block_size.nx2 + 1) / 2 + cng2) : cng) *
            ((ni.ox3 == 0) ? ((pmb->block_size.nx3 + 1) / 2 + cng3) : cng);

  return res;
}

int CellCenteredBoundaryVariable::ComputeVariableBufferSize(const NeighborIndexes &ni,
                                                            int cng) {
  auto sizes = ComputeVariableBufferSizes(ni, cng);
  int size = sizes.same;
  if (pmy_mesh_->multilevel) {
    size = std::max(size, sizes.c2f);
    size = std::max(size, sizes.f2c);
  }
  size *= nu_ + 1;

  // adding 1 to the size to communicate allocation status
  return size + 1;
}

int CellCenteredBoundaryVariable::ComputeFluxCorrectionBufferSize(
    const NeighborIndexes &ni, int cng) {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  int size = 0;
  if (ni.ox1 != 0)
    size = (pmb->block_size.nx2 + 1) / 2 * (pmb->block_size.nx3 + 1) / 2 * (nu_ + 1);
  if (ni.ox2 != 0)
    size = (pmb->block_size.nx1 + 1) / 2 * (pmb->block_size.nx3 + 1) / 2 * (nu_ + 1);
  if (ni.ox3 != 0)
    size = (pmb->block_size.nx1 + 1) / 2 * (pmb->block_size.nx2 + 1) / 2 * (nu_ + 1);

  // adding 1 to the size to communicate allocation status
  return size + 1;
}

void CellCenteredBoundaryVariable::SetupPersistentMPI() {
#ifdef MPI_PARALLEL
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  int &mylevel = pmb->loc.level;

  int ssize, rsize;
  int tag;
  // Initialize non-polar neighbor communications to other ranks
  for (int n = 0; n < pmb->pbval->nneighbor; n++) {
    NeighborBlock &nb = pmb->pbval->neighbor[n];
    if (nb.snb.rank != Globals::my_rank) {
      auto sizes = ComputeVariableBufferSizes(nb.ni, pmb->cnghost);

      if (nb.snb.level == mylevel) { // same
        ssize = rsize = sizes.same;
      } else if (nb.snb.level < mylevel) { // coarser
        ssize = sizes.f2c;
        rsize = sizes.c2f;
      } else { // finer
        ssize = sizes.c2f;
        rsize = sizes.f2c;
      }
      ssize *= (nu_ + 1);
      rsize *= (nu_ + 1);
      // specify the offsets in the view point of the target block: flip ox? signs
      PARTHENON_DEBUG_REQUIRE(ssize > 0, "Send size is 0");
      PARTHENON_DEBUG_REQUIRE(rsize > 0, "Receive size is 0");

      // we add one to the send and receive buffer sizes, which will be used to
      // communicate the allocation status
      ssize += 1;
      rsize += 1;

      bd_var_.recv_size[nb.bufid] = rsize;

      // Initialize persistent communication requests attached to specific BoundaryData
      tag = pmb->pbval->CreateBvalsMPITag(nb.snb.lid, nb.targetid);
      if (bd_var_.req_send[nb.bufid] != MPI_REQUEST_NULL)
        MPI_Request_free(&bd_var_.req_send[nb.bufid]);
      PARTHENON_MPI_CHECK(MPI_Send_init(bd_var_.send[nb.bufid].data(), ssize,
                                        MPI_PARTHENON_REAL, nb.snb.rank, tag, cc_var_comm,
                                        &(bd_var_.req_send[nb.bufid])));
      tag = pmb->pbval->CreateBvalsMPITag(pmb->lid, nb.bufid);
      if (bd_var_.req_recv[nb.bufid] != MPI_REQUEST_NULL)
        MPI_Request_free(&bd_var_.req_recv[nb.bufid]);
      PARTHENON_MPI_CHECK(MPI_Recv_init(bd_var_.recv[nb.bufid].data(), rsize,
                                        MPI_PARTHENON_REAL, nb.snb.rank, tag, cc_var_comm,
                                        &(bd_var_.req_recv[nb.bufid])));

      if (pmy_mesh_->multilevel && nb.ni.type == NeighborConnect::face) {
        // TODO(JL): could we call ComputeFluxCorrectionBufferSize here to reduce code
        // duplication?
        int size;
        if (nb.fid == 0 || nb.fid == 1)
          size = ((pmb->block_size.nx2 + 1) / 2) * ((pmb->block_size.nx3 + 1) / 2);
        else if (nb.fid == 2 || nb.fid == 3)
          size = ((pmb->block_size.nx1 + 1) / 2) * ((pmb->block_size.nx3 + 1) / 2);
        else // (nb.fid == 4 || nb.fid == 5)
          size = ((pmb->block_size.nx1 + 1) / 2) * ((pmb->block_size.nx2 + 1) / 2);
        size *= (nu_ + 1);

        // one more value to communicate if source has variable allocated
        size += 1;

        if (nb.snb.level < mylevel) { // send to coarser
          tag = pmb->pbval->CreateBvalsMPITag(nb.snb.lid, nb.targetid);
          if (bd_var_flcor_.req_send[nb.bufid] != MPI_REQUEST_NULL)
            MPI_Request_free(&bd_var_flcor_.req_send[nb.bufid]);
          PARTHENON_MPI_CHECK(MPI_Send_init(
              bd_var_flcor_.send[nb.bufid].data(), size, MPI_PARTHENON_REAL, nb.snb.rank,
              tag, cc_flcor_comm, &(bd_var_flcor_.req_send[nb.bufid])));
        } else if (nb.snb.level > mylevel) { // receive from finer
          tag = pmb->pbval->CreateBvalsMPITag(pmb->lid, nb.bufid);
          if (bd_var_flcor_.req_recv[nb.bufid] != MPI_REQUEST_NULL)
            MPI_Request_free(&bd_var_flcor_.req_recv[nb.bufid]);
          PARTHENON_MPI_CHECK(MPI_Recv_init(
              bd_var_flcor_.recv[nb.bufid].data(), size, MPI_PARTHENON_REAL, nb.snb.rank,
              tag, cc_var_comm, &(bd_var_flcor_.req_recv[nb.bufid])));
        }
      }
    }
  }
#endif
  return;
}

void CellCenteredBoundaryVariable::StartReceiving(BoundaryCommSubset phase) {
#ifdef MPI_PARALLEL
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  int mylevel = pmb->loc.level;
  for (int n = 0; n < pmb->pbval->nneighbor; n++) {
    NeighborBlock &nb = pmb->pbval->neighbor[n];
    if (nb.snb.rank != Globals::my_rank) {
      pmb->exec_space.fence();
      PARTHENON_REQUIRE_THROWS(bd_var_.req_recv[nb.bufid] != MPI_REQUEST_NULL,
                               "Trying to start a null request");

      PARTHENON_MPI_CHECK(MPI_Start(&(bd_var_.req_recv[nb.bufid])));
      if (phase == BoundaryCommSubset::all && nb.ni.type == NeighborConnect::face &&
          nb.snb.level > mylevel) // opposite condition in ClearBoundary()
        PARTHENON_MPI_CHECK(MPI_Start(&(bd_var_flcor_.req_recv[nb.bufid])));
    }
  }
#endif
  return;
}

void CellCenteredBoundaryVariable::ClearBoundary(BoundaryCommSubset phase) {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  for (int n = 0; n < pmb->pbval->nneighbor; n++) {
    NeighborBlock &nb = pmb->pbval->neighbor[n];
    bd_var_.flag[nb.bufid] = BoundaryStatus::waiting;
    bd_var_.sflag[nb.bufid] = BoundaryStatus::waiting;

    if (nb.ni.type == NeighborConnect::face) {
      bd_var_flcor_.flag[nb.bufid] = BoundaryStatus::waiting;
      bd_var_flcor_.sflag[nb.bufid] = BoundaryStatus::waiting;
    }
#ifdef MPI_PARALLEL
    int mylevel = pmb->loc.level;
    if (nb.snb.rank != Globals::my_rank) {
      pmb->exec_space.fence();
      // Wait for Isend
      PARTHENON_MPI_CHECK(MPI_Wait(&(bd_var_.req_send[nb.bufid]), MPI_STATUS_IGNORE));
      if (phase == BoundaryCommSubset::all && nb.ni.type == NeighborConnect::face &&
          nb.snb.level < mylevel)
        PARTHENON_MPI_CHECK(
            MPI_Wait(&(bd_var_flcor_.req_send[nb.bufid]), MPI_STATUS_IGNORE));
    }
#endif
  }
}

} // namespace parthenon
