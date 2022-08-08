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

#include "parthenon_mpi.hpp"

#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

BoundaryVariable::BoundaryVariable(std::weak_ptr<MeshBlock> pmb, bool is_sparse,
                                   const std::string &label)
    : pmy_block_(pmb), pmy_mesh_(pmb.lock()->pmy_mesh), is_sparse_(is_sparse),
      label_(label) {
#ifdef ENABLE_SPARSE
  // if this is a sparse variable, local neighbor allocation status will be set later, we
  // initialize it to false here. For dense variable we initialize to true, as all local
  // neighbors will always have this variable allocated
  for (int i = 0; i < NMAX_NEIGHBORS; ++i) {
    local_neighbor_allocated[i] = !is_sparse_;
  }
#endif
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
    bd.recv_h[n] = Kokkos::create_mirror_view(bd.recv[n]);
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

void BoundaryVariable::CopyFluxCorrectionBufferSameProcess(NeighborBlock &nb) {
  // Locate target buffer
  // 1) which MeshBlock?
  MeshBlock &target_block = *pmy_mesh_->FindMeshBlock(nb.snb.gid);
  // 2) which element in vector of BoundaryVariable *?
  BoundaryData<> *ptarget_bdata = &(target_block.pbval->bvars.at(label_)->bd_var_flcor_);
  target_block.deep_copy(ptarget_bdata->recv[nb.targetid], bd_var_flcor_.send[nb.bufid]);
  ptarget_bdata->flag[nb.targetid] = BoundaryStatus::arrived;
  return;
}

// Default / shared implementations of 4x BoundaryBuffer public functions

//----------------------------------------------------------------------------------------
//! \fn bool BoundaryVariable::ReceiveBoundaryBuffers()
//  \brief receive the boundary data

bool BoundaryVariable::ReceiveBoundaryBuffers(bool is_allocated) {
  bool bflag = true;

  auto pmb = GetBlockPointer();
  for (int n = 0; n < pmb->pbval->nneighbor; n++) {
    NeighborBlock &nb = pmb->pbval->neighbor[n];
    if (bd_var_.flag[nb.bufid] == BoundaryStatus::arrived) continue;
    if (bd_var_.flag[nb.bufid] == BoundaryStatus::waiting) {
      if (nb.snb.rank == Globals::my_rank) { // on the same process
        // if this variable is allocated, we wait to get boundary data, otherwise we don't
        // care and we'll mark this boundary as complete at the bottom
        if (is_allocated) {
          bflag = false;
          continue;
        }
      } else {
#ifdef MPI_PARALLEL
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
#endif

        if (!is_allocated) {
          // we have to copy flag at the end of recv buffer to host to read it
          const auto idx = bd_var_.recv_size[nb.bufid] - 1;
          // the flag lives on the device, copy it to host
          Real flag;
          Kokkos::deep_copy(flag, Kokkos::subview(bd_var_.recv[nb.bufid], idx));

          // check if we need to allocate this variable if it's not allocated
          if (flag == 1.0) {
            // we need to allocate this variable
            pmb->AllocateSparse(label());
          }
        }
      }

      bd_var_.flag[nb.bufid] = BoundaryStatus::arrived;
    }
  }
  return bflag;
}

} // namespace parthenon
