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

CellCenteredBoundaryVariable::CellCenteredBoundaryVariable(
    std::weak_ptr<MeshBlock> pmb, ParArrayND<Real> var, ParArrayND<Real> coarse_var,
    ParArrayND<Real> var_flux[], bool is_sparse, const std::string &label)
    : BoundaryVariable(pmb, is_sparse), var_cc(var), coarse_buf(coarse_var),
      x1flux(var_flux[X1DIR]), x2flux(var_flux[X2DIR]), x3flux(var_flux[X3DIR]),
      label(label), nl_(0), nu_(var.GetDim(4) - 1) {
  // CellCenteredBoundaryVariable should only be used w/ 4D or 3D (nx4=1) ParArrayND
  // For now, assume that full span of 4th dim of input ParArrayND should be used:
  // ---> get the index limits directly from the input ParArrayND
  // <=nu_ (inclusive), <nx4 (exclusive)
  if (nu_ < 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in CellCenteredBoundaryVariable constructor" << std::endl
        << "An 'ParArrayND<Real> *var' of nx4_ = " << var.GetDim(4) << " was passed\n"
        << "Should be nx4 >= 1 (likely uninitialized)." << std::endl;
    PARTHENON_FAIL(msg);
  }

  InitBoundaryData(bd_var_, BoundaryQuantity::cc);
#ifdef MPI_PARALLEL
  // KGF: dead code, leaving for now:
  // cc_phys_id_ = pmb->pbval->ReserveTagVariableIDs(1);
  cc_phys_id_ = pmb.lock()->pbval->bvars_next_phys_id_;
#endif
  if (pmy_mesh_->multilevel) { // SMR or AMR
    InitBoundaryData(bd_var_flcor_, BoundaryQuantity::cc_flcor);
#ifdef MPI_PARALLEL
    cc_flx_phys_id_ = cc_phys_id_ + 1;
#endif
  }
}

// destructor

CellCenteredBoundaryVariable::~CellCenteredBoundaryVariable() {
  DestroyBoundaryData(bd_var_);
  if (pmy_mesh_->multilevel) DestroyBoundaryData(bd_var_flcor_);
}

int CellCenteredBoundaryVariable::ComputeVariableBufferSize(const NeighborIndexes &ni,
                                                            int cng) {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  int cng1, cng2, cng3;
  cng1 = cng;
  cng2 = cng * (pmb->block_size.nx2 > 1 ? 1 : 0);
  cng3 = cng * (pmb->block_size.nx3 > 1 ? 1 : 0);

  int size = ((ni.ox1 == 0) ? pmb->block_size.nx1 : Globals::nghost) *
             ((ni.ox2 == 0) ? pmb->block_size.nx2 : Globals::nghost) *
             ((ni.ox3 == 0) ? pmb->block_size.nx3 : Globals::nghost);
  if (pmy_mesh_->multilevel) {
    int f2c = ((ni.ox1 == 0) ? ((pmb->block_size.nx1 + 1) / 2) : Globals::nghost) *
              ((ni.ox2 == 0) ? ((pmb->block_size.nx2 + 1) / 2) : Globals::nghost) *
              ((ni.ox3 == 0) ? ((pmb->block_size.nx3 + 1) / 2) : Globals::nghost);
    int c2f = ((ni.ox1 == 0) ? ((pmb->block_size.nx1 + 1) / 2 + cng1) : cng) *
              ((ni.ox2 == 0) ? ((pmb->block_size.nx2 + 1) / 2 + cng2) : cng) *
              ((ni.ox3 == 0) ? ((pmb->block_size.nx3 + 1) / 2 + cng3) : cng);
    size = std::max(size, c2f);
    size = std::max(size, f2c);
  }
  size *= nu_ + 1;
  return size;
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
  return size;
}

//----------------------------------------------------------------------------------------
//! \fn int CellCenteredBoundaryVariable::LoadBoundaryBufferSameLevel(BufArray1D<Real>
//! &buf,
//                                                                const NeighborBlock& nb)
//  \brief Set cell-centered boundary buffers for sending to a block on the same level

int CellCenteredBoundaryVariable::LoadBoundaryBufferSameLevel(BufArray1D<Real> &buf,
                                                              const NeighborBlock &nb) {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  int si, sj, sk, ei, ej, ek;

  IndexDomain interior = IndexDomain::interior;
  const IndexShape &cellbounds = pmb->cellbounds;

  cell_centered_bvars::CalcIndicesLoadSame(nb.ni.ox1, si, ei,
                                           cellbounds.GetBoundsI(interior));
  cell_centered_bvars::CalcIndicesLoadSame(nb.ni.ox2, sj, ej,
                                           cellbounds.GetBoundsJ(interior));
  cell_centered_bvars::CalcIndicesLoadSame(nb.ni.ox3, sk, ek,
                                           cellbounds.GetBoundsK(interior));
  int p = 0;

  ParArray4D<Real> var_cc_ = var_cc.Get<4>(); // automatic template deduction fails
  BufferUtility::PackData(var_cc_, buf, nl_, nu_, si, ei, sj, ej, sk, ek, p, pmb.get());

  return p;
}

//----------------------------------------------------------------------------------------
//! \fn int CellCenteredBoundaryVariable::LoadBoundaryBufferToCoarser(BufArray1D<Real>
//! &buf,
//                                                                const NeighborBlock& nb)
//  \brief Set cell-centered boundary buffers for sending to a block on the coarser level

int CellCenteredBoundaryVariable::LoadBoundaryBufferToCoarser(BufArray1D<Real> &buf,
                                                              const NeighborBlock &nb) {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  int si, sj, sk, ei, ej, ek;

  IndexDomain interior = IndexDomain::interior;
  const IndexShape &c_cellbounds = pmb->c_cellbounds;
  // "Same" logic is the same for loading to a coarse buffer, just using c_cellbounds
  cell_centered_bvars::CalcIndicesLoadSame(nb.ni.ox1, si, ei,
                                           c_cellbounds.GetBoundsI(interior));
  cell_centered_bvars::CalcIndicesLoadSame(nb.ni.ox2, sj, ej,
                                           c_cellbounds.GetBoundsJ(interior));
  cell_centered_bvars::CalcIndicesLoadSame(nb.ni.ox3, sk, ek,
                                           c_cellbounds.GetBoundsK(interior));

  int p = 0;
  pmb->pmr->RestrictCellCenteredValues(var_cc, coarse_buf, nl_, nu_, si, ei, sj, ej, sk,
                                       ek);
  ParArray4D<Real> coarse_buf_ = coarse_buf.Get<4>(); // auto template deduction fails
  BufferUtility::PackData(coarse_buf_, buf, nl_, nu_, si, ei, sj, ej, sk, ek, p,
                          pmb.get());
  return p;
}

//----------------------------------------------------------------------------------------
//! \fn int CellCenteredBoundaryVariable::LoadBoundaryBufferToFiner(BufArray1D<Real> &buf,
//                                                                const NeighborBlock& nb)
//  \brief Set cell-centered boundary buffers for sending to a block on the finer level

int CellCenteredBoundaryVariable::LoadBoundaryBufferToFiner(BufArray1D<Real> &buf,
                                                            const NeighborBlock &nb) {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  int si, sj, sk, ei, ej, ek;

  cell_centered_bvars::CalcIndicesLoadToFiner(si, ei, sj, ej, sk, ek, nb, pmb.get());

  int p = 0;
  ParArray4D<Real> var_cc_ = var_cc.Get<4>(); // auto template deduction fails
  BufferUtility::PackData(var_cc_, buf, nl_, nu_, si, ei, sj, ej, sk, ek, p, pmb.get());
  return p;
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenteredBoundaryVariable::SetBoundarySameLevel(BufArray1D<Real> &buf,
//                                                              const NeighborBlock& nb)
//  \brief Set cell-centered boundary received from a block on the same level

void CellCenteredBoundaryVariable::SetBoundarySameLevel(BufArray1D<Real> &buf,
                                                        const NeighborBlock &nb) {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  int si, sj, sk, ei, ej, ek;

  const IndexShape &cellbounds = pmb->cellbounds;

  IndexDomain interior = IndexDomain::interior;
  cell_centered_bvars::CalcIndicesSetSame(nb.ni.ox1, si, ei,
                                          cellbounds.GetBoundsI(interior));
  cell_centered_bvars::CalcIndicesSetSame(nb.ni.ox2, sj, ej,
                                          cellbounds.GetBoundsJ(interior));
  cell_centered_bvars::CalcIndicesSetSame(nb.ni.ox3, sk, ek,
                                          cellbounds.GetBoundsK(interior));

  int p = 0;

  ParArray4D<Real> var_cc_ = var_cc.Get<4>(); // automatic template deduction fails
  BufferUtility::UnpackData(buf, var_cc_, nl_, nu_, si, ei, sj, ej, sk, ek, p, pmb.get());
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenteredBoundaryVariable::SetBoundaryFromCoarser(BufArray1D<Real> &buf,
//                                                                const NeighborBlock& nb)
//  \brief Set cell-centered prolongation buffer received from a block on a coarser level

void CellCenteredBoundaryVariable::SetBoundaryFromCoarser(BufArray1D<Real> &buf,
                                                          const NeighborBlock &nb) {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  int si, sj, sk, ei, ej, ek;
  int cng = pmb->cnghost;

  const IndexShape &c_cellbounds = pmb->c_cellbounds;

  IndexDomain interior = IndexDomain::interior;
  cell_centered_bvars::CalcIndicesSetFromCoarser(
      nb.ni.ox1, si, ei, c_cellbounds.GetBoundsI(interior), pmb->loc.lx1, cng, true);
  cell_centered_bvars::CalcIndicesSetFromCoarser(
      nb.ni.ox2, sj, ej, c_cellbounds.GetBoundsJ(interior), pmb->loc.lx2, cng,
      pmb->block_size.nx2 > 1);
  cell_centered_bvars::CalcIndicesSetFromCoarser(
      nb.ni.ox3, sk, ek, c_cellbounds.GetBoundsK(interior), pmb->loc.lx3, cng,
      pmb->block_size.nx3 > 1);

  int p = 0;
  ParArray4D<Real> coarse_buf_ = coarse_buf.Get<4>(); // auto template deduction fails
  BufferUtility::UnpackData(buf, coarse_buf_, nl_, nu_, si, ei, sj, ej, sk, ek, p,
                            pmb.get());
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenteredBoundaryVariable::SetBoundaryFromFiner(BufArray1D<Real> &buf,
//                                                              const NeighborBlock& nb)
//  \brief Set cell-centered boundary received from a block on a finer level

void CellCenteredBoundaryVariable::SetBoundaryFromFiner(BufArray1D<Real> &buf,
                                                        const NeighborBlock &nb) {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  // receive already restricted data
  int si, sj, sk, ei, ej, ek;

  cell_centered_bvars::CalcIndicesSetFromFiner(si, ei, sj, ej, sk, ek, nb, pmb.get());
  int p = 0;
  ParArray4D<Real> var_cc_ = var_cc.Get<4>(); // automatic template deduction fails
  BufferUtility::UnpackData(buf, var_cc_, nl_, nu_, si, ei, sj, ej, sk, ek, p, pmb.get());
}

void CellCenteredBoundaryVariable::SetupPersistentMPI() {
#ifdef MPI_PARALLEL
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  int &mylevel = pmb->loc.level;

  int cng, cng1, cng2, cng3;
  cng = cng1 = pmb->cnghost;
  cng2 = (pmy_mesh_->ndim >= 2) ? cng : 0;
  cng3 = (pmy_mesh_->ndim >= 3) ? cng : 0;
  int ssize, rsize;
  int tag;
  // Initialize non-polar neighbor communications to other ranks
  for (int n = 0; n < pmb->pbval->nneighbor; n++) {
    NeighborBlock &nb = pmb->pbval->neighbor[n];
    if (nb.snb.rank != Globals::my_rank) {
      if (nb.snb.level == mylevel) { // same
        ssize = rsize = ((nb.ni.ox1 == 0) ? pmb->block_size.nx1 : Globals::nghost) *
                        ((nb.ni.ox2 == 0) ? pmb->block_size.nx2 : Globals::nghost) *
                        ((nb.ni.ox3 == 0) ? pmb->block_size.nx3 : Globals::nghost);
      } else if (nb.snb.level < mylevel) { // coarser
        ssize = ((nb.ni.ox1 == 0) ? ((pmb->block_size.nx1 + 1) / 2) : Globals::nghost) *
                ((nb.ni.ox2 == 0) ? ((pmb->block_size.nx2 + 1) / 2) : Globals::nghost) *
                ((nb.ni.ox3 == 0) ? ((pmb->block_size.nx3 + 1) / 2) : Globals::nghost);
        rsize = ((nb.ni.ox1 == 0) ? ((pmb->block_size.nx1 + 1) / 2 + cng1) : cng1) *
                ((nb.ni.ox2 == 0) ? ((pmb->block_size.nx2 + 1) / 2 + cng2) : cng2) *
                ((nb.ni.ox3 == 0) ? ((pmb->block_size.nx3 + 1) / 2 + cng3) : cng3);
      } else { // finer
        ssize = ((nb.ni.ox1 == 0) ? ((pmb->block_size.nx1 + 1) / 2 + cng1) : cng1) *
                ((nb.ni.ox2 == 0) ? ((pmb->block_size.nx2 + 1) / 2 + cng2) : cng2) *
                ((nb.ni.ox3 == 0) ? ((pmb->block_size.nx3 + 1) / 2 + cng3) : cng3);
        rsize = ((nb.ni.ox1 == 0) ? ((pmb->block_size.nx1 + 1) / 2) : Globals::nghost) *
                ((nb.ni.ox2 == 0) ? ((pmb->block_size.nx2 + 1) / 2) : Globals::nghost) *
                ((nb.ni.ox3 == 0) ? ((pmb->block_size.nx3 + 1) / 2) : Globals::nghost);
      }
      ssize *= (nu_ + 1);
      rsize *= (nu_ + 1);
      // specify the offsets in the view point of the target block: flip ox? signs

      // Initialize persistent communication requests attached to specific BoundaryData
      tag = pmb->pbval->CreateBvalsMPITag(nb.snb.lid, nb.targetid, cc_phys_id_);
      if (bd_var_.req_send[nb.bufid] != MPI_REQUEST_NULL)
        MPI_Request_free(&bd_var_.req_send[nb.bufid]);
      PARTHENON_MPI_CHECK(MPI_Send_init(bd_var_.send[nb.bufid].data(), ssize,
                                        MPI_PARTHENON_REAL, nb.snb.rank, tag,
                                        MPI_COMM_WORLD, &(bd_var_.req_send[nb.bufid])));
      tag = pmb->pbval->CreateBvalsMPITag(pmb->lid, nb.bufid, cc_phys_id_);
      if (bd_var_.req_recv[nb.bufid] != MPI_REQUEST_NULL)
        MPI_Request_free(&bd_var_.req_recv[nb.bufid]);
      PARTHENON_MPI_CHECK(MPI_Recv_init(bd_var_.recv[nb.bufid].data(), rsize,
                                        MPI_PARTHENON_REAL, nb.snb.rank, tag,
                                        MPI_COMM_WORLD, &(bd_var_.req_recv[nb.bufid])));

      if (pmy_mesh_->multilevel && nb.ni.type == NeighborConnect::face) {
        int size;
        if (nb.fid == 0 || nb.fid == 1)
          size = ((pmb->block_size.nx2 + 1) / 2) * ((pmb->block_size.nx3 + 1) / 2);
        else if (nb.fid == 2 || nb.fid == 3)
          size = ((pmb->block_size.nx1 + 1) / 2) * ((pmb->block_size.nx3 + 1) / 2);
        else // (nb.fid == 4 || nb.fid == 5)
          size = ((pmb->block_size.nx1 + 1) / 2) * ((pmb->block_size.nx2 + 1) / 2);
        size *= (nu_ + 1);
        if (nb.snb.level < mylevel) { // send to coarser
          tag = pmb->pbval->CreateBvalsMPITag(nb.snb.lid, nb.targetid, cc_flx_phys_id_);
          if (bd_var_flcor_.req_send[nb.bufid] != MPI_REQUEST_NULL)
            MPI_Request_free(&bd_var_flcor_.req_send[nb.bufid]);
          PARTHENON_MPI_CHECK(MPI_Send_init(
              bd_var_flcor_.send[nb.bufid].data(), size, MPI_PARTHENON_REAL, nb.snb.rank,
              tag, MPI_COMM_WORLD, &(bd_var_flcor_.req_send[nb.bufid])));
        } else if (nb.snb.level > mylevel) { // receive from finer
          tag = pmb->pbval->CreateBvalsMPITag(pmb->lid, nb.bufid, cc_flx_phys_id_);
          if (bd_var_flcor_.req_recv[nb.bufid] != MPI_REQUEST_NULL)
            MPI_Request_free(&bd_var_flcor_.req_recv[nb.bufid]);
          PARTHENON_MPI_CHECK(MPI_Recv_init(
              bd_var_flcor_.recv[nb.bufid].data(), size, MPI_PARTHENON_REAL, nb.snb.rank,
              tag, MPI_COMM_WORLD, &(bd_var_flcor_.req_recv[nb.bufid])));
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
