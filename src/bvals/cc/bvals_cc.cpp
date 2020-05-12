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

//! \file bvals_cc.cpp
//  \brief functions that apply BCs for CELL_CENTERED variables

#include "bvals/cc/bvals_cc.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>

#include "parthenon_mpi.hpp"

#include "basic_types.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "utils/buffer_utils.hpp"

namespace parthenon {

CellCenteredBoundaryVariable::CellCenteredBoundaryVariable(MeshBlock *pmb,
                                                           ParArrayND<Real> var,
                                                           ParArrayND<Real> coarse_var,
                                                           ParArrayND<Real> var_flux[])
    : BoundaryVariable(pmb), var_cc(var), coarse_buf(coarse_var), x1flux(var_flux[X1DIR]),
      x2flux(var_flux[X2DIR]), x3flux(var_flux[X3DIR]), nl_(0), nu_(var.GetDim(4) - 1) {
  // CellCenteredBoundaryVariable should only be used w/ 4D or 3D (nx4=1) ParArrayND
  // For now, assume that full span of 4th dim of input ParArrayND should be used:
  // ---> get the index limits directly from the input ParArrayND
  // <=nu_ (inclusive), <nx4 (exclusive)
  if (nu_ < 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in CellCenteredBoundaryVariable constructor" << std::endl
        << "An 'ParArrayND<Real> *var' of nx4_ = " << var.GetDim(4) << " was passed\n"
        << "Should be nx4 >= 1 (likely uninitialized)." << std::endl;
    ATHENA_ERROR(msg);
  }

  InitBoundaryData(bd_var_, BoundaryQuantity::cc);
#ifdef MPI_PARALLEL
  // KGF: dead code, leaving for now:
  // cc_phys_id_ = pmb->pbval->ReserveTagVariableIDs(1);
  cc_phys_id_ = pmb->pbval->bvars_next_phys_id_;
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
  MeshBlock *pmb = pmy_block_;
  int cng1, cng2, cng3;
  cng1 = cng;
  cng2 = cng * (pmb->block_size.nx2 > 1 ? 1 : 0);
  cng3 = cng * (pmb->block_size.nx3 > 1 ? 1 : 0);

  int size = ((ni.ox1 == 0) ? pmb->block_size.nx1 : NGHOST) *
             ((ni.ox2 == 0) ? pmb->block_size.nx2 : NGHOST) *
             ((ni.ox3 == 0) ? pmb->block_size.nx3 : NGHOST);
  if (pmy_mesh_->multilevel) {
    int f2c = ((ni.ox1 == 0) ? ((pmb->block_size.nx1 + 1) / 2) : NGHOST) *
              ((ni.ox2 == 0) ? ((pmb->block_size.nx2 + 1) / 2) : NGHOST) *
              ((ni.ox3 == 0) ? ((pmb->block_size.nx3 + 1) / 2) : NGHOST);
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
  MeshBlock *pmb = pmy_block_;
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
//! \fn int CellCenteredBoundaryVariable::LoadBoundaryBufferSameLevel(Real *buf,
//                                                                const NeighborBlock& nb)
//  \brief Set cell-centered boundary buffers for sending to a block on the same level

int CellCenteredBoundaryVariable::LoadBoundaryBufferSameLevel(Real *buf,
                                                              const NeighborBlock &nb) {
  MeshBlock *pmb = pmy_block_;
  int si, sj, sk, ei, ej, ek;

  IndexDomain interior = IndexDomain::interior;
  const IndexShape &cellbounds = pmb->cellbounds;
  si = (nb.ni.ox1 > 0) ? (cellbounds.ie(interior) - NGHOST + 1) : cellbounds.is(interior);
  ei = (nb.ni.ox1 < 0) ? (cellbounds.is(interior) + NGHOST - 1) : cellbounds.ie(interior);
  sj = (nb.ni.ox2 > 0) ? (cellbounds.je(interior) - NGHOST + 1) : cellbounds.js(interior);
  ej = (nb.ni.ox2 < 0) ? (cellbounds.js(interior) + NGHOST - 1) : cellbounds.je(interior);
  sk = (nb.ni.ox3 > 0) ? (cellbounds.ke(interior) - NGHOST + 1) : cellbounds.ks(interior);
  ek = (nb.ni.ox3 < 0) ? (cellbounds.ks(interior) + NGHOST - 1) : cellbounds.ke(interior);
  int p = 0;
  BufferUtility::PackData(var_cc, buf, nl_, nu_, si, ei, sj, ej, sk, ek, p);

  return p;
}

//----------------------------------------------------------------------------------------
//! \fn int CellCenteredBoundaryVariable::LoadBoundaryBufferToCoarser(Real *buf,
//                                                                const NeighborBlock& nb)
//  \brief Set cell-centered boundary buffers for sending to a block on the coarser level

int CellCenteredBoundaryVariable::LoadBoundaryBufferToCoarser(Real *buf,
                                                              const NeighborBlock &nb) {
  MeshBlock *pmb = pmy_block_;
  int si, sj, sk, ei, ej, ek;
  int cn = NGHOST - 1;

  IndexDomain interior = IndexDomain::interior;
  const IndexShape &c_cellbounds = pmb->c_cellbounds;
  si = (nb.ni.ox1 > 0) ? (c_cellbounds.ie(interior) - cn) : c_cellbounds.is(interior);
  ei = (nb.ni.ox1 < 0) ? (c_cellbounds.is(interior) + cn) : c_cellbounds.ie(interior);
  sj = (nb.ni.ox2 > 0) ? (c_cellbounds.je(interior) - cn) : c_cellbounds.js(interior);
  ej = (nb.ni.ox2 < 0) ? (c_cellbounds.js(interior) + cn) : c_cellbounds.je(interior);
  sk = (nb.ni.ox3 > 0) ? (c_cellbounds.ke(interior) - cn) : c_cellbounds.ks(interior);
  ek = (nb.ni.ox3 < 0) ? (c_cellbounds.ks(interior) + cn) : c_cellbounds.ke(interior);

  int p = 0;
  pmb->pmr->RestrictCellCenteredValues(var_cc, coarse_buf, nl_, nu_, si, ei, sj, ej, sk,
                                       ek);
  BufferUtility::PackData(coarse_buf, buf, nl_, nu_, si, ei, sj, ej, sk, ek, p);
  return p;
}

//----------------------------------------------------------------------------------------
//! \fn int CellCenteredBoundaryVariable::LoadBoundaryBufferToFiner(Real *buf,
//                                                                const NeighborBlock& nb)
//  \brief Set cell-centered boundary buffers for sending to a block on the finer level

int CellCenteredBoundaryVariable::LoadBoundaryBufferToFiner(Real *buf,
                                                            const NeighborBlock &nb) {
  MeshBlock *pmb = pmy_block_;
  int si, sj, sk, ei, ej, ek;
  int cn = pmb->cnghost - 1;

  IndexDomain interior = IndexDomain::interior;
  const IndexShape &cellbounds = pmb->cellbounds;
  si = (nb.ni.ox1 > 0) ? (cellbounds.ie(interior) - cn) : cellbounds.is(interior);
  ei = (nb.ni.ox1 < 0) ? (cellbounds.is(interior) + cn) : cellbounds.ie(interior);
  sj = (nb.ni.ox2 > 0) ? (cellbounds.je(interior) - cn) : cellbounds.js(interior);
  ej = (nb.ni.ox2 < 0) ? (cellbounds.js(interior) + cn) : cellbounds.je(interior);
  sk = (nb.ni.ox3 > 0) ? (cellbounds.ke(interior) - cn) : cellbounds.ks(interior);
  ek = (nb.ni.ox3 < 0) ? (cellbounds.ks(interior) + cn) : cellbounds.ke(interior);

  // send the data first and later prolongate on the target block
  // need to add edges for faces, add corners for edges
  if (nb.ni.ox1 == 0) {
    if (nb.ni.fi1 == 1)
      si += pmb->block_size.nx1 / 2 - pmb->cnghost;
    else
      ei -= pmb->block_size.nx1 / 2 - pmb->cnghost;
  }
  if (nb.ni.ox2 == 0 && pmb->block_size.nx2 > 1) {
    if (nb.ni.ox1 != 0) {
      if (nb.ni.fi1 == 1)
        sj += pmb->block_size.nx2 / 2 - pmb->cnghost;
      else
        ej -= pmb->block_size.nx2 / 2 - pmb->cnghost;
    } else {
      if (nb.ni.fi2 == 1)
        sj += pmb->block_size.nx2 / 2 - pmb->cnghost;
      else
        ej -= pmb->block_size.nx2 / 2 - pmb->cnghost;
    }
  }
  if (nb.ni.ox3 == 0 && pmb->block_size.nx3 > 1) {
    if (nb.ni.ox1 != 0 && nb.ni.ox2 != 0) {
      if (nb.ni.fi1 == 1)
        sk += pmb->block_size.nx3 / 2 - pmb->cnghost;
      else
        ek -= pmb->block_size.nx3 / 2 - pmb->cnghost;
    } else {
      if (nb.ni.fi2 == 1)
        sk += pmb->block_size.nx3 / 2 - pmb->cnghost;
      else
        ek -= pmb->block_size.nx3 / 2 - pmb->cnghost;
    }
  }

  int p = 0;
  BufferUtility::PackData(var_cc, buf, nl_, nu_, si, ei, sj, ej, sk, ek, p);
  return p;
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenteredBoundaryVariable::SetBoundarySameLevel(Real *buf,
//                                                              const NeighborBlock& nb)
//  \brief Set cell-centered boundary received from a block on the same level

void CellCenteredBoundaryVariable::SetBoundarySameLevel(Real *buf,
                                                        const NeighborBlock &nb) {
  MeshBlock *pmb = pmy_block_;
  int si, sj, sk, ei, ej, ek;

  const IndexShape &cellbounds = pmb->cellbounds;

  auto CalcIndices = [](int ox, int &s, int &e, const IndexRange &bounds) {
    if (ox == 0) {
      s = bounds.s;
      e = bounds.e;
    } else if (ox > 0) {
      s = bounds.e + 1;
      e = bounds.e + NGHOST;
    } else {
      s = bounds.s - NGHOST;
      e = bounds.s - 1;
    }
  };

  IndexDomain interior = IndexDomain::interior;
  CalcIndices(nb.ni.ox1, si, ei, cellbounds.GetBoundsI(interior));
  CalcIndices(nb.ni.ox2, sj, ej, cellbounds.GetBoundsJ(interior));
  CalcIndices(nb.ni.ox3, sk, ek, cellbounds.GetBoundsK(interior));

  int p = 0;

  BufferUtility::UnpackData(buf, var_cc, nl_, nu_, si, ei, sj, ej, sk, ek, p);
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenteredBoundaryVariable::SetBoundaryFromCoarser(Real *buf,
//                                                                const NeighborBlock& nb)
//  \brief Set cell-centered prolongation buffer received from a block on a coarser level

void CellCenteredBoundaryVariable::SetBoundaryFromCoarser(Real *buf,
                                                          const NeighborBlock &nb) {
  MeshBlock *pmb = pmy_block_;
  int si, sj, sk, ei, ej, ek;
  int cng = pmb->cnghost;

  const IndexShape &c_cellbounds = pmb->c_cellbounds;

  auto CalcIndices = [](const int &ox, int &s, int &e, const IndexRange &bounds,
                        const std::int64_t &lx, const int &cng, const bool include_dim) {
    if (ox == 0) {
      s = bounds.s;
      e = bounds.e;
      if (include_dim) {
        if ((lx & 1LL) == 0LL) {
          e += cng;
        } else {
          s -= cng;
        }
      }
    } else if (ox > 0) {
      s = bounds.e + 1;
      e = bounds.e + cng;
    } else {
      s = bounds.s - cng;
      e = bounds.s - 1;
    }
  };

  IndexDomain interior = IndexDomain::interior;
  CalcIndices(nb.ni.ox1, si, ei, c_cellbounds.GetBoundsI(interior), pmb->loc.lx1, cng,
              true);
  CalcIndices(nb.ni.ox2, sj, ej, c_cellbounds.GetBoundsJ(interior), pmb->loc.lx2, cng,
              pmb->block_size.nx2 > 1);
  CalcIndices(nb.ni.ox3, sk, ek, c_cellbounds.GetBoundsK(interior), pmb->loc.lx3, cng,
              pmb->block_size.nx3 > 1);

  int p = 0;
  BufferUtility::UnpackData(buf, coarse_buf, nl_, nu_, si, ei, sj, ej, sk, ek, p);
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenteredBoundaryVariable::SetBoundaryFromFiner(Real *buf,
//                                                              const NeighborBlock& nb)
//  \brief Set cell-centered boundary received from a block on a finer level

void CellCenteredBoundaryVariable::SetBoundaryFromFiner(Real *buf,
                                                        const NeighborBlock &nb) {
  MeshBlock *pmb = pmy_block_;
  // receive already restricted data
  int si, sj, sk, ei, ej, ek;

  const IndexShape &cellbounds = pmb->cellbounds;
  IndexDomain interior = IndexDomain::interior;

  if (nb.ni.ox1 == 0) {
    si = cellbounds.is(interior);
    ei = cellbounds.ie(interior);
    if (nb.ni.fi1 == 1)
      si += pmb->block_size.nx1 / 2;
    else
      ei -= pmb->block_size.nx1 / 2;
  } else if (nb.ni.ox1 > 0) {
    si = cellbounds.ie(interior) + 1;
    ei = cellbounds.ie(interior) + NGHOST;
  } else {
    si = cellbounds.is(interior) - NGHOST;
    ei = cellbounds.is(interior) - 1;
  }

  if (nb.ni.ox2 == 0) {
    sj = cellbounds.js(interior);
    ej = cellbounds.je(interior);
    if (pmb->block_size.nx2 > 1) {
      if (nb.ni.ox1 != 0) {
        if (nb.ni.fi1 == 1)
          sj += pmb->block_size.nx2 / 2;
        else
          ej -= pmb->block_size.nx2 / 2;
      } else {
        if (nb.ni.fi2 == 1)
          sj += pmb->block_size.nx2 / 2;
        else
          ej -= pmb->block_size.nx2 / 2;
      }
    }
  } else if (nb.ni.ox2 > 0) {
    sj = cellbounds.je(interior) + 1;
    ej = cellbounds.je(interior) + NGHOST;
  } else {
    sj = cellbounds.js(interior) - NGHOST;
    ej = cellbounds.js(interior) - 1;
  }

  if (nb.ni.ox3 == 0) {
    sk = cellbounds.ks(interior);
    ek = cellbounds.ke(interior);
    if (pmb->block_size.nx3 > 1) {
      if (nb.ni.ox1 != 0 && nb.ni.ox2 != 0) {
        if (nb.ni.fi1 == 1)
          sk += pmb->block_size.nx3 / 2;
        else
          ek -= pmb->block_size.nx3 / 2;
      } else {
        if (nb.ni.fi2 == 1)
          sk += pmb->block_size.nx3 / 2;
        else
          ek -= pmb->block_size.nx3 / 2;
      }
    }
  } else if (nb.ni.ox3 > 0) {
    sk = cellbounds.ke(interior) + 1;
    ek = cellbounds.ke(interior) + NGHOST;
  } else {
    sk = cellbounds.ks(interior) - NGHOST;
    ek = cellbounds.ks(interior) - 1;
  }

  int p = 0;
  BufferUtility::UnpackData(buf, var_cc, nl_, nu_, si, ei, sj, ej, sk, ek, p);
}

void CellCenteredBoundaryVariable::SetupPersistentMPI() {
#ifdef MPI_PARALLEL
  MeshBlock *pmb = pmy_block_;
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
        ssize = rsize = ((nb.ni.ox1 == 0) ? pmb->block_size.nx1 : NGHOST) *
                        ((nb.ni.ox2 == 0) ? pmb->block_size.nx2 : NGHOST) *
                        ((nb.ni.ox3 == 0) ? pmb->block_size.nx3 : NGHOST);
      } else if (nb.snb.level < mylevel) { // coarser
        ssize = ((nb.ni.ox1 == 0) ? ((pmb->block_size.nx1 + 1) / 2) : NGHOST) *
                ((nb.ni.ox2 == 0) ? ((pmb->block_size.nx2 + 1) / 2) : NGHOST) *
                ((nb.ni.ox3 == 0) ? ((pmb->block_size.nx3 + 1) / 2) : NGHOST);
        rsize = ((nb.ni.ox1 == 0) ? ((pmb->block_size.nx1 + 1) / 2 + cng1) : cng1) *
                ((nb.ni.ox2 == 0) ? ((pmb->block_size.nx2 + 1) / 2 + cng2) : cng2) *
                ((nb.ni.ox3 == 0) ? ((pmb->block_size.nx3 + 1) / 2 + cng3) : cng3);
      } else { // finer
        ssize = ((nb.ni.ox1 == 0) ? ((pmb->block_size.nx1 + 1) / 2 + cng1) : cng1) *
                ((nb.ni.ox2 == 0) ? ((pmb->block_size.nx2 + 1) / 2 + cng2) : cng2) *
                ((nb.ni.ox3 == 0) ? ((pmb->block_size.nx3 + 1) / 2 + cng3) : cng3);
        rsize = ((nb.ni.ox1 == 0) ? ((pmb->block_size.nx1 + 1) / 2) : NGHOST) *
                ((nb.ni.ox2 == 0) ? ((pmb->block_size.nx2 + 1) / 2) : NGHOST) *
                ((nb.ni.ox3 == 0) ? ((pmb->block_size.nx3 + 1) / 2) : NGHOST);
      }
      ssize *= (nu_ + 1);
      rsize *= (nu_ + 1);
      // specify the offsets in the view point of the target block: flip ox? signs

      // Initialize persistent communication requests attached to specific BoundaryData
      tag = pmb->pbval->CreateBvalsMPITag(nb.snb.lid, nb.targetid, cc_phys_id_);
      if (bd_var_.req_send[nb.bufid] != MPI_REQUEST_NULL)
        MPI_Request_free(&bd_var_.req_send[nb.bufid]);
      MPI_Send_init(bd_var_.send[nb.bufid], ssize, MPI_ATHENA_REAL, nb.snb.rank, tag,
                    MPI_COMM_WORLD, &(bd_var_.req_send[nb.bufid]));
      tag = pmb->pbval->CreateBvalsMPITag(pmb->lid, nb.bufid, cc_phys_id_);
      if (bd_var_.req_recv[nb.bufid] != MPI_REQUEST_NULL)
        MPI_Request_free(&bd_var_.req_recv[nb.bufid]);
      MPI_Recv_init(bd_var_.recv[nb.bufid], rsize, MPI_ATHENA_REAL, nb.snb.rank, tag,
                    MPI_COMM_WORLD, &(bd_var_.req_recv[nb.bufid]));

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
          MPI_Send_init(bd_var_flcor_.send[nb.bufid], size, MPI_ATHENA_REAL, nb.snb.rank,
                        tag, MPI_COMM_WORLD, &(bd_var_flcor_.req_send[nb.bufid]));
        } else if (nb.snb.level > mylevel) { // receive from finer
          tag = pmb->pbval->CreateBvalsMPITag(pmb->lid, nb.bufid, cc_flx_phys_id_);
          if (bd_var_flcor_.req_recv[nb.bufid] != MPI_REQUEST_NULL)
            MPI_Request_free(&bd_var_flcor_.req_recv[nb.bufid]);
          MPI_Recv_init(bd_var_flcor_.recv[nb.bufid], size, MPI_ATHENA_REAL, nb.snb.rank,
                        tag, MPI_COMM_WORLD, &(bd_var_flcor_.req_recv[nb.bufid]));
        }
      }
    }
  }
#endif
  return;
}

void CellCenteredBoundaryVariable::StartReceiving(BoundaryCommSubset phase) {
#ifdef MPI_PARALLEL
  MeshBlock *pmb = pmy_block_;
  int mylevel = pmb->loc.level;
  for (int n = 0; n < pmb->pbval->nneighbor; n++) {
    NeighborBlock &nb = pmb->pbval->neighbor[n];
    if (nb.snb.rank != Globals::my_rank) {
      MPI_Start(&(bd_var_.req_recv[nb.bufid]));
      if (phase == BoundaryCommSubset::all && nb.ni.type == NeighborConnect::face &&
          nb.snb.level > mylevel) // opposite condition in ClearBoundary()
        MPI_Start(&(bd_var_flcor_.req_recv[nb.bufid]));
    }
  }
#endif
  return;
}

void CellCenteredBoundaryVariable::ClearBoundary(BoundaryCommSubset phase) {
  for (int n = 0; n < pmy_block_->pbval->nneighbor; n++) {
    NeighborBlock &nb = pmy_block_->pbval->neighbor[n];
    bd_var_.flag[nb.bufid] = BoundaryStatus::waiting;
    bd_var_.sflag[nb.bufid] = BoundaryStatus::waiting;

    if (nb.ni.type == NeighborConnect::face) {
      bd_var_flcor_.flag[nb.bufid] = BoundaryStatus::waiting;
      bd_var_flcor_.sflag[nb.bufid] = BoundaryStatus::waiting;
    }
#ifdef MPI_PARALLEL
    MeshBlock *pmb = pmy_block_;
    int mylevel = pmb->loc.level;
    if (nb.snb.rank != Globals::my_rank) {
      // Wait for Isend
      MPI_Wait(&(bd_var_.req_send[nb.bufid]), MPI_STATUS_IGNORE);
      if (phase == BoundaryCommSubset::all && nb.ni.type == NeighborConnect::face &&
          nb.snb.level < mylevel)
        MPI_Wait(&(bd_var_flcor_.req_send[nb.bufid]), MPI_STATUS_IGNORE);
    }
#endif
  }
}

} // namespace parthenon
