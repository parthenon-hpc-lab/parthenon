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

// C headers

// C++ headers
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>    // memcpy()
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "coordinates/coordinates.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "utils/buffer_utils.hpp"
#include "bvals_cc.hpp"

// MPI header
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

namespace parthenon {
// constructor

CellCenteredBoundaryVariable::CellCenteredBoundaryVariable(
    MeshBlock *pmb, AthenaArray<Real> *var, AthenaArray<Real> *coarse_var,
    AthenaArray<Real> *var_flux)
    : BoundaryVariable(pmb), var_cc(var), coarse_buf(coarse_var), x1flux(var_flux[X1DIR]),
      x2flux(var_flux[X2DIR]), x3flux(var_flux[X3DIR]), nl_(0), nu_(var->GetDim4() -1) {
  // CellCenteredBoundaryVariable should only be used w/ 4D or 3D (nx4=1) AthenaArray
  // For now, assume that full span of 4th dim of input AthenaArray should be used:
  // ---> get the index limits directly from the input AthenaArray
  // <=nu_ (inclusive), <nx4 (exclusive)
  if (nu_ < 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in CellCenteredBoundaryVariable constructor" << std::endl
        << "An 'AthenaArray<Real> *var' of nx4_ = " << var->GetDim4() << " was passed\n"
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
  if (pmy_mesh_->multilevel)
    DestroyBoundaryData(bd_var_flcor_);
}

int CellCenteredBoundaryVariable::ComputeVariableBufferSize(const NeighborIndexes& ni,
                                                            int cng) {
  MeshBlock *pmb = pmy_block_;
  int cng1, cng2, cng3;
  cng1 = cng;
  cng2 = cng*(pmb->block_size.nx2 > 1 ? 1 : 0);
  cng3 = cng*(pmb->block_size.nx3 > 1 ? 1 : 0);

  int size = ((ni.ox1 == 0) ? pmb->block_size.nx1 : NGHOST)
           *((ni.ox2 == 0) ? pmb->block_size.nx2 : NGHOST)
           *((ni.ox3 == 0) ? pmb->block_size.nx3 : NGHOST);
  if (pmy_mesh_->multilevel) {
    int f2c = ((ni.ox1 == 0) ? ((pmb->block_size.nx1+1)/2) : NGHOST)
            *((ni.ox2 == 0) ? ((pmb->block_size.nx2+1)/2) : NGHOST)
            *((ni.ox3 == 0) ? ((pmb->block_size.nx3+1)/2) : NGHOST);
    int c2f = ((ni.ox1 == 0) ?((pmb->block_size.nx1+1)/2 + cng1) : cng)
            *((ni.ox2 == 0) ? ((pmb->block_size.nx2+1)/2 + cng2) : cng)
            *((ni.ox3 == 0) ? ((pmb->block_size.nx3+1)/2 + cng3) : cng);
    size = std::max(size, c2f);
    size = std::max(size, f2c);
  }
  size *= nu_ + 1;
  return size;
}

int CellCenteredBoundaryVariable::ComputeFluxCorrectionBufferSize(
    const NeighborIndexes& ni, int cng) {
  MeshBlock *pmb = pmy_block_;
  int size = 0;
  if (ni.ox1 != 0)
    size = (pmb->block_size.nx2 + 1)/2*(pmb->block_size.nx3 + 1)/2*(nu_ + 1);
  if (ni.ox2 != 0)
    size = (pmb->block_size.nx1 + 1)/2*(pmb->block_size.nx3 + 1)/2*(nu_ + 1);
  if (ni.ox3 != 0)
    size = (pmb->block_size.nx1 + 1)/2*(pmb->block_size.nx2 + 1)/2*(nu_ + 1);
  return size;
}

//----------------------------------------------------------------------------------------
//! \fn int CellCenteredBoundaryVariable::LoadBoundaryBufferSameLevel(Real *buf,
//                                                                const NeighborBlock& nb)
//  \brief Set cell-centered boundary buffers for sending to a block on the same level

int CellCenteredBoundaryVariable::LoadBoundaryBufferSameLevel(Real *buf,
                                                              const NeighborBlock& nb) {
  MeshBlock *pmb = pmy_block_;
  int si, sj, sk, ei, ej, ek;

  si = (nb.ni.ox1 > 0) ? (pmb->active_cells.x.at(0).e - NGHOST + 1) : pmb->active_cells.x.at(0).s;
  ei = (nb.ni.ox1 < 0) ? (pmb->active_cells.x.at(0).s + NGHOST - 1) : pmb->active_cells.x.at(0).e;
  sj = (nb.ni.ox2 > 0) ? (pmb->active_cells.x.at(1).e - NGHOST + 1) : pmb->active_cells.x.at(1).s;
  ej = (nb.ni.ox2 < 0) ? (pmb->active_cells.x.at(1).s + NGHOST - 1) : pmb->active_cells.x.at(1).e;
  sk = (nb.ni.ox3 > 0) ? (pmb->active_cells.x.at(2).e - NGHOST + 1) : pmb->active_cells.x.at(2).s;
  ek = (nb.ni.ox3 < 0) ? (pmb->active_cells.x.at(2).s + NGHOST - 1) : pmb->active_cells.x.at(2).e;
  int p = 0;
  AthenaArray<Real> &var = *var_cc;
  BufferUtility::PackData(var, buf, nl_, nu_, si, ei, sj, ej, sk, ek, p);

  return p;
}

//----------------------------------------------------------------------------------------
//! \fn int CellCenteredBoundaryVariable::LoadBoundaryBufferToCoarser(Real *buf,
//                                                                const NeighborBlock& nb)
//  \brief Set cell-centered boundary buffers for sending to a block on the coarser level

int CellCenteredBoundaryVariable::LoadBoundaryBufferToCoarser(Real *buf,
                                                              const NeighborBlock& nb) {
  MeshBlock *pmb = pmy_block_;
  int si, sj, sk, ei, ej, ek;
  int cn = NGHOST - 1;
  AthenaArray<Real> &var = *var_cc;
  AthenaArray<Real> &coarse_var = *coarse_buf;

  si = (nb.ni.ox1 > 0) ? (pmb->active_coarse_cells.x.at(0).e - cn) : pmb->active_coarse_cells.x.at(0).s;
  ei = (nb.ni.ox1 < 0) ? (pmb->active_coarse_cells.x.at(0).s + cn) : pmb->active_coarse_cells.x.at(0).e;
  sj = (nb.ni.ox2 > 0) ? (pmb->active_coarse_cells.x.at(1).e - cn) : pmb->active_coarse_cells.x.at(1).s;
  ej = (nb.ni.ox2 < 0) ? (pmb->active_coarse_cells.x.at(1).s + cn) : pmb->active_coarse_cells.x.at(1).e;
  sk = (nb.ni.ox3 > 0) ? (pmb->active_coarse_cells.x.at(2).e - cn) : pmb->active_coarse_cells.x.at(1).s;
  ek = (nb.ni.ox3 < 0) ? (pmb->active_coarse_cells.x.at(2).s + cn) : pmb->active_coarse_cells.x.at(1).e;

  int p = 0;
  pmb->pmr->RestrictCellCenteredValues(var, coarse_var, nl_, nu_, si, ei, sj, ej, sk, ek);
  BufferUtility::PackData(coarse_var, buf, nl_, nu_, si, ei, sj, ej, sk, ek, p);
  return p;
}

//----------------------------------------------------------------------------------------
//! \fn int CellCenteredBoundaryVariable::LoadBoundaryBufferToFiner(Real *buf,
//                                                                const NeighborBlock& nb)
//  \brief Set cell-centered boundary buffers for sending to a block on the finer level

int CellCenteredBoundaryVariable::LoadBoundaryBufferToFiner(Real *buf,
                                                            const NeighborBlock& nb) {
  MeshBlock *pmb = pmy_block_;
  int si, sj, sk, ei, ej, ek;
  int cn = pmb->cnghost - 1;
  AthenaArray<Real> &var = *var_cc;

  si = (nb.ni.ox1 > 0) ? (pmb->active_cells.x.at(0).e - cn) : pmb->active_cells.x.at(0).s;
  ei = (nb.ni.ox1 < 0) ? (pmb->active_cells.x.at(0).s + cn) : pmb->active_cells.x.at(0).e;
  sj = (nb.ni.ox2 > 0) ? (pmb->active_cells.x.at(1).e - cn) : pmb->active_cells.x.at(1).s;
  ej = (nb.ni.ox2 < 0) ? (pmb->active_cells.x.at(1).s + cn) : pmb->active_cells.x.at(1).e;
  sk = (nb.ni.ox3 > 0) ? (pmb->active_cells.x.at(2).e - cn) : pmb->active_cells.x.at(2).s;
  ek = (nb.ni.ox3 < 0) ? (pmb->active_cells.x.at(2).s + cn) : pmb->active_cells.x.at(2).e;

  // send the data first and later prolongate on the target block
  // need to add edges for faces, add corners for edges
  if (nb.ni.ox1 == 0) {
    if (nb.ni.fi1 == 1)   si += pmb->block_size.nx1/2 - pmb->cnghost;
    else            ei -= pmb->block_size.nx1/2 - pmb->cnghost;
  }
  if (nb.ni.ox2 == 0 && pmb->block_size.nx2 > 1) {
    if (nb.ni.ox1 != 0) {
      if (nb.ni.fi1 == 1) sj += pmb->block_size.nx2/2 - pmb->cnghost;
      else          ej -= pmb->block_size.nx2/2 - pmb->cnghost;
    } else {
      if (nb.ni.fi2 == 1) sj += pmb->block_size.nx2/2 - pmb->cnghost;
      else          ej -= pmb->block_size.nx2/2 - pmb->cnghost;
    }
  }
  if (nb.ni.ox3 == 0 && pmb->block_size.nx3 > 1) {
    if (nb.ni.ox1 != 0 && nb.ni.ox2 != 0) {
      if (nb.ni.fi1 == 1) sk += pmb->block_size.nx3/2 - pmb->cnghost;
      else          ek -= pmb->block_size.nx3/2 - pmb->cnghost;
    } else {
      if (nb.ni.fi2 == 1) sk += pmb->block_size.nx3/2 - pmb->cnghost;
      else          ek -= pmb->block_size.nx3/2 - pmb->cnghost;
    }
  }

  int p = 0;
  BufferUtility::PackData(var, buf, nl_, nu_, si, ei, sj, ej, sk, ek, p);
  return p;
}


//----------------------------------------------------------------------------------------
//! \fn void CellCenteredBoundaryVariable::SetBoundarySameLevel(Real *buf,
//                                                              const NeighborBlock& nb)
//  \brief Set cell-centered boundary received from a block on the same level

void CellCenteredBoundaryVariable::SetBoundarySameLevel(Real *buf,
                                                        const NeighborBlock& nb) {
  MeshBlock *pmb = pmy_block_;
  int si, sj, sk, ei, ej, ek;
  AthenaArray<Real> &var = *var_cc;

  if (nb.ni.ox1 == 0)     si = pmb->active_cells.x.at(0).s,        ei = pmb->active_cells.x.at(0).e;
  else if (nb.ni.ox1 > 0) si = pmb->active_cells.x.at(0).e + 1,      ei = pmb->active_cells.x.at(0).e + NGHOST;
  else              si = pmb->active_cells.x.at(0).s - NGHOST, ei = pmb->active_cells.x.at(0).s - 1;
  if (nb.ni.ox2 == 0)     sj = pmb->active_cells.x.at(1).s,        ej = pmb->active_cells.x.at(1).e;
  else if (nb.ni.ox2 > 0) sj = pmb->active_cells.x.at(1).e + 1,      ej = pmb->active_cells.x.at(1).e + NGHOST;
  else              sj = pmb->active_cells.x.at(1).s - NGHOST, ej = pmb->active_cells.x.at(1).s - 1;
  if (nb.ni.ox3 == 0)     sk = pmb->active_cells.x.at(2).s,        ek = pmb->active_cells.x.at(2).e;
  else if (nb.ni.ox3 > 0) sk = pmb->active_cells.x.at(2).e + 1,      ek = pmb->active_cells.x.at(2).e + NGHOST;
  else              sk = pmb->active_cells.x.at(2).s - NGHOST, ek = pmb->active_cells.x.at(2).s - 1;

  int p = 0;

  BufferUtility::UnpackData(buf, var, nl_, nu_, si, ei, sj, ej, sk, ek, p);
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenteredBoundaryVariable::SetBoundaryFromCoarser(Real *buf,
//                                                                const NeighborBlock& nb)
//  \brief Set cell-centered prolongation buffer received from a block on a coarser level

void CellCenteredBoundaryVariable::SetBoundaryFromCoarser(Real *buf,
                                                          const NeighborBlock& nb) {
  MeshBlock *pmb = pmy_block_;
  int si, sj, sk, ei, ej, ek;
  int cng = pmb->cnghost;
  AthenaArray<Real> &coarse_var = *coarse_buf;

  if (nb.ni.ox1 == 0) {
    si = pmb->active_coarse_cells.x.at(0).s;
    ei = pmb->active_coarse_cells.x.at(0).e;
    if ((pmb->loc.lx1 & 1LL) == 0LL) ei += cng;
    else                             si -= cng;
  } else if (nb.ni.ox1 > 0)  {
    si = pmb->active_coarse_cells.x.at(0).e + 1;
    ei = pmb->active_coarse_cells.x.at(0).e + cng;
  } else {
    si = pmb->active_coarse_cells.x.at(0).s - cng;
    ei = pmb->active_coarse_cells.x.at(0).s - 1;
  }
  if (nb.ni.ox2 == 0) {
    sj = pmb->active_coarse_cells.x.at(1).s, ej = pmb->active_coarse_cells.x.at(1).e;
    if (pmb->block_size.nx2 > 1) {
      if ((pmb->loc.lx2 & 1LL) == 0LL) ej += cng;
      else                             sj -= cng;
    }
  } else if (nb.ni.ox2 > 0) {
    sj = pmb->active_coarse_cells.x.at(1).e + 1;
    ej = pmb->active_coarse_cells.x.at(1).e + cng;
  } else {
    sj = pmb->active_coarse_cells.x.at(1).s - cng;
    ej = pmb->active_coarse_cells.x.at(1).s - 1;
  }
  if (nb.ni.ox3 == 0) {
    sk = pmb->active_coarse_cells.x.at(2).s;
    ek = pmb->active_coarse_cells.x.at(2).e;
    if (pmb->block_size.nx3 > 1) {
      if ((pmb->loc.lx3 & 1LL) == 0LL) ek += cng;
      else                             sk -= cng;
    }
  } else if (nb.ni.ox3 > 0)  {
    sk = pmb->active_coarse_cells.x.at(2).e + 1;
    ek = pmb->active_coarse_cells.x.at(2).e + cng;
  } else {
    sk = pmb->active_coarse_cells.x.at(2).s - cng;
    ek = pmb->active_coarse_cells.x.at(2).s - 1;
  }

  int p = 0;
  BufferUtility::UnpackData(buf, coarse_var, nl_, nu_, si, ei, sj, ej, sk, ek, p);
  //pmb->pmr->ProlongateCellCenteredValues(coarse_var, *var_cc, nl_, nu_, si, ei, sj, ej, sk, ek);
}


//----------------------------------------------------------------------------------------
//! \fn void CellCenteredBoundaryVariable::SetBoundaryFromFiner(Real *buf,
//                                                              const NeighborBlock& nb)
//  \brief Set cell-centered boundary received from a block on a finer level

void CellCenteredBoundaryVariable::SetBoundaryFromFiner(Real *buf,
                                                        const NeighborBlock& nb) {
  MeshBlock *pmb = pmy_block_;
  AthenaArray<Real> &var = *var_cc;
  // receive already restricted data
  int si, sj, sk, ei, ej, ek;

  if (nb.ni.ox1 == 0) {
    si = pmb->active_cells.x.at(0).s; 
    ei = pmb->active_cells.x.at(0).e;
    if (nb.ni.fi1 == 1)   si += pmb->block_size.nx1/2;
    else            ei -= pmb->block_size.nx1/2;
  } else if (nb.ni.ox1 > 0) {
    si = pmb->active_cells.x.at(0).e + 1,      ei = pmb->active_cells.x.at(0).e + NGHOST;
  } else {
    si = pmb->active_cells.x.at(0).s - NGHOST, ei = pmb->active_cells.x.at(0).s - 1;
  }
  if (nb.ni.ox2 == 0) {
    sj = pmb->active_cells.x.at(1).s;
    ej = pmb->active_cells.x.at(1).e;
    if (pmb->block_size.nx2 > 1) {
      if (nb.ni.ox1 != 0) {
        if (nb.ni.fi1 == 1) sj += pmb->block_size.nx2/2;
        else          ej -= pmb->block_size.nx2/2;
      } else {
        if (nb.ni.fi2 == 1) sj += pmb->block_size.nx2/2;
        else          ej -= pmb->block_size.nx2/2;
      }
    }
  } else if (nb.ni.ox2 > 0) {
    sj = pmb->active_cells.x.at(1).e + 1;
    ej = pmb->active_cells.x.at(1).e + NGHOST;
  } else {
    sj = pmb->active_cells.x.at(1).s - NGHOST;
    ej = pmb->active_cells.x.at(1).s - 1;
  }
  if (nb.ni.ox3 == 0) {
    sk = pmb->active_cells.x.at(2).s, ek = pmb->active_cells.x.at(2).e;
    if (pmb->block_size.nx3 > 1) {
      if (nb.ni.ox1 != 0 && nb.ni.ox2 != 0) {
        if (nb.ni.fi1 == 1) sk += pmb->block_size.nx3/2;
        else          ek -= pmb->block_size.nx3/2;
      } else {
        if (nb.ni.fi2 == 1) sk += pmb->block_size.nx3/2;
        else          ek -= pmb->block_size.nx3/2;
      }
    }
  } else if (nb.ni.ox3 > 0) {
    sk = pmb->active_cells.x.at(2).e + 1,      ek = pmb->active_cells.x.at(2).e + NGHOST;
  } else {
    sk = pmb->active_cells.x.at(2).s - NGHOST, ek = pmb->active_cells.x.at(2).s - 1;
  }

  int p = 0;
  BufferUtility::UnpackData(buf, var, nl_, nu_, si, ei, sj, ej, sk, ek, p);
}

void CellCenteredBoundaryVariable::SetupPersistentMPI() {
#ifdef MPI_PARALLEL
  MeshBlock* pmb = pmy_block_;
  int &mylevel = pmb->loc.level;

  int cng, cng1, cng2, cng3;
  cng  = cng1 = pmb->cnghost;
  cng2 = (pmy_mesh_->ndim) ? cng : 0;
  cng3 = (pmy_mesh_->ndim) ? cng : 0;
  int ssize, rsize;
  int tag;
  // Initialize non-polar neighbor communications to other ranks
  for (int n=0; n < pmb->pbval->nneighbor; n++) {
    NeighborBlock& nb = pmb->pbval->neighbor[n];
    if (nb.snb.rank != Globals::my_rank) {
      if (nb.snb.level == mylevel) { // same
        ssize = rsize = ((nb.ni.ox1 == 0) ? pmb->block_size.nx1 : NGHOST)
              *((nb.ni.ox2 == 0) ? pmb->block_size.nx2 : NGHOST)
              *((nb.ni.ox3 == 0) ? pmb->block_size.nx3 : NGHOST);
      } else if (nb.snb.level < mylevel) { // coarser
        ssize = ((nb.ni.ox1 == 0) ? ((pmb->block_size.nx1 + 1)/2) : NGHOST)
              *((nb.ni.ox2 == 0) ? ((pmb->block_size.nx2 + 1)/2) : NGHOST)
              *((nb.ni.ox3 == 0) ? ((pmb->block_size.nx3 + 1)/2) : NGHOST);
        rsize = ((nb.ni.ox1 == 0) ? ((pmb->block_size.nx1 + 1)/2 + cng1) : cng1)
              *((nb.ni.ox2 == 0) ? ((pmb->block_size.nx2 + 1)/2 + cng2) : cng2)
              *((nb.ni.ox3 == 0) ? ((pmb->block_size.nx3 + 1)/2 + cng3) : cng3);
      } else { // finer
        ssize = ((nb.ni.ox1 == 0) ? ((pmb->block_size.nx1 + 1)/2 + cng1) : cng1)
              *((nb.ni.ox2 == 0) ? ((pmb->block_size.nx2 + 1)/2 + cng2) : cng2)
              *((nb.ni.ox3 == 0) ? ((pmb->block_size.nx3 + 1)/2 + cng3) : cng3);
        rsize = ((nb.ni.ox1 == 0) ? ((pmb->block_size.nx1 + 1)/2) : NGHOST)
              *((nb.ni.ox2 == 0) ? ((pmb->block_size.nx2 + 1)/2) : NGHOST)
              *((nb.ni.ox3 == 0) ? ((pmb->block_size.nx3 + 1)/2) : NGHOST);
      }
      ssize *= (nu_ + 1); rsize *= (nu_ + 1);
      // specify the offsets in the view point of the target block: flip ox? signs

      // Initialize persistent communication requests attached to specific BoundaryData
      tag = pmb->pbval->CreateBvalsMPITag(nb.snb.lid, nb.targetid, cc_phys_id_);
      if (bd_var_.req_send[nb.bufid] != MPI_REQUEST_NULL)
        MPI_Request_free(&bd_var_.req_send[nb.bufid]);
      MPI_Send_init(bd_var_.send[nb.bufid], ssize, MPI_ATHENA_REAL,
                    nb.snb.rank, tag, MPI_COMM_WORLD, &(bd_var_.req_send[nb.bufid]));
      tag = pmb->pbval->CreateBvalsMPITag(pmb->lid, nb.bufid, cc_phys_id_);
      if (bd_var_.req_recv[nb.bufid] != MPI_REQUEST_NULL)
        MPI_Request_free(&bd_var_.req_recv[nb.bufid]);
      MPI_Recv_init(bd_var_.recv[nb.bufid], rsize, MPI_ATHENA_REAL,
                    nb.snb.rank, tag, MPI_COMM_WORLD, &(bd_var_.req_recv[nb.bufid]));

      if (pmy_mesh_->multilevel && nb.ni.type == NeighborConnect::face) {
        int size;
        if (nb.fid == 0 || nb.fid == 1)
          size = ((pmb->block_size.nx2 + 1)/2)*((pmb->block_size.nx3 + 1)/2);
        else if (nb.fid == 2 || nb.fid == 3)
          size = ((pmb->block_size.nx1 + 1)/2)*((pmb->block_size.nx3 + 1)/2);
        else // (nb.fid == 4 || nb.fid == 5)
          size = ((pmb->block_size.nx1 + 1)/2)*((pmb->block_size.nx2 + 1)/2);
        size *= (nu_ + 1);
        if (nb.snb.level < mylevel) { // send to coarser
          tag = pmb->pbval->CreateBvalsMPITag(nb.snb.lid, nb.targetid, cc_flx_phys_id_);
          if (bd_var_flcor_.req_send[nb.bufid] != MPI_REQUEST_NULL)
            MPI_Request_free(&bd_var_flcor_.req_send[nb.bufid]);
          MPI_Send_init(bd_var_flcor_.send[nb.bufid], size, MPI_ATHENA_REAL,
                        nb.snb.rank, tag, MPI_COMM_WORLD,
                        &(bd_var_flcor_.req_send[nb.bufid]));
        } else if (nb.snb.level > mylevel) { // receive from finer
          tag = pmb->pbval->CreateBvalsMPITag(pmb->lid, nb.bufid, cc_flx_phys_id_);
          if (bd_var_flcor_.req_recv[nb.bufid] != MPI_REQUEST_NULL)
            MPI_Request_free(&bd_var_flcor_.req_recv[nb.bufid]);
          MPI_Recv_init(bd_var_flcor_.recv[nb.bufid], size, MPI_ATHENA_REAL,
                        nb.snb.rank, tag, MPI_COMM_WORLD,
                        &(bd_var_flcor_.req_recv[nb.bufid]));
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
  for (int n=0; n < pmb->pbval->nneighbor; n++) {
    NeighborBlock& nb = pmb->pbval->neighbor[n];
    if (nb.snb.rank != Globals::my_rank) {
      MPI_Start(&(bd_var_.req_recv[nb.bufid]));
      if (phase == BoundaryCommSubset::all && nb.ni.type == NeighborConnect::face
          && nb.snb.level > mylevel) // opposite condition in ClearBoundary()
        MPI_Start(&(bd_var_flcor_.req_recv[nb.bufid]));
    }
  }
#endif
  return;
}


void CellCenteredBoundaryVariable::ClearBoundary(BoundaryCommSubset phase) {
  for (int n=0; n < pmy_block_->pbval->nneighbor; n++) {
    NeighborBlock& nb = pmy_block_->pbval->neighbor[n];
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
      if (phase == BoundaryCommSubset::all && nb.ni.type == NeighborConnect::face
          && nb.snb.level < mylevel)
        MPI_Wait(&(bd_var_flcor_.req_send[nb.bufid]), MPI_STATUS_IGNORE);
    }
#endif
  }
}
}
