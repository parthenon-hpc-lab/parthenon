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
//! \file flux_correction_cc.cpp
//  \brief functions that perform flux correction for CELL_CENTERED variables

// C headers

// C++ headers
#include <algorithm>  // min
#include <cmath>
#include <cstdlib>
#include <cstring>    // std::memcpy
#include <iomanip>
#include <iostream>   // endl
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
//----------------------------------------------------------------------------------------
//! \fn void CellCenteredBoundaryVariable::SendFluxCorrection()
//  \brief Restrict, pack and send the surface flux to the coarse neighbor(s)

void CellCenteredBoundaryVariable::SendFluxCorrection() {
  MeshBlock *pmb = pmy_block_;
  auto &pco = pmb->pcoord;

  // cache pointers to surface area arrays (BoundaryBase protected variable)
  AthenaArray<Real> &sarea0 = pmb->pbval->sarea_[0];
  AthenaArray<Real> &sarea1 = pmb->pbval->sarea_[1];

  for (int n=0; n < pmb->pbval->nneighbor; n++) {
    NeighborBlock& nb = pmb->pbval->neighbor[n];
    if (nb.ni.type != NeighborConnect::face) break;
    if (bd_var_flcor_.sflag[nb.bufid] == BoundaryStatus::completed) continue;
    if (nb.snb.level == pmb->loc.level - 1) {
      int is, ie, js, je, ks, ke;
      pmb->cells.GetIndices(interior,is,ie,js,je,ks,ke);
      int p = 0;
      Real *sbuf = bd_var_flcor_.send[nb.bufid];
      // x1 direction
      if (nb.fid == BoundaryFace::inner_x1 || nb.fid == BoundaryFace::outer_x1) {
        int i = is + (ie - is + 1)*nb.fid;
        if (pmb->block_size.nx3>1) { // 3D
          for (int nn=nl_; nn<=nu_; nn++) {
            for (int k=ks; k<=ke; k+=2) {
              for (int j=js; j<=je; j+=2) {
                Real amm = pco->GetFace1Area(k,   j,   i);
                Real amp = pco->GetFace1Area(k,   j+1, i);
                Real apm = pco->GetFace1Area(k+1, j,   i);
                Real app = pco->GetFace1Area(k+1, j+1, i);
                Real tarea = amm + amp + apm + app;
                sbuf[p++] = (x1flux(nn, k  , j  , i)*amm
                            + x1flux(nn, k  , j+1, i)*amp
                            + x1flux(nn, k+1, j  , i)*apm
                            + x1flux(nn, k+1, j+1, i)*app)/tarea;
              }
            }
          }
        } else if (pmb->block_size.nx2>1) { // 2D
          int k = ks;
          for (int nn=nl_; nn<=nu_; nn++) {
            for (int j=js; j<=je; j+=2) {
              Real am = pco->GetFace1Area(k, j,   i);
              Real ap = pco->GetFace1Area(k, j+1, i);
              Real tarea = am + ap;
              sbuf[p++] = (x1flux(nn, k, j  , i)*am + x1flux(nn, k, j+1, i)*ap)/tarea;
            }
          }
        } else { // 1D
          int k = ks, j = js;
          for (int nn=nl_; nn<=nu_; nn++)
            sbuf[p++] = x1flux(nn, k, j, i);
        }
        // x2 direction
      } else if (nb.fid == BoundaryFace::inner_x2 || nb.fid == BoundaryFace::outer_x2) {
        int j = js + (je - js + 1)*(nb.fid & 1);
        if (pmb->block_size.nx3>1) { // 3D
          for (int nn=nl_; nn<=nu_; nn++) {
            for (int k=ks; k<=ke; k+=2) {
              pco->Face2Area(k  , j, is, ie, sarea0);
              pco->Face2Area(k+1, j, is, ie, sarea1);
              for (int i=is; i<=ie; i+=2) {
                Real tarea = sarea0(i) + sarea0(i+1) + sarea1(i) + sarea1(i+1);
                sbuf[p++] = (x2flux(nn, k  , j, i  )*sarea0(i  )
                            + x2flux(nn, k  , j, i+1)*sarea0(i+1)
                            + x2flux(nn, k+1, j, i  )*sarea1(i  )
                            + x2flux(nn, k+1, j, i+1)*sarea1(i+1))/tarea;
              }
            }
          }
        } else if (pmb->block_size.nx2>1) { // 2D
          int k = ks;
          for (int nn=nl_; nn<=nu_; nn++) {
            pco->Face2Area(0, j, is ,ie, sarea0);
            for (int i=is; i<=ie; i+=2) {
              Real tarea = sarea0(i) + sarea0(i+1);
              sbuf[p++] = (x2flux(nn, k, j, i  )*sarea0(i  )
                          + x2flux(nn, k, j, i+1)*sarea0(i+1))/tarea;
            }
          }
        }
        // x3 direction - 3D onl_y
      } else if (nb.fid == BoundaryFace::inner_x3 || nb.fid == BoundaryFace::outer_x3) {
        int k = ks + (ke - ks + 1)*(nb.fid & 1);
        for (int nn=nl_; nn<=nu_; nn++) {
          for (int j=js; j<=je; j+=2) {
            pco->Face3Area(k, j,   is, ie, sarea0);
            pco->Face3Area(k, j+1, is, ie, sarea1);
            for (int i=is; i<=ie; i+=2) {
              Real tarea = sarea0(i) + sarea0(i+1) + sarea1(i) + sarea1(i+1);
              sbuf[p++] = (x3flux(nn, k, j  , i  )*sarea0(i  )
                           + x3flux(nn, k, j  , i+1)*sarea0(i+1)
                           + x3flux(nn, k, j+1, i  )*sarea1(i  )
                           + x3flux(nn, k, j+1, i+1)*sarea1(i+1))/tarea;
            }
          }
        }
      }
      if (nb.snb.rank == Globals::my_rank) { // on the same node
        CopyFluxCorrectionBufferSameProcess(nb, p);
      }
#ifdef MPI_PARALLEL
      else
        MPI_Start(&(bd_var_flcor_.req_send[nb.bufid]));
#endif
      bd_var_flcor_.sflag[nb.bufid] = BoundaryStatus::completed;
    }
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn bool CellCenteredBoundaryVariable::ReceiveFluxCorrection()
//  \brief Receive and apply the surface flux from the finer neighbor(s)

bool CellCenteredBoundaryVariable::ReceiveFluxCorrection() {
  MeshBlock *pmb = pmy_block_;
  bool bflag=true;

  for (int n=0; n < pmb->pbval->nneighbor; n++) {
    NeighborBlock& nb = pmb->pbval->neighbor[n];
    if (nb.ni.type != NeighborConnect::face) break;
    if (nb.snb.level == pmb->loc.level+1) {
      if (bd_var_flcor_.flag[nb.bufid] == BoundaryStatus::completed) continue;
      if (bd_var_flcor_.flag[nb.bufid] == BoundaryStatus::waiting) {
        if (nb.snb.rank == Globals::my_rank) {// on the same process
          bflag = false;
          continue;
        }
#ifdef MPI_PARALLEL
        else { // NOLINT
          int test;
          MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &test,
                     MPI_STATUS_IGNORE);
          MPI_Test(&(bd_var_flcor_.req_recv[nb.bufid]), &test, MPI_STATUS_IGNORE);
          if (!static_cast<bool>(test)) {
            bflag = false;
            continue;
          }
          bd_var_flcor_.flag[nb.bufid] = BoundaryStatus::arrived;
        }
#endif
      }
      // boundary arrived; apply flux correction
      int p = 0;
      Real *rbuf=bd_var_flcor_.recv[nb.bufid];

      int is, ie, js, je, ks, ke;
      pmb->cells.GetIndices(interior,is,ie,js,je,ks,ke);
      if (nb.fid == BoundaryFace::inner_x1 || nb.fid == BoundaryFace::outer_x1) {
        int il = is + (ie - is)*nb.fid+nb.fid;
        int jl = js, ju = je, kl = ks, ku = ke;
        if (nb.ni.fi1 == 0) ju -= pmb->block_size.nx2/2;
        else          jl += pmb->block_size.nx2/2;
        if (nb.ni.fi2 == 0) ku -= pmb->block_size.nx3/2;
        else          kl += pmb->block_size.nx3/2;
        for (int nn=nl_; nn<=nu_; nn++) {
          for (int k=kl; k<=ku; k++) {
            for (int j=jl; j<=ju; j++)
              x1flux(nn,k,j,il) = rbuf[p++];
          }
        }
      } else if (nb.fid == BoundaryFace::inner_x2 || nb.fid == BoundaryFace::outer_x2) {
        int jl = js + (je - js)*(nb.fid & 1) + (nb.fid & 1);
        int il = is, iu = ie, kl = ks, ku = ke;
        if (nb.ni.fi1 == 0) iu -= pmb->block_size.nx1/2;
        else          il += pmb->block_size.nx1/2;
        if (nb.ni.fi2 == 0) ku -= pmb->block_size.nx3/2;
        else          kl += pmb->block_size.nx3/2;
        for (int nn=nl_; nn<=nu_; nn++) {
          for (int k=kl; k<=ku; k++) {
            for (int i=il; i<=iu; i++)
              x2flux(nn,k,jl,i) = rbuf[p++];
          }
        }
      } else if (nb.fid == BoundaryFace::inner_x3 || nb.fid == BoundaryFace::outer_x3) {
        int kl = ks + (ke - ks)*(nb.fid & 1) + (nb.fid & 1);
        int il = is, iu = ie, jl = js, ju = je;
        if (nb.ni.fi1 == 0) iu -= pmb->block_size.nx1/2;
        else          il += pmb->block_size.nx1/2;
        if (nb.ni.fi2 == 0) ju -= pmb->block_size.nx2/2;
        else          jl += pmb->block_size.nx2/2;
        for (int nn=nl_; nn<=nu_; nn++) {
          for (int j=jl; j<=ju; j++) {
            for (int i=il; i<=iu; i++)
              x3flux(nn,kl,j,i) = rbuf[p++];
          }
        }
      }
      bd_var_flcor_.flag[nb.bufid] = BoundaryStatus::completed;
    }
  }
  return bflag;
}
}
