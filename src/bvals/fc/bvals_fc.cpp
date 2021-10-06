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
//! \file bvals_fc.cpp
//  \brief functions that apply BCs for FACE_CENTERED variables

#include "bvals/fc/bvals_fc.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
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

namespace parthenon {

FaceCenteredBoundaryVariable::FaceCenteredBoundaryVariable(std::weak_ptr<MeshBlock> pmb,
                                                           FaceField *var,
                                                           FaceField &coarse_buf,
                                                           EdgeField &var_flux)
    // TODO(JL): Add a label for face variables
    : BoundaryVariable(
          pmb, false,
          "TODO: Give label to face variables"), // sparse variables are not supported yet
      var_fc(var), coarse_buf(coarse_buf) {
  // assuming Field, not generic FaceCenteredBoundaryVariable:

  InitBoundaryData(bd_var_, BoundaryQuantity::fc);
  InitBoundaryData(bd_var_flcor_, BoundaryQuantity::fc_flcor);

#ifdef MPI_PARALLEL
  // KGF: dead code, leaving for now:
  // fc_phys_id_ = pmb->pbval->ReserveTagVariableIDs(2);
  fc_phys_id_ = pmb.lock()->pbval->bvars_next_phys_id_;
  fc_flx_phys_id_ = fc_phys_id_ + 1;
#endif
}

// destructor

FaceCenteredBoundaryVariable::~FaceCenteredBoundaryVariable() {
  DestroyBoundaryData(bd_var_);
  DestroyBoundaryData(bd_var_flcor_);
}

int FaceCenteredBoundaryVariable::ComputeVariableBufferSize(const NeighborIndexes &ni,
                                                            int cng) {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  int nx1 = pmb->block_size.nx1;
  int nx2 = pmb->block_size.nx2;
  int nx3 = pmb->block_size.nx3;
  const int f2 = (pmy_mesh_->ndim >= 2) ? 1 : 0; // extra cells/faces from being 2d
  const int f3 = (pmy_mesh_->ndim >= 3) ? 1 : 0; // extra cells/faces from being 3d
  int cng1, cng2, cng3;
  cng1 = cng;
  cng2 = cng * f2;
  cng3 = cng * f3;

  int size1 = ((ni.ox1 == 0) ? (nx1 + 1) : Globals::nghost) *
              ((ni.ox2 == 0) ? (nx2) : Globals::nghost) *
              ((ni.ox3 == 0) ? (nx3) : Globals::nghost);
  int size2 = ((ni.ox1 == 0) ? (nx1) : Globals::nghost) *
              ((ni.ox2 == 0) ? (nx2 + f2) : Globals::nghost) *
              ((ni.ox3 == 0) ? (nx3) : Globals::nghost);
  int size3 = ((ni.ox1 == 0) ? (nx1) : Globals::nghost) *
              ((ni.ox2 == 0) ? (nx2) : Globals::nghost) *
              ((ni.ox3 == 0) ? (nx3 + f3) : Globals::nghost);
  int size = size1 + size2 + size3;
  if (pmy_mesh_->multilevel) {
    if (ni.type != NeighborConnect::face) {
      if (ni.ox1 != 0) size1 = size1 / Globals::nghost * (Globals::nghost + 1);
      if (ni.ox2 != 0) size2 = size2 / Globals::nghost * (Globals::nghost + 1);
      if (ni.ox3 != 0) size3 = size3 / Globals::nghost * (Globals::nghost + 1);
    }
    size = size1 + size2 + size3;
    int f2c1 = ((ni.ox1 == 0) ? ((nx1 + 1) / 2 + 1) : Globals::nghost) *
               ((ni.ox2 == 0) ? ((nx2 + 1) / 2) : Globals::nghost) *
               ((ni.ox3 == 0) ? ((nx3 + 1) / 2) : Globals::nghost);
    int f2c2 = ((ni.ox1 == 0) ? ((nx1 + 1) / 2) : Globals::nghost) *
               ((ni.ox2 == 0) ? ((nx2 + 1) / 2 + f2) : Globals::nghost) *
               ((ni.ox3 == 0) ? ((nx3 + 1) / 2) : Globals::nghost);
    int f2c3 = ((ni.ox1 == 0) ? ((nx1 + 1) / 2) : Globals::nghost) *
               ((ni.ox2 == 0) ? ((nx2 + 1) / 2) : Globals::nghost) *
               ((ni.ox3 == 0) ? ((nx3 + 1) / 2 + f3) : Globals::nghost);
    if (ni.type != NeighborConnect::face) {
      if (ni.ox1 != 0) f2c1 = f2c1 / Globals::nghost * (Globals::nghost + 1);
      if (ni.ox2 != 0) f2c2 = f2c2 / Globals::nghost * (Globals::nghost + 1);
      if (ni.ox3 != 0) f2c3 = f2c3 / Globals::nghost * (Globals::nghost + 1);
    }
    int fsize = f2c1 + f2c2 + f2c3;
    int c2f1 = ((ni.ox1 == 0) ? ((nx1 + 1) / 2 + cng1 + 1) : cng + 1) *
               ((ni.ox2 == 0) ? ((nx2 + 1) / 2 + cng2) : cng) *
               ((ni.ox3 == 0) ? ((nx3 + 1) / 2 + cng3) : cng);
    int c2f2 = ((ni.ox1 == 0) ? ((nx1 + 1) / 2 + cng1) : cng) *
               ((ni.ox2 == 0) ? ((nx2 + 1) / 2 + cng2 + f2) : cng + 1) *
               ((ni.ox3 == 0) ? ((nx3 + 1) / 2 + cng3) : cng);
    int c2f3 = ((ni.ox1 == 0) ? ((nx1 + 1) / 2 + cng1) : cng) *
               ((ni.ox2 == 0) ? ((nx2 + 1) / 2 + cng2) : cng) *
               ((ni.ox3 == 0) ? ((nx3 + 1) / 2 + cng3 + f3) : cng + 1);
    int csize = c2f1 + c2f2 + c2f3;
    size = std::max(size, std::max(csize, fsize));
  }
  return size;
}

int FaceCenteredBoundaryVariable::ComputeFluxCorrectionBufferSize(
    const NeighborIndexes &ni, int cng) {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  int nx1 = pmb->block_size.nx1;
  int nx2 = pmb->block_size.nx2;
  int nx3 = pmb->block_size.nx3;
  int size = 0;

  if (ni.type == NeighborConnect::face) {
    if (nx3 > 1) { // 3D
      if (ni.ox1 != 0)
        size = (nx2 + 1) * (nx3) + (nx2) * (nx3 + 1);
      else if (ni.ox2 != 0)
        size = (nx1 + 1) * (nx3) + (nx1) * (nx3 + 1);
      else
        size = (nx1 + 1) * (nx2) + (nx1) * (nx2 + 1);
    } else if (nx2 > 1) { // 2D
      if (ni.ox1 != 0)
        size = (nx2 + 1) + nx2;
      else
        size = (nx1 + 1) + nx1;
    } else { // 1D
      size = 2;
    }
  } else if (ni.type == NeighborConnect::edge) {
    if (nx3 > 1) { // 3D
      if (ni.ox3 == 0) size = nx3;
      if (ni.ox2 == 0) size = nx2;
      if (ni.ox1 == 0) size = nx1;
    } else if (nx2 > 1) {
      size = 1;
    }
  }
  return size;
}

void FaceCenteredBoundaryVariable::CountFineEdges() {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  int &mylevel = pmb->loc.level;

  // count the number of the fine meshblocks contacting on each edge
  int eid = 0;
  if (pmb->block_size.nx2 > 1) {
    for (int ox2 = -1; ox2 <= 1; ox2 += 2) {
      for (int ox1 = -1; ox1 <= 1; ox1 += 2) {
        int nis, nie, njs, nje;
        nis = std::max(ox1 - 1, -1), nie = std::min(ox1 + 1, 1);
        njs = std::max(ox2 - 1, -1), nje = std::min(ox2 + 1, 1);
        int nf = 0, fl = mylevel;
        for (int nj = njs; nj <= nje; nj++) {
          for (int ni = nis; ni <= nie; ni++) {
            if (pmb->pbval->nblevel[1][nj + 1][ni + 1] > fl) fl++, nf = 0;
            if (pmb->pbval->nblevel[1][nj + 1][ni + 1] == fl) nf++;
          }
        }
        edge_flag_[eid] = (fl == mylevel);
        nedge_fine_[eid++] = nf;
      }
    }
  }

  if (pmb->block_size.nx3 > 1) {
    for (int ox3 = -1; ox3 <= 1; ox3 += 2) {
      for (int ox1 = -1; ox1 <= 1; ox1 += 2) {
        int nis, nie, nks, nke;
        nis = std::max(ox1 - 1, -1), nie = std::min(ox1 + 1, 1);
        nks = std::max(ox3 - 1, -1), nke = std::min(ox3 + 1, 1);
        int nf = 0, fl = mylevel;
        for (int nk = nks; nk <= nke; nk++) {
          for (int ni = nis; ni <= nie; ni++) {
            if (pmb->pbval->nblevel[nk + 1][1][ni + 1] > fl) fl++, nf = 0;
            if (pmb->pbval->nblevel[nk + 1][1][ni + 1] == fl) nf++;
          }
        }
        edge_flag_[eid] = (fl == mylevel);
        nedge_fine_[eid++] = nf;
      }
    }

    for (int ox3 = -1; ox3 <= 1; ox3 += 2) {
      for (int ox2 = -1; ox2 <= 1; ox2 += 2) {
        int njs, nje, nks, nke;
        njs = std::max(ox2 - 1, -1), nje = std::min(ox2 + 1, 1);
        nks = std::max(ox3 - 1, -1), nke = std::min(ox3 + 1, 1);
        int nf = 0, fl = mylevel;
        for (int nk = nks; nk <= nke; nk++) {
          for (int nj = njs; nj <= nje; nj++) {
            if (pmb->pbval->nblevel[nk + 1][nj + 1][1] > fl) fl++, nf = 0;
            if (pmb->pbval->nblevel[nk + 1][nj + 1][1] == fl) nf++;
          }
        }
        edge_flag_[eid] = (fl == mylevel);
        nedge_fine_[eid++] = nf;
      }
    }
  }
}

void FaceCenteredBoundaryVariable::SetupPersistentMPI() {
  CountFineEdges();

#ifdef MPI_PARALLEL
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  int nx1 = pmb->block_size.nx1;
  int nx2 = pmb->block_size.nx2;
  int nx3 = pmb->block_size.nx3;
  int &mylevel = pmb->loc.level;

  const int f2 = (pmy_mesh_->ndim >= 2) ? 1 : 0; // extra cells/faces from being 2d
  const int f3 = (pmy_mesh_->ndim >= 3) ? 1 : 0; // extra cells/faces from being 3d
  int cng, cng1, cng2, cng3;
  cng = cng1 = pmb->cnghost;
  cng2 = cng * f2;
  cng3 = cng * f3;
  int ssize, rsize;
  int tag;
  // Initialize non-polar neighbor communications to other ranks
  for (int n = 0; n < pmb->pbval->nneighbor; n++) {
    NeighborBlock &nb = pmb->pbval->neighbor[n];
    if (nb.snb.rank != Globals::my_rank) {
      int size, csize, fsize;
      int size1 = ((nb.ni.ox1 == 0) ? (nx1 + 1) : Globals::nghost) *
                  ((nb.ni.ox2 == 0) ? (nx2) : Globals::nghost) *
                  ((nb.ni.ox3 == 0) ? (nx3) : Globals::nghost);
      int size2 = ((nb.ni.ox1 == 0) ? (nx1) : Globals::nghost) *
                  ((nb.ni.ox2 == 0) ? (nx2 + f2) : Globals::nghost) *
                  ((nb.ni.ox3 == 0) ? (nx3) : Globals::nghost);
      int size3 = ((nb.ni.ox1 == 0) ? (nx1) : Globals::nghost) *
                  ((nb.ni.ox2 == 0) ? (nx2) : Globals::nghost) *
                  ((nb.ni.ox3 == 0) ? (nx3 + f3) : Globals::nghost);
      size = size1 + size2 + size3;
      if (pmy_mesh_->multilevel) {
        if (nb.ni.type != NeighborConnect::face) {
          if (nb.ni.ox1 != 0) size1 = size1 / Globals::nghost * (Globals::nghost + 1);
          if (nb.ni.ox2 != 0) size2 = size2 / Globals::nghost * (Globals::nghost + 1);
          if (nb.ni.ox3 != 0) size3 = size3 / Globals::nghost * (Globals::nghost + 1);
        }
        size = size1 + size2 + size3;
        int f2c1 = ((nb.ni.ox1 == 0) ? ((nx1 + 1) / 2 + 1) : Globals::nghost) *
                   ((nb.ni.ox2 == 0) ? ((nx2 + 1) / 2) : Globals::nghost) *
                   ((nb.ni.ox3 == 0) ? ((nx3 + 1) / 2) : Globals::nghost);
        int f2c2 = ((nb.ni.ox1 == 0) ? ((nx1 + 1) / 2) : Globals::nghost) *
                   ((nb.ni.ox2 == 0) ? ((nx2 + 1) / 2 + f2) : Globals::nghost) *
                   ((nb.ni.ox3 == 0) ? ((nx3 + 1) / 2) : Globals::nghost);
        int f2c3 = ((nb.ni.ox1 == 0) ? ((nx1 + 1) / 2) : Globals::nghost) *
                   ((nb.ni.ox2 == 0) ? ((nx2 + 1) / 2) : Globals::nghost) *
                   ((nb.ni.ox3 == 0) ? ((nx3 + 1) / 2 + f3) : Globals::nghost);
        if (nb.ni.type != NeighborConnect::face) {
          if (nb.ni.ox1 != 0) f2c1 = f2c1 / Globals::nghost * (Globals::nghost + 1);
          if (nb.ni.ox2 != 0) f2c2 = f2c2 / Globals::nghost * (Globals::nghost + 1);
          if (nb.ni.ox3 != 0) f2c3 = f2c3 / Globals::nghost * (Globals::nghost + 1);
        }
        fsize = f2c1 + f2c2 + f2c3;
        int c2f1 = ((nb.ni.ox1 == 0) ? ((nx1 + 1) / 2 + cng1 + 1) : cng + 1) *
                   ((nb.ni.ox2 == 0) ? ((nx2 + 1) / 2 + cng2) : cng) *
                   ((nb.ni.ox3 == 0) ? ((nx3 + 1) / 2 + cng3) : cng);
        int c2f2 = ((nb.ni.ox1 == 0) ? ((nx1 + 1) / 2 + cng1) : cng) *
                   ((nb.ni.ox2 == 0) ? ((nx2 + 1) / 2 + cng2 + f2) : cng + 1) *
                   ((nb.ni.ox3 == 0) ? ((nx3 + 1) / 2 + cng3) : cng);
        int c2f3 = ((nb.ni.ox1 == 0) ? ((nx1 + 1) / 2 + cng1) : cng) *
                   ((nb.ni.ox2 == 0) ? ((nx2 + 1) / 2 + cng2) : cng) *
                   ((nb.ni.ox3 == 0) ? ((nx3 + 1) / 2 + cng3 + f3) : cng + 1);
        csize = c2f1 + c2f2 + c2f3;
      }                            // end of multilevel
      if (nb.snb.level == mylevel) // same refinement level
        ssize = size, rsize = size;
      else if (nb.snb.level < mylevel) // coarser
        ssize = fsize, rsize = csize;
      else // finer
        ssize = csize, rsize = fsize;

      // face-centered field: bd_var_
      tag = pmb->pbval->CreateBvalsMPITag(nb.snb.lid, nb.targetid, fc_phys_id_);
      if (bd_var_.req_send[nb.bufid] != MPI_REQUEST_NULL)
        MPI_Request_free(&bd_var_.req_send[nb.bufid]);
      PARTHENON_MPI_CHECK(MPI_Send_init(bd_var_.send[nb.bufid].data(), ssize,
                                        MPI_PARTHENON_REAL, nb.snb.rank, tag,
                                        MPI_COMM_WORLD, &(bd_var_.req_send[nb.bufid])));
      tag = pmb->pbval->CreateBvalsMPITag(pmb->lid, nb.bufid, fc_phys_id_);
      if (bd_var_.req_recv[nb.bufid] != MPI_REQUEST_NULL)
        MPI_Request_free(&bd_var_.req_recv[nb.bufid]);
      PARTHENON_MPI_CHECK(MPI_Recv_init(bd_var_.recv[nb.bufid].data(), rsize,
                                        MPI_PARTHENON_REAL, nb.snb.rank, tag,
                                        MPI_COMM_WORLD, &(bd_var_.req_recv[nb.bufid])));

      // set up flux correction MPI communication buffers
      int f2csize;
      if (nb.ni.type == NeighborConnect::face) { // face
        if (nx3 > 1) {                           // 3D
          if (nb.fid == BoundaryFace::inner_x1 || nb.fid == BoundaryFace::outer_x1) {
            size = (nx2 + 1) * (nx3) + (nx2) * (nx3 + 1);
            f2csize = (nx2 / 2 + 1) * (nx3 / 2) + (nx2 / 2) * (nx3 / 2 + 1);
          } else if (nb.fid == BoundaryFace::inner_x2 ||
                     nb.fid == BoundaryFace::outer_x2) {
            size = (nx1 + 1) * (nx3) + (nx1) * (nx3 + 1);
            f2csize = (nx1 / 2 + 1) * (nx3 / 2) + (nx1 / 2) * (nx3 / 2 + 1);
          } else if (nb.fid == BoundaryFace::inner_x3 ||
                     nb.fid == BoundaryFace::outer_x3) {
            size = (nx1 + 1) * (nx2) + (nx1) * (nx2 + 1);
            f2csize = (nx1 / 2 + 1) * (nx2 / 2) + (nx1 / 2) * (nx2 / 2 + 1);
          }
        } else if (nx2 > 1) { // 2D
          if (nb.fid == BoundaryFace::inner_x1 || nb.fid == BoundaryFace::outer_x1) {
            size = (nx2 + 1) + nx2;
            f2csize = (nx2 / 2 + 1) + nx2 / 2;
          } else if (nb.fid == BoundaryFace::inner_x2 ||
                     nb.fid == BoundaryFace::outer_x2) {
            size = (nx1 + 1) + nx1;
            f2csize = (nx1 / 2 + 1) + nx1 / 2;
          }
        } else { // 1D
          size = f2csize = 2;
        }
      } else if (nb.ni.type == NeighborConnect::edge) { // edge
        if (nx3 > 1) {                                  // 3D
          if (nb.eid >= 0 && nb.eid < 4) {
            size = nx3;
            f2csize = nx3 / 2;
          } else if (nb.eid >= 4 && nb.eid < 8) {
            size = nx2;
            f2csize = nx2 / 2;
          } else if (nb.eid >= 8 && nb.eid < 12) {
            size = nx1;
            f2csize = nx1 / 2;
          }
        } else if (nx2 > 1) { // 2D
          size = f2csize = 1;
        }
      } else { // corner
        continue;
      }

      if (nb.snb.level == mylevel) { // the same level
        if ((nb.ni.type == NeighborConnect::face) ||
            ((nb.ni.type == NeighborConnect::edge) && (edge_flag_[nb.eid]))) {
          tag = pmb->pbval->CreateBvalsMPITag(nb.snb.lid, nb.targetid, fc_flx_phys_id_);
          if (bd_var_flcor_.req_send[nb.bufid] != MPI_REQUEST_NULL)
            MPI_Request_free(&bd_var_flcor_.req_send[nb.bufid]);
          PARTHENON_MPI_CHECK(MPI_Send_init(
              bd_var_flcor_.send[nb.bufid].data(), size, MPI_PARTHENON_REAL, nb.snb.rank,
              tag, MPI_COMM_WORLD, &(bd_var_flcor_.req_send[nb.bufid])));
          tag = pmb->pbval->CreateBvalsMPITag(pmb->lid, nb.bufid, fc_flx_phys_id_);
          if (bd_var_flcor_.req_recv[nb.bufid] != MPI_REQUEST_NULL)
            MPI_Request_free(&bd_var_flcor_.req_recv[nb.bufid]);
          PARTHENON_MPI_CHECK(MPI_Recv_init(
              bd_var_flcor_.recv[nb.bufid].data(), size, MPI_PARTHENON_REAL, nb.snb.rank,
              tag, MPI_COMM_WORLD, &(bd_var_flcor_.req_recv[nb.bufid])));
        }
      }
      if (nb.snb.level > mylevel) { // finer neighbor
        tag = pmb->pbval->CreateBvalsMPITag(pmb->lid, nb.bufid, fc_flx_phys_id_);
        if (bd_var_flcor_.req_recv[nb.bufid] != MPI_REQUEST_NULL)
          MPI_Request_free(&bd_var_flcor_.req_recv[nb.bufid]);
        PARTHENON_MPI_CHECK(MPI_Recv_init(
            bd_var_flcor_.recv[nb.bufid].data(), f2csize, MPI_PARTHENON_REAL, nb.snb.rank,
            tag, MPI_COMM_WORLD, &(bd_var_flcor_.req_recv[nb.bufid])));
      }
      if (nb.snb.level < mylevel) { // coarser neighbor
        tag = pmb->pbval->CreateBvalsMPITag(nb.snb.lid, nb.targetid, fc_flx_phys_id_);
        if (bd_var_flcor_.req_send[nb.bufid] != MPI_REQUEST_NULL)
          MPI_Request_free(&bd_var_flcor_.req_send[nb.bufid]);
        PARTHENON_MPI_CHECK(MPI_Send_init(
            bd_var_flcor_.send[nb.bufid].data(), f2csize, MPI_PARTHENON_REAL, nb.snb.rank,
            tag, MPI_COMM_WORLD, &(bd_var_flcor_.req_send[nb.bufid])));
      }
    } // neighbor block is on separate MPI process
  }   // end loop over neighbors
#endif
  return;
}

void FaceCenteredBoundaryVariable::StartReceiving(BoundaryCommSubset phase) {
  if (phase == BoundaryCommSubset::all) recv_flx_same_lvl_ = true;
#ifdef MPI_PARALLEL
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  int mylevel = pmb->loc.level;
  for (int n = 0; n < pmb->pbval->nneighbor; n++) {
    NeighborBlock &nb = pmb->pbval->neighbor[n];
    if (nb.snb.rank != Globals::my_rank) {
      pmb->exec_space.fence();
      PARTHENON_MPI_CHECK(MPI_Start(&(bd_var_.req_recv[nb.bufid])));
      if (phase == BoundaryCommSubset::all &&
          (nb.ni.type == NeighborConnect::face || nb.ni.type == NeighborConnect::edge)) {
        if ((nb.snb.level > mylevel) ||
            ((nb.snb.level == mylevel) &&
             ((nb.ni.type == NeighborConnect::face) ||
              ((nb.ni.type == NeighborConnect::edge) && (edge_flag_[nb.eid])))))
          PARTHENON_MPI_CHECK(MPI_Start(&(bd_var_flcor_.req_recv[nb.bufid])));
      }
    }
  }
#endif
  return;
}

void FaceCenteredBoundaryVariable::ClearBoundary(BoundaryCommSubset phase) {
  // Clear non-polar boundary communications
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  for (int n = 0; n < pmb->pbval->nneighbor; n++) {
    NeighborBlock &nb = pmb->pbval->neighbor[n];
    bd_var_.flag[nb.bufid] = BoundaryStatus::waiting;
    bd_var_.sflag[nb.bufid] = BoundaryStatus::waiting;
    if (((nb.ni.type == NeighborConnect::face) ||
         (nb.ni.type == NeighborConnect::edge)) &&
        phase == BoundaryCommSubset::all) {
      bd_var_flcor_.flag[nb.bufid] = BoundaryStatus::waiting;
      bd_var_flcor_.sflag[nb.bufid] = BoundaryStatus::waiting;
    }
#ifdef MPI_PARALLEL
    int mylevel = pmb->loc.level;
    if (nb.snb.rank != Globals::my_rank) {
      pmb->exec_space.fence();
      // Wait for Isend
      PARTHENON_MPI_CHECK(MPI_Wait(&(bd_var_.req_send[nb.bufid]), MPI_STATUS_IGNORE));

      if (phase == BoundaryCommSubset::all) {
        if (nb.ni.type == NeighborConnect::face || nb.ni.type == NeighborConnect::edge) {
          if (nb.snb.level < mylevel)
            PARTHENON_MPI_CHECK(
                MPI_Wait(&(bd_var_flcor_.req_send[nb.bufid]), MPI_STATUS_IGNORE));
          else if ((nb.snb.level == mylevel) &&
                   ((nb.ni.type == NeighborConnect::face) ||
                    ((nb.ni.type == NeighborConnect::edge) && (edge_flag_[nb.eid]))))
            PARTHENON_MPI_CHECK(
                MPI_Wait(&(bd_var_flcor_.req_send[nb.bufid]), MPI_STATUS_IGNORE));
        }
      }
    }
#endif
  }
}

} // namespace parthenon
