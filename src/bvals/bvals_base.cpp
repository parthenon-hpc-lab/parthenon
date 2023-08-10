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
//! \file bvals_base.cpp
//  \brief utility functions for BoundaryBase neighbors and buffers

#include "bvals/bvals.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring> // memcpy()
#include <iomanip>
#include <iostream>  // endl
#include <sstream>   // stringstream
#include <stdexcept> // runtime_error
#include <string>    // c_str()

#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "utils/buffer_utils.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

// required definitions of static data members of BoundaryBase outside class definition
// (zero-initialization is performed for all static storage duration variables)
// scalar types: integral constant 0 is explicitly converted to type
bool BoundaryBase::called_;
int BoundaryBase::maxneighbor_;
// array types: each element is zero-initialized
int BoundaryBase::bufid[NMAX_NEIGHBORS];
// struct type: zero-initializes each non-static data member (this case: all scalar types)
NeighborIndexes BoundaryBase::ni[NMAX_NEIGHBORS];

//----------------------------------------------------------------------------------------
// \!fn void NeighborBlock::SetNeighbor(int irank, int ilevel, int igid, int ilid,
//                          int iox1, int iox2, int iox3, NeighborConnect itype,
//                          int ibid, int itargetid, int ifi1=0, int ifi2=0)
// \brief Set neighbor information

void NeighborBlock::SetNeighbor(LogicalLocation inloc, int irank, int ilevel, int igid,
                                int ilid, int iox1, int iox2, int iox3,
                                NeighborConnect itype, int ibid, int itargetid,
                                int ifi1, // =0
                                int ifi2  // =0
) {
  snb.rank = irank;
  snb.level = ilevel;
  snb.gid = igid;
  snb.lid = ilid;
  ni.ox1 = iox1;
  ni.ox2 = iox2;
  ni.ox3 = iox3;
  ni.type = itype;
  ni.fi1 = ifi1;
  ni.fi2 = ifi2;
  bufid = ibid;
  targetid = itargetid;
  loc = inloc;
  if (ni.type == NeighborConnect::face) {
    if (ni.ox1 == -1)
      fid = BoundaryFace::inner_x1;
    else if (ni.ox1 == 1)
      fid = BoundaryFace::outer_x1;
    else if (ni.ox2 == -1)
      fid = BoundaryFace::inner_x2;
    else if (ni.ox2 == 1)
      fid = BoundaryFace::outer_x2;
    else if (ni.ox3 == -1)
      fid = BoundaryFace::inner_x3;
    else if (ni.ox3 == 1)
      fid = BoundaryFace::outer_x3;
  }
  if (ni.type == NeighborConnect::edge) {
    if (ni.ox3 == 0)
      eid = ((((ni.ox1 + 1) >> 1) | ((ni.ox2 + 1) & 2)));
    else if (ni.ox2 == 0)
      eid = (4 + (((ni.ox1 + 1) >> 1) | ((ni.ox3 + 1) & 2)));
    else if (ni.ox1 == 0)
      eid = (8 + (((ni.ox2 + 1) >> 1) | ((ni.ox3 + 1) & 2)));
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn BoundaryBase::BoundaryBase(Mesh *pm, LogicalLocation iloc, RegionSize isize,
//                                 BoundaryFlag *input_bcs)
//  \brief constructor of BoundaryBase
BoundaryBase::BoundaryBase(Mesh *pm, LogicalLocation iloc, RegionSize isize,
                           BoundaryFlag *input_bcs) {
  loc = iloc;
  block_size_ = isize;
  pmy_mesh_ = pm;
  if (!called_) {
    maxneighbor_ = BufferID(pmy_mesh_->ndim, pmy_mesh_->multilevel);
    called_ = true;
  }
  // copy/set in class the input 6x BoundaryFlag for this local MeshBlock boundaries
  for (int i = 0; i < 6; i++)
    block_bcs[i] = input_bcs[i];

  if (pmy_mesh_->multilevel) { // SMR or AMR
    // allocate surface area array
    int nc1 = block_size_.nx1() + 2 * Globals::nghost;
    sarea_[0] = ParArrayND<Real>(PARARRAY_TEMP, nc1);
    sarea_[1] = ParArrayND<Real>(PARARRAY_TEMP, nc1);
  }
}

//----------------------------------------------------------------------------------------
//! \fn unsigned int BoundaryBase::CreateBufferID(int ox1, int ox2, int ox3,
//                                                       int fi1, int fi2)
//  \brief calculate a buffer identifier

int BoundaryBase::CreateBufferID(int ox1, int ox2, int ox3, int fi1, int fi2) {
  // WARN: highly unsafe conversion if signed (oxi+1) are negative (they shouldn't be)
  // See comments on BoundaryBase::CreateBvalsMPITag()
  int ux1 = (ox1 + 1);
  int ux2 = (ox2 + 1);
  int ux3 = (ox3 + 1);

  return (ux1 << 6) | (ux2 << 4) | (ux3 << 2) | (fi1 << 1) | fi2;
}

//----------------------------------------------------------------------------------------
//! \fn int BoundaryBase::BufferID(int dim, bool multilevel)
//  \brief calculate neighbor indexes and target buffer IDs

int BoundaryBase::BufferID(int dim, bool multilevel) {
  int nf1 = 1, nf2 = 1;
  if (multilevel) {
    if (dim >= 2) nf1 = 2;
    if (dim >= 3) nf2 = 2;
  }
  int b = 0;
  // x1 face
  for (int n = -1; n <= 1; n += 2) {
    for (int f2 = 0; f2 < nf2; f2++) {
      for (int f1 = 0; f1 < nf1; f1++) {
        ni[b].ox1 = n;
        ni[b].ox2 = 0;
        ni[b].ox3 = 0;
        ni[b].fi1 = f1;
        ni[b].fi2 = f2;
        ni[b].type = NeighborConnect::face;
        b++;
      }
    }
  }
  // x2 face
  if (dim >= 2) {
    for (int n = -1; n <= 1; n += 2) {
      for (int f2 = 0; f2 < nf2; f2++) {
        for (int f1 = 0; f1 < nf1; f1++) {
          ni[b].ox1 = 0;
          ni[b].ox2 = n;
          ni[b].ox3 = 0;
          ni[b].fi1 = f1;
          ni[b].fi2 = f2;
          ni[b].type = NeighborConnect::face;
          b++;
        }
      }
    }
  }
  if (dim == 3) {
    // x3 face
    for (int n = -1; n <= 1; n += 2) {
      for (int f2 = 0; f2 < nf2; f2++) {
        for (int f1 = 0; f1 < nf1; f1++) {
          ni[b].ox1 = 0;
          ni[b].ox2 = 0;
          ni[b].ox3 = n;
          ni[b].fi1 = f1;
          ni[b].fi2 = f2;
          ni[b].type = NeighborConnect::face;
          b++;
        }
      }
    }
  }
  // edges
  // x1x2
  if (dim >= 2) {
    for (int m = -1; m <= 1; m += 2) {
      for (int n = -1; n <= 1; n += 2) {
        for (int f1 = 0; f1 < nf2; f1++) {
          ni[b].ox1 = n;
          ni[b].ox2 = m;
          ni[b].ox3 = 0;
          ni[b].fi1 = f1;
          ni[b].fi2 = 0;
          ni[b].type = NeighborConnect::edge;
          b++;
        }
      }
    }
  }
  if (dim == 3) {
    // x1x3
    for (int m = -1; m <= 1; m += 2) {
      for (int n = -1; n <= 1; n += 2) {
        for (int f1 = 0; f1 < nf1; f1++) {
          ni[b].ox1 = n;
          ni[b].ox2 = 0;
          ni[b].ox3 = m;
          ni[b].fi1 = f1;
          ni[b].fi2 = 0;
          ni[b].type = NeighborConnect::edge;
          b++;
        }
      }
    }
    // x2x3
    for (int m = -1; m <= 1; m += 2) {
      for (int n = -1; n <= 1; n += 2) {
        for (int f1 = 0; f1 < nf1; f1++) {
          ni[b].ox1 = 0;
          ni[b].ox2 = n;
          ni[b].ox3 = m;
          ni[b].fi1 = f1;
          ni[b].fi2 = 0;
          ni[b].type = NeighborConnect::edge;
          b++;
        }
      }
    }
    // corners
    for (int l = -1; l <= 1; l += 2) {
      for (int m = -1; m <= 1; m += 2) {
        for (int n = -1; n <= 1; n += 2) {
          ni[b].ox1 = n;
          ni[b].ox2 = m;
          ni[b].ox3 = l;
          ni[b].fi1 = 0;
          ni[b].fi2 = 0;
          ni[b].type = NeighborConnect::corner;
          b++;
        }
      }
    }
  }

  for (int n = 0; n < b; n++)
    bufid[n] = CreateBufferID(ni[n].ox1, ni[n].ox2, ni[n].ox3, ni[n].fi1, ni[n].fi2);

  return b;
}

//----------------------------------------------------------------------------------------
//! \fn int BoundaryBase::FindBufferID(int ox1, int ox2, int ox3, int fi1, int fi2)
//  \brief find the boundary buffer ID from the direction

int BoundaryBase::FindBufferID(int ox1, int ox2, int ox3, int fi1, int fi2) {
  int bid = CreateBufferID(ox1, ox2, ox3, fi1, fi2);

  for (int i = 0; i < maxneighbor_; i++) {
    if (bid == bufid[i]) return i;
  }
  return -1;
}

//----------------------------------------------------------------------------------------
//! \fn int BoundaryBase::CreateBvalsMPITag(int lid, int bufid)
//  \brief calculate an MPI tag for Bval communications
//  MPI tag = local id of destination (remaining bits) + bufid(6 bits)

// WARN: The below procedure of generating unsigned integer bitfields from signed integer
// types and converting output to signed integer tags (required by MPI) is tricky and may
// lead to unsafe conversions (and overflows from built-in types and MPI_TAG_UB).  Note,
// the MPI standard requires signed int tag, with MPI_TAG_UB>= 2^15-1 = 32,767 (inclusive)

int BoundaryBase::CreateBvalsMPITag(int lid, int bufid) {
  PARTHENON_REQUIRE_THROWS(lid >= 0 && bufid >= 0, "Ids are expected to be positive.")
  int tag = (lid << 6) | bufid;
  // Technically we could query MPI_TAG_UB here. However, using a fixed (lower) bound
  // guaranteed by the standard for now to reduce the additional functions calls or
  // adding a new internal upper lim variable.
  PARTHENON_REQUIRE_THROWS(tag >= 0 && tag < 32768, "MPI tag out of bounds.")
  return tag;
}

//----------------------------------------------------------------------------------------
// \!fn void BoundaryBase::SearchAndSetNeighbors(MeshBlockTree &tree,
//                                               int *ranklist, int *nslist)
// \brief Search and set all the neighbor blocks

// TODO(felker): break-up this long function

void BoundaryBase::SearchAndSetNeighbors(MeshBlockTree &tree, int *ranklist,
                                         int *nslist) {
  Kokkos::Profiling::pushRegion("SearchAndSetNeighbors");
  MeshBlockTree *neibt;
  int myox1, myox2 = 0, myox3 = 0, myfx1, myfx2, myfx3;
  myfx1 = ((loc.lx1() & 1LL) == 1LL);
  myfx2 = ((loc.lx2() & 1LL) == 1LL);
  myfx3 = ((loc.lx3() & 1LL) == 1LL);
  myox1 = ((loc.lx1() & 1LL) == 1LL) * 2 - 1;
  if (block_size_.nx2() > 1) myox2 = ((loc.lx2() & 1LL) == 1LL) * 2 - 1;
  if (block_size_.nx3() > 1) myox3 = ((loc.lx3() & 1LL) == 1LL) * 2 - 1;

  int nf1 = 1, nf2 = 1;
  if (pmy_mesh_->multilevel) {
    if (block_size_.nx2() > 1) nf1 = 2;
    if (block_size_.nx3() > 1) nf2 = 2;
  }
  int bufid = 0;
  nneighbor = 0;
  for (int k = 0; k <= 2; k++) {
    for (int j = 0; j <= 2; j++) {
      for (int i = 0; i <= 2; i++)
        nblevel[k][j][i] = -1;
    }
  }
  nblevel[1][1][1] = loc.level();

  // x1 face
  for (int n = -1; n <= 1; n += 2) {
    neibt = tree.FindNeighbor(loc, n, 0, 0);
    if (neibt == nullptr) {
      bufid += nf1 * nf2;
      continue;
    }
    if (neibt->pleaf_ != nullptr) { // neighbor at finer level
      int fface = 1 - (n + 1) / 2;  // 0 for BoundaryFace::outer_x1, 1 for inner_x1
      nblevel[1][1][n + 1] = neibt->loc_.level() + 1;
      for (int f2 = 0; f2 < nf2; f2++) {
        for (int f1 = 0; f1 < nf1; f1++) {
          MeshBlockTree *nf = neibt->GetLeaf(fface, f1, f2);
          int fid = nf->gid_;
          int nlevel = nf->loc_.level();
          int tbid = FindBufferID(-n, 0, 0, 0, 0);
          neighbor[nneighbor].SetNeighbor(nf->loc_, ranklist[fid], nlevel, fid,
                                          fid - nslist[ranklist[fid]], n, 0, 0,
                                          NeighborConnect::face, bufid, tbid, f1, f2);
          bufid++;
          nneighbor++;
        }
      }
    } else { // neighbor at same or coarser level
      int nlevel = neibt->loc_.level();
      int nid = neibt->gid_;
      nblevel[1][1][n + 1] = nlevel;
      int tbid;
      if (nlevel == loc.level()) { // neighbor at same level
        tbid = FindBufferID(-n, 0, 0, 0, 0);
      } else { // neighbor at coarser level
        tbid = FindBufferID(-n, 0, 0, myfx2, myfx3);
      }
      neighbor[nneighbor].SetNeighbor(neibt->loc_, ranklist[nid], nlevel, nid,
                                      nid - nslist[ranklist[nid]], n, 0, 0,
                                      NeighborConnect::face, bufid, tbid);
      bufid += nf1 * nf2;
      nneighbor++;
    }
  }
  if (block_size_.nx2() == 1) {
    SetNeighborOwnership();
    Kokkos::Profiling::popRegion(); // SearchAndSetNeighbors
    return;
  }

  // x2 face
  for (int n = -1; n <= 1; n += 2) {
    neibt = tree.FindNeighbor(loc, 0, n, 0);
    if (neibt == nullptr) {
      bufid += nf1 * nf2;
      continue;
    }
    if (neibt->pleaf_ != nullptr) { // neighbor at finer level
      int fface = 1 - (n + 1) / 2;  // 0 for BoundaryFace::outer_x2, 1 for inner_x2
      nblevel[1][n + 1][1] = neibt->loc_.level() + 1;
      for (int f2 = 0; f2 < nf2; f2++) {
        for (int f1 = 0; f1 < nf1; f1++) {
          MeshBlockTree *nf = neibt->GetLeaf(f1, fface, f2);
          int fid = nf->gid_;
          int nlevel = nf->loc_.level();
          int tbid = FindBufferID(0, -n, 0, 0, 0);
          neighbor[nneighbor].SetNeighbor(nf->loc_, ranklist[fid], nlevel, fid,
                                          fid - nslist[ranklist[fid]], 0, n, 0,
                                          NeighborConnect::face, bufid, tbid, f1, f2);
          bufid++;
          nneighbor++;
        }
      }
    } else { // neighbor at same or coarser level
      int nlevel = neibt->loc_.level();
      int nid = neibt->gid_;
      nblevel[1][n + 1][1] = nlevel;
      int tbid;
      if (nlevel == loc.level()) { // neighbor at same level
        tbid = FindBufferID(0, -n, 0, 0, 0);
      } else { // neighbor at coarser level
        tbid = FindBufferID(0, -n, 0, myfx1, myfx3);
      }
      neighbor[nneighbor].SetNeighbor(neibt->loc_, ranklist[nid], nlevel, nid,
                                      nid - nslist[ranklist[nid]], 0, n, 0,
                                      NeighborConnect::face, bufid, tbid);
      bufid += nf1 * nf2;
      nneighbor++;
    }
  }

  // x3 face
  if (block_size_.nx3() > 1) {
    for (int n = -1; n <= 1; n += 2) {
      neibt = tree.FindNeighbor(loc, 0, 0, n);
      if (neibt == nullptr) {
        bufid += nf1 * nf2;
        continue;
      }
      if (neibt->pleaf_ != nullptr) { // neighbor at finer level
        int fface = 1 - (n + 1) / 2;  // 0 for BoundaryFace::outer_x3, 1 for inner_x3
        nblevel[n + 1][1][1] = neibt->loc_.level() + 1;
        for (int f2 = 0; f2 < nf2; f2++) {
          for (int f1 = 0; f1 < nf1; f1++) {
            MeshBlockTree *nf = neibt->GetLeaf(f1, f2, fface);
            int fid = nf->gid_;
            int nlevel = nf->loc_.level();
            int tbid = FindBufferID(0, 0, -n, 0, 0);
            neighbor[nneighbor].SetNeighbor(nf->loc_, ranklist[fid], nlevel, fid,
                                            fid - nslist[ranklist[fid]], 0, 0, n,
                                            NeighborConnect::face, bufid, tbid, f1, f2);
            bufid++;
            nneighbor++;
          }
        }
      } else { // neighbor at same or coarser level
        int nlevel = neibt->loc_.level();
        int nid = neibt->gid_;
        nblevel[n + 1][1][1] = nlevel;
        int tbid;
        if (nlevel == loc.level()) { // neighbor at same level
          tbid = FindBufferID(0, 0, -n, 0, 0);
        } else { // neighbor at coarser level
          tbid = FindBufferID(0, 0, -n, myfx1, myfx2);
        }
        neighbor[nneighbor].SetNeighbor(neibt->loc_, ranklist[nid], nlevel, nid,
                                        nid - nslist[ranklist[nid]], 0, 0, n,
                                        NeighborConnect::face, bufid, tbid);
        bufid += nf1 * nf2;
        nneighbor++;
      }
    }
  }

  // x1x2 edge
  for (int m = -1; m <= 1; m += 2) {
    for (int n = -1; n <= 1; n += 2) {
      neibt = tree.FindNeighbor(loc, n, m, 0);
      if (neibt == nullptr) {
        bufid += nf2;
        continue;
      }
      if (neibt->pleaf_ != nullptr) { // neighbor at finer level
        int ff1 = 1 - (n + 1) / 2;    // 0 for BoundaryFace::outer_x1, 1 for inner_x1
        int ff2 = 1 - (m + 1) / 2;    // 0 for BoundaryFace::outer_x2, 1 for inner_x2

        nblevel[1][m + 1][n + 1] = neibt->loc_.level() + 1;
        for (int f1 = 0; f1 < nf2; f1++) {
          MeshBlockTree *nf = neibt->GetLeaf(ff1, ff2, f1);
          int fid = nf->gid_;
          int nlevel = nf->loc_.level();
          int tbid = FindBufferID(-n, -m, 0, 0, 0);
          neighbor[nneighbor].SetNeighbor(nf->loc_, ranklist[fid], nlevel, fid,
                                          fid - nslist[ranklist[fid]], n, m, 0,
                                          NeighborConnect::edge, bufid, tbid, f1, 0);
          bufid++;
          nneighbor++;
        }
      } else { // neighbor at same or coarser level
        int nlevel = neibt->loc_.level();
        int nid = neibt->gid_;
        nblevel[1][m + 1][n + 1] = nlevel;
        int tbid;
        if (nlevel == loc.level()) { // neighbor at same level
          tbid = FindBufferID(-n, -m, 0, 0, 0);
        } else { // neighbor at coarser level
          tbid = FindBufferID(-n, -m, 0, myfx3, 0);
        }
        if (nlevel >= loc.level() || (myox1 == n && myox2 == m)) {
          neighbor[nneighbor].SetNeighbor(neibt->loc_, ranklist[nid], nlevel, nid,
                                          nid - nslist[ranklist[nid]], n, m, 0,
                                          NeighborConnect::edge, bufid, tbid);
          nneighbor++;
        }
        bufid += nf2;
      }
    }
  }

  if (block_size_.nx3() == 1) {
    SetNeighborOwnership();
    Kokkos::Profiling::popRegion(); // SearchAndSetNeighbors
    return;
  }

  // x1x3 edge
  for (int m = -1; m <= 1; m += 2) {
    for (int n = -1; n <= 1; n += 2) {
      neibt = tree.FindNeighbor(loc, n, 0, m);
      if (neibt == nullptr) {
        bufid += nf1;
        continue;
      }
      if (neibt->pleaf_ != nullptr) { // neighbor at finer level
        int ff1 = 1 - (n + 1) / 2;    // 0 for BoundaryFace::outer_x1, 1 for inner_x1
        int ff2 = 1 - (m + 1) / 2;    // 0 for BoundaryFace::outer_x3, 1 for inner_x3
        nblevel[m + 1][1][n + 1] = neibt->loc_.level() + 1;
        for (int f1 = 0; f1 < nf1; f1++) {
          MeshBlockTree *nf = neibt->GetLeaf(ff1, f1, ff2);
          int fid = nf->gid_;
          int nlevel = nf->loc_.level();
          int tbid = FindBufferID(-n, 0, -m, 0, 0);
          neighbor[nneighbor].SetNeighbor(nf->loc_, ranklist[fid], nlevel, fid,
                                          fid - nslist[ranklist[fid]], n, 0, m,
                                          NeighborConnect::edge, bufid, tbid, f1, 0);
          bufid++;
          nneighbor++;
        }
      } else { // neighbor at same or coarser level
        int nlevel = neibt->loc_.level();
        int nid = neibt->gid_;
        nblevel[m + 1][1][n + 1] = nlevel;
        int tbid;
        if (nlevel == loc.level()) { // neighbor at same level
          tbid = FindBufferID(-n, 0, -m, 0, 0);
        } else { // neighbor at coarser level
          tbid = FindBufferID(-n, 0, -m, myfx2, 0);
        }
        if (nlevel >= loc.level() || (myox1 == n && myox3 == m)) {
          neighbor[nneighbor].SetNeighbor(neibt->loc_, ranklist[nid], nlevel, nid,
                                          nid - nslist[ranklist[nid]], n, 0, m,
                                          NeighborConnect::edge, bufid, tbid);
          nneighbor++;
        }
        bufid += nf1;
      }
    }
  }

  // x2x3 edge
  for (int m = -1; m <= 1; m += 2) {
    for (int n = -1; n <= 1; n += 2) {
      neibt = tree.FindNeighbor(loc, 0, n, m);
      if (neibt == nullptr) {
        bufid += nf1;
        continue;
      }
      if (neibt->pleaf_ != nullptr) { // neighbor at finer level
        int ff1 = 1 - (n + 1) / 2;    // 0 for BoundaryFace::outer_x2, 1 for inner_x2
        int ff2 = 1 - (m + 1) / 2;    // 0 for BoundaryFace::outer_x3, 1 for inner_x3
        nblevel[m + 1][n + 1][1] = neibt->loc_.level() + 1;
        for (int f1 = 0; f1 < nf1; f1++) {
          MeshBlockTree *nf = neibt->GetLeaf(f1, ff1, ff2);
          int fid = nf->gid_;
          int nlevel = nf->loc_.level();
          int tbid = FindBufferID(0, -n, -m, 0, 0);
          neighbor[nneighbor].SetNeighbor(nf->loc_, ranklist[fid], nlevel, fid,
                                          fid - nslist[ranklist[fid]], 0, n, m,
                                          NeighborConnect::edge, bufid, tbid, f1, 0);
          bufid++;
          nneighbor++;
        }
      } else { // neighbor at same or coarser level
        int nlevel = neibt->loc_.level();
        int nid = neibt->gid_;
        nblevel[m + 1][n + 1][1] = nlevel;
        int tbid;
        if (nlevel == loc.level()) { // neighbor at same level
          tbid = FindBufferID(0, -n, -m, 0, 0);
        } else { // neighbor at coarser level
          tbid = FindBufferID(0, -n, -m, myfx1, 0);
        }
        if (nlevel >= loc.level() || (myox2 == n && myox3 == m)) {
          neighbor[nneighbor].SetNeighbor(neibt->loc_, ranklist[nid], nlevel, nid,
                                          nid - nslist[ranklist[nid]], 0, n, m,
                                          NeighborConnect::edge, bufid, tbid);
          nneighbor++;
        }
        bufid += nf1;
      }
    }
  }

  // corners
  for (int l = -1; l <= 1; l += 2) {
    for (int m = -1; m <= 1; m += 2) {
      for (int n = -1; n <= 1; n += 2) {
        neibt = tree.FindNeighbor(loc, n, m, l);
        if (neibt == nullptr) {
          bufid++;
          continue;
        }
        if (neibt->pleaf_ != nullptr) { // neighbor at finer level
          int ff1 = 1 - (n + 1) / 2;    // 0 for BoundaryFace::outer_x1, 1 for inner_x1
          int ff2 = 1 - (m + 1) / 2;    // 0 for BoundaryFace::outer_x2, 1 for inner_x2
          int ff3 = 1 - (l + 1) / 2;    // 0 for BoundaryFace::outer_x3, 1 for inner_x3
          neibt = neibt->GetLeaf(ff1, ff2, ff3);
        }
        int nlevel = neibt->loc_.level();
        nblevel[l + 1][m + 1][n + 1] = nlevel;
        if (nlevel >= loc.level() || (myox1 == n && myox2 == m && myox3 == l)) {
          int nid = neibt->gid_;
          int tbid = FindBufferID(-n, -m, -l, 0, 0);
          neighbor[nneighbor].SetNeighbor(neibt->loc_, ranklist[nid], nlevel, nid,
                                          nid - nslist[ranklist[nid]], n, m, l,
                                          NeighborConnect::corner, bufid, tbid);
          nneighbor++;
        }
        bufid++;
      }
    }
  }

  SetNeighborOwnership();
  Kokkos::Profiling::popRegion(); // SearchAndSetNeighbors
}

void BoundaryBase::SetNeighborOwnership() {
  // Set neighbor block ownership
  std::set<LogicalLocation> allowed_neighbors;
  allowed_neighbors.insert(loc); // Insert the location of this block
  for (int n = 0; n < nneighbor; ++n)
    allowed_neighbors.insert(neighbor[n].loc);
  // Although the neighbor blocks abut more blocks than are contained in this
  // list, the unaccounted for blocks cannot impact the ownership of elements
  // that are shared with *this
  RootGridInfo rg_info = pmy_mesh_->GetRootGridInfo();
  for (int n = 0; n < nneighbor; ++n) {
    neighbor[n].ownership =
        DetermineOwnership(neighbor[n].loc, allowed_neighbors, rg_info);
    neighbor[n].ownership.initialized = true;
  }
}
} // namespace parthenon
