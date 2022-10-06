//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
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
//! \file mesh_refinement.cpp
//  \brief implements functions for static/adaptive mesh refinement

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>

#include "coordinates/coordinates.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "interface/variable.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/refinement_cc_in_one.hpp"
#include "parameter_input.hpp"
#include "parthenon_arrays.hpp"
#include "refinement/refinement.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

//----------------------------------------------------------------------------------------
//! \fn MeshRefinement::MeshRefinement(MeshBlock *pmb, ParameterInput *pin)
//  \brief constructor

MeshRefinement::MeshRefinement(std::weak_ptr<MeshBlock> pmb, ParameterInput *pin)
    : pmy_block_(pmb), deref_count_(0),
      deref_threshold_(pin->GetOrAddInteger("parthenon/mesh", "derefine_count", 10)) {
  // Create coarse mesh object for parent grid
  coarse_coords = Coordinates_t(pmb.lock()->coords, 2);

  if ((Globals::nghost % 2) != 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in MeshRefinement constructor" << std::endl
        << "Selected --nghost=" << Globals::nghost
        << " is incompatible with mesh refinement because it is not a multiple of 2.\n"
        << "Rerun with an even number of ghost cells " << std::endl;
    PARTHENON_FAIL(msg);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::RestrictCellCenteredValues(const ParArrayND<Real> &fine,
//                           ParArrayND<Real> &coarse, int sn, int en,
//                           int csi, int cei, int csj, int cej, int csk, int cek)
//  \brief restrict cell centered values

void MeshRefinement::RestrictCellCenteredValues(const ParArrayND<Real> &fine,
                                                ParArrayND<Real> &coarse, int sn, int en,
                                                int csi, int cei, int csj, int cej,
                                                int csk, int cek) {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();

  int b = 0;
  int nbuffers = 1;
  cell_centered_bvars::BufferCacheHost_t info_h("refinement info", nbuffers);
  // buff and var unused.
  info_h(b).si = csi;
  info_h(b).ei = cei;
  info_h(b).sj = csj;
  info_h(b).ej = cej;
  info_h(b).sk = csk;
  info_h(b).ek = cek;
  info_h(b).Nt = fine.GetDim(6);
  info_h(b).Nu = fine.GetDim(5);
  info_h(b).Nv = fine.GetDim(4);
  info_h(b).refinement_op = RefinementOp_t::Restriction;
  info_h(b).coords = pmb->coords;
  info_h(b).coarse_coords = this->coarse_coords;
  info_h(b).fine = fine.Get();
  info_h(b).coarse = coarse.Get();
  cell_centered_refinement::Restrict(info_h, pmb->cellbounds, pmb->c_cellbounds);
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::RestrictFieldX1(const ParArrayND<Real> &fine
//      ParArrayND<Real> &coarse, int csi, int cei, int csj, int cej, int csk, int cek)
//  \brief restrict the x1 field data and set them into the coarse buffer

void MeshRefinement::RestrictFieldX1(const ParArrayND<Real> &fine,
                                     ParArrayND<Real> &coarse, int csi, int cei, int csj,
                                     int cej, int csk, int cek) {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  auto &coords = pmb->coords;
  const IndexDomain interior = IndexDomain::interior;
  // int si = (csi - pmb->c_cellbounds.is(interior)) * 2 + pmb->cellbounds.is(interior);
  // int ei = (cei - pmb->c_cellbounds.is(interior)) * 2 + pmb->cellbounds.is(interior);

  // store the restricted data in the prolongation buffer for later use
  if (pmb->block_size.nx3 > 1) { // 3D
    for (int ck = csk; ck <= cek; ck++) {
      int k = (ck - pmb->c_cellbounds.ks(interior)) * 2 + pmb->cellbounds.ks(interior);
      for (int cj = csj; cj <= cej; cj++) {
        int j = (cj - pmb->c_cellbounds.js(interior)) * 2 + pmb->cellbounds.js(interior);
        for (int ci = csi; ci <= cei; ci++) {
          int i =
              (ci - pmb->c_cellbounds.is(interior)) * 2 + pmb->cellbounds.is(interior);
          const Real area00 = coords.Area(X1DIR, k, j, i);
          const Real area01 = coords.Area(X1DIR, k, j + 1, i);
          const Real area10 = coords.Area(X1DIR, k + 1, j, i);
          const Real area11 = coords.Area(X1DIR, k + 1, j + 1, i);
          const Real tarea = area00 + area01 + area10 + area11;
          coarse(ck, cj, ci) =
              (fine(k, j, i) * area00 + fine(k, j + 1, i) * area01 +
               fine(k + 1, j, i) * area10 + fine(k + 1, j + 1, i) * area11) /
              tarea;
        }
      }
    }
  } else if (pmb->block_size.nx2 > 1) { // 2D
    int k = pmb->cellbounds.ks(interior);
    for (int cj = csj; cj <= cej; cj++) {
      int j = (cj - pmb->c_cellbounds.js(interior)) * 2 + pmb->cellbounds.js(interior);
      for (int ci = csi; ci <= cei; ci++) {
        int i = (ci - pmb->c_cellbounds.is(interior)) * 2 + pmb->cellbounds.is(interior);
        const Real area0 = coords.Area(X1DIR, k, j, i);
        const Real area1 = coords.Area(X1DIR, k, j + 1, i);
        const Real tarea = area0 + area1;
        coarse(csk, cj, ci) = (fine(k, j, i) * area0 + fine(k, j + 1, i) * area1) / tarea;
      }
    }

  } else { // 1D - no restriction, just copy
    for (int ci = csi; ci <= cei; ci++) {
      int i = (ci - pmb->c_cellbounds.is(interior)) * 2 + pmb->cellbounds.is(interior);
      coarse(csk, csj, ci) =
          fine(pmb->cellbounds.ks(interior), pmb->cellbounds.js(interior), i);
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::RestrictFieldX2(const ParArrayND<Real> &fine
//      ParArrayND<Real> &coarse, int csi, int cei, int csj, int cej, int csk, int cek)
//  \brief restrict the x2 field data and set them into the coarse buffer

void MeshRefinement::RestrictFieldX2(const ParArrayND<Real> &fine,
                                     ParArrayND<Real> &coarse, int csi, int cei, int csj,
                                     int cej, int csk, int cek) {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  auto &coords = pmb->coords;
  const IndexDomain interior = IndexDomain::interior;
  int si = (csi - pmb->c_cellbounds.is(interior)) * 2 + pmb->cellbounds.is(interior);
  int ei = (cei - pmb->c_cellbounds.is(interior)) * 2 + pmb->cellbounds.is(interior) + 1;

  // store the restricted data in the prolongation buffer for later use
  if (pmb->block_size.nx3 > 1) { // 3D
    for (int ck = csk; ck <= cek; ck++) {
      int k = (ck - pmb->c_cellbounds.ks(interior)) * 2 + pmb->cellbounds.ks(interior);
      for (int cj = csj; cj <= cej; cj++) {
        int j = (cj - pmb->c_cellbounds.js(interior)) * 2 + pmb->cellbounds.js(interior);
        for (int ci = csi; ci <= cei; ci++) {
          int i =
              (ci - pmb->c_cellbounds.is(interior)) * 2 + pmb->cellbounds.is(interior);
          const Real area00 = coords.Area(X2DIR, k, j, i);
          const Real area01 = coords.Area(X2DIR, k, j, i + 1);
          const Real area10 = coords.Area(X2DIR, k + 1, j, i);
          const Real area11 = coords.Area(X2DIR, k + 1, j, i + 1);
          const Real tarea = area00 + area01 + area10 + area11;
          coarse(ck, cj, ci) =
              (fine(k, j, i) * area00 + fine(k, j, i + 1) * area01 +
               fine(k + 1, j, i) * area10 + fine(k + 1, j, i + 1) * area11) /
              tarea;
        }
      }
    }
  } else if (pmb->block_size.nx2 > 1) { // 2D
    int k = pmb->cellbounds.ks(interior);
    for (int cj = csj; cj <= cej; cj++) {
      int j = (cj - pmb->c_cellbounds.js(interior)) * 2 + pmb->cellbounds.js(interior);
      for (int ci = csi; ci <= cei; ci++) {
        int i = (ci - pmb->c_cellbounds.is(interior)) * 2 + pmb->cellbounds.is(interior);
        const Real area0 = coords.Area(X2DIR, k, j, i);
        const Real area1 = coords.Area(X2DIR, k, j, i + 1);
        const Real tarea = area0 + area1;
        coarse(pmb->c_cellbounds.ks(interior), cj, ci) =
            (fine(k, j, i) * area0 + fine(k, j, i + 1) * area1) / tarea;
      }
    }
  } else { // 1D
    int k = pmb->cellbounds.ks(interior), j = pmb->cellbounds.js(interior);
    for (int ci = csi; ci <= cei; ci++) {
      int i = (ci - pmb->c_cellbounds.is(interior)) * 2 + pmb->cellbounds.is(interior);
      const Real area0 = coords.Area(X2DIR, k, j, i);
      const Real area1 = coords.Area(X2DIR, k, j, i + 1);
      const Real tarea = area0 + area1;
      coarse(pmb->c_cellbounds.ks(interior), pmb->c_cellbounds.js(interior), ci) =
          (fine(k, j, i) * area0 + fine(k, j, i + 1) * area1) / tarea;
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::RestrictFieldX3(const ParArrayND<Real> &fine
//      ParArrayND<Real> &coarse, int csi, int cei, int csj, int cej, int csk, int cek)
//  \brief restrict the x3 field data and set them into the coarse buffer

void MeshRefinement::RestrictFieldX3(const ParArrayND<Real> &fine,
                                     ParArrayND<Real> &coarse, int csi, int cei, int csj,
                                     int cej, int csk, int cek) {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  auto &coords = pmb->coords;
  const IndexDomain interior = IndexDomain::interior;
  int si = (csi - pmb->c_cellbounds.is(interior)) * 2 + pmb->cellbounds.is(interior),
      ei = (cei - pmb->c_cellbounds.is(interior)) * 2 + pmb->cellbounds.is(interior) + 1;

  // store the restricted data in the prolongation buffer for later use
  if (pmb->block_size.nx3 > 1) { // 3D
    for (int ck = csk; ck <= cek; ck++) {
      int k = (ck - pmb->c_cellbounds.ks(interior)) * 2 + pmb->cellbounds.ks(interior);
      for (int cj = csj; cj <= cej; cj++) {
        int j = (cj - pmb->c_cellbounds.js(interior)) * 2 + pmb->cellbounds.js(interior);
        for (int ci = csi; ci <= cei; ci++) {
          int i =
              (ci - pmb->c_cellbounds.is(interior)) * 2 + pmb->cellbounds.is(interior);
          const Real area00 = coords.Area(X3DIR, k, j, i);
          const Real area01 = coords.Area(X3DIR, k, j, i + 1);
          const Real area10 = coords.Area(X3DIR, k, j + 1, i);
          const Real area11 = coords.Area(X3DIR, k, j + 1, i + 1);
          const Real tarea = area00 + area01 + area10 + area11;
          coarse(ck, cj, ci) =
              (fine(k, j, i) * area00 + fine(k, j, i + 1) * area01 +
               fine(k, j + 1, i) * area10 + fine(k, j + 1, i + 1) * area11) /
              tarea;
        }
      }
    }
  } else if (pmb->block_size.nx2 > 1) { // 2D
    int k = pmb->cellbounds.ks(interior);
    for (int cj = csj; cj <= cej; cj++) {
      int j = (cj - pmb->c_cellbounds.js(interior)) * 2 + pmb->cellbounds.js(interior);
      for (int ci = csi; ci <= cei; ci++) {
        int i = (ci - pmb->c_cellbounds.is(interior)) * 2 + pmb->cellbounds.is(interior);
        const Real area00 = coords.Area(X3DIR, k, j, i);
        const Real area01 = coords.Area(X3DIR, k, j, i + 1);
        const Real area10 = coords.Area(X3DIR, k, j + 1, i);
        const Real area11 = coords.Area(X3DIR, k, j + 1, i + 1);
        const Real tarea = area00 + area01 + area10 + area11;
        coarse(pmb->c_cellbounds.ks(interior), cj, ci) =
            (fine(k, j, i) * area00 + fine(k, j, i + 1) * area01 +
             fine(k, j + 1, i) * area10 + fine(k, j + 1, i + 1) * area11) /
            tarea;
      }
    }
  } else { // 1D
    int k = pmb->cellbounds.ks(interior), j = pmb->cellbounds.js(interior);
    for (int ci = csi; ci <= cei; ci++) {
      int i = (ci - pmb->c_cellbounds.is(interior)) * 2 + pmb->cellbounds.is(interior);
      const Real area0 = coords.Area(X3DIR, k, j, i);
      const Real area1 = coords.Area(X3DIR, k, j, i + 1);
      const Real tarea = area0 + area1;
      coarse(pmb->c_cellbounds.ks(interior), pmb->c_cellbounds.js(interior), ci) =
          (fine(k, j, i) * area0 + fine(k, j, i + 1) * area1) / tarea;
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::ProlongateCellCenteredValues(
//        const ParArrayND<Real> &coarse,ParArrayND<Real> &fine, int sn, int en,,
//        int si, int ei, int sj, int ej, int sk, int ek)
//  \brief Prolongate cell centered values

void MeshRefinement::ProlongateCellCenteredValues(const ParArrayND<Real> &coarse,
                                                  ParArrayND<Real> &fine, int sn, int en,
                                                  int si, int ei, int sj, int ej, int sk,
                                                  int ek) {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  int b = 0;
  int nbuffers = 1;
  cell_centered_bvars::BufferCacheHost_t info_h("refinement info", nbuffers);
  // buff and var unused
  info_h(b).si = si;
  info_h(b).ei = ei;
  info_h(b).sj = sj;
  info_h(b).ej = ej;
  info_h(b).sk = sk;
  info_h(b).ek = ek;
  info_h(b).Nt = coarse.GetDim(6);
  info_h(b).Nu = coarse.GetDim(5);
  info_h(b).Nv = coarse.GetDim(4);
  info_h(b).refinement_op = RefinementOp_t::Prolongation;
  info_h(b).coords = pmb->coords;
  info_h(b).coarse_coords = this->coarse_coords;
  info_h(b).fine = fine.Get();
  info_h(b).coarse = coarse.Get();
  cell_centered_refinement::Prolongate(info_h, pmb->cellbounds, pmb->c_cellbounds);
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::ProlongateSharedFieldX1(const ParArrayND<Real> &coarse,
//      ParArrayND<Real> &fine, int si, int ei, int sj, int ej, int sk, int ek)
//  \brief prolongate x1 face-centered fields shared between coarse and fine levels

void MeshRefinement::ProlongateSharedFieldX1(const ParArrayND<Real> &coarse,
                                             ParArrayND<Real> &fine, int si, int ei,
                                             int sj, int ej, int sk, int ek) {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  auto &coords = pmb->coords;
  const IndexDomain interior = IndexDomain::interior;
  if (pmb->block_size.nx3 > 1) {
    for (int k = sk; k <= ek; k++) {
      int fk = (k - pmb->c_cellbounds.ks(interior)) * 2 + pmb->cellbounds.ks(interior);
      const Real x3m = coarse_coords.x3s1(k - 1);
      const Real x3c = coarse_coords.x3s1(k);
      const Real x3p = coarse_coords.x3s1(k + 1);
      Real dx3m = x3c - x3m;
      Real dx3p = x3p - x3c;
      const Real fx3m = coords.x3s1(fk);
      const Real fx3p = coords.x3s1(fk + 1);
      for (int j = sj; j <= ej; j++) {
        int fj = (j - pmb->c_cellbounds.js(interior)) * 2 + pmb->cellbounds.js(interior);
        const Real x2m = coarse_coords.x2s1(j - 1);
        const Real x2c = coarse_coords.x2s1(j);
        const Real x2p = coarse_coords.x2s1(j + 1);
        Real dx2m = x2c - x2m;
        Real dx2p = x2p - x2c;
        const Real fx2m = coords.x2s1(fj);
        const Real fx2p = coords.x2s1(fj + 1);
        for (int i = si; i <= ei; i++) {
          int fi =
              (i - pmb->c_cellbounds.is(interior)) * 2 + pmb->cellbounds.is(interior);
          Real ccval = coarse(k, j, i);

          Real gx2m = (ccval - coarse(k, j - 1, i)) / dx2m;
          Real gx2p = (coarse(k, j + 1, i) - ccval) / dx2p;
          Real gx2c =
              0.5 * (SIGN(gx2m) + SIGN(gx2p)) * std::min(std::abs(gx2m), std::abs(gx2p));
          Real gx3m = (ccval - coarse(k - 1, j, i)) / dx3m;
          Real gx3p = (coarse(k + 1, j, i) - ccval) / dx3p;
          Real gx3c =
              0.5 * (SIGN(gx3m) + SIGN(gx3p)) * std::min(std::abs(gx3m), std::abs(gx3p));

          fine(fk, fj, fi) = ccval - gx2c * (x2c - fx2m) - gx3c * (x3c - fx3m);
          fine(fk, fj + 1, fi) = ccval + gx2c * (fx2p - x2c) - gx3c * (x3c - fx3m);
          fine(fk + 1, fj, fi) = ccval - gx2c * (x2c - fx2m) + gx3c * (fx3p - x3c);
          fine(fk + 1, fj + 1, fi) = ccval + gx2c * (fx2p - x2c) + gx3c * (fx3p - x3c);
        }
      }
    }
  } else if (pmb->block_size.nx2 > 1) {
    int k = pmb->c_cellbounds.ks(interior), fk = pmb->cellbounds.ks(interior);
    for (int j = sj; j <= ej; j++) {
      int fj = (j - pmb->c_cellbounds.js(interior)) * 2 + pmb->cellbounds.js(interior);
      const Real x2m = coarse_coords.x2s1(j - 1);
      const Real x2c = coarse_coords.x2s1(j);
      const Real x2p = coarse_coords.x2s1(j + 1);
      Real dx2m = x2c - x2m;
      Real dx2p = x2p - x2c;
      const Real fx2m = coords.x2s1(fj);
      const Real fx2p = coords.x2s1(fj + 1);
      for (int i = si; i <= ei; i++) {
        int fi = (i - pmb->c_cellbounds.is(interior)) * 2 + pmb->cellbounds.is(interior);
        Real ccval = coarse(k, j, i);

        Real gx2m = (ccval - coarse(k, j - 1, i)) / dx2m;
        Real gx2p = (coarse(k, j + 1, i) - ccval) / dx2p;
        Real gx2c =
            0.5 * (SIGN(gx2m) + SIGN(gx2p)) * std::min(std::abs(gx2m), std::abs(gx2p));

        fine(fk, fj, fi) = ccval - gx2c * (x2c - fx2m);
        fine(fk, fj + 1, fi) = ccval + gx2c * (fx2p - x2c);
      }
    }
  } else { // 1D
    for (int i = si; i <= ei; i++) {
      int fi = (i - pmb->c_cellbounds.is(interior)) * 2 + pmb->cellbounds.is(interior);
      fine(0, 0, fi) = coarse(0, 0, i);
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::ProlongateSharedFieldX2(const ParArrayND<Real> &coarse,
//      ParArrayND<Real> &fine, int si, int ei, int sj, int ej, int sk, int ek)
//  \brief prolongate x2 face-centered fields shared between coarse and fine levels

void MeshRefinement::ProlongateSharedFieldX2(const ParArrayND<Real> &coarse,
                                             ParArrayND<Real> &fine, int si, int ei,
                                             int sj, int ej, int sk, int ek) {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  const IndexDomain interior = IndexDomain::interior;
  auto &coords = pmb->coords;
  if (pmb->block_size.nx3 > 1) {
    for (int k = sk; k <= ek; k++) {
      int fk = (k - pmb->c_cellbounds.ks(interior)) * 2 + pmb->cellbounds.ks(interior);
      const Real x3m = coarse_coords.x3s2(k - 1);
      const Real x3c = coarse_coords.x3s2(k);
      const Real x3p = coarse_coords.x3s2(k + 1);
      Real dx3m = x3c - x3m;
      Real dx3p = x3p - x3c;
      const Real fx3m = coords.x3s2(fk);
      const Real fx3p = coords.x3s2(fk + 1);
      for (int j = sj; j <= ej; j++) {
        int fj = (j - pmb->c_cellbounds.js(interior)) * 2 + pmb->cellbounds.js(interior);
        for (int i = si; i <= ei; i++) {
          int fi =
              (i - pmb->c_cellbounds.is(interior)) * 2 + pmb->cellbounds.is(interior);
          const Real x1m = coarse_coords.x1s2(i - 1);
          const Real x1c = coarse_coords.x1s2(i);
          const Real x1p = coarse_coords.x1s2(i + 1);
          Real dx1m = x1c - x1m;
          Real dx1p = x1p - x1c;
          const Real fx1m = coords.x1s2(fi);
          const Real fx1p = coords.x1s2(fi + 1);
          Real ccval = coarse(k, j, i);

          Real gx1m = (ccval - coarse(k, j, i - 1)) / dx1m;
          Real gx1p = (coarse(k, j, i + 1) - ccval) / dx1p;
          Real gx1c =
              0.5 * (SIGN(gx1m) + SIGN(gx1p)) * std::min(std::abs(gx1m), std::abs(gx1p));
          Real gx3m = (ccval - coarse(k - 1, j, i)) / dx3m;
          Real gx3p = (coarse(k + 1, j, i) - ccval) / dx3p;
          Real gx3c =
              0.5 * (SIGN(gx3m) + SIGN(gx3p)) * std::min(std::abs(gx3m), std::abs(gx3p));

          fine(fk, fj, fi) = ccval - gx1c * (x1c - fx1m) - gx3c * (x3c - fx3m);
          fine(fk, fj, fi + 1) = ccval + gx1c * (fx1p - x1c) - gx3c * (x3c - fx3m);
          fine(fk + 1, fj, fi) = ccval - gx1c * (x1c - fx1m) + gx3c * (fx3p - x3c);
          fine(fk + 1, fj, fi + 1) = ccval + gx1c * (fx1p - x1c) + gx3c * (fx3p - x3c);
        }
      }
    }
  } else if (pmb->block_size.nx2 > 1) {
    int k = pmb->c_cellbounds.ks(interior), fk = pmb->cellbounds.ks(interior);
    for (int j = sj; j <= ej; j++) {
      int fj = (j - pmb->c_cellbounds.js(interior)) * 2 + pmb->cellbounds.js(interior);
      for (int i = si; i <= ei; i++) {
        int fi = (i - pmb->c_cellbounds.is(interior)) * 2 + pmb->cellbounds.is(interior);
        const Real x1m = coarse_coords.x1s2(i - 1);
        const Real x1c = coarse_coords.x1s2(i);
        const Real x1p = coarse_coords.x1s2(i + 1);
        const Real fx1m = coords.x1s2(fi);
        const Real fx1p = coords.x1s2(fi + 1);
        Real ccval = coarse(k, j, i);

        Real gx1m = (ccval - coarse(k, j, i - 1)) / (x1c - x1m);
        Real gx1p = (coarse(k, j, i + 1) - ccval) / (x1p - x1c);
        Real gx1c =
            0.5 * (SIGN(gx1m) + SIGN(gx1p)) * std::min(std::abs(gx1m), std::abs(gx1p));

        fine(fk, fj, fi) = ccval - gx1c * (x1c - fx1m);
        fine(fk, fj, fi + 1) = ccval + gx1c * (fx1p - x1c);
      }
    }
  } else {
    for (int i = si; i <= ei; i++) {
      int fi = (i - pmb->c_cellbounds.is(interior)) * 2 + pmb->cellbounds.is(interior);
      Real gxm = (coarse(0, 0, i) - coarse(0, 0, i - 1)) /
                 (coarse_coords.x1s2(i) - coarse_coords.x1s2(i - 1));
      Real gxp = (coarse(0, 0, i + 1) - coarse(0, 0, i)) /
                 (coarse_coords.x1s2(i + 1) - coarse_coords.x1s2(i));
      Real gxc = 0.5 * (SIGN(gxm) + SIGN(gxp)) * std::min(std::abs(gxm), std::abs(gxp));
      fine(0, 0, fi) = fine(0, 1, fi) =
          coarse(0, 0, i) - gxc * (coarse_coords.x1s2(i) - coords.x1s2(fi));
      fine(0, 0, fi + 1) = fine(0, 1, fi + 1) =
          coarse(0, 0, i) + gxc * (coords.x1s2(fi + 1) - coarse_coords.x1s2(i));
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::ProlongateSharedFieldX3(const ParArrayND<Real> &coarse,
//      ParArrayND<Real> &fine, int si, int ei, int sj, int ej, int sk, int ek)
//  \brief prolongate x3 face-centered fields shared between coarse and fine levels

void MeshRefinement::ProlongateSharedFieldX3(const ParArrayND<Real> &coarse,
                                             ParArrayND<Real> &fine, int si, int ei,
                                             int sj, int ej, int sk, int ek) {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  const IndexDomain interior = IndexDomain::interior;
  auto &coords = pmb->coords;
  if (pmb->block_size.nx3 > 1) {
    for (int k = sk; k <= ek; k++) {
      int fk = (k - pmb->c_cellbounds.ks(interior)) * 2 + pmb->cellbounds.ks(interior);
      for (int j = sj; j <= ej; j++) {
        int fj = (j - pmb->c_cellbounds.js(interior)) * 2 + pmb->cellbounds.js(interior);
        const Real x2m = coarse_coords.x2s3(j - 1);
        const Real x2c = coarse_coords.x2s3(j);
        const Real x2p = coarse_coords.x2s3(j + 1);
        const Real dx2m = x2c - x2m;
        const Real dx2p = x2p - x2c;
        const Real fx2m = coords.x2s3(fj);
        const Real fx2p = coords.x2s3(fj + 1);
        for (int i = si; i <= ei; i++) {
          int fi =
              (i - pmb->c_cellbounds.is(interior)) * 2 + pmb->cellbounds.is(interior);
          const Real x1m = coarse_coords.x1s3(i - 1);
          const Real x1c = coarse_coords.x1s3(i);
          const Real x1p = coarse_coords.x1s3(i + 1);
          const Real dx1m = x1c - x1m;
          const Real dx1p = x1p - x1c;
          const Real fx1m = coords.x1s3(fi);
          const Real fx1p = coords.x1s3(fi + 1);
          Real ccval = coarse(k, j, i);

          Real gx1m = (ccval - coarse(k, j, i - 1)) / dx1m;
          Real gx1p = (coarse(k, j, i + 1) - ccval) / dx1p;
          Real gx1c =
              0.5 * (SIGN(gx1m) + SIGN(gx1p)) * std::min(std::abs(gx1m), std::abs(gx1p));
          Real gx2m = (ccval - coarse(k, j - 1, i)) / dx2m;
          Real gx2p = (coarse(k, j + 1, i) - ccval) / dx2p;
          Real gx2c =
              0.5 * (SIGN(gx2m) + SIGN(gx2p)) * std::min(std::abs(gx2m), std::abs(gx2p));

          fine(fk, fj, fi) = ccval - gx1c * (x1c - fx1m) - gx2c * (x2c - fx2m);
          fine(fk, fj, fi + 1) = ccval + gx1c * (fx1p - x1c) - gx2c * (x2c - fx2m);
          fine(fk, fj + 1, fi) = ccval - gx1c * (x1c - fx1m) + gx2c * (fx2p - x2c);
          fine(fk, fj + 1, fi + 1) = ccval + gx1c * (fx1p - x1c) + gx2c * (fx2p - x2c);
        }
      }
    }
  } else if (pmb->block_size.nx2 > 1) {
    int k = pmb->c_cellbounds.ks(interior), fk = pmb->cellbounds.ks(interior);
    for (int j = sj; j <= ej; j++) {
      int fj = (j - pmb->c_cellbounds.js(interior)) * 2 + pmb->cellbounds.js(interior);
      const Real x2m = coarse_coords.x2s3(j - 1);
      const Real x2c = coarse_coords.x2s3(j);
      const Real x2p = coarse_coords.x2s3(j + 1);
      Real dx2m = x2c - x2m;
      Real dx2p = x2p - x2c;
      const Real fx2m = coords.x2s3(fj);
      const Real fx2p = coords.x2s3(fj + 1);
      Real dx2fm = x2c - fx2m;
      Real dx2fp = fx2p - x2c;
      for (int i = si; i <= ei; i++) {
        int fi = (i - pmb->c_cellbounds.is(interior)) * 2 + pmb->cellbounds.is(interior);
        const Real x1m = coarse_coords.x1s3(i - 1);
        const Real x1c = coarse_coords.x1s3(i);
        const Real x1p = coarse_coords.x1s3(i + 1);
        Real dx1m = x1c - x1m;
        Real dx1p = x1p - x1c;
        const Real fx1m = coords.x1s3(fi);
        const Real fx1p = coords.x1s3(fi + 1);
        Real dx1fm = x1c - fx1m;
        Real dx1fp = fx1p - x1c;
        Real ccval = coarse(k, j, i);

        // calculate 2D gradients using the minmod limiter
        Real gx1m = (ccval - coarse(k, j, i - 1)) / dx1m;
        Real gx1p = (coarse(k, j, i + 1) - ccval) / dx1p;
        Real gx1c =
            0.5 * (SIGN(gx1m) + SIGN(gx1p)) * std::min(std::abs(gx1m), std::abs(gx1p));
        Real gx2m = (ccval - coarse(k, j - 1, i)) / dx2m;
        Real gx2p = (coarse(k, j + 1, i) - ccval) / dx2p;
        Real gx2c =
            0.5 * (SIGN(gx2m) + SIGN(gx2p)) * std::min(std::abs(gx2m), std::abs(gx2p));

        // interpolate on to the finer grid
        fine(fk, fj, fi) = fine(fk + 1, fj, fi) = ccval - gx1c * dx1fm - gx2c * dx2fm;
        fine(fk, fj, fi + 1) = fine(fk + 1, fj, fi + 1) =
            ccval + gx1c * dx1fp - gx2c * dx2fm;
        fine(fk, fj + 1, fi) = fine(fk + 1, fj + 1, fi) =
            ccval - gx1c * dx1fm + gx2c * dx2fp;
        fine(fk, fj + 1, fi + 1) = fine(fk + 1, fj + 1, fi + 1) =
            ccval + gx1c * dx1fp + gx2c * dx2fp;
      }
    }
  } else {
    for (int i = si; i <= ei; i++) {
      int fi = (i - pmb->c_cellbounds.is(interior)) * 2 + pmb->cellbounds.is(interior);
      Real gxm = (coarse(0, 0, i) - coarse(0, 0, i - 1)) /
                 (coarse_coords.x1s3(i) - coarse_coords.x1s3(i - 1));
      Real gxp = (coarse(0, 0, i + 1) - coarse(0, 0, i)) /
                 (coarse_coords.x1s3(i + 1) - coarse_coords.x1s3(i));
      Real gxc = 0.5 * (SIGN(gxm) + SIGN(gxp)) * std::min(std::abs(gxm), std::abs(gxp));
      fine(0, 0, fi) = fine(1, 0, fi) =
          coarse(0, 0, i) - gxc * (coarse_coords.x1s3(i) - coords.x1s3(fi));
      fine(0, 0, fi + 1) = fine(1, 0, fi + 1) =
          coarse(0, 0, i) + gxc * (coords.x1s3(fi + 1) - coarse_coords.x1s3(i));
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::ProlongateInternalField(FaceField &fine,
//                           int si, int ei, int sj, int ej, int sk, int ek)
//  \brief prolongate the internal face-centered fields

void MeshRefinement::ProlongateInternalField(FaceField &fine, int si, int ei, int sj,
                                             int ej, int sk, int ek) {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  auto &coords = pmb->coords;
  const IndexDomain interior = IndexDomain::interior;
  int fsi = (si - pmb->c_cellbounds.is(interior)) * 2 + pmb->cellbounds.is(interior),
      fei = (ei - pmb->c_cellbounds.is(interior)) * 2 + pmb->cellbounds.is(interior) + 1;
  if (pmb->block_size.nx3 > 1) {
    for (int k = sk; k <= ek; k++) {
      int fk = (k - pmb->c_cellbounds.ks(interior)) * 2 + pmb->cellbounds.ks(interior);
      for (int j = sj; j <= ej; j++) {
        int fj = (j - pmb->c_cellbounds.js(interior)) * 2 + pmb->cellbounds.js(interior);
        for (int i = si; i <= ei; i++) {
          int fi =
              (i - pmb->c_cellbounds.is(interior)) * 2 + pmb->cellbounds.is(interior);
          Real Uxx = 0.0, Vyy = 0.0, Wzz = 0.0;
          Real Uxyz = 0.0, Vxyz = 0.0, Wxyz = 0.0;
#pragma unroll
          for (int jj = 0; jj < 2; jj++) {
            int js = 2 * jj - 1, fjj = fj + jj, fjp = fj + 2 * jj;
#pragma unroll
            for (int ii = 0; ii < 2; ii++) {
              int is = 2 * ii - 1, fii = fi + ii, fip = fi + 2 * ii;

              Uxx += is *
                     (js * (fine.x2f(fk, fjp, fii) * coords.Area(X2DIR, fk, fjp, fii) +
                            fine.x2f(fk + 1, fjp, fii) *
                                coords.Area(X2DIR, fk + 1, fjp, fii)) +
                      (fine.x3f(fk + 2, fjj, fii) * coords.Area(X3DIR, fk + 2, fjj, fii) -
                       fine.x3f(fk, fjj, fii) * coords.Area(X3DIR, fk, fjj, fii)));
              Vyy += js *
                     ((fine.x3f(fk + 2, fjj, fii) * coords.Area(X3DIR, fk + 2, fjj, fii) -
                       fine.x3f(fk, fjj, fii) * coords.Area(X3DIR, fk, fjj, fii)) +
                      is * (fine.x1f(fk, fjj, fip) * coords.Area(X1DIR, fk, fjj, fip) +
                            fine.x1f(fk + 1, fjj, fip) *
                                coords.Area(X1DIR, fk + 1, fjj, fip)));
              Wzz +=
                  is *
                      (fine.x1f(fk + 1, fjj, fip) * coords.Area(X1DIR, fk + 1, fjj, fip) -
                       fine.x1f(fk, fjj, fip) * coords.Area(X1DIR, fk, fjj, fip)) +
                  js *
                      (fine.x2f(fk + 1, fjp, fii) * coords.Area(X2DIR, fk + 1, fjp, fii) -
                       fine.x2f(fk, fjp, fii) * coords.Area(X2DIR, fk, fjp, fii));
              Uxyz += is * js *
                      (fine.x1f(fk + 1, fjj, fip) * coords.Area(X1DIR, fk + 1, fjj, fip) -
                       fine.x1f(fk, fjj, fip) * coords.Area(X1DIR, fk, fjj, fip));
              Vxyz += is * js *
                      (fine.x2f(fk + 1, fjp, fii) * coords.Area(X2DIR, fk + 1, fjp, fii) -
                       fine.x2f(fk, fjp, fii) * coords.Area(X2DIR, fk, fjp, fii));
              Wxyz += is * js *
                      (fine.x3f(fk + 2, fjj, fii) * coords.Area(X3DIR, fk + 2, fjj, fii) -
                       fine.x3f(fk, fjj, fii) * coords.Area(X3DIR, fk, fjj, fii));
            }
          }
          Real Sdx1 = SQR(coords.dx1f(fi) + coords.dx1f(fi + 1));
          Real Sdx2 = SQR(coords.EdgeLength(X2DIR, fk + 1, fj, fi + 1) +
                          coords.EdgeLength(X2DIR, fk + 1, fj + 1, fi + 1));
          Real Sdx3 = SQR(coords.EdgeLength(X3DIR, fk, fj + 1, fi + 1) +
                          coords.EdgeLength(X3DIR, fk + 1, fj + 1, fi + 1));
          Uxx *= 0.125;
          Vyy *= 0.125;
          Wzz *= 0.125;
          Uxyz *= 0.125 / (Sdx2 + Sdx3);
          Vxyz *= 0.125 / (Sdx1 + Sdx3);
          Wxyz *= 0.125 / (Sdx1 + Sdx2);
          fine.x1f(fk, fj, fi + 1) =
              (0.5 * (fine.x1f(fk, fj, fi) * coords.Area(X1DIR, fk, fj, fi) +
                      fine.x1f(fk, fj, fi + 2) * coords.Area(X1DIR, fk, fj, fi + 2)) +
               Uxx - Sdx3 * Vxyz - Sdx2 * Wxyz) /
              coords.Area(X1DIR, fk, fj, fi + 1);
          fine.x1f(fk, fj + 1, fi + 1) =
              (0.5 * (fine.x1f(fk, fj + 1, fi) * coords.Area(X1DIR, fk, fj + 1, fi) +
                      fine.x1f(fk, fj + 1, fi + 2) *
                          coords.Area(X1DIR, fk, fj + 1, fi + 2)) +
               Uxx - Sdx3 * Vxyz + Sdx2 * Wxyz) /
              coords.Area(X1DIR, fk, fj + 1, fi + 1);
          fine.x1f(fk + 1, fj, fi + 1) =
              (0.5 * (fine.x1f(fk + 1, fj, fi) * coords.Area(X1DIR, fk + 1, fj, fi) +
                      fine.x1f(fk + 1, fj, fi + 2) *
                          coords.Area(X1DIR, fk + 1, fj, fi + 2)) +
               Uxx + Sdx3 * Vxyz - Sdx2 * Wxyz) /
              coords.Area(X1DIR, fk + 1, fj, fi + 1);
          fine.x1f(fk + 1, fj + 1, fi + 1) =
              (0.5 * (fine.x1f(fk + 1, fj + 1, fi) *
                          coords.Area(X1DIR, fk + 1, fj + 1, fi) +
                      fine.x1f(fk + 1, fj + 1, fi + 2) *
                          coords.Area(X1DIR, fk + 1, fj + 1, fi + 2)) +
               Uxx + Sdx3 * Vxyz + Sdx2 * Wxyz) /
              coords.Area(X1DIR, fk + 1, fj + 1, fi + 1);

          fine.x2f(fk, fj + 1, fi) =
              (0.5 * (fine.x2f(fk, fj, fi) * coords.Area(X2DIR, fk, fj, fi) +
                      fine.x2f(fk, fj + 2, fi) * coords.Area(X2DIR, fk, fj + 2, fi)) +
               Vyy - Sdx3 * Uxyz - Sdx1 * Wxyz) /
              coords.Area(X2DIR, fk, fj + 1, fi);
          fine.x2f(fk, fj + 1, fi + 1) =
              (0.5 * (fine.x2f(fk, fj, fi + 1) * coords.Area(X2DIR, fk, fj, fi + 1) +
                      fine.x2f(fk, fj + 2, fi + 1) *
                          coords.Area(X2DIR, fk, fj + 2, fi + 1)) +
               Vyy - Sdx3 * Uxyz + Sdx1 * Wxyz) /
              coords.Area(X2DIR, fk, fj + 1, fi + 1);
          fine.x2f(fk + 1, fj + 1, fi) =
              (0.5 * (fine.x2f(fk + 1, fj, fi) * coords.Area(X2DIR, fk + 1, fj, fi) +
                      fine.x2f(fk + 1, fj + 2, fi) *
                          coords.Area(X2DIR, fk + 1, fj + 2, fi)) +
               Vyy + Sdx3 * Uxyz - Sdx1 * Wxyz) /
              coords.Area(X2DIR, fk + 1, fj + 1, fi);
          fine.x2f(fk + 1, fj + 1, fi + 1) =
              (0.5 * (fine.x2f(fk + 1, fj, fi + 1) *
                          coords.Area(X2DIR, fk + 1, fj, fi + 1) +
                      fine.x2f(fk + 1, fj + 2, fi + 1) *
                          coords.Area(X2DIR, fk + 1, fj + 2, fi + 1)) +
               Vyy + Sdx3 * Uxyz + Sdx1 * Wxyz) /
              coords.Area(X2DIR, fk + 1, fj + 1, fi + 1);

          fine.x3f(fk + 1, fj, fi) =
              (0.5 * (fine.x3f(fk + 2, fj, fi) * coords.Area(X3DIR, fk + 2, fj, fi) +
                      fine.x3f(fk, fj, fi) * coords.Area(X3DIR, fk, fj, fi)) +
               Wzz - Sdx2 * Uxyz - Sdx1 * Vxyz) /
              coords.Area(X3DIR, fk + 1, fj, fi);
          fine.x3f(fk + 1, fj, fi + 1) =
              (0.5 * (fine.x3f(fk + 2, fj, fi + 1) *
                          coords.Area(X3DIR, fk + 2, fj, fi + 1) +
                      fine.x3f(fk, fj, fi + 1) * coords.Area(X3DIR, fk, fj, fi + 1)) +
               Wzz - Sdx2 * Uxyz + Sdx1 * Vxyz) /
              coords.Area(X3DIR, fk + 1, fj, fi + 1);
          fine.x3f(fk + 1, fj + 1, fi) =
              (0.5 * (fine.x3f(fk + 2, fj + 1, fi) *
                          coords.Area(X3DIR, fk + 2, fj + 1, fi) +
                      fine.x3f(fk, fj + 1, fi) * coords.Area(X3DIR, fk, fj + 1, fi)) +
               Wzz + Sdx2 * Uxyz - Sdx1 * Vxyz) /
              coords.Area(X3DIR, fk + 1, fj + 1, fi);
          fine.x3f(fk + 1, fj + 1, fi + 1) =
              (0.5 * (fine.x3f(fk + 2, fj + 1, fi + 1) *
                          coords.Area(X3DIR, fk + 2, fj + 1, fi + 1) +
                      fine.x3f(fk, fj + 1, fi + 1) *
                          coords.Area(X3DIR, fk, fj + 1, fi + 1)) +
               Wzz + Sdx2 * Uxyz + Sdx1 * Vxyz) /
              coords.Area(X3DIR, fk + 1, fj + 1, fi + 1);
        }
      }
    }
  } else if (pmb->block_size.nx2 > 1) {
    int fk = pmb->cellbounds.ks(interior);
    for (int j = sj; j <= ej; j++) {
      int fj = (j - pmb->c_cellbounds.js(interior)) * 2 + pmb->cellbounds.js(interior);
      for (int i = si; i <= ei; i++) {
        int fi = (i - pmb->c_cellbounds.is(interior)) * 2 + pmb->cellbounds.is(interior);
        Real tmp1 =
            0.25 *
            (fine.x2f(fk, fj + 2, fi + 1) * coords.Area(X2DIR, fk, fj + 2, fi + 1) -
             fine.x2f(fk, fj, fi + 1) * coords.Area(X2DIR, fk, fj, fi + 1) -
             fine.x2f(fk, fj + 2, fi) * coords.Area(X2DIR, fk, fj + 2, fi) +
             fine.x2f(fk, fj, fi) * coords.Area(X2DIR, fk, fj, fi));
        Real tmp2 =
            0.25 *
            (fine.x1f(fk, fj, fi) * coords.Area(X1DIR, fk, fj, fi) -
             fine.x1f(fk, fj, fi + 2) * coords.Area(X1DIR, fk, fj, fi + 2) -
             fine.x1f(fk, fj + 1, fi) * coords.Area(X1DIR, fk, fj + 1, fi) +
             fine.x1f(fk, fj + 1, fi + 2) * coords.Area(X1DIR, fk, fj + 1, fi + 2));
        fine.x1f(fk, fj, fi + 1) =
            (0.5 * (fine.x1f(fk, fj, fi) * coords.Area(X1DIR, fk, fj, fi) +
                    fine.x1f(fk, fj, fi + 2) * coords.Area(X1DIR, fk, fj, fi + 2)) +
             tmp1) /
            coords.Area(X1DIR, fk, fj, fi + 1);
        fine.x1f(fk, fj + 1, fi + 1) =
            (0.5 *
                 (fine.x1f(fk, fj + 1, fi) * coords.Area(X1DIR, fk, fj + 1, fi) +
                  fine.x1f(fk, fj + 1, fi + 2) * coords.Area(X1DIR, fk, fj + 1, fi + 2)) +
             tmp1) /
            coords.Area(X1DIR, fk, fj + 1, fi + 1);
        fine.x2f(fk, fj + 1, fi) =
            (0.5 * (fine.x2f(fk, fj, fi) * coords.Area(X2DIR, fk, fj, fi) +
                    fine.x2f(fk, fj + 2, fi) * coords.Area(X2DIR, fk, fj + 2, fi)) +
             tmp2) /
            coords.Area(X2DIR, fk, fj + 1, fi);
        fine.x2f(fk, fj + 1, fi + 1) =
            (0.5 *
                 (fine.x2f(fk, fj, fi + 1) * coords.Area(X2DIR, fk, fj, fi + 1) +
                  fine.x2f(fk, fj + 2, fi + 1) * coords.Area(X2DIR, fk, fj + 2, fi + 1)) +
             tmp2) /
            coords.Area(X2DIR, fk, fj + 1, fi + 1);
      }
    }
  } else {
    for (int i = si; i <= ei; i++) {
      int fi = (i - pmb->c_cellbounds.is(interior)) * 2 + pmb->cellbounds.is(interior);
      Real ph = coords.Area(X1DIR, 0, 0, fi) * fine.x1f(0, 0, fi);
      fine.x1f(0, 0, fi + 1) = ph / coords.Area(X1DIR, 0, 0, fi + 1);
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::CheckRefinementCondition()
//  \brief Check refinement criteria

void MeshRefinement::CheckRefinementCondition() {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  auto &rc = pmb->meshblock_data.Get();
  AmrTag ret = Refinement::CheckAllRefinement(rc.get());
  SetRefinement(ret);
}

void MeshRefinement::SetRefinement(AmrTag flag) {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  int aret = std::max(-1, static_cast<int>(flag));

  if (aret == 0) refine_flag_ = 0;

  if (aret >= 0) deref_count_ = 0;
  if (aret > 0) {
    if (pmb->loc.level == pmb->pmy_mesh->max_level) {
      refine_flag_ = 0;
    } else {
      refine_flag_ = 1;
    }
  } else if (aret < 0) {
    if (pmb->loc.level == pmb->pmy_mesh->root_level) {
      refine_flag_ = 0;
      deref_count_ = 0;
    } else {
      deref_count_++;
      int ec = 0, js, je, ks, ke;
      if (pmb->block_size.nx2 > 1) {
        js = -1;
        je = 1;
      } else {
        js = 0;
        je = 0;
      }
      if (pmb->block_size.nx3 > 1) {
        ks = -1;
        ke = 1;
      } else {
        ks = 0;
        ke = 0;
      }
      for (int k = ks; k <= ke; k++) {
        for (int j = js; j <= je; j++) {
          for (int i = -1; i <= 1; i++)
            if (pmb->pbval->nblevel[k + 1][j + 1][i + 1] > pmb->loc.level) ec++;
        }
      }
      if (ec > 0) {
        refine_flag_ = 0;
      } else {
        if (deref_count_ >= deref_threshold_) {
          refine_flag_ = -1;
        } else {
          refine_flag_ = 0;
        }
      }
    }
  }

  return;
}

// TODO(felker): consider merging w/ MeshBlock::pvars_cc, etc. See meshblock.cpp

int MeshRefinement::AddToRefinement(std::shared_ptr<Variable<Real>> pvar) {
  pvars_cc_.push_back(pvar);
  return static_cast<int>(pvars_cc_.size() - 1);
}

int MeshRefinement::AddToRefinement(FaceField *pvar_fc, FaceField *pcoarse_fc) {
  pvars_fc_.push_back(std::make_tuple(pvar_fc, pcoarse_fc));
  return static_cast<int>(pvars_fc_.size() - 1);
}

} // namespace parthenon
