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
//! \file bvals_refine.cpp
//  \brief constructor/destructor and utility functions for BoundaryValues class

#include "bvals/bvals_interfaces.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>

#include "bvals/cc/bvals_cc_in_one.hpp"
#include "fc/bvals_fc.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock.hpp"

namespace parthenon {

// -----------
// NOTE ON SWITCHING BETWEEN PRIMITIVE VS. CONSERVED AND STANDARD VS. COARSE BUFFERS HERE:
// -----------

// -----------
// There are several sets of variable pointers used in this file:
// 1) MeshRefinement tuples of pointers: pvars_cc_
// -- Used in RestrictGhostCellsOnSameLevel_() and ProlongateGhostCells_()

// 2) Hardcoded pointers through MeshBlock members
// -- Used in ProlongateGhostCells_() where
// physical quantities are coupled through EquationOfState

// NOTE(JMM): ProlongateBounds has been split into RestrictBoundaries
// and ProlongateBoundaries,
// which are now called together as ProlongateBoundaries in
// `bvals/bondary_conditions.hpp`. This allows us to loop over all variables in a
// container.

int BoundaryValues::NumRestrictions() {
  int nbuffs = 0;
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  MeshRefinement *pmr = pmb->pmr.get();
  int &mylevel = pmb->loc.level;
  for (int n = 0; n < nneighbor; n++) {
    NeighborBlock &nb = neighbor[n];
    if (nb.snb.level >= mylevel) continue;

    IndexRange bni, bnj, bnk;
    ComputeRestrictionBounds_(nb, bni, bnj, bnk);

    for (int nk = bnk.s; nk <= bnk.e; nk++) {
      for (int nj = bnj.s; nj <= bnj.e; nj++) {
        for (int ni = bni.s; ni <= bni.e; ni++) {
          int ntype = std::abs(ni) + std::abs(nj) + std::abs(nk);
          // skip myself or coarse levels; only the same level must be restricted
          if (ntype == 0 || nblevel[nk + 1][nj + 1][ni + 1] != mylevel) continue;
          nbuffs += 1;
        }
      }
    }
  }
  return nbuffs;
}

void BoundaryValues::FillRestrictionMetadata(cell_centered_bvars::BufferCacheHost_t &info,
                                             int &idx, ParArray4D<Real> &fine,
                                             ParArray4D<Real> &coarse, int Nv) {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  MeshRefinement *pmr = pmb->pmr.get();
  int &mylevel = pmb->loc.level;
  for (int n = 0; n < nneighbor; n++) {
    NeighborBlock &nb = neighbor[n];
    if (nb.snb.level >= mylevel) continue;

    IndexRange bni, bnj, bnk;
    ComputeRestrictionBounds_(nb, bni, bnj, bnk);

    for (int nk = bnk.s; nk <= bnk.e; nk++) {
      for (int nj = bnj.s; nj <= bnj.e; nj++) {
        for (int ni = bni.s; ni <= bni.e; ni++) {
          int ntype = std::abs(ni) + std::abs(nj) + std::abs(nk);
          // skip myself or coarse levels; only the same level must be restricted
          if (ntype == 0 || nblevel[nk + 1][nj + 1][ni + 1] != mylevel) continue;
          ComputeRestrictionIndices_(nb, nk, nj, ni, info(idx).si, info(idx).ei,
                                     info(idx).sj, info(idx).ej, info(idx).sk,
                                     info(idx).ek);
          info(idx).coords = pmb->coords;
          info(idx).coarse_coords = pmb->pmr->coarse_coords;
          info(idx).fine = fine;
          info(idx).coarse = coarse;
          info(idx).restriction = true;
          info(idx).Nv = Nv;
          idx++;
        }
      }
    }
  }
}

void BoundaryValues::ProlongateBoundaries() {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  int &mylevel = pmb->loc.level;
  for (int n = 0; n < nneighbor; n++) {
    NeighborBlock &nb = neighbor[n];
    if (nb.snb.level >= mylevel) continue;
    // calculate the loop limits for the ghost zones
    IndexRange bi, bj, bk;
    ComputeProlongationBounds_(nb, bi, bj, bk);
    ProlongateGhostCells_(nb, bi.s, bi.e, bj.s, bj.e, bk.s, bk.e);
  } // end loop over nneighbor
}

void BoundaryValues::RestrictGhostCellsOnSameLevel_(const NeighborBlock &nb, int nk,
                                                    int nj, int ni) {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  MeshRefinement *pmr = pmb->pmr.get();

  const IndexDomain interior = IndexDomain::interior;
  IndexRange cib = pmb->c_cellbounds.GetBoundsI(interior);
  IndexRange cjb = pmb->c_cellbounds.GetBoundsJ(interior);
  IndexRange ckb = pmb->c_cellbounds.GetBoundsK(interior);

  int ris, rie, rjs, rje, rks, rke;
  ComputeRestrictionIndices_(nb, nk, nj, ni, ris, rie, rjs, rje, rks, rke);

  for (auto cc_var : pmr->pvars_cc_) {
    if (!cc_var->IsAllocated()) continue;
    ParArrayND<Real> var_cc = cc_var->data;
    ParArrayND<Real> coarse_cc = cc_var->coarse_s;
    int nu = var_cc.GetDim(4) - 1;
    pmb->pmr->RestrictCellCenteredValues(var_cc, coarse_cc, 0, nu, ris, rie, rjs, rje,
                                         rks, rke);
  }

  for (auto fc_pair : pmr->pvars_fc_) {
    FaceField *var_fc = std::get<0>(fc_pair);
    FaceField *coarse_fc = std::get<1>(fc_pair);
    int &mylevel = pmb->loc.level;
    int rs = ris, re = rie + 1;
    if (rs == cib.s && nblevel[nk + 1][nj + 1][ni] < mylevel) rs++;
    if (re == cib.e + 1 && nblevel[nk + 1][nj + 1][ni + 2] < mylevel) re--;
    pmr->RestrictFieldX1((*var_fc).x1f, (*coarse_fc).x1f, rs, re, rjs, rje, rks, rke);
    if (pmb->block_size.nx2 > 1) {
      rs = rjs, re = rje + 1;
      if (rs == cjb.s && nblevel[nk + 1][nj][ni + 1] < mylevel) rs++;
      if (re == cjb.e + 1 && nblevel[nk + 1][nj + 2][ni + 1] < mylevel) re--;
      pmr->RestrictFieldX2((*var_fc).x2f, (*coarse_fc).x2f, ris, rie, rs, re, rks, rke);
    } else { // 1D
      pmr->RestrictFieldX2((*var_fc).x2f, (*coarse_fc).x2f, ris, rie, rjs, rje, rks, rke);
      for (int i = ris; i <= rie; i++)
        (*coarse_fc).x2f(rks, rjs + 1, i) = (*coarse_fc).x2f(rks, rjs, i);
    }

    if (pmb->block_size.nx3 > 1) {
      rs = rks, re = rke + 1;
      if (rs == ckb.s && nblevel[nk][nj + 1][ni + 1] < mylevel) rs++;
      if (re == ckb.e + 1 && nblevel[nk + 2][nj + 1][ni + 1] < mylevel) re--;
      pmr->RestrictFieldX3((*var_fc).x3f, (*coarse_fc).x3f, ris, rie, rjs, rje, rs, re);
    } else { // 1D or 2D
      pmr->RestrictFieldX3((*var_fc).x3f, (*coarse_fc).x3f, ris, rie, rjs, rje, rks, rke);
      for (int j = rjs; j <= rje; j++) {
        for (int i = ris; i <= rie; i++)
          (*coarse_fc).x3f(rks + 1, j, i) = (*coarse_fc).x3f(rks, j, i);
      }
    }
  } // end loop over pvars_fc_
  return;
}

void BoundaryValues::ProlongateGhostCells_(const NeighborBlock &nb, int si, int ei,
                                           int sj, int ej, int sk, int ek) {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  auto &pmr = pmb->pmr;

  for (auto cc_var : pmr->pvars_cc_) {
    if (!cc_var->IsAllocated()) continue;
    ParArrayND<Real> var_cc = cc_var->data;
    ParArrayND<Real> coarse_cc = cc_var->coarse_s;
    int nu = var_cc.GetDim(4) - 1;
    pmr->ProlongateCellCenteredValues(coarse_cc, var_cc, 0, nu, si, ei, sj, ej, sk, ek);
  }

  // prolongate face-centered S/AMR-enrolled quantities (magnetic fields)
  int &mylevel = pmb->loc.level;
  int il, iu, jl, ju, kl, ku;
  il = si, iu = ei + 1;
  if ((nb.ni.ox1 >= 0) && (nblevel[nb.ni.ox3 + 1][nb.ni.ox2 + 1][nb.ni.ox1] >= mylevel)) {
    il++;
  }
  if ((nb.ni.ox1 <= 0) &&
      (nblevel[nb.ni.ox3 + 1][nb.ni.ox2 + 1][nb.ni.ox1 + 2] >= mylevel)) {
    iu--;
  }

  if (pmb->block_size.nx2 > 1) {
    jl = sj, ju = ej + 1;
    if ((nb.ni.ox2 >= 0) && (nblevel[nb.ni.ox3 + 1][nb.ni.ox2][nb.ni.ox1 + 1] >= mylevel))
      jl++;
    if ((nb.ni.ox2 <= 0) &&
        (nblevel[nb.ni.ox3 + 1][nb.ni.ox2 + 2][nb.ni.ox1 + 1] >= mylevel))
      ju--;
  } else {
    jl = sj;
    ju = ej;
  }

  if (pmb->block_size.nx3 > 1) {
    kl = sk, ku = ek + 1;
    if ((nb.ni.ox3 >= 0) &&
        (nblevel[nb.ni.ox3][nb.ni.ox2 + 1][nb.ni.ox1 + 1] >= mylevel)) {
      kl++;
    }
    if ((nb.ni.ox3 <= 0) &&
        (nblevel[nb.ni.ox3 + 2][nb.ni.ox2 + 1][nb.ni.ox1 + 1] >= mylevel)) {
      ku--;
    }
  } else {
    kl = sk;
    ku = ek;
  }

  for (auto fc_pair : pmr->pvars_fc_) {
    FaceField *var_fc = std::get<0>(fc_pair);
    FaceField *coarse_fc = std::get<1>(fc_pair);

    // step 1. calculate x1 outer surface fields and slopes
    pmr->ProlongateSharedFieldX1((*coarse_fc).x1f, (*var_fc).x1f, il, iu, sj, ej, sk, ek);
    // step 2. calculate x2 outer surface fields and slopes
    pmr->ProlongateSharedFieldX2((*coarse_fc).x2f, (*var_fc).x2f, si, ei, jl, ju, sk, ek);
    // step 3. calculate x3 outer surface fields and slopes
    pmr->ProlongateSharedFieldX3((*coarse_fc).x3f, (*var_fc).x3f, si, ei, sj, ej, kl, ku);

    // step 4. calculate the internal finer fields using the Toth & Roe method
    pmr->ProlongateInternalField((*var_fc), si, ei, sj, ej, sk, ek);
  }
}

void BoundaryValues::ComputeRestrictionIndices_(const NeighborBlock &nb, int nk, int nj,
                                                int ni, int &ris, int &rie, int &rjs,
                                                int &rje, int &rks, int &rke) {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  const IndexDomain interior = IndexDomain::interior;
  IndexRange cib = pmb->c_cellbounds.GetBoundsI(interior);
  IndexRange cjb = pmb->c_cellbounds.GetBoundsJ(interior);
  IndexRange ckb = pmb->c_cellbounds.GetBoundsK(interior);

  auto CalcIndices = [](int &rs, int &re, int n, int ox, const IndexRange &b) {
    if (n == 0) {
      rs = b.s;
      re = b.e;
      if (ox == 1) {
        rs = b.e;
      } else if (ox == -1) {
        re = b.s;
      }
    } else if (n == 1) {
      rs = b.e + 1;
      re = b.e + 1;
    } else { //(n ==  - 1)
      rs = b.s - 1;
      re = b.s - 1;
    }
  };

  CalcIndices(ris, rie, ni, nb.ni.ox1, cib);
  CalcIndices(rjs, rje, nj, nb.ni.ox2, cjb);
  CalcIndices(rks, rke, nk, nb.ni.ox3, ckb);
}

void BoundaryValues::ComputeRestrictionBounds_(const NeighborBlock &nb, IndexRange &ni,
                                               IndexRange &nj, IndexRange &nk) {
  auto getbounds = [](const int nbx, IndexRange &n) {
    n.s = std::max(nbx - 1, -1);
    n.e = std::min(nbx + 1, 1);
  };

  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  getbounds(nb.ni.ox1, ni);
  if (pmb->block_size.nx2 == 1) {
    nj.s = nj.e = 0;
  } else {
    getbounds(nb.ni.ox2, nj);
  }

  if (pmb->block_size.nx3 == 1) {
    nk.s = nk.e = 0;
  } else {
    getbounds(nb.ni.ox3, nk);
  }
}

void BoundaryValues::ComputeProlongationBounds_(const NeighborBlock &nb, IndexRange &bi,
                                                IndexRange &bj, IndexRange &bk) {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  const IndexDomain interior = IndexDomain::interior;
  int cn = pmb->cnghost - 1;

  auto getbounds = [=](const int nbx, const std::int64_t &lx, const IndexRange bblock,
                       IndexRange &bprol) {
    if (nbx == 0) {
      bprol.s = bblock.s;
      bprol.e = bblock.e;
      if ((lx & 1LL) == 0LL) {
        bprol.e += cn;
      } else {
        bprol.s -= cn;
      }
    } else if (nbx > 0) {
      bprol.s = bblock.e + 1;
      bprol.e = bblock.e + cn;
    } else {
      bprol.s = bblock.s - cn;
      bprol.e = bblock.s - 1;
    }
  };

  getbounds(nb.ni.ox1, pmb->loc.lx1, pmb->c_cellbounds.GetBoundsI(interior), bi);
  getbounds(nb.ni.ox2, pmb->loc.lx2, pmb->c_cellbounds.GetBoundsJ(interior), bj);
  getbounds(nb.ni.ox3, pmb->loc.lx3, pmb->c_cellbounds.GetBoundsK(interior), bk);
}

} // namespace parthenon
