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
//! \file bvals_refine.cpp
//  \brief constructor/destructor and utility functions for BoundaryValues class

#include "bvals/bvals_interfaces.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>

#include "fc/bvals_fc.hpp"
#include "mesh/mesh.hpp"

namespace parthenon {

// -----------
// NOTE ON SWITCHING BETWEEN PRIMITIVE VS. CONSERVED AND STANDARD VS. COARSE BUFFERS HERE:
// -----------

// In both Mesh::Initialize and time_integartor.cpp, this wrapper function
// ProlongateBoundaries expects to have associated
// BoundaryVariable objects with member pointers pointing to their CONSERVED VARIABLE
// ARRAYS (standard and coarse buffers) by the time this function is called.

// E.g. in time_integrator.cpp, the PROLONG task is called after SEND_HYD, SETB_HYD,
// SEND_SCLR, SETB_SCLR, all of which indepedently switch to their associated CONSERVED
// VARIABLE ARRAYS and before CON2PRIM which switches to PRIMITIVE VARIABLE ARRAYS.

// However, this is currently not a strict requirement, since all below
// MeshRefinement::Prolongate*() and Restrict*() calls refer directly to
// MeshRefinement::pvars_cc_, pvars_fc_ vectors, NOT the var_cc, coarse_buf ptr members of
// CellCenteredBoundaryVariable objects

// -----------
// There are three sets of variable pointers used in this file:
// 1) BoundaryVariable pointer members: var_cc, coarse_buf
// -- Only used in ApplyPhysicalBoundariesOnCoarseLevel()

// 2) MeshRefinement tuples of pointers: pvars_cc_
// -- Used in RestrictGhostCellsOnSameLevel() and ProlongateGhostCells()

// 3) Hardcoded pointers through MeshBlock members
// -- Used in ApplyPhysicalBoundariesOnCoarseLevel() and ProlongateGhostCells() where
// physical quantities are coupled through EquationOfState

// -----------
// SUMMARY OF BELOW PTR CHANGES:
// -----------
// 1. RestrictGhostCellsOnSameLevel (MeshRefinement::pvars_cc)
// --- change standard and coarse CONSERVED
// (also temporarily change to standard and coarse PRIMITIVE for GR simulations)

// 2. ApplyPhysicalBoundariesOnCoarseLevel (CellCenteredBoundaryVariable::var_cc)
// --- ONLY var_cc (var_fc) is changed to = coarse_buf, PRIMITIVE
// (automatically switches var_cc to standard and coarse_buf to coarse primitive
// arrays after fn returns)

// 3. ProlongateGhostCells (MeshRefinement::pvars_cc)
// --- change to standard and coarse PRIMITIVE
// (automatically switches back to conserved variables at the end of fn)

void BoundaryValues::ProlongateBoundaries(const Real time, const Real dt) {
  MeshBlock *pmb = pmy_block_;
  int &mylevel = pmb->loc.level;

  // This hardcoded technique is also used to manually specify the coupling between
  // physical variables in:
  // - step 2, ApplyPhysicalBoundariesOnCoarseLevel(): calls to W(U) and user BoundaryFunc
  // - step 3, ProlongateGhostCells(): calls to calculate bcc and U(W)

  // downcast BoundaryVariable pointers to known derived class pointer types:
  // RTTI via dynamic_case

  // For each finer neighbor, to prolongate a boundary we need to fill one more cell
  // surrounding the boundary zone to calculate the slopes ("ghost-ghost zone"). 3x steps:
  for (int n = 0; n < nneighbor; n++) {
    NeighborBlock &nb = neighbor[n];
    if (nb.snb.level >= mylevel) continue;
    // fill the required ghost-ghost zone
    int nis, nie, njs, nje, nks, nke;
    nis = std::max(nb.ni.ox1 - 1, -1);
    nie = std::min(nb.ni.ox1 + 1, 1);
    if (pmb->block_size.nx2 == 1) {
      njs = 0;
      nje = 0;
    } else {
      njs = std::max(nb.ni.ox2 - 1, -1);
      nje = std::min(nb.ni.ox2 + 1, 1);
    }

    if (pmb->block_size.nx3 == 1) {
      nks = 0;
      nke = 0;
    } else {
      nks = std::max(nb.ni.ox3 - 1, -1);
      nke = std::min(nb.ni.ox3 + 1, 1);
    }

    // Step 1. Apply necessary variable restrictions when ghost-ghost zone is on same lvl
    for (int nk = nks; nk <= nke; nk++) {
      for (int nj = njs; nj <= nje; nj++) {
        for (int ni = nis; ni <= nie; ni++) {
          int ntype = std::abs(ni) + std::abs(nj) + std::abs(nk);
          // skip myself or coarse levels; only the same level must be restricted
          if (ntype == 0 || nblevel[nk + 1][nj + 1][ni + 1] != mylevel) continue;

          // this neighbor block is on the same level
          // and needs to be restricted for prolongation
          RestrictGhostCellsOnSameLevel(nb, nk, nj, ni);
        }
      }
    }

    const IndexDomain interior = IndexDomain::interior;
    // calculate the loop limits for the ghost zones
    int cn = pmb->cnghost - 1;
    int si, ei, sj, ej, sk, ek;
    if (nb.ni.ox1 == 0) {
      std::int64_t &lx1 = pmb->loc.lx1;
      si = pmb->c_cellbounds.is(interior);
      ei = pmb->c_cellbounds.ie(interior);
      if ((lx1 & 1LL) == 0LL) {
        ei += cn;
      } else {
        si -= cn;
      }
    } else if (nb.ni.ox1 > 0) {
      si = pmb->c_cellbounds.ie(interior) + 1;
      ei = pmb->c_cellbounds.ie(interior) + cn;
    } else {
      si = pmb->c_cellbounds.is(interior) - cn;
      ei = pmb->c_cellbounds.is(interior) - 1;
    }

    if (nb.ni.ox2 == 0) {
      sj = pmb->c_cellbounds.js(interior);
      ej = pmb->c_cellbounds.je(interior);
      if (pmb->block_size.nx2 > 1) {
        std::int64_t &lx2 = pmb->loc.lx2;
        if ((lx2 & 1LL) == 0LL) {
          ej += cn;
        } else {
          sj -= cn;
        }
      }
    } else if (nb.ni.ox2 > 0) {
      sj = pmb->c_cellbounds.je(interior) + 1;
      ej = pmb->c_cellbounds.je(interior) + cn;
    } else {
      sj = pmb->c_cellbounds.js(interior) - cn;
      ej = pmb->c_cellbounds.js(interior) - 1;
    }

    if (nb.ni.ox3 == 0) {
      sk = pmb->c_cellbounds.ks(interior);
      ek = pmb->c_cellbounds.ke(interior);
      if (pmb->block_size.nx3 > 1) {
        std::int64_t &lx3 = pmb->loc.lx3;
        if ((lx3 & 1LL) == 0LL) {
          ek += cn;
        } else {
          sk -= cn;
        }
      }
    } else if (nb.ni.ox3 > 0) {
      sk = pmb->c_cellbounds.ke(interior) + 1;
      ek = pmb->c_cellbounds.ke(interior) + cn;
    } else {
      sk = pmb->c_cellbounds.ks(interior) - cn;
      ek = pmb->c_cellbounds.ks(interior) - 1;
    }

    // (temp workaround) to automatically call all BoundaryFunction_[] on coarse_prim/b
    // instead of previous targets var_cc=cons, var_fc=b

    // Step 2. Re-apply physical boundaries on the coarse boundary:
    // ApplyPhysicalBoundariesOnCoarseLevel(nb, time, dt, si, ei, sj, ej, sk, ek);

    // (temp workaround) swap BoundaryVariable var_cc/fc to standard primitive variable
    // arrays (not coarse) from coarse primitive variables arrays

    // Step 3. Finally, the ghost-ghost zones are ready for prolongation:
    ProlongateGhostCells(nb, si, ei, sj, ej, sk, ek);
  } // end loop over nneighbor
  return;
}

void BoundaryValues::RestrictGhostCellsOnSameLevel(const NeighborBlock &nb, int nk,
                                                   int nj, int ni) {
  MeshBlock *pmb = pmy_block_;
  MeshRefinement *pmr = pmb->pmr.get();

  const IndexDomain interior = IndexDomain::interior;
  IndexRange cib = pmb->c_cellbounds.GetBoundsI(interior);
  IndexRange cjb = pmb->c_cellbounds.GetBoundsJ(interior);
  IndexRange ckb = pmb->c_cellbounds.GetBoundsK(interior);

  auto CalcRestricedIndices = [](int & rs, int & re, int n, int ox, const IndexRange & b){
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

  int ris, rie, rjs, rje, rks, rke;
  CalcRestricedIndices(ris, rie, ni, nb.ni.ox1, cib);
  CalcRestricedIndices(rjs, rje, nj, nb.ni.ox2, cjb);
  CalcRestricedIndices(rks, rke, nk, nb.ni.ox3, ckb);

  for (auto cc_pair : pmr->pvars_cc_) {
    ParArrayND<Real> var_cc = std::get<0>(cc_pair);
    ParArrayND<Real> coarse_cc = std::get<1>(cc_pair);
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

//----------------------------------------------------------------------------------------
//! \fn void BoundaryValues::ApplyPhysicalBoundariesOnCoarseLevel(
//           const NeighborBlock& nb, const Real time, const Real dt,
//           int si, int ei, int sj, int ej, int sk, int ek)
//  \brief

void BoundaryValues::ApplyPhysicalBoundariesOnCoarseLevel(const NeighborBlock &nb,
                                                          const Real time, const Real dt,
                                                          int si, int ei, int sj, int ej,
                                                          int sk, int ek) {
  // TODO(SS)
  // Write code to take a container as input and apply
  // appropriate boundary condiditions
  throw std::runtime_error(std::string(__func__) + " is not implemented");
}

void BoundaryValues::ProlongateGhostCells(const NeighborBlock &nb, int si, int ei, int sj,
                                          int ej, int sk, int ek) {
  MeshBlock *pmb = pmy_block_;
  auto &pmr = pmb->pmr;

  for (auto cc_pair : pmr->pvars_cc_) {
    ParArrayND<Real> var_cc = std::get<0>(cc_pair);
    ParArrayND<Real> coarse_cc = std::get<1>(cc_pair);
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

  // now that the ghost-ghost zones are filled and prolongated,
  // calculate the loop limits for the finer grid
  int fsi, fei, fsj, fej, fsk, fek;

  const IndexDomain interior = IndexDomain::interior;
  IndexRange cib = pmb->c_cellbounds.GetBoundsI(interior);
  IndexRange cjb = pmb->c_cellbounds.GetBoundsJ(interior);
  IndexRange ckb = pmb->c_cellbounds.GetBoundsK(interior);

  fsi = (si - cib.s) * 2 + pmb->cellbounds.is(interior);
  fei = (ei - cib.s) * 2 + pmb->cellbounds.is(interior) + 1;
  if (pmb->block_size.nx2 > 1) {
    fsj = (sj - cjb.s) * 2 + pmb->cellbounds.js(interior);
    fej = (ej - cjb.s) * 2 + pmb->cellbounds.js(interior) + 1;
  } else {
    fsj = pmb->cellbounds.js(interior);
    fej = pmb->cellbounds.je(interior);
  }

  if (pmb->block_size.nx3 > 1) {
    fsk = (sk - ckb.s) * 2 + pmb->cellbounds.ks(interior);
    fek = (ek - ckb.s) * 2 + pmb->cellbounds.ks(interior) + 1;
  } else {
    fsk = pmb->cellbounds.ks(interior);
    fek = pmb->cellbounds.ke(interior);
  }

  // KGF: COUPLING OF QUANTITIES (must be manually specified)
  // Field prolongation completed, calculate cell centered fields
  // TODO(KGF): passing nullptrs (pf) if no MHD (coarse_* no longer in MeshRefinement)
  // (may be fine to unconditionally directly set to pmb->pfield now. see above comment)

  // KGF: COUPLING OF QUANTITIES (must be manually specified)
  // calculate conservative variables
  // pmb->peos->PrimitiveToConserved(ph->w, pf->bcc, ph->u, pmb->pcoord,
  //                                fsi, fei, fsj, fej, fsk, fek);
  return;
}

} // namespace parthenon
