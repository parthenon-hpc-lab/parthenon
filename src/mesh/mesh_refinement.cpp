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
//! \file mesh_refinement.cpp
//  \brief implements functions for static/adaptive mesh refinement

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "parameter_input.hpp"
#include "parthenon_arrays.hpp"
#include "refinement/refinement.hpp"

namespace parthenon {

//----------------------------------------------------------------------------------------
//! \fn MeshRefinement::MeshRefinement(MeshBlock *pmb, ParameterInput *pin)
//  \brief constructor

MeshRefinement::MeshRefinement(MeshBlock *pmb, ParameterInput *pin)
    : pmy_block_(pmb), deref_count_(0),
      deref_threshold_(pin->GetOrAddInteger("parthenon/mesh", "derefine_count", 10)),
      AMRFlag_(pmb->pmy_mesh->AMRFlag_) {
  // Create coarse mesh object for parent grid

  if (NGHOST % 2) {
    std::stringstream msg;
    msg << "### FATAL ERROR in MeshRefinement constructor" << std::endl
        << "Selected --nghost=" << NGHOST << " is incompatible with mesh refinement.\n"
        << "Reconfigure with an even number of ghost cells " << std::endl;
    ATHENA_ERROR(msg);
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
  MeshBlock *pmb = pmy_block_;
  int si = (csi - pmb->cis) * 2 + pmb->is, ei = (cei - pmb->cis) * 2 + pmb->is + 1;

  // store the restricted data in the prolongation buffer for later use
  if (pmb->block_size.nx3 > 1) { // 3D
    for (int n = sn; n <= en; ++n) {
      for (int ck = csk; ck <= cek; ck++) {
        int k = (ck - pmb->cks) * 2 + pmb->ks;
        for (int cj = csj; cj <= cej; cj++) {
          int j = (cj - pmb->cjs) * 2 + pmb->js;
          for (int ci = csi; ci <= cei; ci++) {
            int i = (ci - pmb->cis) * 2 + pmb->is;
            // KGF: add the off-centered quantities first to preserve FP symmetry
            coarse(n, ck, cj, ci) =
                (((fine(n, k, j, i) + fine(n, k, j + 1, i)) +
                  (fine(n, k, j, i + 1) + fine(n, k, j + 1, i + 1))) +
                 ((fine(n, k + 1, j, i) + fine(n, k + 1, j + 1, i)) +
                  (fine(n, k + 1, j, i + 1) + fine(n, k + 1, j + 1, i + 1)))) / 8.0;
          }
        }
      }
    }
  } else if (pmb->block_size.nx2 > 1) { // 2D
    for (int n = sn; n <= en; ++n) {
      for (int cj = csj; cj <= cej; cj++) {
        int j = (cj - pmb->cjs) * 2 + pmb->js;
        for (int ci = csi; ci <= cei; ci++) {
          int i = (ci - pmb->cis) * 2 + pmb->is;
          // KGF: add the off-centered quantities first to preserve FP symmetry
          coarse(n, 0, cj, ci) = ((fine(n, 0, j, i) + fine(n, 0, j + 1, i)) +
                                  (fine(n, 0, j, i + 1) + fine(n, 0, j + 1, i + 1)))
                                  /4.0;
        }
      }
    }
  } else { // 1D
    int j = pmb->js, cj = pmb->cjs, k = pmb->ks, ck = pmb->cks;
    for (int n = sn; n <= en; ++n) {
      for (int ci = csi; ci <= cei; ci++) {
        int i = (ci - pmb->cis) * 2 + pmb->is;
        coarse(n, ck, cj, ci) = (fine(n, k, j, i) + fine(n, k, j, i + 1)) / 2.0;
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::RestrictFieldX1(const ParArrayND<Real> &fine
//      ParArrayND<Real> &coarse, int csi, int cei, int csj, int cej, int csk, int cek)
//  \brief restrict the x1 field data and set them into the coarse buffer

void MeshRefinement::RestrictFieldX1(const ParArrayND<Real> &fine,
                                     ParArrayND<Real> &coarse, int csi, int cei, int csj,
                                     int cej, int csk, int cek) {
  MeshBlock *pmb = pmy_block_;
  int si = (csi - pmb->cis) * 2 + pmb->is, ei = (cei - pmb->cis) * 2 + pmb->is;

  // store the restricted data in the prolongation buffer for later use
  if (pmb->block_size.nx3 > 1) { // 3D
    for (int ck = csk; ck <= cek; ck++) {
      int k = (ck - pmb->cks) * 2 + pmb->ks;
      for (int cj = csj; cj <= cej; cj++) {
        int j = (cj - pmb->cjs) * 2 + pmb->js;
        for (int ci = csi; ci <= cei; ci++) {
          int i = (ci - pmb->cis) * 2 + pmb->is;
          coarse(ck, cj, ci) = (fine(k, j, i) + fine(k, j + 1, i) +
                                fine(k + 1, j, i) + fine(k + 1, j + 1, i)) / 4.0;
        }
      }
    }
  } else if (pmb->block_size.nx2 > 1) { // 2D
    int k = pmb->ks;
    for (int cj = csj; cj <= cej; cj++) {
      int j = (cj - pmb->cjs) * 2 + pmb->js;
      for (int ci = csi; ci <= cei; ci++) {
        int i = (ci - pmb->cis) * 2 + pmb->is;
        coarse(csk, cj, ci) = (fine(k, j, i) + fine(k, j + 1, i)) / 2.0;
      }
    }
  } else { // 1D - no restriction, just copy
    for (int ci = csi; ci <= cei; ci++) {
      int i = (ci - pmb->cis) * 2 + pmb->is;
      coarse(csk, csj, ci) = fine(pmb->ks, pmb->js, i);
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
  MeshBlock *pmb = pmy_block_;
  int si = (csi - pmb->cis) * 2 + pmb->is, ei = (cei - pmb->cis) * 2 + pmb->is + 1;

  // store the restricted data in the prolongation buffer for later use
  if (pmb->block_size.nx3 > 1) { // 3D
    for (int ck = csk; ck <= cek; ck++) {
      int k = (ck - pmb->cks) * 2 + pmb->ks;
      for (int cj = csj; cj <= cej; cj++) {
        int j = (cj - pmb->cjs) * 2 + pmb->js;
        for (int ci = csi; ci <= cei; ci++) {
          int i = (ci - pmb->cis) * 2 + pmb->is;
          coarse(ck, cj, ci) = (fine(k, j, i) + fine(k, j, i + 1) +
                                fine(k + 1, j, i) + fine(k + 1, j, i + 1)) / 4.0;
        }
      }
    }
  } else if (pmb->block_size.nx2 > 1) { // 2D
    int k = pmb->ks;
    for (int cj = csj; cj <= cej; cj++) {
      int j = (cj - pmb->cjs) * 2 + pmb->js;
      for (int ci = csi; ci <= cei; ci++) {
        int i = (ci - pmb->cis) * 2 + pmb->is;
        coarse(pmb->cks, cj, ci) = (fine(k, j, i) + fine(k, j, i + 1)) / 2.0;
      }
    }
  } else { // 1D
    int k = pmb->ks, j = pmb->js;
    for (int ci = csi; ci <= cei; ci++) {
      int i = (ci - pmb->cis) * 2 + pmb->is;
      coarse(pmb->cks, pmb->cjs, ci) = (fine(k, j, i) + fine(k, j, i + 1)) / 2.0;
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
  MeshBlock *pmb = pmy_block_;
  int si = (csi - pmb->cis) * 2 + pmb->is, ei = (cei - pmb->cis) * 2 + pmb->is + 1;

  // store the restricted data in the prolongation buffer for later use
  if (pmb->block_size.nx3 > 1) { // 3D
    for (int ck = csk; ck <= cek; ck++) {
      int k = (ck - pmb->cks) * 2 + pmb->ks;
      for (int cj = csj; cj <= cej; cj++) {
        int j = (cj - pmb->cjs) * 2 + pmb->js;
        for (int ci = csi; ci <= cei; ci++) {
          int i = (ci - pmb->cis) * 2 + pmb->is;
          coarse(ck, cj, ci) = (fine(k, j, i) + fine(k, j, i + 1) +
                                fine(k, j + 1, i) + fine(k, j + 1, i + 1)) / 4.0;
        }
      }
    }
  } else if (pmb->block_size.nx2 > 1) { // 2D
    int k = pmb->ks;
    for (int cj = csj; cj <= cej; cj++) {
      int j = (cj - pmb->cjs) * 2 + pmb->js;
      for (int ci = csi; ci <= cei; ci++) {
        int i = (ci - pmb->cis) * 2 + pmb->is;
        coarse(pmb->cks, cj, ci) = (fine(k, j, i) + fine(k, j, i + 1) +
                                    fine(k, j + 1, i) + fine(k, j + 1, i + 1)) / 4.0;
      }
    }
  } else { // 1D
    int k = pmb->ks, j = pmb->js;
    for (int ci = csi; ci <= cei; ci++) {
      int i = (ci - pmb->cis) * 2 + pmb->is;
      coarse(pmb->cks, pmb->cjs, ci) = (fine(k, j, i) + fine(k, j, i + 1)) / 2.0;
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
  MeshBlock *pmb = pmy_block_;
  if (pmb->block_size.nx3 > 1) {
    for (int n = sn; n <= en; n++) {
      for (int k = sk; k <= ek; k++) {
        int fk = (k - pmb->cks) * 2 + pmb->ks;
        for (int j = sj; j <= ej; j++) {
          int fj = (j - pmb->cjs) * 2 + pmb->js;
          for (int i = si; i <= ei; i++) {
            int fi = (i - pmb->cis) * 2 + pmb->is;
            Real ccval = coarse(n, k, j, i);

            // calculate 3D gradients using the minmod limiter
            // TODO(jcd): confirm 0.25 factor is correct
            Real gx1m = (ccval - coarse(n, k, j, i - 1));
            Real gx1p = (coarse(n, k, j, i + 1) - ccval);
            Real gx1c = 0.25*0.5 * (SIGN(gx1m) + SIGN(gx1p)) *
                        std::min(std::abs(gx1m), std::abs(gx1p));
            Real gx2m = (ccval - coarse(n, k, j - 1, i));
            Real gx2p = (coarse(n, k, j + 1, i) - ccval);
            Real gx2c = 0.25*0.5 * (SIGN(gx2m) + SIGN(gx2p)) *
                        std::min(std::abs(gx2m), std::abs(gx2p));
            Real gx3m = (ccval - coarse(n, k - 1, j, i));
            Real gx3p = (coarse(n, k + 1, j, i) - ccval);
            Real gx3c = 0.25*0.5 * (SIGN(gx3m) + SIGN(gx3p)) *
                        std::min(std::abs(gx3m), std::abs(gx3p));

            // KGF: add the off-centered quantities first to preserve FP symmetry
            // interpolate onto the finer grid
            fine(n, fk, fj, fi) = ccval - (gx1c + gx2c + gx3c);
            fine(n, fk, fj, fi + 1) = ccval + (gx1c - gx2c - gx3c);
            fine(n, fk, fj + 1, fi) = ccval - (gx1c - gx2c + gx3c);
            fine(n, fk, fj + 1, fi + 1) = ccval + (gx1c + gx2c - gx3c);
            fine(n, fk + 1, fj, fi) = ccval - (gx1c + gx2c - gx3c);
            fine(n, fk + 1, fj, fi + 1) = ccval + (gx1c - gx2c + gx3c);
            fine(n, fk + 1, fj + 1, fi) = ccval - (gx1c - gx2c - gx3c);
            fine(n, fk + 1, fj + 1, fi + 1) = ccval + (gx1c + gx2c + gx3c);
          }
        }
      }
    }
  } else if (pmb->block_size.nx2 > 1) {
    int k = pmb->cks, fk = pmb->ks;
    for (int n = sn; n <= en; n++) {
      for (int j = sj; j <= ej; j++) {
        int fj = (j - pmb->cjs) * 2 + pmb->js;
        for (int i = si; i <= ei; i++) {
          int fi = (i - pmb->cis) * 2 + pmb->is;
          Real ccval = coarse(n, k, j, i);

          // calculate 2D gradients using the minmod limiter
          Real gx1m = (ccval - coarse(n, k, j, i - 1));
          Real gx1p = (coarse(n, k, j, i + 1) - ccval);
          Real gx1c = 0.25*0.5 * (SIGN(gx1m) + SIGN(gx1p))
                        * std::min(std::abs(gx1m), std::abs(gx1p));
          Real gx2m = (ccval - coarse(n, k, j - 1, i));
          Real gx2p = (coarse(n, k, j + 1, i) - ccval);
          Real gx2c = 0.25*0.5 * (SIGN(gx2m) + SIGN(gx2p))
                        * std::min(std::abs(gx2m), std::abs(gx2p));

          // interpolate onto the finer grid
          fine(n, fk, fj, fi) = ccval - (gx1c + gx2c);
          fine(n, fk, fj, fi + 1) = ccval + (gx1c - gx2c);
          fine(n, fk, fj + 1, fi) = ccval - (gx1c - gx2c);
          fine(n, fk, fj + 1, fi + 1) = ccval + (gx1c + gx2c);
        }
      }
    }
  } else { // 1D
    int k = pmb->cks, fk = pmb->ks, j = pmb->cjs, fj = pmb->js;
    for (int n = sn; n <= en; n++) {
      for (int i = si; i <= ei; i++) {
        int fi = (i - pmb->cis) * 2 + pmb->is;
        Real ccval = coarse(n, k, j, i);

        // calculate 1D gradient using the min-mod limiter
        Real gx1m = (ccval - coarse(n, k, j, i - 1));
        Real gx1p = (coarse(n, k, j, i + 1) - ccval);
        Real gx1c = 0.25*0.5 * (SIGN(gx1m) + SIGN(gx1p))
                      * std::min(std::abs(gx1m), std::abs(gx1p));

        // interpolate on to the finer grid
        fine(n, fk, fj, fi) = ccval - gx1c;
        fine(n, fk, fj, fi + 1) = ccval + gx1c;
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::ProlongateSharedFieldX1(const ParArrayND<Real> &coarse,
//      ParArrayND<Real> &fine, int si, int ei, int sj, int ej, int sk, int ek)
//  \brief prolongate x1 face-centered fields shared between coarse and fine levels

void MeshRefinement::ProlongateSharedFieldX1(const ParArrayND<Real> &coarse,
                                             ParArrayND<Real> &fine, int si, int ei,
                                             int sj, int ej, int sk, int ek) {
  MeshBlock *pmb = pmy_block_;
  if (pmb->block_size.nx3 > 1) {
    for (int k = sk; k <= ek; k++) {
      int fk = (k - pmb->cks) * 2 + pmb->ks;
      for (int j = sj; j <= ej; j++) {
        int fj = (j - pmb->cjs) * 2 + pmb->js;
        for (int i = si; i <= ei; i++) {
          int fi = (i - pmb->cis) * 2 + pmb->is;
          Real ccval = coarse(k, j, i);

          Real gx2m = (ccval - coarse(k, j - 1, i));
          Real gx2p = (coarse(k, j + 1, i) - ccval);
          Real gx2c = 0.25*0.5 * (SIGN(gx2m) + SIGN(gx2p))
                        * std::min(std::abs(gx2m), std::abs(gx2p));
          Real gx3m = (ccval - coarse(k - 1, j, i));
          Real gx3p = (coarse(k + 1, j, i) - ccval);
          Real gx3c = 0.25*0.5 * (SIGN(gx3m) + SIGN(gx3p))
                        * std::min(std::abs(gx3m), std::abs(gx3p));

          fine(fk, fj, fi) = ccval - gx2c - gx3c;
          fine(fk, fj + 1, fi) = ccval + gx2c - gx3c;
          fine(fk + 1, fj, fi) = ccval - gx2c + gx3c;
          fine(fk + 1, fj + 1, fi) = ccval + gx2c + gx3c;
        }
      }
    }
  } else if (pmb->block_size.nx2 > 1) {
    int k = pmb->cks, fk = pmb->ks;
    for (int j = sj; j <= ej; j++) {
      int fj = (j - pmb->cjs) * 2 + pmb->js;
      for (int i = si; i <= ei; i++) {
        int fi = (i - pmb->cis) * 2 + pmb->is;
        Real ccval = coarse(k, j, i);

        Real gx2m = (ccval - coarse(k, j - 1, i));
        Real gx2p = (coarse(k, j + 1, i) - ccval);
        Real gx2c = 0.25*0.5 * (SIGN(gx2m) + SIGN(gx2p))
                      * std::min(std::abs(gx2m), std::abs(gx2p));

        fine(fk, fj, fi) = ccval - gx2c;
        fine(fk, fj + 1, fi) = ccval + gx2c;
      }
    }
  } else { // 1D
    for (int i = si; i <= ei; i++) {
      int fi = (i - pmb->cis) * 2 + pmb->is;
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
  MeshBlock *pmb = pmy_block_;
  if (pmb->block_size.nx3 > 1) {
    for (int k = sk; k <= ek; k++) {
      int fk = (k - pmb->cks) * 2 + pmb->ks;
      for (int j = sj; j <= ej; j++) {
        int fj = (j - pmb->cjs) * 2 + pmb->js;
        for (int i = si; i <= ei; i++) {
          int fi = (i - pmb->cis) * 2 + pmb->is;
          Real ccval = coarse(k, j, i);

          Real gx1m = (ccval - coarse(k, j, i - 1));
          Real gx1p = (coarse(k, j, i + 1) - ccval);
          Real gx1c = 0.25*0.5 * (SIGN(gx1m) + SIGN(gx1p))
                        * std::min(std::abs(gx1m), std::abs(gx1p));
          Real gx3m = (ccval - coarse(k - 1, j, i));
          Real gx3p = (coarse(k + 1, j, i) - ccval);
          Real gx3c = 0.25*0.5 * (SIGN(gx3m) + SIGN(gx3p))
                        * std::min(std::abs(gx3m), std::abs(gx3p));

          fine(fk, fj, fi) = ccval - gx1c - gx3c;
          fine(fk, fj, fi + 1) = ccval + gx1c - gx3c;
          fine(fk + 1, fj, fi) = ccval - gx1c + gx3c;
          fine(fk + 1, fj, fi + 1) = ccval + gx1c + gx3c;
        }
      }
    }
  } else if (pmb->block_size.nx2 > 1) {
    int k = pmb->cks, fk = pmb->ks;
    for (int j = sj; j <= ej; j++) {
      int fj = (j - pmb->cjs) * 2 + pmb->js;
      for (int i = si; i <= ei; i++) {
        int fi = (i - pmb->cis) * 2 + pmb->is;
        Real ccval = coarse(k, j, i);

        Real gx1m = (ccval - coarse(k, j, i - 1));
        Real gx1p = (coarse(k, j, i + 1) - ccval);
        Real gx1c = 0.25*0.5 * (SIGN(gx1m) + SIGN(gx1p))
                      * std::min(std::abs(gx1m), std::abs(gx1p));

        fine(fk, fj, fi) = ccval - gx1c;
        fine(fk, fj, fi + 1) = ccval + gx1c;
      }
    }
  } else {
    for (int i = si; i <= ei; i++) {
      int fi = (i - pmb->cis) * 2 + pmb->is;
      Real gxm = coarse(0, 0, i) - coarse(0, 0, i - 1);
      Real gxp = coarse(0, 0, i + 1) - coarse(0, 0, i);
      Real gxc = 0.25*0.5 * (SIGN(gxm) + SIGN(gxp))
                    * std::min(std::abs(gxm), std::abs(gxp));
      fine(0, 0, fi) = fine(0, 1, fi) = coarse(0, 0, i) - gxc;
      fine(0, 0, fi + 1) = fine(0, 1, fi + 1) = coarse(0, 0, i) + gxc;
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
  MeshBlock *pmb = pmy_block_;
  if (pmb->block_size.nx3 > 1) {
    for (int k = sk; k <= ek; k++) {
      int fk = (k - pmb->cks) * 2 + pmb->ks;
      for (int j = sj; j <= ej; j++) {
        int fj = (j - pmb->cjs) * 2 + pmb->js;
        for (int i = si; i <= ei; i++) {
          int fi = (i - pmb->cis) * 2 + pmb->is;
          Real ccval = coarse(k, j, i);

          Real gx1m = (ccval - coarse(k, j, i - 1));
          Real gx1p = (coarse(k, j, i + 1) - ccval);
          Real gx1c = 0.25*0.5 * (SIGN(gx1m) + SIGN(gx1p))
                        * std::min(std::abs(gx1m), std::abs(gx1p));
          Real gx2m = (ccval - coarse(k, j - 1, i));
          Real gx2p = (coarse(k, j + 1, i) - ccval);
          Real gx2c = 0.25*0.5 * (SIGN(gx2m) + SIGN(gx2p))
                        * std::min(std::abs(gx2m), std::abs(gx2p));

          fine(fk, fj, fi) = ccval - gx1c - gx2c;
          fine(fk, fj, fi + 1) = ccval + gx1c - gx2c;
          fine(fk, fj + 1, fi) = ccval - gx1c + gx2c;
          fine(fk, fj + 1, fi + 1) = ccval + gx1c + gx2c;
        }
      }
    }
  } else if (pmb->block_size.nx2 > 1) {
    int k = pmb->cks, fk = pmb->ks;
    for (int j = sj; j <= ej; j++) {
      int fj = (j - pmb->cjs) * 2 + pmb->js;
      for (int i = si; i <= ei; i++) {
        int fi = (i - pmb->cis) * 2 + pmb->is;
        Real ccval = coarse(k, j, i);

        // calculate 2D gradients using the minmod limiter
        Real gx1m = (ccval - coarse(k, j, i - 1));
        Real gx1p = (coarse(k, j, i + 1) - ccval);
        Real gx1c = 0.25*0.5 * (SIGN(gx1m) + SIGN(gx1p))
                      * std::min(std::abs(gx1m), std::abs(gx1p));
        Real gx2m = (ccval - coarse(k, j - 1, i));
        Real gx2p = (coarse(k, j + 1, i) - ccval);
        Real gx2c = 0.25*0.5 * (SIGN(gx2m) + SIGN(gx2p))
                      * std::min(std::abs(gx2m), std::abs(gx2p));

        // interpolate on to the finer grid
        fine(fk, fj, fi) = fine(fk + 1, fj, fi) = ccval - gx1c - gx2c;
        fine(fk, fj, fi + 1) = fine(fk + 1, fj, fi + 1) = ccval + gx1c - gx2c;
        fine(fk, fj + 1, fi) = fine(fk + 1, fj + 1, fi) = ccval - gx1c + gx2c;
        fine(fk, fj + 1, fi + 1) = fine(fk + 1, fj + 1, fi + 1) = ccval + gx1c + gx2c;
      }
    }
  } else {
    for (int i = si; i <= ei; i++) {
      int fi = (i - pmb->cis) * 2 + pmb->is;
      Real gxm = coarse(0, 0, i) - coarse(0, 0, i - 1);
      Real gxp = coarse(0, 0, i + 1) - coarse(0, 0, i);
      Real gxc = 0.25*0.5 * (SIGN(gxm) + SIGN(gxp))
                    * std::min(std::abs(gxm), std::abs(gxp));
      fine(0, 0, fi) = fine(1, 0, fi) = coarse(0, 0, i) - gxc;
      fine(0, 0, fi + 1) = fine(1, 0, fi + 1) = coarse(0, 0, i) + gxc;
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
  MeshBlock *pmb = pmy_block_;
  auto dx = pmb->GetDx();
  auto area = pmb->GetArea();
  int fsi = (si - pmb->cis) * 2 + pmb->is, fei = (ei - pmb->cis) * 2 + pmb->is + 1;
  if (pmb->block_size.nx3 > 1) {
    for (int k = sk; k <= ek; k++) {
      int fk = (k - pmb->cks) * 2 + pmb->ks;
      for (int j = sj; j <= ej; j++) {
        int fj = (j - pmb->cjs) * 2 + pmb->js;
        for (int i = si; i <= ei; i++) {
          int fi = (i - pmb->cis) * 2 + pmb->is;
          Real Uxx = 0.0, Vyy = 0.0, Wzz = 0.0;
          Real Uxyz = 0.0, Vxyz = 0.0, Wxyz = 0.0;
#pragma unroll
          for (int jj = 0; jj < 2; jj++) {
            int js = 2 * jj - 1, fjj = fj + jj, fjp = fj + 2 * jj;
#pragma unroll
            for (int ii = 0; ii < 2; ii++) {
              int is = 2 * ii - 1, fii = fi + ii, fip = fi + 2 * ii;
              Uxx += is * (js * (fine.x2f(fk, fjp, fii) * area[1] +
                                 fine.x2f(fk + 1, fjp, fii) * area[1]) +
                           (fine.x3f(fk + 2, fjj, fii) * area[2] -
                            fine.x3f(fk, fjj, fii) * area[2]));
              Vyy += js * ((fine.x3f(fk + 2, fjj, fii) * area[2] -
                            fine.x3f(fk, fjj, fii) * area[2]) +
                           is * (fine.x1f(fk, fjj, fip) * area[0] +
                                 fine.x1f(fk + 1, fjj, fip) * area[0]));
              Wzz += is * (fine.x1f(fk + 1, fjj, fip) * area[0] -
                           fine.x1f(fk, fjj, fip) * area[0]) +
                     js * (fine.x2f(fk + 1, fjp, fii) * area[1] -
                           fine.x2f(fk, fjp, fii) * area[1]);
              Uxyz += is * js *
                      (fine.x1f(fk + 1, fjj, fip) * area[0] -
                       fine.x1f(fk, fjj, fip) * area[0]);
              Vxyz += is * js *
                      (fine.x2f(fk + 1, fjp, fii) * area[1] -
                       fine.x2f(fk, fjp, fii) * area[1]);
              Wxyz += is * js *
                      (fine.x3f(fk + 2, fjj, fii) * area[2] -
                       fine.x3f(fk, fjj, fii) * area[2]);
            }
          }
          Real Sdx1 = 4.0*dx[0]*dx[0];
          Real Sdx2 = 4.0*dx[1]*dx[1];
          Real Sdx3 = 4.0*dx[2]*dx[2];

          Uxx *= 0.125;
          Vyy *= 0.125;
          Wzz *= 0.125;
          Uxyz *= 0.125 / (Sdx2 + Sdx3);
          Vxyz *= 0.125 / (Sdx1 + Sdx3);
          Wxyz *= 0.125 / (Sdx1 + Sdx2);
          fine.x1f(fk, fj, fi + 1) =
              (0.5 * area[0] * (fine.x1f(fk, fj, fi) +
                      fine.x1f(fk, fj, fi + 2)) +
               Uxx - Sdx3 * Vxyz - Sdx2 * Wxyz) /
              area[0];
          fine.x1f(fk, fj + 1, fi + 1) =
              (0.5 * area[0] * (fine.x1f(fk, fj + 1, fi) +
                      fine.x1f(fk, fj + 1, fi + 2)) +
               Uxx - Sdx3 * Vxyz + Sdx2 * Wxyz) /
              area[0];
          fine.x1f(fk + 1, fj, fi + 1) =
              (0.5 * area[0] * (fine.x1f(fk + 1, fj, fi) +
                      fine.x1f(fk + 1, fj, fi + 2)) +
               Uxx + Sdx3 * Vxyz - Sdx2 * Wxyz) /
              area[0];
          fine.x1f(fk + 1, fj + 1, fi + 1) =
              (0.5 * area[0] * (fine.x1f(fk + 1, fj + 1, fi) +
                      fine.x1f(fk + 1, fj + 1, fi + 2)) +
               Uxx + Sdx3 * Vxyz + Sdx2 * Wxyz) /
              area[0];

          fine.x2f(fk, fj + 1, fi) =
              (0.5 * area[1] * (fine.x2f(fk, fj, fi) +
                      fine.x2f(fk, fj + 2, fi)) +
               Vyy - Sdx3 * Uxyz - Sdx1 * Wxyz) /
              area[1];
          fine.x2f(fk, fj + 1, fi + 1) =
              (0.5 * area[1] * (fine.x2f(fk, fj, fi + 1) +
                      fine.x2f(fk, fj + 2, fi + 1)) +
               Vyy - Sdx3 * Uxyz + Sdx1 * Wxyz) /
              area[1];
          fine.x2f(fk + 1, fj + 1, fi) =
              (0.5 * area[1] * (fine.x2f(fk + 1, fj, fi) +
                      fine.x2f(fk + 1, fj + 2, fi)) +
               Vyy + Sdx3 * Uxyz - Sdx1 * Wxyz) /
              area[1];
          fine.x2f(fk + 1, fj + 1, fi + 1) =
              (0.5 * area[1] * (fine.x2f(fk + 1, fj, fi + 1) +
                      fine.x2f(fk + 1, fj + 2, fi + 1)) +
               Vyy + Sdx3 * Uxyz + Sdx1 * Wxyz) /
              area[1];

          fine.x3f(fk + 1, fj, fi) =
              (0.5 * area[2] * (fine.x3f(fk + 2, fj, fi) +
                      fine.x3f(fk, fj, fi)) +
               Wzz - Sdx2 * Uxyz - Sdx1 * Vxyz) /
              area[2];
          fine.x3f(fk + 1, fj, fi + 1) =
              (0.5 * area[2] * (fine.x3f(fk + 2, fj, fi + 1) +
                      fine.x3f(fk, fj, fi + 1)) +
               Wzz - Sdx2 * Uxyz + Sdx1 * Vxyz) /
              area[2];
          fine.x3f(fk + 1, fj + 1, fi) =
              (0.5 * area[2] * (fine.x3f(fk + 2, fj + 1, fi) +
                      fine.x3f(fk, fj + 1, fi)) +
               Wzz + Sdx2 * Uxyz - Sdx1 * Vxyz) /
              area[2];
          fine.x3f(fk + 1, fj + 1, fi + 1) =
              (0.5 * area[2] * (fine.x3f(fk + 2, fj + 1, fi + 1) +
                      fine.x3f(fk, fj + 1, fi + 1)) +
               Wzz + Sdx2 * Uxyz + Sdx1 * Vxyz) /
              area[2];
        }
      }
    }
  } else if (pmb->block_size.nx2 > 1) {
    int fk = pmb->ks;
    for (int j = sj; j <= ej; j++) {
      int fj = (j - pmb->cjs) * 2 + pmb->js;
      for (int i = si; i <= ei; i++) {
        int fi = (i - pmb->cis) * 2 + pmb->is;
        Real tmp1 = 0.25 * (fine.x2f(fk, fj + 2, fi + 1) * area[1] -
                            fine.x2f(fk, fj, fi + 1) * area[1] -
                            fine.x2f(fk, fj + 2, fi) * area[1] +
                            fine.x2f(fk, fj, fi) * area[1]);
        Real tmp2 = 0.25 * (fine.x1f(fk, fj, fi) * area[0] -
                            fine.x1f(fk, fj, fi + 2) * area[0] -
                            fine.x1f(fk, fj + 1, fi) * area[0] +
                            fine.x1f(fk, fj + 1, fi + 2) * area[0]);
        fine.x1f(fk, fj, fi + 1) =
            (0.5 * area[0] * (fine.x1f(fk, fj, fi) +
                    fine.x1f(fk, fj, fi + 2)) +
             tmp1) /
            area[0];
        fine.x1f(fk, fj + 1, fi + 1) =
            (0.5 * area[0] * (fine.x1f(fk, fj + 1, fi) +
                    fine.x1f(fk, fj + 1, fi + 2)) +
             tmp1) /
            area[0];
        fine.x2f(fk, fj + 1, fi) =
            (0.5 * area[1] * (fine.x2f(fk, fj, fi) +
                    fine.x2f(fk, fj + 2, fi)) +
             tmp2) /
            area[1];
        fine.x2f(fk, fj + 1, fi + 1) =
            (0.5 * area[1] * (fine.x2f(fk, fj, fi + 1) +
                    fine.x2f(fk, fj + 2, fi + 1)) +
             tmp2) /
            area[1];
      }
    }
  } else {
    for (int i = si; i <= ei; i++) {
      int fi = (i - pmb->cis) * 2 + pmb->is;
      fine.x1f(0, 0, fi + 1) = fine.x1f(0, 0, fi);
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::CheckRefinementCondition()
//  \brief Check refinement criteria

void MeshRefinement::CheckRefinementCondition() {
  MeshBlock *pmb = pmy_block_;
  Container<Real> &rc = pmb->real_containers.Get();
  AmrTag ret = Refinement::CheckAllRefinement(rc);
  // if (AMRFlag_ != nullptr) ret = AMRFlag_(pmb);
  SetRefinement(ret);
}

void MeshRefinement::SetRefinement(AmrTag flag) {
  MeshBlock *pmb = pmy_block_;
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

int MeshRefinement::AddToRefinement(ParArrayND<Real> pvar_cc,
                                    ParArrayND<Real> pcoarse_cc) {
  pvars_cc_.push_back(std::make_tuple(pvar_cc, pcoarse_cc));
  return static_cast<int>(pvars_cc_.size() - 1);
}

int MeshRefinement::AddToRefinement(FaceField *pvar_fc, FaceField *pcoarse_fc) {
  pvars_fc_.push_back(std::make_tuple(pvar_fc, pcoarse_fc));
  return static_cast<int>(pvars_fc_.size() - 1);
}

} // namespace parthenon
