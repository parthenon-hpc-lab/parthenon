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
#include <memory>

#include "bvals/cc/bnd_info.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock.hpp"

namespace parthenon {

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

void BoundaryValues::ProlongateGhostCells_(const NeighborBlock &nb, int si, int ei,
                                           int sj, int ej, int sk, int ek) {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  auto &pmr = pmb->pmr;

  for (auto cc_var : pmr->pvars_cc_) {
    if (!cc_var->IsAllocated()) continue;
    int nu = cc_var->GetDim(4) - 1;
    pmr->ProlongateCellCenteredValues(cc_var.get(), cc_var.get(), 0, nu, si, ei, sj, ej,
                                      sk, ek);
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
