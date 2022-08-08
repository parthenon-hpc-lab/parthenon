//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2022 The Parthenon collaboration
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

#include <algorithm>
#include <iostream> // debug
#include <memory>
#include <string>
#include <vector>

#include "basic_types.hpp"
#include "bvals/bvals_interfaces.hpp"
#include "bvals_cc_in_one.hpp"
#include "config.hpp"
#include "globals.hpp"
#include "interface/variable.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/refinement_cc_in_one.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

namespace cell_centered_bvars {

//----------------------------------------------------------------------------------------
//! \fn void cell_centered_bvars::CalcIndicesSetSame(int ox, int &s, int &e,
//                                                   const IndexRange &bounds)
//  \brief Calculate indices for SetBoundary routines for buffers on the same level

void CalcIndicesSetSame(int ox, int &s, int &e, const IndexRange &bounds) {
  if (ox == 0) {
    s = bounds.s;
    e = bounds.e;
  } else if (ox > 0) {
    s = bounds.e + 1;
    e = bounds.e + Globals::nghost;
  } else {
    s = bounds.s - Globals::nghost;
    e = bounds.s - 1;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void cell_centered_bvars::CalcIndicesSetFomCoarser(const int &ox, int &s, int &e,
//                                                         const IndexRange &bounds,
//                                                         const std::int64_t &lx,
//                                                         const int &cng,
//                                                         const bool include_dim)
//  \brief Calculate indices for SetBoundary routines for buffers from coarser levels

void CalcIndicesSetFromCoarser(const int &ox, int &s, int &e, const IndexRange &bounds,
                               const std::int64_t &lx, const int &cng,
                               const bool include_dim) {
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
}

//----------------------------------------------------------------------------------------
//! \fn void cell_centered_bvars::CalcIndicesSetFromFiner(int &si, int &ei, int &sj,
//                                                        int &ej, int &sk, int &ek,
//                                                        const NeighborBlock &nb,
//                                                        MeshBlock *pmb)
//  \brief Calculate indices for SetBoundary routines for buffers from finer levels

void CalcIndicesSetFromFiner(int &si, int &ei, int &sj, int &ej, int &sk, int &ek,
                             const NeighborBlock &nb, MeshBlock *pmb) {
  IndexDomain interior = IndexDomain::interior;
  const IndexShape &cellbounds = pmb->cellbounds;
  if (nb.ni.ox1 == 0) {
    si = cellbounds.is(interior);
    ei = cellbounds.ie(interior);
    if (nb.ni.fi1 == 1)
      si += pmb->block_size.nx1 / 2;
    else
      ei -= pmb->block_size.nx1 / 2;
  } else if (nb.ni.ox1 > 0) {
    si = cellbounds.ie(interior) + 1;
    ei = cellbounds.ie(interior) + Globals::nghost;
  } else {
    si = cellbounds.is(interior) - Globals::nghost;
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
    ej = cellbounds.je(interior) + Globals::nghost;
  } else {
    sj = cellbounds.js(interior) - Globals::nghost;
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
    ek = cellbounds.ke(interior) + Globals::nghost;
  } else {
    sk = cellbounds.ks(interior) - Globals::nghost;
    ek = cellbounds.ks(interior) - 1;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void cell_centered_bvars::CalcIndicesLoadSame(int ox, int &s, int &e,
//                                                    const IndexRange &bounds)
//  \brief Calculate indices for LoadBoundary routines for buffers on the same level
//         and to coarser.

void CalcIndicesLoadSame(int ox, int &s, int &e, const IndexRange &bounds) {
  if (ox == 0) {
    s = bounds.s;
    e = bounds.e;
  } else if (ox > 0) {
    s = bounds.e - Globals::nghost + 1;
    e = bounds.e;
  } else {
    s = bounds.s;
    e = bounds.s + Globals::nghost - 1;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void cell_centered_bvars::CalcIndicesLoadToFiner(int &si, int &ei, int &sj,
//                                                       int &ej, int &sk, int &ek,
//                                                       const NeighborBlock &nb,
//                                                       MeshBlock *pmb)
//  \brief Calculate indices for LoadBoundary routines for buffers to finer levels

void CalcIndicesLoadToFiner(int &si, int &ei, int &sj, int &ej, int &sk, int &ek,
                            const NeighborBlock &nb, MeshBlock *pmb) {
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
}

int GetBufferSize(std::shared_ptr<MeshBlock> pmb, const NeighborBlock &nb,
                  std::shared_ptr<CellVariable<Real>> v) {
  auto &cb = pmb->cellbounds;
  const IndexDomain in = IndexDomain::interior;
  const int isize = cb.ie(in) - cb.is(in) + 1;
  const int jsize = cb.je(in) - cb.js(in) + 1;
  const int ksize = cb.ke(in) - cb.ks(in) + 1;
  return (nb.ni.ox1 == 0 ? isize : Globals::nghost) *
         (nb.ni.ox2 == 0 ? jsize : Globals::nghost) *
         (nb.ni.ox3 == 0 ? ksize : Globals::nghost) * v->GetDim(6) * v->GetDim(5) *
         v->GetDim(4);
}

BndInfo BndInfo::GetSendBndInfo(std::shared_ptr<MeshBlock> pmb, const NeighborBlock &nb,
                                std::shared_ptr<CellVariable<Real>> v) {
  BndInfo out;
  out.Nv = v->GetDim(4);
  out.Nu = v->GetDim(5);
  out.Nt = v->GetDim(6);

  int mylevel = pmb->loc.level;
  out.coords = pmb->coords;

  if (pmb->pmr) out.coarse_coords = pmb->pmr->GetCoarseCoords();

  out.fine = v->data.Get();
  out.coarse = v->coarse_s.Get();

  IndexDomain interior = IndexDomain::interior;
  if (nb.snb.level == mylevel) {
    const parthenon::IndexShape &cellbounds = pmb->cellbounds;
    CalcIndicesLoadSame(nb.ni.ox1, out.si, out.ei, cellbounds.GetBoundsI(interior));
    CalcIndicesLoadSame(nb.ni.ox2, out.sj, out.ej, cellbounds.GetBoundsJ(interior));
    CalcIndicesLoadSame(nb.ni.ox3, out.sk, out.ek, cellbounds.GetBoundsK(interior));
    out.var = v->data.Get();
  } else if (nb.snb.level < mylevel) {
    // "Same" logic is the same for loading to a coarse buffer, just using
    // c_cellbounds
    const IndexShape &c_cellbounds = pmb->c_cellbounds;
    CalcIndicesLoadSame(nb.ni.ox1, out.si, out.ei, c_cellbounds.GetBoundsI(interior));
    CalcIndicesLoadSame(nb.ni.ox2, out.sj, out.ej, c_cellbounds.GetBoundsJ(interior));
    CalcIndicesLoadSame(nb.ni.ox3, out.sk, out.ek, c_cellbounds.GetBoundsK(interior));
    out.refinement_op = RefinementOp_t::Restriction;
    out.var = v->coarse_s.Get();
  } else {
    CalcIndicesLoadToFiner(out.si, out.ei, out.sj, out.ej, out.sk, out.ek, nb, pmb.get());
    out.var = v->data.Get();
  }
  return out;
}

BndInfo BndInfo::GetSetBndInfo(std::shared_ptr<MeshBlock> pmb, const NeighborBlock &nb,
                               std::shared_ptr<CellVariable<Real>> v) {
  BndInfo out;

  out.Nv = v->GetDim(4);
  out.Nu = v->GetDim(5);
  out.Nt = v->GetDim(6);

  int mylevel = pmb->loc.level;
  IndexDomain interior = IndexDomain::interior;
  if (nb.snb.level == mylevel) {
    const parthenon::IndexShape &cellbounds = pmb->cellbounds;
    CalcIndicesSetSame(nb.ni.ox1, out.si, out.ei, cellbounds.GetBoundsI(interior));
    CalcIndicesSetSame(nb.ni.ox2, out.sj, out.ej, cellbounds.GetBoundsJ(interior));
    CalcIndicesSetSame(nb.ni.ox3, out.sk, out.ek, cellbounds.GetBoundsK(interior));
    out.var = v->data.Get();
  } else if (nb.snb.level < mylevel) {
    const IndexShape &c_cellbounds = pmb->c_cellbounds;
    const auto &cng = pmb->cnghost;
    CalcIndicesSetFromCoarser(nb.ni.ox1, out.si, out.ei,
                              c_cellbounds.GetBoundsI(interior), pmb->loc.lx1, cng, true);
    CalcIndicesSetFromCoarser(nb.ni.ox2, out.sj, out.ej,
                              c_cellbounds.GetBoundsJ(interior), pmb->loc.lx2, cng,
                              pmb->block_size.nx2 > 1);
    CalcIndicesSetFromCoarser(nb.ni.ox3, out.sk, out.ek,
                              c_cellbounds.GetBoundsK(interior), pmb->loc.lx3, cng,
                              pmb->block_size.nx3 > 1);

    out.var = v->coarse_s.Get();
  } else {
    CalcIndicesSetFromFiner(out.si, out.ei, out.sj, out.ej, out.sk, out.ek, nb,
                            pmb.get());
    out.var = v->data.Get();
  }
  return out;
}

} // namespace cell_centered_bvars
} // namespace parthenon
