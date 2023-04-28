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
#include "bvals/bnd_flux_communication/bnd_info.hpp"
#include "config.hpp"
#include "globals.hpp"
#include "interface/variable.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock.hpp"
#include "prolong_restrict/prolong_restrict.hpp"
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

void ComputeRestrictionBounds(IndexRange &ni, IndexRange &nj, IndexRange &nk,
                              const NeighborBlock &nb,
                              const std::shared_ptr<MeshBlock> &pmb) {
  auto getbounds = [](const int nbx, IndexRange &n) {
    n.s = std::max(nbx - 1, -1); // can be -1 or 0
    n.e = std::min(nbx + 1, 1);  // can be 0 or 1
  };
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

// JMM: Finds the pieces of the coarse buffer, both interior and in
// ghost halo, needed to be restricted to enable prolongation. This is
// both the boundary buffer itself, and the regions *around* it.
//
// Here nk, nj, ni are offset indices. They indicate offsets from this
// piece of the ghost halo to other pieces that may be relevant for
// getting physical boundary conditions right.
// They point to other neighbor blocks.
// ris, rie, rjs, rje, rks, rke are the start and end i,j,k, indices
// of the region of the ghost halo to restrict.
void CalcIndicesRestrict(int nk, int nj, int ni, int &ris, int &rie, int &rjs, int &rje,
                         int &rks, int &rke, const NeighborBlock &nb,
                         std::shared_ptr<MeshBlock> &pmb) {
  const IndexDomain interior = IndexDomain::interior;
  IndexRange cib = pmb->c_cellbounds.GetBoundsI(interior);
  IndexRange cjb = pmb->c_cellbounds.GetBoundsJ(interior);
  IndexRange ckb = pmb->c_cellbounds.GetBoundsK(interior);

  // JMM: rs and re are the bounds of the region to restrict
  // n is the offset index from this boundary/ghost halo to other
  // regions of the coarse buffer
  // ox is the offset index of the neighbor block this boundary/ghost
  // halo communicates with.
  // note this func is called *per axis* so interior here might still
  // be an edge or corner.
  auto CalcIndices = [](int &rs, int &re, int n, int ox, const IndexRange &b) {
    if (n == 0) { // need to fill "interior" of coarse buffer on this axis
      rs = b.s;
      re = b.e;
      if (ox == 1) {
        rs = b.e;
      } else if (ox == -1) {
        re = b.s;
      }
    } else if (n == 1) { // need to fill "edges" or "corners" on this axis
      rs = b.e + 1;      // TODO(JMM): Is this always true?
      re = b.e + 1;      // should this end at b.e + NG - 1?
    } else {             //(n ==  - 1)
      rs = b.s - 1;      // TODO(JMM): should this start at b.s - NG + 1?
      re = b.s - 1;      // or something similar?
    }
  };

  CalcIndices(ris, rie, ni, nb.ni.ox1, cib);
  CalcIndices(rjs, rje, nj, nb.ni.ox2, cjb);
  CalcIndices(rks, rke, nk, nb.ni.ox3, ckb);
}

int GetBufferSize(std::shared_ptr<MeshBlock> pmb, const NeighborBlock &nb,
                  std::shared_ptr<Variable<Real>> v) {
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
                                std::shared_ptr<Variable<Real>> v,
                                CommBuffer<buf_pool_t<Real>::owner_t> *buf,
                                const OffsetIndices &) {
  BndInfo out;

  out.allocated = v->IsAllocated();
  if (!out.allocated) return out;

  out.buf = buf->buffer();

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
                               std::shared_ptr<Variable<Real>> v,
                               CommBuffer<buf_pool_t<Real>::owner_t> *buf,
                               const OffsetIndices &) {
  BndInfo out;
  out.buf = buf->buffer();
  auto buf_state = buf->GetState();
  if (buf_state == BufferState::received) {
    out.allocated = true;
    PARTHENON_DEBUG_REQUIRE(v->IsAllocated(), "Variable must be allocated to receive");
  } else if (buf_state == BufferState::received_null) {
    out.allocated = false;
  } else {
    PARTHENON_FAIL("Buffer should be in a received state.");
  }

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

  if (buf_state == BufferState::received) {
    // With control variables, we can end up in a state where a
    // variable that is not receiving null data is unallocated.
    // for allocated to be set, the buffer must be sending non-null
    // data and the receiving variable must be allocated
    out.allocated = v->IsAllocated();
  } else if (buf_state == BufferState::received_null) {
    out.allocated = false;
  } else {
    PARTHENON_FAIL("Buffer should be in a received state.");
  }
  return out;
}

BndInfo BndInfo::GetSendCCFluxCor(std::shared_ptr<MeshBlock> pmb, const NeighborBlock &nb,
                                  std::shared_ptr<Variable<Real>> v,
                                  CommBuffer<buf_pool_t<Real>::owner_t> *buf,
                                  const OffsetIndices &) {
  BndInfo out;
  out.allocated = v->IsAllocated();
  if (!v->IsAllocated()) {
    // Not going to actually do anything with this buffer
    return out;
  }
  out.buf = buf->buffer();

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // This is the index range for the coarse field
  out.sk = kb.s;
  out.ek = out.sk + std::max((kb.e - kb.s + 1) / 2, 1) - 1;
  out.sj = jb.s;
  out.ej = out.sj + std::max((jb.e - jb.s + 1) / 2, 1) - 1;
  out.si = ib.s;
  out.ei = out.si + std::max((ib.e - ib.s + 1) / 2, 1) - 1;

  if (nb.fid == BoundaryFace::inner_x1 || nb.fid == BoundaryFace::outer_x1) {
    out.dir = X1DIR;
    if (nb.fid == BoundaryFace::inner_x1)
      out.si = ib.s;
    else
      out.si = ib.e + 1;
    out.ei = out.si;
  } else if (nb.fid == BoundaryFace::inner_x2 || nb.fid == BoundaryFace::outer_x2) {
    out.dir = X2DIR;
    if (nb.fid == BoundaryFace::inner_x2)
      out.sj = jb.s;
    else
      out.sj = jb.e + 1;
    out.ej = out.sj;
  } else if (nb.fid == BoundaryFace::inner_x3 || nb.fid == BoundaryFace::outer_x3) {
    out.dir = X3DIR;
    if (nb.fid == BoundaryFace::inner_x3)
      out.sk = kb.s;
    else
      out.sk = kb.e + 1;
    out.ek = out.sk;
  } else {
    PARTHENON_FAIL("Flux corrections only occur on faces for CC variables.");
  }

  out.var = v->flux[out.dir];

  out.Nv = out.var.GetDim(4);
  out.Nu = out.var.GetDim(5);
  out.Nt = out.var.GetDim(6);
  out.coords = pmb->coords;

  return out;
}

BndInfo BndInfo::GetSetCCFluxCor(std::shared_ptr<MeshBlock> pmb, const NeighborBlock &nb,
                                 std::shared_ptr<Variable<Real>> v,
                                 CommBuffer<buf_pool_t<Real>::owner_t> *buf,
                                 const OffsetIndices &) {
  BndInfo out;

  if (!v->IsAllocated() || buf->GetState() != BufferState::received) {
    out.allocated = false;
    return out;
  }
  out.allocated = true;
  out.buf = buf->buffer();

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  out.sk = kb.s;
  out.sj = jb.s;
  out.si = ib.s;
  out.ek = kb.e;
  out.ej = jb.e;
  out.ei = ib.e;
  if (nb.fid == BoundaryFace::inner_x1 || nb.fid == BoundaryFace::outer_x1) {
    out.dir = X1DIR;
    if (nb.fid == BoundaryFace::inner_x1)
      out.ei = out.si;
    else
      out.si = ++out.ei;
    if (nb.ni.fi1 == 0)
      out.ej -= pmb->block_size.nx2 / 2;
    else
      out.sj += pmb->block_size.nx2 / 2;
    if (nb.ni.fi2 == 0)
      out.ek -= pmb->block_size.nx3 / 2;
    else
      out.sk += pmb->block_size.nx3 / 2;
  } else if (nb.fid == BoundaryFace::inner_x2 || nb.fid == BoundaryFace::outer_x2) {
    out.dir = X2DIR;
    if (nb.fid == BoundaryFace::inner_x2)
      out.ej = out.sj;
    else
      out.sj = ++out.ej;
    if (nb.ni.fi1 == 0)
      out.ei -= pmb->block_size.nx1 / 2;
    else
      out.si += pmb->block_size.nx1 / 2;
    if (nb.ni.fi2 == 0)
      out.ek -= pmb->block_size.nx3 / 2;
    else
      out.sk += pmb->block_size.nx3 / 2;
  } else if (nb.fid == BoundaryFace::inner_x3 || nb.fid == BoundaryFace::outer_x3) {
    out.dir = X3DIR;
    if (nb.fid == BoundaryFace::inner_x3)
      out.ek = out.sk;
    else
      out.sk = ++out.ek;
    if (nb.ni.fi1 == 0)
      out.ei -= pmb->block_size.nx1 / 2;
    else
      out.si += pmb->block_size.nx1 / 2;
    if (nb.ni.fi2 == 0)
      out.ej -= pmb->block_size.nx2 / 2;
    else
      out.sj += pmb->block_size.nx2 / 2;
  } else {
    PARTHENON_FAIL("Flux corrections only occur on faces for CC variables.");
  }

  out.var = v->flux[out.dir];

  out.Nv = out.var.GetDim(4);
  out.Nu = out.var.GetDim(5);
  out.Nt = out.var.GetDim(6);

  out.coords = pmb->coords;

  return out;
}

BndInfo BndInfo::GetCCRestrictInfo(std::shared_ptr<MeshBlock> pmb,
                                   const NeighborBlock &nb,
                                   std::shared_ptr<Variable<Real>> v,
                                   CommBuffer<buf_pool_t<Real>::owner_t> *buf,
                                   const OffsetIndices &no) {
  BndInfo out;
  if (!v->IsAllocated()) {
    out.allocated = false;
    return out;
  }
  out.allocated = true;
  CalcIndicesRestrict(no.nk, no.nj, no.ni, out.si, out.ei, out.sj, out.ej, out.sk, out.ek,
                      nb, pmb);
  out.coords = pmb->coords;
  out.coarse_coords = pmb->pmr->GetCoarseCoords();
  out.fine = v->data.Get();
  out.coarse = v->coarse_s.Get();
  out.refinement_op = RefinementOp_t::Restriction;
  out.Nt = v->GetDim(6);
  out.Nu = v->GetDim(5);
  out.Nv = v->GetDim(4);
  return out;
}
} // namespace cell_centered_bvars
} // namespace parthenon
