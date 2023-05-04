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
#include "bvals/comms/bnd_info.hpp"
#include "config.hpp"
#include "globals.hpp"
#include "interface/variable.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock.hpp"
#include "prolong_restrict/prolong_restrict.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

Indexer6D CalcLoadIndices(const NeighborIndexes &ni, bool c2f, TopologicalElement el,
                          std::array<int, 3> tensor_shape,
                          const parthenon::IndexShape &shape) {
  IndexDomain interior = IndexDomain::interior;
  std::array<IndexRange, 3> bounds{shape.GetBoundsI(interior, el),
                                   shape.GetBoundsJ(interior, el),
                                   shape.GetBoundsK(interior, el)};

  // Account for the fact that the neighbor block may duplicate
  // some active zones on the loading block for face, edge, and nodal
  // fields, so the boundary of the neighbor block is one deeper into
  // the current block in some cases
  std::array<int, 3> top_offset{TopologicalOffsetI(el), TopologicalOffsetJ(el),
                                TopologicalOffsetK(el)};
  std::array<int, 3> block_offset{ni.ox1, ni.ox2, ni.ox3};
  std::array<int, 2> face_offset{ni.fi1, ni.fi2};

  int off_idx = 0;
  std::array<int, 3> s, e;
  for (int dir = 0; dir < 3; ++dir) {
    if (block_offset[dir] == 0) {
      s[dir] = bounds[dir].s;
      e[dir] = bounds[dir].e;
      if (c2f &&
          bounds[dir].e > bounds[dir].s) { // Check that this dimension has ghost zones
        // We are sending from a coarser level to the coarse buffer of a finer level,
        // so we need to only send the approximately half of the indices that overlap
        // with the other block. We also send nghost "extra" zones in the interior
        // to ensure there is enough information for prolongation. Also note for
        // non-cell centered values the number of grid points may be odd, so we
        // pick up an extra zone that is communicated. I think this is ok, but
        // something to keep in mind if there are issues.
        const int half_grid = (bounds[dir].e - bounds[dir].s + 1) / 2;
        s[dir] += face_offset[off_idx] == 1 ? half_grid - Globals::nghost : 0;
        e[dir] -= face_offset[off_idx] == 0 ? half_grid - Globals::nghost : 0;
      }
      ++off_idx; // Offsets are listed in X1,X2,X3 order, should never try to access
                 // with off_idx > 1 since all neighbors must have a non-zero block
                 // offset in some direction
    } else if (block_offset[dir] > 0) {
      s[dir] = bounds[dir].e - Globals::nghost + 1 - top_offset[dir];
      e[dir] = bounds[dir].e - top_offset[dir];
    } else {
      s[dir] = bounds[dir].s + top_offset[dir];
      e[dir] = bounds[dir].s + Globals::nghost - 1 + top_offset[dir];
    }
  }
  return Indexer6D({0, tensor_shape[0] - 1}, {0, tensor_shape[1] - 1},
                   {0, tensor_shape[2] - 1}, {s[2], e[2]}, {s[1], e[1]}, {s[0], e[0]});
}

Indexer6D CalcSetIndices(const NeighborIndexes &ni, LogicalLocation loc, bool c2f,
                         bool f2c, TopologicalElement el, std::array<int, 3> tensor_shape,
                         const parthenon::IndexShape &shape) {
  IndexDomain interior = IndexDomain::interior;
  std::array<IndexRange, 3> bounds{shape.GetBoundsI(interior, el),
                                   shape.GetBoundsJ(interior, el),
                                   shape.GetBoundsK(interior, el)};

  std::array<int, 3> block_offset{ni.ox1, ni.ox2, ni.ox3};
  // This is gross, but the face offsets do not contain the correct
  // information for going from coarse to fine and the neighbor block
  // structure does not contain the logical location of the neighbor
  // block
  std::array<std::int64_t, 3> logic_loc{loc.lx1, loc.lx2, loc.lx3};
  std::array<int, 2> face_offset{ni.fi1, ni.fi2};
  std::array<int, 3> s, e;

  // This is the inverse of CalcLoadIndices, but we don't require any topological element
  // information beyond what we have in the IndexRanges
  int off_idx = 0;
  for (int dir = 0; dir < 3; ++dir) {
    if (block_offset[dir] == 0) {
      s[dir] = bounds[dir].s;
      e[dir] = bounds[dir].e;
      if (c2f && bounds[dir].e > bounds[dir].s) {
        s[dir] -= logic_loc[dir] % 2 == 1 ? Globals::nghost : 0;
        e[dir] += logic_loc[dir] % 2 == 0 ? Globals::nghost : 0;
      } else if (f2c) {
        const int half_grid = (bounds[dir].e - bounds[dir].s + 1) / 2;
        s[dir] += face_offset[off_idx] == 1 ? half_grid : 0;
        e[dir] -= face_offset[off_idx] == 0 ? half_grid : 0;
      }
      ++off_idx;
    } else if (block_offset[dir] > 0) {
      s[dir] = bounds[dir].e + 1;
      e[dir] = bounds[dir].e + Globals::nghost;
    } else {
      s[dir] = bounds[dir].s - Globals::nghost;
      e[dir] = bounds[dir].s - 1;
    }
  }
  return Indexer6D({0, tensor_shape[0] - 1}, {0, tensor_shape[1] - 1},
                   {0, tensor_shape[2] - 1}, {s[2], e[2]}, {s[1], e[1]}, {s[0], e[0]});
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

  using TE = TopologicalElement;
  std::vector<TopologicalElement> elements = {TE::C};
  if (v->IsSet(Metadata::Face)) elements = {TE::FX, TE::FY, TE::FZ};
  if (v->IsSet(Metadata::Edge)) elements = {TE::EXY, TE::EXZ, TE::EYZ};
  if (v->IsSet(Metadata::Node)) elements = {TE::NXYZ};

  for (auto el : elements) {
    int idx = static_cast<int>(el) % 3;
    if (nb.snb.level == mylevel) {
      out.idxer[idx] =
          CalcLoadIndices(nb.ni, false, el, {out.Nt, out.Nu, out.Nv}, pmb->cellbounds);
      out.var = v->data.Get();
    } else if (nb.snb.level < mylevel) {
      // "Same" logic is the same for loading to a coarse buffer, just using
      // c_cellbounds
      out.idxer[idx] =
          CalcLoadIndices(nb.ni, false, el, {out.Nt, out.Nu, out.Nv}, pmb->c_cellbounds);
      out.refinement_op = RefinementOp_t::Restriction;
      out.var = v->coarse_s.Get();
    } else {
      out.idxer[idx] =
          CalcLoadIndices(nb.ni, true, el, {out.Nt, out.Nu, out.Nv}, pmb->cellbounds);
      out.var = v->data.Get();
    }
  }
  // Still don't understand why, but these have to be set
  out.si = out.idxer[0].template StartIdx<5>();
  out.ei = out.idxer[0].template EndIdx<5>();
  out.sj = out.idxer[0].template StartIdx<4>();
  out.ej = out.idxer[0].template EndIdx<4>();
  out.sk = out.idxer[0].template StartIdx<3>();
  out.ek = out.idxer[0].template EndIdx<3>();
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

  using TE = TopologicalElement;
  std::vector<TopologicalElement> elements = {TE::C};
  if (v->IsSet(Metadata::Face)) elements = {TE::FX, TE::FY, TE::FZ};
  if (v->IsSet(Metadata::Edge)) elements = {TE::EXY, TE::EXZ, TE::EYZ};
  if (v->IsSet(Metadata::Node)) elements = {TE::NXYZ};

  int mylevel = pmb->loc.level;

  for (auto el : elements) {
    int idx = static_cast<int>(el) % 3;
    if (nb.snb.level == mylevel) {
      out.var = v->data.Get();
      out.idxer[idx] = CalcSetIndices(nb.ni, pmb->loc, false, false, el,
                                      {out.Nt, out.Nu, out.Nv}, pmb->cellbounds);
    } else if (nb.snb.level < mylevel) {
      out.idxer[idx] = CalcSetIndices(nb.ni, pmb->loc, true, false, el,
                                      {out.Nt, out.Nu, out.Nv}, pmb->c_cellbounds);
      out.var = v->coarse_s.Get();
    } else {
      out.idxer[idx] = CalcSetIndices(nb.ni, pmb->loc, false, true, el,
                                      {out.Nt, out.Nu, out.Nv}, pmb->cellbounds);
      out.var = v->data.Get();
    }
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

  // Still don't understand why, but these have to be set
  out.si = out.idxer[0].template StartIdx<5>();
  out.ei = out.idxer[0].template EndIdx<5>();
  out.sj = out.idxer[0].template StartIdx<4>();
  out.ej = out.idxer[0].template EndIdx<4>();
  out.sk = out.idxer[0].template StartIdx<3>();
  out.ek = out.idxer[0].template EndIdx<3>();

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
  out.idxer[0] = Indexer6D({0, out.Nt - 1}, {0, out.Nu - 1}, {0, out.Nv - 1},
                           {out.sk, out.ek}, {out.sj, out.ej}, {out.si, out.ei});
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
  out.idxer[0] = Indexer6D({0, out.Nt - 1}, {0, out.Nu - 1}, {0, out.Nv - 1},
                           {out.sk, out.ek}, {out.sj, out.ej}, {out.si, out.ei});
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
  out.idxer[0] = Indexer6D({0, out.Nt - 1}, {0, out.Nu - 1}, {0, out.Nv - 1},
                           {out.sk, out.ek}, {out.sj, out.ej}, {out.si, out.ei});
  return out;
}
} // namespace parthenon
