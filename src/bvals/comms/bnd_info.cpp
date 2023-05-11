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

namespace {
enum class InterfaceType { SameToSame, CoarseToFine, FineToCoarse };
}
namespace parthenon {

template <InterfaceType INTERFACE>
Indexer6D CalcLoadIndices(const NeighborIndexes &ni, TopologicalElement el,
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
      if ((INTERFACE == InterfaceType::CoarseToFine) &&
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

template <InterfaceType INTERFACE, bool RESTRICT = false>
Indexer6D CalcSetIndices(const NeighborIndexes &ni, LogicalLocation loc,
                         TopologicalElement el, std::array<int, 3> tensor_shape,
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
      if ((INTERFACE == InterfaceType::CoarseToFine) && bounds[dir].e > bounds[dir].s) {
        s[dir] -= logic_loc[dir] % 2 == 1 ? Globals::nghost : 0;
        e[dir] += logic_loc[dir] % 2 == 0 ? Globals::nghost : 0;
      } else if (INTERFACE == InterfaceType::FineToCoarse) {
        const int half_grid = (bounds[dir].e - bounds[dir].s + 1) / 2;
        s[dir] += face_offset[off_idx] == 1 ? half_grid : 0;
        e[dir] -= face_offset[off_idx] == 0 ? half_grid : 0;
      }
      ++off_idx;
    } else if (block_offset[dir] > 0) {
      s[dir] = bounds[dir].e + 1;
      e[dir] = bounds[dir].e + (RESTRICT ? Globals::nghost / 2 : Globals::nghost);
    } else {
      s[dir] = bounds[dir].s - (RESTRICT ? Globals::nghost / 2 : Globals::nghost);
      e[dir] = bounds[dir].s - 1;
    }
  }
  return Indexer6D({0, tensor_shape[0] - 1}, {0, tensor_shape[1] - 1},
                   {0, tensor_shape[2] - 1}, {s[2], e[2]}, {s[1], e[1]}, {s[0], e[0]});
}

int GetBufferSize(std::shared_ptr<MeshBlock> pmb, const NeighborBlock &nb,
                  std::shared_ptr<Variable<Real>> v) {
  // This does not do a careful job of calculating the buffer size, in many
  // cases there will be some extra storage that is not required, but there
  // will always be enough storage
  auto &cb = pmb->cellbounds;
  int topo_comp = (v->IsSet(Metadata::Face) || v->IsSet(Metadata::Edge)) ? 3 : 1;
  const IndexDomain in = IndexDomain::interior;
  // The plus 2 instead of 1 is to account for the possible size of face, edge, and nodal
  // fields
  const int isize = cb.ie(in) - cb.is(in) + 2;
  const int jsize = cb.je(in) - cb.js(in) + 2;
  const int ksize = cb.ke(in) - cb.ks(in) + 2;
  return (nb.ni.ox1 == 0 ? isize : Globals::nghost) *
         (nb.ni.ox2 == 0 ? jsize : Globals::nghost) *
         (nb.ni.ox3 == 0 ? ksize : Globals::nghost) * v->GetDim(6) * v->GetDim(5) *
         v->GetDim(4) * topo_comp;
}

BndInfo BndInfo::GetSendBndInfo(std::shared_ptr<MeshBlock> pmb, const NeighborBlock &nb,
                                std::shared_ptr<Variable<Real>> v,
                                CommBuffer<buf_pool_t<Real>::owner_t> *buf) {
  BndInfo out;

  out.allocated = v->IsAllocated();
  if (!out.allocated) return out;

  out.buf = buf->buffer();

  int Nv = v->GetDim(4);
  int Nu = v->GetDim(5);
  int Nt = v->GetDim(6);

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

  out.ntopological_elements = elements.size();
  for (auto el : elements) {
    int idx = static_cast<int>(el) % 3;
    if (nb.snb.level == mylevel) {
      out.idxer[idx] = CalcLoadIndices<InterfaceType::SameToSame>(nb.ni, el, {Nt, Nu, Nv},
                                                                  pmb->cellbounds);
      out.var = v->data.Get();
    } else if (nb.snb.level < mylevel) {
      // "Same" logic is the same for loading to a coarse buffer, just using
      // c_cellbounds
      out.idxer[idx] = CalcLoadIndices<InterfaceType::FineToCoarse>(
          nb.ni, el, {Nt, Nu, Nv}, pmb->c_cellbounds);
      out.prores_idxer[idx] = out.idxer[idx];
      out.refinement_op = RefinementOp_t::Restriction;
      out.var = v->coarse_s.Get();
    } else {
      out.idxer[idx] = CalcLoadIndices<InterfaceType::CoarseToFine>(
          nb.ni, el, {Nt, Nu, Nv}, pmb->cellbounds);
      out.var = v->data.Get();
    }
  }
  return out;
}

BndInfo BndInfo::GetSetBndInfo(std::shared_ptr<MeshBlock> pmb, const NeighborBlock &nb,
                               std::shared_ptr<Variable<Real>> v,
                               CommBuffer<buf_pool_t<Real>::owner_t> *buf) {
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

  int Nv = v->GetDim(4);
  int Nu = v->GetDim(5);
  int Nt = v->GetDim(6);

  using TE = TopologicalElement;
  std::vector<TopologicalElement> elements = {TE::C};
  if (v->IsSet(Metadata::Face)) elements = {TE::FX, TE::FY, TE::FZ};
  if (v->IsSet(Metadata::Edge)) elements = {TE::EXY, TE::EXZ, TE::EYZ};
  if (v->IsSet(Metadata::Node)) elements = {TE::NXYZ};

  int mylevel = pmb->loc.level;
  out.coords = pmb->coords;
  if (pmb->pmr) out.coarse_coords = pmb->pmr->GetCoarseCoords();
  out.fine = v->data.Get();
  out.coarse = v->coarse_s.Get();

  // This will select a superset of the boundaries that actually need to be restricted,
  // more logic could be added to only restrict boundary regions that abut boundary
  // regions that were filled by coarser neighbors
  bool restricted = false;
  if (mylevel > 0) {
    for (int k = 0; k < 3; ++k) {
      for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
          restricted = restricted || (pmb->pbval->nblevel[k][j][i] == (mylevel - 1));
        }
      }
    }
  }

  out.ntopological_elements = elements.size();
  for (auto el : elements) {
    int idx = static_cast<int>(el) % 3;
    if (nb.snb.level == mylevel) {
      out.var = v->data.Get();
      out.idxer[idx] = CalcSetIndices<InterfaceType::SameToSame>(
          nb.ni, pmb->loc, el, {Nt, Nu, Nv}, pmb->cellbounds);
      if (restricted) {
        out.refinement_op = RefinementOp_t::Restriction;
        out.prores_idxer[idx] = CalcSetIndices<InterfaceType::SameToSame, true>(
            nb.ni, pmb->loc, el, {Nt, Nu, Nv}, pmb->c_cellbounds);
      }
    } else if (nb.snb.level < mylevel) {
      out.idxer[idx] = CalcSetIndices<InterfaceType::CoarseToFine>(
          nb.ni, pmb->loc, el, {Nt, Nu, Nv}, pmb->c_cellbounds);
      out.var = v->coarse_s.Get();
      // TODO(LFR): These are regions that need to be registered for prolongation (which
      // can be done after physical boundaries are filled on coarse buffers)
    } else {
      out.var = v->data.Get();
      out.idxer[idx] = CalcSetIndices<InterfaceType::FineToCoarse>(
          nb.ni, pmb->loc, el, {Nt, Nu, Nv}, pmb->cellbounds);
      if (restricted) {
        out.refinement_op = RefinementOp_t::Restriction;
        out.prores_idxer[idx] = CalcSetIndices<InterfaceType::FineToCoarse, true>(
            nb.ni, pmb->loc, el, {Nt, Nu, Nv}, pmb->c_cellbounds);
      }
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
  return out;
}

BndInfo BndInfo::GetSendCCFluxCor(std::shared_ptr<MeshBlock> pmb, const NeighborBlock &nb,
                                  std::shared_ptr<Variable<Real>> v,
                                  CommBuffer<buf_pool_t<Real>::owner_t> *buf) {
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
  int sk = kb.s;
  int ek = sk + std::max((kb.e - kb.s + 1) / 2, 1) - 1;
  int sj = jb.s;
  int ej = sj + std::max((jb.e - jb.s + 1) / 2, 1) - 1;
  int si = ib.s;
  int ei = si + std::max((ib.e - ib.s + 1) / 2, 1) - 1;

  if (nb.fid == BoundaryFace::inner_x1 || nb.fid == BoundaryFace::outer_x1) {
    out.dir = X1DIR;
    if (nb.fid == BoundaryFace::inner_x1)
      si = ib.s;
    else
      si = ib.e + 1;
    ei = si;
  } else if (nb.fid == BoundaryFace::inner_x2 || nb.fid == BoundaryFace::outer_x2) {
    out.dir = X2DIR;
    if (nb.fid == BoundaryFace::inner_x2)
      sj = jb.s;
    else
      sj = jb.e + 1;
    ej = sj;
  } else if (nb.fid == BoundaryFace::inner_x3 || nb.fid == BoundaryFace::outer_x3) {
    out.dir = X3DIR;
    if (nb.fid == BoundaryFace::inner_x3)
      sk = kb.s;
    else
      sk = kb.e + 1;
    ek = sk;
  } else {
    PARTHENON_FAIL("Flux corrections only occur on faces for CC variables.");
  }

  out.var = v->flux[out.dir];
  out.coords = pmb->coords;
  out.idxer[0] = Indexer6D({0, out.var.GetDim(6) - 1}, {0, out.var.GetDim(5) - 1},
                           {0, out.var.GetDim(4) - 1}, {sk, ek}, {sj, ej}, {si, ei});
  return out;
}

BndInfo BndInfo::GetSetCCFluxCor(std::shared_ptr<MeshBlock> pmb, const NeighborBlock &nb,
                                 std::shared_ptr<Variable<Real>> v,
                                 CommBuffer<buf_pool_t<Real>::owner_t> *buf) {
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

  int sk = kb.s;
  int sj = jb.s;
  int si = ib.s;
  int ek = kb.e;
  int ej = jb.e;
  int ei = ib.e;
  if (nb.fid == BoundaryFace::inner_x1 || nb.fid == BoundaryFace::outer_x1) {
    out.dir = X1DIR;
    if (nb.fid == BoundaryFace::inner_x1)
      ei = si;
    else
      si = ++ei;
    if (nb.ni.fi1 == 0)
      ej -= pmb->block_size.nx2 / 2;
    else
      sj += pmb->block_size.nx2 / 2;
    if (nb.ni.fi2 == 0)
      ek -= pmb->block_size.nx3 / 2;
    else
      sk += pmb->block_size.nx3 / 2;
  } else if (nb.fid == BoundaryFace::inner_x2 || nb.fid == BoundaryFace::outer_x2) {
    out.dir = X2DIR;
    if (nb.fid == BoundaryFace::inner_x2)
      ej = sj;
    else
      sj = ++ej;
    if (nb.ni.fi1 == 0)
      ei -= pmb->block_size.nx1 / 2;
    else
      si += pmb->block_size.nx1 / 2;
    if (nb.ni.fi2 == 0)
      ek -= pmb->block_size.nx3 / 2;
    else
      sk += pmb->block_size.nx3 / 2;
  } else if (nb.fid == BoundaryFace::inner_x3 || nb.fid == BoundaryFace::outer_x3) {
    out.dir = X3DIR;
    if (nb.fid == BoundaryFace::inner_x3)
      ek = sk;
    else
      sk = ++ek;
    if (nb.ni.fi1 == 0)
      ei -= pmb->block_size.nx1 / 2;
    else
      si += pmb->block_size.nx1 / 2;
    if (nb.ni.fi2 == 0)
      ej -= pmb->block_size.nx2 / 2;
    else
      sj += pmb->block_size.nx2 / 2;
  } else {
    PARTHENON_FAIL("Flux corrections only occur on faces for CC variables.");
  }

  out.var = v->flux[out.dir];

  out.coords = pmb->coords;
  out.idxer[0] = Indexer6D({0, out.var.GetDim(6) - 1}, {0, out.var.GetDim(5) - 1},
                           {0, out.var.GetDim(4) - 1}, {sk, ek}, {sj, ej}, {si, ei});
  return out;
}

} // namespace parthenon
