//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2022 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2023. Triad National Security, LLC. All rights reserved.
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
#include "bvals/comms/bnd_info.hpp"
#include "bvals/neighbor_block.hpp"
#include "config.hpp"
#include "globals.hpp"
#include "interface/state_descriptor.hpp"
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
enum class IndexRangeType {
  BoundaryInteriorSend,
  BoundaryExteriorRecv,
  InteriorSend,
  InteriorRecv
};

using namespace parthenon;

} // namespace
namespace parthenon {

void ProResCache_t::Initialize(int n_regions, StateDescriptor *pkg) {
  prores_info = ParArray1D<ProResInfo>("prores_info", n_regions);
  prores_info_h = Kokkos::create_mirror_view(prores_info);
  int nref_funcs = pkg->NumRefinementFuncs();
  // Note that assignment of Kokkos views resets them, but
  // buffer_subset_sizes is a std::vector. It must be cleared, then
  // re-filled.
  buffer_subset_sizes.clear();
  buffer_subset_sizes.resize(nref_funcs, 0);
  buffer_subsets = ParArray2D<std::size_t>("buffer_subsets", nref_funcs, n_regions);
  buffer_subsets_h = Kokkos::create_mirror_view(buffer_subsets);
}

void ProResCache_t::RegisterRegionHost(int region, ProResInfo pri, Variable<Real> *v,
                                       StateDescriptor *pkg) {
  prores_info_h(region) = pri;
  if (v->IsRefined()) {
    // var must be registered for refinement
    // note this condition means that each subset contains
    // both prolongation and restriction conditions. The
    // `RefinementOp_t` in `BndInfo` is assumed to
    // differentiate.
    std::size_t rfid = pkg->RefinementFuncID((v->GetRefinementFunctions()));
    buffer_subsets_h(rfid, buffer_subset_sizes[rfid]++) = region;
  }
}

SpatiallyMaskedIndexer6D CalcIndices(const NeighborBlock &nb, MeshBlock *pmb,
                                     TopologicalElement el, IndexRangeType ir_type,
                                     bool prores, std::array<int, 3> tensor_shape) {
  const auto &ni = nb.ni;
  const auto &loc = pmb->loc;
  auto shape = pmb->cellbounds;
  // Both prolongation and restriction always operate in the coarse
  // index space. Also need to use the coarse index space if the
  // neighbor is coarser than you, wether or not you are setting
  // interior or exterior cells
  if (prores || nb.loc.level() < loc.level()) shape = pmb->c_cellbounds;

  // Re-create the index space for the neighbor block (either the main block or
  // the coarse buffer as required)
  int coarse_fac = 1;
  if (nb.loc.level() > loc.level()) coarse_fac = 2;
  auto neighbor_shape = IndexShape(nb.block_size.nx(X3DIR) / coarse_fac,
                                   nb.block_size.nx(X2DIR) / coarse_fac,
                                   nb.block_size.nx(X1DIR) / coarse_fac, Globals::nghost);

  IndexDomain interior = IndexDomain::interior;
  std::array<IndexRange, 3> bounds{shape.GetBoundsI(interior, el),
                                   shape.GetBoundsJ(interior, el),
                                   shape.GetBoundsK(interior, el)};
  std::array<IndexRange, 3> neighbor_bounds{neighbor_shape.GetBoundsI(interior, el),
                                            neighbor_shape.GetBoundsJ(interior, el),
                                            neighbor_shape.GetBoundsK(interior, el)};

  std::array<bool, 3> not_symmetry{!pmb->block_size.symmetry(X1DIR),
                                   !pmb->block_size.symmetry(X2DIR),
                                   !pmb->block_size.symmetry(X3DIR)};
  // Account for the fact that the neighbor block may duplicate
  // some active zones on the loading block for face, edge, and nodal
  // fields, so the boundary of the neighbor block is one deeper into
  // the current block in some cases
  std::array<int, 3> top_offset{TopologicalOffsetI(el), TopologicalOffsetJ(el),
                                TopologicalOffsetK(el)};
  std::array<int, 3> block_offset = {ni.ox1, ni.ox2, ni.ox3};

  int interior_offset =
      ir_type == IndexRangeType::BoundaryInteriorSend ? Globals::nghost : 0;
  int exterior_offset =
      ir_type == IndexRangeType::BoundaryExteriorRecv ? Globals::nghost : 0;
  if (prores) {
    // The coarse ghosts cover twice as much volume as the fine ghosts, so when working in
    // the exterior (i.e. ghosts) we must only go over the coarse ghosts that have
    // corresponding fine ghosts
    exterior_offset /= 2;
  }

  std::array<int, 3> s, e;
  for (int dir = 0; dir < 3; ++dir) {
    if (block_offset[dir] == 0) {
      s[dir] = bounds[dir].s;
      e[dir] = bounds[dir].e;
      if ((loc.level() < nb.loc.level()) &&
          not_symmetry[dir]) { // Check that this dimension has ghost zones
        // The requested neighbor block is at a finer level, so it only abuts
        // approximately half of the zones in any given direction with offset zero. If we
        // are asking for an interior index range, we also send nghost "extra" zones in
        // the interior to ensure there is enough information for prolongation. Also note
        // for non-cell centered values the number of grid points may be odd, so we pick
        // up an extra zone that is communicated. I think this is ok, but something to
        // keep in mind if there are issues.
        const int extra_zones = (bounds[dir].e - bounds[dir].s + 1) -
                                (neighbor_bounds[dir].e - neighbor_bounds[dir].s + 1);
        s[dir] += nb.loc.l(dir) % 2 == 1 ? extra_zones - interior_offset : 0;
        e[dir] -= nb.loc.l(dir) % 2 == 0 ? extra_zones - interior_offset : 0;
        if (ir_type == IndexRangeType::InteriorSend) {
          // Include ghosts of finer block coarse array in message
          s[dir] -= Globals::nghost;
          e[dir] += Globals::nghost;
        }
      }
      if (loc.level() > nb.loc.level() && not_symmetry[dir]) {
        // If we are setting (i.e. have non-zero exterior_offset) from a neighbor block
        // that is coarser, we got extra ghost zones from the neighbor (see inclusion of
        // interior_offset in the above if block)
        s[dir] -= loc.l(dir) % 2 == 1 ? exterior_offset : 0;
        e[dir] += loc.l(dir) % 2 == 0 ? exterior_offset : 0;
        if (ir_type == IndexRangeType::InteriorRecv) {
          // Include ghosts of finer block coarse array in message
          s[dir] -= Globals::nghost;
          e[dir] += Globals::nghost;
        }
      }
      // Prolongate into ghosts of interior receiver since we have the data available,
      // having this is important for AMR MG
      if (prores && not_symmetry[dir] && IndexRangeType::InteriorRecv == ir_type) {
        s[dir] -= Globals::nghost / 2;
        e[dir] += Globals::nghost / 2;
      }
    } else if (block_offset[dir] > 0) {
      s[dir] = bounds[dir].e - interior_offset + 1 - top_offset[dir];
      e[dir] = bounds[dir].e + exterior_offset;
    } else {
      s[dir] = bounds[dir].s - exterior_offset;
      e[dir] = bounds[dir].s + interior_offset - 1 + top_offset[dir];
    }
  }

  block_ownership_t owns(true);
  // Although it wouldn't hurt to include ownership when producing an interior
  // index range, it is unecessary. This is probably not immediately obvious,
  // but it is possible to convince oneself that dealing with ownership in
  // only exterior index ranges works correctly
  if (ir_type == IndexRangeType::BoundaryExteriorRecv) {
    int sox1 = -ni.ox1;
    int sox2 = -ni.ox2;
    int sox3 = -ni.ox3;
    if (nb.loc.level() < loc.level()) {
      // For coarse to fine interfaces, we are passing zones from only an
      // interior corner of the cell, never an entire face or edge
      if (sox1 == 0) sox1 = loc.l(0) % 2 == 1 ? 1 : -1;
      if (sox2 == 0) sox2 = loc.l(1) % 2 == 1 ? 1 : -1;
      if (sox3 == 0) sox3 = loc.l(2) % 2 == 1 ? 1 : -1;
    }
    owns = GetIndexRangeMaskFromOwnership(el, nb.ownership, sox1, sox2, sox3);
  }
  return SpatiallyMaskedIndexer6D(owns, {0, tensor_shape[0] - 1},
                                  {0, tensor_shape[1] - 1}, {0, tensor_shape[2] - 1},
                                  {s[2], e[2]}, {s[1], e[1]}, {s[0], e[0]});
}

int GetBufferSize(MeshBlock *pmb, const NeighborBlock &nb,
                  std::shared_ptr<Variable<Real>> v) {
  // This does not do a careful job of calculating the buffer size, in many
  // cases there will be some extra storage that is not required, but there
  // will always be enough storage
  auto &cb = pmb->cellbounds;
  int topo_comp = (v->IsSet(Metadata::Face) || v->IsSet(Metadata::Edge)) ? 3 : 1;
  const IndexDomain in = IndexDomain::entire;
  // The plus 2 instead of 1 is to account for the possible size of face, edge, and nodal
  // fields
  const int isize = cb.ie(in) - cb.is(in) + 2;
  const int jsize = cb.je(in) - cb.js(in) + 2;
  const int ksize = cb.ke(in) - cb.ks(in) + 2;
  return (nb.ni.ox1 == 0 ? isize : Globals::nghost + 1) *
         (nb.ni.ox2 == 0 ? jsize : Globals::nghost + 1) *
         (nb.ni.ox3 == 0 ? ksize : Globals::nghost + 1) * v->GetDim(6) * v->GetDim(5) *
         v->GetDim(4) * topo_comp;
}

void BndInfo::LockedBufferCopy(CommBuffer<buf_pool_t<Real>::owner_t> *input_buf) {
  //std::lock_guard<std::mutex> lock(mutex);
  buf = input_buf->buffer();
}

BndInfo BndInfo::GetSendBndInfo(MeshBlock *pmb, const NeighborBlock &nb,
                                std::shared_ptr<Variable<Real>> v,
                                CommBuffer<buf_pool_t<Real>::owner_t> *buf) {
  BndInfo out;

  out.allocated = v->IsAllocated();
  out.alloc_status = v->GetAllocationStatus();
  if (!out.allocated) return out;

  out.LockedBufferCopy(buf);
  //out.buf = buf->buffer();

  int Nv = v->GetDim(4);
  int Nu = v->GetDim(5);
  int Nt = v->GetDim(6);
  int mylevel = pmb->loc.level();

  auto elements = v->GetTopologicalElements();
  out.ntopological_elements = elements.size();
  auto idx_range_type = IndexRangeType::BoundaryInteriorSend;
  if (std::abs(nb.ni.ox1) + std::abs(nb.ni.ox2) + std::abs(nb.ni.ox3) == 0)
    idx_range_type = IndexRangeType::InteriorSend;
  for (auto el : elements) {
    int idx = static_cast<int>(el) % 3;
    out.idxer[idx] = CalcIndices(nb, pmb, el, idx_range_type, false, {Nt, Nu, Nv});
  }
  if (nb.snb.level < mylevel) {
    out.var = v->coarse_s.Get();
  } else {
    out.var = v->data.Get();
  }
  return out;
}

BndInfo BndInfo::GetSetBndInfo(MeshBlock *pmb, const NeighborBlock &nb,
                               std::shared_ptr<Variable<Real>> v,
                               CommBuffer<buf_pool_t<Real>::owner_t> *buf) {
  BndInfo out;
  //out.buf = buf->buffer();
  out.LockedBufferCopy(buf);
  auto buf_state = buf->GetState();
  if (buf_state == BufferState::received) {
    out.buf_allocated = true;
  } else if (buf_state == BufferState::received_null) {
    out.buf_allocated = false;
  } else {
    PARTHENON_FAIL("Buffer should be in a received state.");
  }
  out.allocated = v->IsAllocated();
  out.alloc_status = v->GetAllocationStatus();

  int Nv = v->GetDim(4);
  int Nu = v->GetDim(5);
  int Nt = v->GetDim(6);

  int mylevel = pmb->loc.level();

  auto elements = v->GetTopologicalElements();
  out.ntopological_elements = elements.size();
  auto idx_range_type = IndexRangeType::BoundaryExteriorRecv;
  if (std::abs(nb.ni.ox1) + std::abs(nb.ni.ox2) + std::abs(nb.ni.ox3) == 0)
    idx_range_type = IndexRangeType::InteriorRecv;
  for (auto el : elements) {
    int idx = static_cast<int>(el) % 3;
    out.idxer[idx] = CalcIndices(nb, pmb, el, idx_range_type, false, {Nt, Nu, Nv});
  }
  if (nb.snb.level < mylevel) {
    out.var = v->coarse_s.Get();
  } else {
    out.var = v->data.Get();
  }

  return out;
}

ProResInfo ProResInfo::GetInteriorRestrict(MeshBlock *pmb, const NeighborBlock & /*nb*/,
                                           std::shared_ptr<Variable<Real>> v) {
  ProResInfo out;

  out.allocated = v->IsAllocated();
  out.alloc_status = v->GetAllocationStatus();
  if (!out.allocated) return out;

  int Nv = v->GetDim(4);
  int Nu = v->GetDim(5);
  int Nt = v->GetDim(6);

  int mylevel = pmb->loc.level();
  out.coords = pmb->coords;

  if (pmb->pmr) out.coarse_coords = pmb->pmr->GetCoarseCoords();

  out.fine = v->data.Get();
  out.coarse = v->coarse_s.Get();
  NeighborBlock nb(pmb->pmy_mesh, pmb->loc, Globals::my_rank, 0, 0, {0, 0, 0},
                   NeighborConnect::none, 0, 0, 0, 0);

  auto elements = v->GetTopologicalElements();
  out.ntopological_elements = elements.size();
  for (auto el : elements) {
    out.idxer[static_cast<int>(el)] =
        CalcIndices(nb, pmb, el, IndexRangeType::InteriorSend, true, {Nt, Nu, Nv});
  }
  out.refinement_op = RefinementOp_t::Restriction;
  return out;
}

ProResInfo ProResInfo::GetInteriorProlongate(MeshBlock *pmb, const NeighborBlock & /*nb*/,
                                             std::shared_ptr<Variable<Real>> v) {
  ProResInfo out;

  out.allocated = v->IsAllocated();
  out.alloc_status = v->GetAllocationStatus();
  if (!out.allocated) return out;

  int Nv = v->GetDim(4);
  int Nu = v->GetDim(5);
  int Nt = v->GetDim(6);

  int mylevel = pmb->loc.level();
  out.coords = pmb->coords;

  if (pmb->pmr) out.coarse_coords = pmb->pmr->GetCoarseCoords();

  out.fine = v->data.Get();
  out.coarse = v->coarse_s.Get();
  NeighborBlock nb(pmb->pmy_mesh, pmb->loc, Globals::my_rank, 0, 0, {0, 0, 0},
                   NeighborConnect::none, 0, 0, 0, 0);

  auto elements = v->GetTopologicalElements();
  out.ntopological_elements = elements.size();
  for (auto el : {TE::CC, TE::F1, TE::F2, TE::F3, TE::E1, TE::E2, TE::E3, TE::NN})
    out.idxer[static_cast<int>(el)] =
        CalcIndices(nb, pmb, el, IndexRangeType::InteriorRecv, true, {Nt, Nu, Nv});
  out.refinement_op = RefinementOp_t::Prolongation;
  return out;
}

ProResInfo ProResInfo::GetSend(MeshBlock *pmb, const NeighborBlock &nb,
                               std::shared_ptr<Variable<Real>> v) {
  ProResInfo out;

  out.allocated = v->IsAllocated();
  out.alloc_status = v->GetAllocationStatus();
  if (!out.allocated) return out;

  int Nv = v->GetDim(4);
  int Nu = v->GetDim(5);
  int Nt = v->GetDim(6);

  int mylevel = pmb->loc.level();
  out.coords = pmb->coords;

  if (pmb->pmr) out.coarse_coords = pmb->pmr->GetCoarseCoords();

  out.fine = v->data.Get();
  out.coarse = v->coarse_s.Get();

  auto elements = v->GetTopologicalElements();
  out.ntopological_elements = elements.size();
  if (nb.snb.level < mylevel) {
    for (auto el : elements) {
      out.idxer[static_cast<int>(el)] = CalcIndices(
          nb, pmb, el, IndexRangeType::BoundaryInteriorSend, true, {Nt, Nu, Nv});
      out.refinement_op = RefinementOp_t::Restriction;
    }
  }
  return out;
}

ProResInfo ProResInfo::GetSet(MeshBlock *pmb, const NeighborBlock &nb,
                              std::shared_ptr<Variable<Real>> v) {
  ProResInfo out;
  out.allocated = v->IsAllocated();
  out.alloc_status = v->GetAllocationStatus();
  int Nv = v->GetDim(4);
  int Nu = v->GetDim(5);
  int Nt = v->GetDim(6);

  int mylevel = pmb->loc.level();
  out.coords = pmb->coords;
  if (pmb->pmr) out.coarse_coords = pmb->pmr->GetCoarseCoords();
  out.fine = v->data.Get();
  out.coarse = v->coarse_s.Get();

  // This will select a superset of the boundaries that actually need to be restricted,
  // more logic could be added to only restrict boundary regions that abut boundary
  // regions that were filled by coarser neighbors
  bool restricted = false;
  if (mylevel > 0) {
    for (const auto &nb : pmb->neighbors) {
      restricted = restricted || (nb.loc.level() == (mylevel - 1));
    }
  }

  auto elements = v->GetTopologicalElements();
  out.ntopological_elements = elements.size();
  for (auto el : elements) {
    if (nb.snb.level < mylevel) {
      out.refinement_op = RefinementOp_t::Prolongation;
    } else {
      if (restricted) {
        out.refinement_op = RefinementOp_t::Restriction;
        out.idxer[static_cast<int>(el)] = CalcIndices(
            nb, pmb, el, IndexRangeType::BoundaryExteriorRecv, true, {Nt, Nu, Nv});
      }
    }
  }

  // LFR: All of these are not necessarily required, but some subset are for internal
  // prolongation.
  //      if the variable is NXYZ we require (C, FX, FY, FZ, EXY, EXZ, EYZ, NXYZ)
  //      if the variable is EXY we require (C, FX, FY, EXY), etc.
  //      if the variable is FX we require (C, FX), etc.
  //      if the variable is C we require (C)
  //      I doubt that the extra calculations matter, but the storage overhead could
  //      matter since each 6D indexer contains 18 ints and we are always carrying around
  //      10 indexers per bound info even if the field isn't allocated
  if (nb.snb.level < mylevel) {
    for (auto el : {TE::CC, TE::F1, TE::F2, TE::F3, TE::E1, TE::E2, TE::E3, TE::NN})
      out.idxer[static_cast<int>(el)] = CalcIndices(
          nb, pmb, el, IndexRangeType::BoundaryExteriorRecv, true, {Nt, Nu, Nv});
  }
  return out;
}

BndInfo BndInfo::GetSendCCFluxCor(MeshBlock *pmb, const NeighborBlock &nb,
                                  std::shared_ptr<Variable<Real>> v,
                                  CommBuffer<buf_pool_t<Real>::owner_t> *buf) {
  BndInfo out;
  out.allocated = v->IsAllocated();
  out.alloc_status = v->GetAllocationStatus();
  if (!v->IsAllocated()) {
    // Not going to actually do anything with this buffer
    return out;
  }
  //out.buf = buf->buffer();
  out.LockedBufferCopy(buf);

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
  block_ownership_t owns(true);
  out.idxer[0] = SpatiallyMaskedIndexer6D(
      owns, {0, out.var.GetDim(6) - 1}, {0, out.var.GetDim(5) - 1},
      {0, out.var.GetDim(4) - 1}, {sk, ek}, {sj, ej}, {si, ei});
  return out;
}

BndInfo BndInfo::GetSetCCFluxCor(MeshBlock *pmb, const NeighborBlock &nb,
                                 std::shared_ptr<Variable<Real>> v,
                                 CommBuffer<buf_pool_t<Real>::owner_t> *buf) {
  BndInfo out;

  if (!v->IsAllocated() || buf->GetState() != BufferState::received) {
    out.allocated = false;
    out.alloc_status = v->GetAllocationStatus();
    return out;
  }
  out.allocated = true;
  out.alloc_status = v->GetAllocationStatus();
  //out.buf = buf->buffer();
  out.LockedBufferCopy(buf);

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
      ej -= pmb->block_size.nx(X2DIR) / 2;
    else
      sj += pmb->block_size.nx(X2DIR) / 2;
    if (nb.ni.fi2 == 0)
      ek -= pmb->block_size.nx(X3DIR) / 2;
    else
      sk += pmb->block_size.nx(X3DIR) / 2;
  } else if (nb.fid == BoundaryFace::inner_x2 || nb.fid == BoundaryFace::outer_x2) {
    out.dir = X2DIR;
    if (nb.fid == BoundaryFace::inner_x2)
      ej = sj;
    else
      sj = ++ej;
    if (nb.ni.fi1 == 0)
      ei -= pmb->block_size.nx(X1DIR) / 2;
    else
      si += pmb->block_size.nx(X1DIR) / 2;
    if (nb.ni.fi2 == 0)
      ek -= pmb->block_size.nx(X3DIR) / 2;
    else
      sk += pmb->block_size.nx(X3DIR) / 2;
  } else if (nb.fid == BoundaryFace::inner_x3 || nb.fid == BoundaryFace::outer_x3) {
    out.dir = X3DIR;
    if (nb.fid == BoundaryFace::inner_x3)
      ek = sk;
    else
      sk = ++ek;
    if (nb.ni.fi1 == 0)
      ei -= pmb->block_size.nx(X1DIR) / 2;
    else
      si += pmb->block_size.nx(X1DIR) / 2;
    if (nb.ni.fi2 == 0)
      ej -= pmb->block_size.nx(X2DIR) / 2;
    else
      sj += pmb->block_size.nx(X2DIR) / 2;
  } else {
    PARTHENON_FAIL("Flux corrections only occur on faces for CC variables.");
  }

  out.var = v->flux[out.dir];

  out.coords = pmb->coords;
  block_ownership_t owns(true);
  out.idxer[0] = SpatiallyMaskedIndexer6D(
      owns, {0, out.var.GetDim(6) - 1}, {0, out.var.GetDim(5) - 1},
      {0, out.var.GetDim(4) - 1}, {sk, ek}, {sj, ej}, {si, ei});
  return out;
}

} // namespace parthenon
