//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2022 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2024. Triad National Security, LLC. All rights reserved.
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
#include <cstdio>
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

namespace parthenon {

void ProResCache_t::Initialize(int n_regions, StateDescriptor *pkg) {
  prores_info = ProResInfoArr_t(ViewOfViewAlloc("prores_info"), n_regions);
  prores_info_h = create_view_of_view_mirror(prores_info);
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
  if (v->HasRefinementOps()) {
    // var must be registered for refinement
    // note this condition means that each subset contains
    // both prolongation and restriction conditions. The
    // `RefinementOp_t` in `BndInfo` is assumed to
    // differentiate.
    std::size_t rfid = pkg->RefinementFuncID((v->GetRefinementFunctions()));
    buffer_subsets_h(rfid, buffer_subset_sizes[rfid]++) = region;
  }
}

// Determines which topological elements need to be restricted and communicated for flux
// correction, which only occurs on shared elements between two blocks
std::vector<TopologicalElement>
GetFluxCorrectionElements(const std::shared_ptr<Variable<Real>> &v,
                          const CellCentOffsets &offsets) {
  using TE = TopologicalElement;
  std::vector<TopologicalElement> elements;
  if (v->IsSet(Metadata::Face)) {
    if (offsets.IsFace()) {
      if (std::abs(offsets(X1DIR))) elements = {TE::F1};
      if (std::abs(offsets(X2DIR))) elements = {TE::F2};
      if (std::abs(offsets(X3DIR))) elements = {TE::F3};
    } else {
      PARTHENON_FAIL("Flux correction for face fluxes only occurs on shared faces.");
    }
  } else if (v->IsSet(Metadata::Edge)) {
    if (offsets.IsFace()) {
      if (std::abs(offsets(X1DIR))) elements = {TE::E2, TE::E3};
      if (std::abs(offsets(X2DIR))) elements = {TE::E3, TE::E1};
      if (std::abs(offsets(X3DIR))) elements = {TE::E1, TE::E2};
    } else if (offsets.IsEdge()) {
      if (offsets(X1DIR) == 0) elements = {TE::E1};
      if (offsets(X2DIR) == 0) elements = {TE::E2};
      if (offsets(X3DIR) == 0) elements = {TE::E3};
    } else {
      PARTHENON_FAIL(
          "Flux correction for edge fluxes only occurs on shared faces and edges.");
    }
  } else if (v->IsSet(Metadata::Node)) {
    elements = {TE::NN};
  } else {
    PARTHENON_FAIL("Only faces, edges, and nodes can be fluxes.");
  }
  return elements;
}

SpatiallyMaskedIndexer6D
CalcIndices(const NeighborBlock &nb, MeshBlock *pmb,
            const std::shared_ptr<Variable<Real>> &v, TopologicalElement el,
            IndexRangeType ir_type, bool prores,
            const forest::LogicalCoordinateTransformation &lcoord_trans =
                forest::LogicalCoordinateTransformation()) {
  std::array<int, 3> tensor_shape{v->GetDim(6), v->GetDim(5), v->GetDim(4)};
  const bool flux = v->IsSet(Metadata::Flux);

  const auto &loc = pmb->loc;
  bool is_fine_field = v->IsSet(Metadata::Fine);
  auto shape = is_fine_field ? pmb->f_cellbounds : pmb->cellbounds;
  // Both prolongation and restriction always operate in the coarse
  // index space. Also need to use the coarse index space if the
  // neighbor is coarser than you, wether or not you are setting
  // interior or exterior cells
  if (prores || nb.loc.level() < loc.level())
    shape = is_fine_field ? pmb->cellbounds : pmb->c_cellbounds;

  // Re-create the index space for the neighbor block (either the main block or
  // the coarse buffer as required)
  int fine_field_fac = is_fine_field ? 2 : 1;
  int coarse_fac = nb.loc.level() > loc.level() ? 2 : 1;
  auto neighbor_shape =
      IndexShape(nb.block_size.nx(X3DIR) * fine_field_fac / coarse_fac,
                 nb.block_size.nx(X2DIR) * fine_field_fac / coarse_fac,
                 nb.block_size.nx(X1DIR) * fine_field_fac / coarse_fac, Globals::nghost);

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
  std::array<int, 3> block_offset = nb.offsets;

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
      if ((loc.level() < nb.origin_loc.level()) &&
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
        s[dir] += nb.origin_loc.l(dir) % 2 == 1 ? extra_zones - interior_offset : 0;
        e[dir] -= nb.origin_loc.l(dir) % 2 == 0 ? extra_zones - interior_offset : 0;
        if (ir_type == IndexRangeType::InteriorSend && !prores) {
          // Include ghosts of finer block coarse array in message
          s[dir] -= Globals::nghost;
          e[dir] += Globals::nghost;
        }
      }
      if (loc.level() > nb.origin_loc.level() && not_symmetry[dir]) {
        // If we are setting (i.e. have non-zero exterior_offset) from a neighbor block
        // that is coarser, we got extra ghost zones from the neighbor (see inclusion of
        // interior_offset in the above if block)
        s[dir] -= loc.l(dir) % 2 == 1 ? exterior_offset : 0;
        e[dir] += loc.l(dir) % 2 == 0 ? exterior_offset : 0;
        if (ir_type == IndexRangeType::InteriorRecv && !prores) {
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
      // Fluxes are only communicated on shared elements
      s[dir] = bounds[dir].e + (flux ? 0 : -interior_offset + 1 - top_offset[dir]);
      e[dir] = bounds[dir].e + (flux ? 0 : exterior_offset);
    } else {
      s[dir] = bounds[dir].s + (flux ? 0 : -exterior_offset);
      e[dir] = bounds[dir].s + (flux ? 0 : interior_offset - 1 + top_offset[dir]);
    }
  }

  // Transform to logical coordinates of neighbor block if this
  // is a receiving block
  if (ir_type == IndexRangeType::BoundaryExteriorRecv) {
    s = lcoord_trans.Transform(s);
    e = lcoord_trans.Transform(e);
    // Transformation can flip the order of the upper and
    // lower index, so make sure they are increasing
    for (int dir = 0; dir < 3; ++dir) {
      if (s[dir] > e[dir]) {
        int temp = s[dir];
        s[dir] = e[dir];
        e[dir] = temp;
      }
    }
  }

  block_ownership_t owns(true);
  // Although it wouldn't hurt to include ownership when producing an interior
  // index range, it is unecessary. This is probably not immediately obvious,
  // but it is possible to convince oneself that dealing with ownership in
  // only exterior index ranges works correctly
  if (ir_type == IndexRangeType::BoundaryExteriorRecv) {
    int sox1 = -block_offset[0];
    int sox2 = -block_offset[1];
    int sox3 = -block_offset[2];
    if (nb.origin_loc.level() < loc.level()) {
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
  auto &cb = v->IsSet(Metadata::Fine) ? pmb->f_cellbounds : pmb->cellbounds;
  int topo_comp = (v->IsSet(Metadata::Face) || v->IsSet(Metadata::Edge)) ? 3 : 1;
  const IndexDomain in = IndexDomain::entire;
  // The plus 2 instead of 1 is to account for the possible size of face, edge, and nodal
  // fields
  const int isize = cb.ie(in) - cb.is(in) + 2;
  const int jsize = cb.je(in) - cb.js(in) + 2;
  const int ksize = cb.ke(in) - cb.ks(in) + 2;
  return (nb.offsets(X1DIR) == 0 ? isize : Globals::nghost + 1) *
         (nb.offsets(X2DIR) == 0 ? jsize : Globals::nghost + 1) *
         (nb.offsets(X3DIR) == 0 ? ksize : Globals::nghost + 1) * v->GetDim(6) *
         v->GetDim(5) * v->GetDim(4) * topo_comp;
}

BndInfo::BndInfo(MeshBlock *pmb, const NeighborBlock &nb,
                 std::shared_ptr<Variable<Real>> v,
                 CommBuffer<buf_pool_t<Real>::owner_t> *combuf,
                 IndexRangeType idx_range_type) {
  allocated = v->IsAllocated();
  alloc_status = v->GetAllocationStatus();

  buf = combuf->buffer();
  same_to_same = pmb->gid == nb.gid && nb.offsets.IsCell();
  lcoord_trans = nb.lcoord_trans;
  if (!allocated) return;

  if (nb.origin_loc.level() < pmb->loc.level()) {
    var = v->coarse_s.Get();
  } else {
    var = v->data.Get();
  }

  coords = pmb->coords;

  auto elements = v->GetTopologicalElements();
  if (v->IsSet(Metadata::Flux)) elements = GetFluxCorrectionElements(v, nb.offsets);
  ntopological_elements = elements.size();

  lcoord_trans.ncell = var.GetDim(1);
  int idx{0};
  for (auto el : elements) {
    topo_idx[idx] = el;
    if (idx_range_type == IndexRangeType::BoundaryExteriorRecv)
      el = std::get<0>(lcoord_trans.InverseTransform(el));
    idxer[idx] = CalcIndices(nb, pmb, v, el, idx_range_type, false, lcoord_trans);
    idx++;
  }
}

BndInfo BndInfo::GetSendBndInfo(MeshBlock *pmb, const NeighborBlock &nb,
                                std::shared_ptr<Variable<Real>> v,
                                CommBuffer<buf_pool_t<Real>::owner_t> *buf) {
  auto idx_range_type = IndexRangeType::BoundaryInteriorSend;
  // Test if the neighbor block is not offset from this block (i.e. is a
  // parent or daughter block of pmb), and change the IndexRangeType
  // accordingly
  if (nb.offsets.IsCell()) idx_range_type = IndexRangeType::InteriorSend;
  return BndInfo(pmb, nb, v, buf, idx_range_type);
}

BndInfo BndInfo::GetSetBndInfo(MeshBlock *pmb, const NeighborBlock &nb,
                               std::shared_ptr<Variable<Real>> v,
                               CommBuffer<buf_pool_t<Real>::owner_t> *buf) {
  auto idx_range_type = IndexRangeType::BoundaryExteriorRecv;
  // Test if the neighbor block is not offset from this block (i.e. is a
  // parent or daughter block of pmb), and change the IndexRangeType
  // accordingly
  if (nb.offsets.IsCell()) idx_range_type = IndexRangeType::InteriorRecv;
  BndInfo out(pmb, nb, v, buf, idx_range_type);

  auto buf_state = buf->GetState();
  if (buf_state == BufferState::received) {
    out.buf_allocated = true;
  } else if (buf_state == BufferState::received_null) {
    out.buf_allocated = false;
  } else {
    printf("%i [rank: %i] -> %i [rank: %i] (Set %s) is in state %i.\n", nb.gid, nb.rank,
           pmb->gid, Globals::my_rank, v->label().c_str(), static_cast<int>(buf_state));
    PARTHENON_FAIL("Buffer should be in a received state.");
  }
  return out;
}

ProResInfo::ProResInfo(MeshBlock *pmb, const NeighborBlock &nb,
                       std::shared_ptr<Variable<Real>> v) {
  allocated = v->IsAllocated();
  alloc_status = v->GetAllocationStatus();
  ntopological_elements = v->GetTopologicalElements().size();
  coords = pmb->coords;

  if (pmb->pmr) coarse_coords = pmb->pmr->GetCoarseCoords();

  fine = v->data.Get();
  coarse = v->coarse_s.Get();
}

ProResInfo ProResInfo::GetInteriorRestrict(MeshBlock *pmb, const NeighborBlock &nb,
                                           std::shared_ptr<Variable<Real>> v) {
  ProResInfo out(pmb, nb, v);
  if (!out.allocated) return out;

  if (nb.loc.level() < pmb->loc.level()) {
    for (auto el : v->GetTopologicalElements()) {
      out.IncludeTopoEl(el) = true;
      out.idxer[static_cast<int>(el)] =
          CalcIndices(nb, pmb, v, el, IndexRangeType::InteriorSend, true);
    }
    out.refinement_op = RefinementOp_t::Restriction;
  }
  return out;
}

ProResInfo ProResInfo::GetInteriorProlongate(MeshBlock *pmb, const NeighborBlock &nb,
                                             std::shared_ptr<Variable<Real>> v) {
  ProResInfo out(pmb, nb, v);
  if (!out.allocated) return out;

  if (nb.loc.level() < pmb->loc.level()) {
    for (auto el : v->GetTopologicalElements())
      out.IncludeTopoEl(el) = true;
    for (auto el : {TE::CC, TE::F1, TE::F2, TE::F3, TE::E1, TE::E2, TE::E3, TE::NN})
      out.idxer[static_cast<int>(el)] =
          CalcIndices(nb, pmb, v, el, IndexRangeType::InteriorRecv, true);
    out.refinement_op = RefinementOp_t::Prolongation;
  }
  return out;
}

ProResInfo ProResInfo::GetSend(MeshBlock *pmb, const NeighborBlock &nb,
                               std::shared_ptr<Variable<Real>> v) {
  ProResInfo out(pmb, nb, v);
  if (!out.allocated) return out;

  if (nb.origin_loc.level() < pmb->loc.level()) {
    auto elements = v->GetTopologicalElements();
    if (v->IsSet(Metadata::Flux)) elements = GetFluxCorrectionElements(v, nb.offsets);
    for (auto el : elements) {
      out.IncludeTopoEl(el) = true;
      out.idxer[static_cast<int>(el)] =
          CalcIndices(nb, pmb, v, el, IndexRangeType::BoundaryInteriorSend, true);
    }
    out.refinement_op = RefinementOp_t::Restriction;
  }
  return out;
}

ProResInfo ProResInfo::GetSet(MeshBlock *pmb, const NeighborBlock &nb,
                              std::shared_ptr<Variable<Real>> v) {
  ProResInfo out(pmb, nb, v);

  // This will select a superset of the boundaries that actually need to be restricted,
  // more logic could be added to only restrict boundary regions that abut boundary
  // regions that were filled by coarser neighbors
  bool restricted = false;
  int mylevel = pmb->loc.level();
  if (mylevel > 0) {
    for (const auto &nb : pmb->neighbors) {
      restricted = restricted || (nb.origin_loc.level() == (mylevel - 1));
    }
  }

  for (auto el : v->GetTopologicalElements()) {
    out.IncludeTopoEl(el) = true;
    if (nb.origin_loc.level() < mylevel) {
      out.refinement_op = RefinementOp_t::Prolongation;
    } else {
      if (restricted) {
        out.refinement_op = RefinementOp_t::Restriction;
        out.idxer[static_cast<int>(el)] =
            CalcIndices(nb, pmb, v, el, IndexRangeType::BoundaryExteriorRecv, true);
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
  if (nb.origin_loc.level() < mylevel) {
    for (auto el : {TE::CC, TE::F1, TE::F2, TE::F3, TE::E1, TE::E2, TE::E3, TE::NN})
      out.idxer[static_cast<int>(el)] =
          CalcIndices(nb, pmb, v, el, IndexRangeType::BoundaryExteriorRecv, true);
  }
  return out;
}
} // namespace parthenon
