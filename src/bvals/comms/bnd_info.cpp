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
#include "bvals/comms/bvals_utils.hpp"
#include "bvals/comms/calc_indices.hpp"
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

bool BvarsSubCache_t::RequiresReinitialize(Mesh *pmesh) const {
  return buf_vec.size() == 0 || epoch != pmesh->boundary_comm_map.GetCurrentEpoch();
}

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

bool NeighborIsCoarser(MeshBlock *pmb, const NeighborBlock &nb) {
  return nb.loc.level() < pmb->loc.level() ||
         nb.block_coarsenings > pmb->block_coarsenings;
}

bool NeighborIsFiner(MeshBlock *pmb, const NeighborBlock &nb) {
  return nb.loc.level() > pmb->loc.level() ||
         nb.block_coarsenings < pmb->block_coarsenings;
}

bool NeighborIsSame(MeshBlock *pmb, const NeighborBlock &nb) {
  return nb.loc.level() == pmb->loc.level() &&
         nb.block_coarsenings == pmb->block_coarsenings;
}

// Thin wrapper preserving the historical MeshBlock-taking signature. Packs the block's
// geometry into a BlockInfo and forwards to the CalcIndices in calc_indices.hpp. The
// receiving-block box can be computed without a live MeshBlock via the templated routine
// directly.
SpatiallyMaskedIndexer6D
CalcIndices(const NeighborBlock &nb, MeshBlock *pmb,
            const std::shared_ptr<Variable<Real>> &v, TopologicalElement el,
            IndexRangeType ir_type, bool prores,
            const forest::LogicalCoordinateTransformation &lcoord_trans =
                forest::LogicalCoordinateTransformation()) {
  return CalcIndices(nb, BlockInfo(pmb), pmb->pmy_mesh->multilevel, v, el, ir_type,
                     prores, lcoord_trans);
}

int GetBufferSize(const MeshBlock *const pmb, const NeighborBlock &nb,
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

  // Sometimes we may build a BndInfo object just to get the
  // size of the index space associated with the boundary. In
  // that case an associated communication buffer may not exist
  // and a nullptr will be passed instead.
  if (combuf != nullptr) buf = combuf->buffer();
  same_to_same = pmb->gid == nb.gid && nb.offsets.IsCell();
  lcoord_trans = nb.lcoord_trans;

  if (NeighborIsCoarser(pmb, nb)) {
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

  auto buf_state = buf != nullptr ? buf->GetState() : BufferState::received;
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

  if (NeighborIsCoarser(pmb, nb)) {
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

  if (NeighborIsCoarser(pmb, nb)) {
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

  if (NeighborIsCoarser(pmb, nb)) {
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
  bool restricted = pmb->HasCoarserNeighbors();

  for (auto el : v->GetTopologicalElements()) {
    out.IncludeTopoEl(el) = true;
    if (NeighborIsCoarser(pmb, nb)) {
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
  if (NeighborIsCoarser(pmb, nb)) {
    for (auto el : {TE::CC, TE::F1, TE::F2, TE::F3, TE::E1, TE::E2, TE::E3, TE::NN})
      out.idxer[static_cast<int>(el)] =
          CalcIndices(nb, pmb, v, el, IndexRangeType::BoundaryExteriorRecv, true);
  }
  return out;
}
} // namespace parthenon
