//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2026. Triad National Security, LLC. All rights reserved.
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

#ifndef BVALS_COMMS_CALC_INDICES_HPP_
#define BVALS_COMMS_CALC_INDICES_HPP_

#include <algorithm>
#include <array>

#include "basic_types.hpp"
#include "bvals/comms/bnd_info.hpp"
#include "bvals/neighbor_block.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "interface/metadata.hpp"
#include "mesh/domain.hpp"
#include "mesh/forest/block_ownership.hpp"
#include "mesh/forest/logical_coordinate_transformation.hpp"
#include "utils/error_checking.hpp"
#include "utils/indexer.hpp"

namespace parthenon {

// Neighbor level relative to a block, derived from loc level and coarsening count.
inline bool NeighborIsCoarser(const BlockInfo &binfo, const NeighborBlock &nb) {
  return nb.loc.level() < binfo.loc.level() ||
         nb.block_coarsenings > binfo.block_coarsenings;
}
inline bool NeighborIsFiner(const BlockInfo &binfo, const NeighborBlock &nb) {
  return nb.loc.level() > binfo.loc.level() ||
         nb.block_coarsenings < binfo.block_coarsenings;
}
inline bool NeighborIsSame(const BlockInfo &binfo, const NeighborBlock &nb) {
  return nb.loc.level() == binfo.loc.level() &&
         nb.block_coarsenings == binfo.block_coarsenings;
}

// Reconstruct a block's fine/regular/coarse index shapes from its size alone, matching
// GetIndexShapes in meshblock.cpp (using block_size.symmetry in place of mesh_size).
inline std::array<IndexShape, 3> CalcIndexShapes(const RegionSize &block_size,
                                                 bool multilevel) {
  // Symmetry directions carry nx==1 in RegionSize but must be treated as zero-dimensional
  // (as MeshBlock does by passing 0 for inactive directions to InitializeIndexShapes).
  const int nx1 = block_size.symmetry(X1DIR) ? 0 : block_size.nx(X1DIR);
  const int nx2 = block_size.symmetry(X2DIR) ? 0 : block_size.nx(X2DIR);
  const int nx3 = block_size.symmetry(X3DIR) ? 0 : block_size.nx(X3DIR);
  IndexShape cellbounds(nx3, nx2, nx1, Globals::nghost);
  IndexShape f_cellbounds(2 * nx3, 2 * nx2, 2 * nx1, Globals::nghost);
  IndexShape c_cellbounds(nx3 / 2, nx2 / 2, nx1 / 2, 0);
  if (multilevel) {
    // Prevent the coarse bounds from going to zero
    int cnx1 = block_size.symmetry(X1DIR) ? 0 : std::max(1, nx1 / 2);
    int cnx2 = block_size.symmetry(X2DIR) ? 0 : std::max(1, nx2 / 2);
    int cnx3 = block_size.symmetry(X3DIR) ? 0 : std::max(1, nx3 / 2);
    c_cellbounds = IndexShape(cnx3, cnx2, cnx1, Globals::nghost);
  }
  return {cellbounds, f_cellbounds, c_cellbounds};
}

// Compute the 6D index range (and ownership mask) of a boundary region on a block. The
// block is described by a lightweight BlockInfo (no live MeshBlock is required), the
// neighbor by a NeighborBlock descriptor, and the field by any pointer-like type
// providing GetDim(4..6) and IsSet(MetadataFlag) (a std::shared_ptr<Variable<Real>>
// today; a TensorTrainT in the future).
template <class Field>
SpatiallyMaskedIndexer6D
CalcIndices(const NeighborBlock &nb, const BlockInfo &binfo, bool multilevel,
            const Field &v, TopologicalElement el, IndexRangeType ir_type, bool prores,
            const forest::LogicalCoordinateTransformation &lcoord_trans =
                forest::LogicalCoordinateTransformation()) {
  std::array<int, 3> tensor_shape{v->GetDim(6), v->GetDim(5), v->GetDim(4)};
  const bool flux = v->IsSet(Metadata::Flux);

  const auto &loc = binfo.loc;
  bool is_fine_field = v->IsSet(Metadata::Fine);
  auto shapes = CalcIndexShapes(binfo.block_size, multilevel);
  const IndexShape &cellbounds = shapes[0];
  const IndexShape &f_cellbounds = shapes[1];
  const IndexShape &c_cellbounds = shapes[2];
  auto shape = is_fine_field ? f_cellbounds : cellbounds;

  const bool nb_is_coarser = NeighborIsCoarser(binfo, nb);
  const bool nb_is_finer = NeighborIsFiner(binfo, nb);
  const bool nb_is_same = NeighborIsSame(binfo, nb);
  PARTHENON_REQUIRE(nb_is_coarser + nb_is_finer + nb_is_same == 1,
                    "Only one should be set.");
  // Both prolongation and restriction always operate in the coarse
  // index space. Also need to use the coarse index space if the
  // neighbor is coarser than you, wether or not you are setting
  // interior or exterior cells
  if (prores || nb_is_coarser)
    shape = is_fine_field ? cellbounds : c_cellbounds;

  // Re-create the index space for the neighbor block (either the main block or
  // the coarse buffer as required)
  int fine_field_fac = is_fine_field ? 2 : 1;
  int coarse_fac = nb_is_finer ? 2 : 1;
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

  std::array<bool, 3> not_symmetry{!binfo.block_size.symmetry(X1DIR),
                                   !binfo.block_size.symmetry(X2DIR),
                                   !binfo.block_size.symmetry(X3DIR)};
  // Account for the fact that the neighbor block may duplicate
  // some active zones on the loading block for face, edge, and nodal
  // fields, so the boundary of the neighbor block is one deeper into
  // the current block in some cases
  std::array<int, 3> top_offset{TopologicalOffsetI(el), TopologicalOffsetJ(el),
                                TopologicalOffsetK(el)};
  std::array<int, 3> block_offset = nb.offsets;

  int communicated_ghosts = Globals::nghost;
  if (!prores && nb_is_same && v->IsSet(Metadata::CommunicateOne))
    communicated_ghosts = 1;
  int interior_offset =
      ir_type == IndexRangeType::BoundaryInteriorSend ? communicated_ghosts : 0;
  int exterior_offset =
      ir_type == IndexRangeType::BoundaryExteriorRecv ? communicated_ghosts : 0;
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
      // Check that this dimension has ghost zones
      if (nb_is_finer && not_symmetry[dir]) {
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
      if (nb_is_coarser && not_symmetry[dir]) {
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
    if (nb_is_coarser) {
      // For coarse to fine interfaces, we are passing zones from only an
      // interior corner of the cell, never an entire face or edge
      if (sox1 == 0) sox1 = loc.l(0) % 2 == 1 ? 1 : -1;
      if (sox2 == 0) sox2 = loc.l(1) % 2 == 1 ? 1 : -1;
      if (sox3 == 0) sox3 = loc.l(2) % 2 == 1 ? 1 : -1;
    }
    owns = GetIndexRangeMaskFromOwnership(el, nb.ownership, sox1, sox2, sox3);
  } else if (ir_type == IndexRangeType::InteriorRecv) {
    // Also need to set ownership when a parent block receives from a daughter
    // block during multigrid operations
    owns = GetIndexRangeMaskFromOwnership(el, nb.ownership, 0, 0, 0);
  }
  return SpatiallyMaskedIndexer6D(owns, {0, tensor_shape[0] - 1},
                                  {0, tensor_shape[1] - 1}, {0, tensor_shape[2] - 1},
                                  {s[2], e[2]}, {s[1], e[1]}, {s[0], e[0]});
}

} // namespace parthenon

#endif // BVALS_COMMS_CALC_INDICES_HPP_
