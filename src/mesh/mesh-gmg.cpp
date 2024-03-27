//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2024 The Parthenon collaboration
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
//! \file mesh_amr.cpp
//  \brief implementation of Mesh::AdaptiveMeshRefinement() and related utilities

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_set>

#include "parthenon_mpi.hpp"

#include "bvals/boundary_conditions.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "interface/update.hpp"
#include "mesh/forest/forest.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock.hpp"
#include "parthenon_arrays.hpp"
#include "utils/bit_hacks.hpp"
#include "utils/buffer_utils.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

void Mesh::SetMeshBlockNeighbors(
    GridIdentifier grid_id, BlockList_t &block_list, const std::vector<int> &ranklist,
    const std::unordered_set<LogicalLocation> &newly_refined) {
  Indexer3D offsets({ndim > 0 ? -1 : 0, ndim > 0 ? 1 : 0},
                    {ndim > 1 ? -1 : 0, ndim > 1 ? 1 : 0},
                    {ndim > 2 ? -1 : 0, ndim > 2 ? 1 : 0});
  BufferID buffer_id(ndim, multilevel);

  for (auto &pmb : block_list) {
    std::vector<NeighborBlock> all_neighbors;
    const auto &loc = pmb->loc;
    auto neighbors = forest.FindNeighbors(loc, grid_id);

    // Build NeighborBlocks for unique neighbors
    int buf_id = 0;
    for (const auto &nloc : neighbors) {
      auto gid = forest.GetGid(nloc.global_loc);
      auto offsets = loc.GetSameLevelOffsets(nloc.origin_loc);
      auto f =
          loc.GetAthenaXXFaceOffsets(nloc.origin_loc, offsets[0], offsets[1], offsets[2]);
      int bid = buffer_id.GetID(offsets[0], offsets[1], offsets[2], f[0], f[1]);

      // TODO(LFR): This will only give the correct buffer index if the two trees have the
      // same coordinate orientation. We really need to transform loc into the logical
      // coord system of the tree nloc.global_loc to get the true tid
      auto fn = nloc.origin_loc.GetAthenaXXFaceOffsets(loc, -offsets[0], -offsets[1],
                                                       -offsets[2]);
      int tid = buffer_id.GetID(-offsets[0], -offsets[1], -offsets[2], fn[0], fn[1]);
      int lgid = forest.GetLeafGid(nloc.global_loc);
      all_neighbors.emplace_back(pmb->pmy_mesh, nloc.global_loc, ranklist[lgid], gid,
                                 offsets, bid, tid, f[0], f[1]);

      // Set neighbor block ownership
      auto &nb = all_neighbors.back();
      auto neighbor_neighbors = forest.FindNeighbors(nloc.global_loc);

      nb.ownership =
          DetermineOwnership(nloc.global_loc, neighbor_neighbors, newly_refined);
      nb.ownership.initialized = true;
    }

    if (grid_id.type == GridType::leaf) {
      pmb->neighbors = all_neighbors;
    } else if (grid_id.type == GridType::two_level_composite &&
               pmb->loc.level() == grid_id.logical_level) {
      pmb->gmg_same_neighbors = all_neighbors;
    } else if (grid_id.type == GridType::two_level_composite &&
               pmb->loc.level() == grid_id.logical_level - 1) {
      pmb->gmg_composite_finer_neighbors = all_neighbors;
    }
  }
}

void Mesh::BuildGMGBlockLists(ParameterInput *pin, ApplicationInput *app_in) {
  if (!multigrid) return;

  // See how many times we can go below logical level zero based on the
  // number of times a blocks zones can be reduced by 2^D
  int gmg_level_offset = std::numeric_limits<int>::max();
  auto block_size_default = GetDefaultBlockSize();
  for (auto dir : {X1DIR, X2DIR, X3DIR}) {
    if (!mesh_size.symmetry(dir)) {
      int dir_allowed_levels = NumberOfBinaryTrailingZeros(block_size_default.nx(dir));
      gmg_level_offset = std::min(dir_allowed_levels, gmg_level_offset);
    }
  }

  const int gmg_min_level = -gmg_level_offset;
  gmg_min_logical_level_ = gmg_min_level;
  for (int level = gmg_min_level; level <= current_level; ++level) {
    gmg_block_lists[level] = BlockList_t();
    gmg_mesh_data[level] = DataCollection<MeshData<Real>>();
  }

  // Create MeshData objects for GMG
  for (auto &[l, mdc] : gmg_mesh_data)
    mdc.SetMeshPointer(this);

  // Fill/create gmg block lists based on this ranks block list
  for (auto &pmb : block_list) {
    const int level = pmb->loc.level();
    // Add the leaf block to its level
    gmg_block_lists[level].push_back(pmb);

    // Add the leaf block to the next finer level if required
    if (level < current_level) {
      gmg_block_lists[level + 1].push_back(pmb);
    }

    // Create internal blocks that share a Morton number with this block
    // and add them to gmg two-level composite grid block lists. This
    // determines which process internal blocks live on
    auto loc = pmb->loc.GetParent();
    while (loc.level() >= gmg_min_level && loc.morton() == pmb->loc.morton()) {
      RegionSize block_size = GetDefaultBlockSize();
      BoundaryFlag block_bcs[6];
      SetBlockSizeAndBoundaries(loc, block_size, block_bcs);
      gmg_block_lists[loc.level()].push_back(
          MeshBlock::Make(forest.GetGid(loc), -1, loc, block_size, block_bcs, this, pin,
                          app_in, packages, resolved_packages, gflag));
      loc = loc.GetParent();
    }
  }

  // Sort the gmg block lists by gid
  for (auto &[level, bl] : gmg_block_lists) {
    std::sort(bl.begin(), bl.end(), [](auto &a, auto &b) { return a->gid < b->gid; });
  }
}

void Mesh::SetGMGNeighbors() {
  if (!multigrid) return;
  const int gmg_min_level = GetGMGMinLevel();
  // Sort the gmg block lists by gid and find neighbors
  for (auto &[level, bl] : gmg_block_lists) {
    for (auto &pmb : bl) {
      // Coarser neighbor
      pmb->gmg_coarser_neighbors.clear();
      if (pmb->loc.level() > gmg_min_level) {
        auto ploc = pmb->loc.GetParent();
        int gid = forest.GetGid(ploc);
        if (gid >= 0) {
          int leaf_gid = forest.GetLeafGid(ploc);
          pmb->gmg_coarser_neighbors.emplace_back(pmb->pmy_mesh, ploc, ranklist[leaf_gid],
                                                  gid, std::array<int, 3>{0, 0, 0}, 0, 0,
                                                  0, 0);
        }
      }

      // Finer neighbor(s)
      pmb->gmg_finer_neighbors.clear();
      if (pmb->loc.level() < current_level) {
        auto dlocs = pmb->loc.GetDaughters(ndim);
        for (auto &d : dlocs) {
          int gid = forest.GetGid(d);
          if (gid >= 0) {
            int leaf_gid = forest.GetLeafGid(d);
            pmb->gmg_finer_neighbors.emplace_back(pmb->pmy_mesh, d, ranklist[leaf_gid],
                                                  gid, std::array<int, 3>{0, 0, 0}, 0, 0,
                                                  0, 0);
          }
        }
      }

      // Same level neighbors
      SetMeshBlockNeighbors(GridIdentifier::two_level_composite(level), bl, ranklist);
    }
  }
}
} // namespace parthenon
