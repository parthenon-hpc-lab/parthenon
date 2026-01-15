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
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

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
    for (const auto &nloc : neighbors) {
      auto gid = forest.GetGid(nloc.global_loc, pmb->block_coarsenings);
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
      all_neighbors.emplace_back(pmb->pmy_mesh, nloc.global_loc, nloc.origin_loc,
                                 ranklist[lgid], gid, offsets, bid, tid, f[0], f[1], pmb->block_coarsenings);
      
      // Set neighbor block ownership
      auto &nb = all_neighbors.back();
      auto neighbor_neighbors = forest.FindNeighbors(nloc.global_loc, grid_id);

      nb.ownership =
          DetermineOwnership(nloc.global_loc, neighbor_neighbors, newly_refined);
      nb.ownership.initialized = true;

      // Set logical coordinate transformation from this block to the neighbor
      nb.lcoord_trans = nloc.lcoord_trans;

    }

    if (grid_id.type() == GridType::leaf) {
      pmb->neighbors = all_neighbors;
    } else if (grid_id.type() == GridType::two_level_composite &&
               pmb->loc.level() == grid_id.logical_level()) {
      pmb->gmg_same_neighbors = all_neighbors;
    } else if (grid_id.type() == GridType::two_level_composite &&
               pmb->loc.level() == grid_id.logical_level() - 1) {
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
  gmg_min_level_ = -gmg_level_offset;

  // Populate a list of multigrid grids from coarsest to finest level
  gmg_block_lists_.clear();
  gmg_grids_.clear();
  // Number of block_coarsenings to perform on the leaf grid
  const std::size_t base_block_coarsenings{0}; // TODO(LFR): Change this from zero

  // Build up the gmg grids
  for (int gmg_level = GetGMGMinLevel(); gmg_level <= GetGMGMaxLevel(); ++gmg_level) {
    int logical_level = use_negative_levels ? gmg_level : std::max(gmg_level, 0);
    std::size_t block_coarsenings = use_negative_levels ? 0 : std::max(-gmg_level, 0);
    gmg_block_lists_[gmg_level] = BlockList_t();
    gmg_grids_[gmg_level] = GridIdentifier::two_level_composite(logical_level, base_block_coarsenings + block_coarsenings);
    gmg_grids_[gmg_level].multigrid_level() = gmg_level;
  }
  
  // Only want to create one of each block at each position in the octree location - block_coarsenings
  // space, so store them in a map and populate it initially with the base blocks
  std::map<std::pair<LogicalLocation, int>, std::shared_ptr<MeshBlock>> all_blocks;
  for (auto &b : block_list)
    all_blocks[{b->loc, b->block_coarsenings}] = b;
  
  const std::size_t nnodes = forest.CountLeafMeshBlock() + forest.CountInternalMeshBlock();

  for (int gmg_level = GetGMGMaxLevel(); gmg_level >= GetGMGMinLevel(); --gmg_level) {
    auto grid = gmg_grids_[gmg_level];
    auto &cur_block_list = gmg_block_lists_[gmg_level];
    // Algorithm for building a two level composite grid
    // Loop over leaf blocks on this rank since we want parent blocks of leaf blocks stored on the 
    // same rank 
    for (auto &plmb: block_list) {
      auto leaf_loc = plmb->loc;
      if (leaf_loc.level() >= grid.logical_level() - 1) {
        // Logical location of the parent block, should naturally give leaf locations on the two-level
        // composite grid as well as internal node location on the two-level composite grid
        auto loc = leaf_loc.GetParent(std::max(leaf_loc.level() - grid.logical_level(), 0));
         // Only want to work with this block if the leaf is the lower left corner of the parent block
         // (otherwise all daughters would add the same parent and blocks could be duplicated across ranks)
        if (loc.morton() == leaf_loc.morton()) {
          if (all_blocks.count({loc, grid.block_coarsenings()})) { 
            // Block already exists, so just add pointer to list
            cur_block_list.push_back(all_blocks[{loc, grid.block_coarsenings()}]);
          } else {
            // Current block needs to be created
            RegionSize block_size = GetDefaultBlockSize();
            BoundaryFlag block_bcs[6];
            SetBlockSizeAndBoundaries(loc, block_size, block_bcs, grid.block_coarsenings());
            const int gid = forest.GetGid(loc, grid.block_coarsenings());
            auto new_block = MeshBlock::Make(gid, -1, loc, block_size, block_bcs,
                                             this, pin, app_in, packages, resolved_packages, gflag);
            new_block->block_coarsenings = grid.block_coarsenings();
            cur_block_list.push_back(new_block);
            all_blocks[{loc, grid.block_coarsenings()}] = new_block; 
          }
        }
      }
    }
  }
  
  // Sort the gmg block lists by gid
  for (auto &[level, bl] : gmg_block_lists_) {
    std::sort(bl.begin(), bl.end(), [](auto &a, auto &b) { return a->gid < b->gid; });
    BuildBlockPartitions(gmg_grids_[level]);
  }
}

void Mesh::SetGMGNeighbors() {
  if (!multigrid) return;
  const int gmg_min_level = GetGMGMinLevel();
  // Sort the gmg block lists by gid and find neighbors
  for (auto &[level, bl] : gmg_block_lists_) {
    auto cur_grid = gmg_grids_[level];
    for (auto &pmb : bl) {
      const bool boundary_block = cur_grid.type() == GridType::two_level_composite && pmb->loc.level() != cur_grid.logical_level();
      if (boundary_block) continue;
      // Coarser neighbor
      pmb->gmg_coarser_neighbors.clear();
      if (gmg_grids_.count(level - 1)) {
        auto coarse_grid = gmg_grids_.at(level - 1);
        // By default assume that there is a block level coarsening between the two
        // grids so that the blocks on both multi-grid levels have the same 
        // logical location
        auto coarse_loc = pmb->loc;
        // If they have the same number of block level block_coarsenings, replace with 
        // the logical location of the parent block
        if (coarse_grid.block_coarsenings() == cur_grid.block_coarsenings())
            coarse_loc = pmb->loc.GetParent();
        
        int gid = forest.GetGid(coarse_loc, coarse_grid.block_coarsenings());
        if (gid >= 0) {
          int leaf_gid = forest.GetLeafGid(coarse_loc);
          pmb->gmg_coarser_neighbors.emplace_back(
              pmb->pmy_mesh, coarse_loc, coarse_loc, ranklist[leaf_gid], gid,
              Kokkos::Array<int, 3>{0, 0, 0}, 0, 0, 0, 0, coarse_grid.block_coarsenings());
          // No need to explicitly set ownership (which defaults to
          // true), since the coarse block owns all elements of all
          // of its daughter blocks
        }
      }

      // Finer neighbor(s)
      pmb->gmg_finer_neighbors.clear();
      pmb->gmg_self_neighbors.clear();
      // Check if there is a finer grid below this one
      if (gmg_grids_.count(level + 1)) {
        auto fine_grid = gmg_grids_.at(level + 1);
        // There must be an internal coarsening between the two grids
        if (fine_grid.logical_level() == cur_grid.logical_level()) {
          PARTHENON_REQUIRE(fine_grid.block_coarsenings() == cur_grid.block_coarsenings() - 1, "Must be related by a single coarsening");
          int gid = forest.GetGid(pmb->loc, fine_grid.block_coarsenings());
          pmb->gmg_finer_neighbors.emplace_back(pmb->pmy_mesh, pmb->loc, pmb->loc, Globals::my_rank,
                                                  gid, Kokkos::Array<int, 3>{0, 0, 0}, 0,
                                                  0, 0, 0, fine_grid.block_coarsenings());
        } else { 
          for (auto &d : pmb->loc.GetDaughters(ndim)) {
            int gid = forest.GetGid(d, fine_grid.block_coarsenings());
            if (gid >= 0) {
              int leaf_gid = forest.GetLeafGid(d);
              pmb->gmg_finer_neighbors.emplace_back(pmb->pmy_mesh, d, d, ranklist[leaf_gid],
                                                    gid, Kokkos::Array<int, 3>{0, 0, 0}, 0,
                                                    0, 0, 0, fine_grid.block_coarsenings());
            }
          } 
        }
      }
       
      // Add the block as its own neighbor 
      // so that when restricting/prolongating between two two_level_composite 
      // grids, a block that lives on both levels communicates a message to itself, 
      // which must be received before operations can be performed on the next level
      pmb->gmg_self_neighbors.emplace_back(
          pmb->pmy_mesh, pmb->loc, pmb->loc, Globals::my_rank, pmb->gid,
          Kokkos::Array<int, 3>{0, 0, 0}, 0, 0, 0, 0, pmb->block_coarsenings);
      
      if (pmb->gmg_finer_neighbors.size() > 1) {
        // This block has multiple finer neighbors, so we need to set ownership
        // on shared elements in the interior of the coarse block. We do not need
        // to worry about coordinate transformations, since all daughter blocks
        // are guaranteed to be in the same logical coordinate system.
        std::vector<forest::NeighborLocation> neighbor_locs;
        for (const auto &n : pmb->gmg_finer_neighbors)
          neighbor_locs.emplace_back(n.loc, n.loc,
                                     forest::LogicalCoordinateTransformation());
        for (auto &n : pmb->gmg_finer_neighbors)
          n.ownership = DetermineOwnership(n.loc, neighbor_locs);
      }

      // Same level neighbors
      SetMeshBlockNeighbors(gmg_grids_[level], bl, ranklist);
    }
  }
}
} // namespace parthenon
