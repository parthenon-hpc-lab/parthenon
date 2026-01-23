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
#include <unistd.h>

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

// Terminology and GridIdentifier conventions for Parthenon GMG on a forest-of-octrees AMR mesh
//
// The GMG hierarchy is constructed over two independent notions of coarsening:
//
//   (1) AMR/tree refinement, indexed by logical_level (Athena++ / Parthenon convention):
//         - Larger logical_level => finer AMR blocks.
//         - For a domain of size L in a given direction, the physical block size is
//               L_block = L / 2^logical_level
//           (ignoring that the domain is represented by a forest of octrees).
//
//   (2) In-block geometric coarsening, indexed by block_coarsenings:
//         - Starting from an N^3 block, after block_coarsenings=k the block resolution is
//               N_block = N / 2^k
//           in each direction (conceptually N^3 -> (N/2)^3 -> (N/4)^3 -> ...).
//
// GridIdentifier specifies which *set of blocks* constitutes a grid, and whether that
// grid participates in the GMG hierarchy (is_multigrid_).
//
// Important: multigrid_level ordering
//   In this implementation, larger multigrid_level values are *finer*.
//   The finest GMG grid corresponds to the finest level of the default Parthenon mesh,
//   stored in Mesh::current_level. Coarser GMG levels decrease multigrid_level.
//
// Grid types:
//
//   GridType::leaf
//     - Grid consisting only of AMR leaf blocks.
//     - logical_level_ is not meaningful (sentinel), since leaves span multiple AMR levels.
//     - block_coarsenings_ specifies in-block GMG coarsening.
//     - Two cases exist:
//         * Non-multigrid leaf grid (GridIdentifier::leaf()):
//             - is_multigrid_ = false
//             - Used for the base Parthenon grid on which the final solution is defined.
//         * Multigrid leaf grid (GridIdentifier::leaf(mg_level, k)):
//             - is_multigrid_ = true
//             - Leaf-only GMG level with block_coarsenings = k.
//
//   GridType::two_level_composite
//     - Composite GMG grid anchored at a target logical_level = ell.
//     - Contains (all at the specified block_coarsenings):
//         * level ell:
//             - leaf blocks at logical_level ell, and
//             - internal-node blocks at ell that are parents of finer-level leaf blocks
//               (to maintain coverage and coupling across refinement boundaries);
//         * level ell-1:
//             - leaf blocks at logical_level ell-1, used to provide boundary conditions
//               for the level-ell blocks.
//     - logical_level_ stores ell (the fine level of the composite).
//     - block_coarsenings_ specifies in-block GMG coarsening.
//     - is_multigrid_ = true.
//
// GMG hierarchy construction:
//
//   The number of initial multigrid leaf grids is controlled by
//
//       parthenon/mesh/base_block_coarsenings = n_leaf
//
//   Let max_level denote the finest GMG multigrid_level (typically Mesh::current_level).
//   The hierarchy consists of:
//
//     (A) Base (non-multigrid) leaf grid:
//           leaf()
//         This corresponds to the default Parthenon mesh resolution and is not part of GMG.
//
//     (B) n_leaf multigrid leaf grids with explicit coarsenings, ordered from fine to coarse
//         by decreasing multigrid_level:
//
//           leaf(max_level,         0),
//           leaf(max_level - 1,     1),
//           ...
//           leaf(max_level - (n_leaf - 1),  n_leaf - 1)
//
//     (C) Two-level composite grids, starting after the last leaf grid. The composite grids
//         begin with block_coarsenings fixed at n_leaf (i.e., one more than the last leaf grid
//         when n_leaf > 0), and proceed by moving up the AMR tree in logical_level.
//
//         During this phase, multigrid_level continues to decrease as grids become coarser,
//         while block_coarsenings remains fixed at n_leaf and logical_level decreases
//         (ell_max -> ... -> 0), e.g.:
//
//           two_level_composite(mg, ell_max,   n_leaf),
//           two_level_composite(mg-1, ell_max-1, n_leaf),
//           ...
//           two_level_composite(..., 0,       n_leaf)
//
//     (D) After reaching the root (logical_level = 0), further GMG coarsening proceeds by
//         increasing block_coarsenings at ell = 0:
//
//           two_level_composite(..., 0, n_leaf + 1),
//           two_level_composite(..., 0, n_leaf + 2),
//           ...
//
// Notes:
//   - block_coarsenings_ refers exclusively to in-block GMG coarsening, not AMR refinement.
//   - logical_level refers to AMR level; multigrid_level refers to GMG hierarchy index.
//   - operator< orders GridIdentifier by GridType, then block_coarsenings, then logical_level,
//     providing a stable ordering for associative containers.

void Mesh::BuildGMGBlockLists(ParameterInput *pin, ApplicationInput *app_in) {
  if (!multigrid) return;

  // See how many times we can go below logical level zero based on the
  // number of times a blocks zones can be reduced by 2^D
  int max_block_coarsenings = std::numeric_limits<int>::max();
  auto block_size_default = GetDefaultBlockSize();
  for (auto dir : {X1DIR, X2DIR, X3DIR}) {
    if (!mesh_size.symmetry(dir)) {
      int dir_allowed_levels = NumberOfBinaryTrailingZeros(block_size_default.nx(dir));
      max_block_coarsenings = std::min(dir_allowed_levels, max_block_coarsenings);
    }
  }
  gmg_min_level_ = -max_block_coarsenings;

  // Populate a list of multigrid
  gmg_block_lists_.clear();
  gmg_grids_.clear();
  
  PARTHENON_REQUIRE(base_block_coarsenings <= max_block_coarsenings,
                    "Asking for more block coarsenings than are allowed by the chosen meshblock size.");

  // Add initially coarsened leaf grids first
  for (int c = 0; c < base_block_coarsenings; ++c) {
    const int gmg_level = GetGMGMaxLevel() - c;
    gmg_grids_[gmg_level] = GridIdentifier::leaf(gmg_level, c);
  }

  // Build up the subsequent two-level composite grids 
  for (int gmg_level = GetGMGMinLevel(); gmg_level <= (GetGMGMaxLevel() - base_block_coarsenings); ++gmg_level) {
    int logical_level = std::max(gmg_level + base_block_coarsenings, 0);
    std::size_t block_coarsenings = std::max(-base_block_coarsenings - gmg_level, 0);
    gmg_block_lists_[gmg_level] = BlockList_t();
    gmg_grids_[gmg_level] = GridIdentifier::two_level_composite(gmg_level, logical_level, base_block_coarsenings + block_coarsenings);
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
    if (grid.type() == GridType::two_level_composite) {
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
    } else if (grid.type() == GridType::leaf) {
      for (auto &plmb : block_list) { 
        // Just add every block at this coarsening
        if (all_blocks.count({plmb->loc, grid.block_coarsenings()})) {
          cur_block_list.push_back(all_blocks[{plmb->loc, grid.block_coarsenings()}]);
        } else { 
          // Current block needs to be created
          RegionSize block_size = GetDefaultBlockSize();
          BoundaryFlag block_bcs[6];
          SetBlockSizeAndBoundaries(plmb->loc, block_size, block_bcs, grid.block_coarsenings());
          const int gid = forest.GetGid(plmb->loc, grid.block_coarsenings());
          auto new_block = MeshBlock::Make(gid, -1, plmb->loc, block_size, block_bcs,
                                           this, pin, app_in, packages, resolved_packages, gflag);
          new_block->block_coarsenings = grid.block_coarsenings();
          cur_block_list.push_back(new_block);
          all_blocks[{plmb->loc, grid.block_coarsenings()}] = new_block;
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
  auto write_block_str = [](const auto &in) { 
    return std::to_string(in.gid) + " " + in.loc.label() + "{" + std::to_string(in.block_coarsenings) + "}"; 
  };
  if (!multigrid) return;
  const int gmg_min_level = GetGMGMinLevel();
  // Sort the gmg block lists by gid and find neighbors
  for (auto &[gmg_level, bl] : gmg_block_lists_) {
    auto cur_grid = gmg_grids_[gmg_level];
    for (auto &pmb : bl) {
      // Set relevant neighbors for this grid on pmb
      SetMeshBlockNeighbors(gmg_grids_[gmg_level], bl, ranklist);
      
      // Don't set inter-grid neighbors if this is a boundary block, since it is shared by another 
      // multigrid level and its inter-grid neighbors will be set there
      const bool boundary_block = cur_grid.type() == GridType::two_level_composite && pmb->loc.level() != cur_grid.logical_level();
      if (boundary_block) continue;

      // Coarser neighbor
      pmb->gmg_coarser_neighbors.clear();
      if (gmg_grids_.count(gmg_level - 1)) {
        auto coarse_grid = gmg_grids_.at(gmg_level - 1);
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
      const bool is_leaf = forest.GetGid(pmb->loc) == forest.GetLeafGid(pmb->loc);
      
      // Possibilities:
      // 1. This is a leaf block 
      //   a. It has zero internal coarsenings, so no finer neighbors
      //   b. It has internal coarsenings, so one finer neighbor at the same logical location but with one less internal coarsening
      // 2. This is an internal block, which implies it must be on a two-level composite grid and the next finer grid must also
      //    be a two-level composite grid
      //   a. If the next finer grid has the same number of block coarsenings, it must have 2^D finer neighbors on that grid
      //   b. Otherwise, there is a coarsening between the two-level composite grids
      if (is_leaf) {
        if (pmb->block_coarsenings > 0) {
          int gid = forest.GetGid(pmb->loc, pmb->block_coarsenings - 1);
          pmb->gmg_finer_neighbors.emplace_back(pmb->pmy_mesh, pmb->loc, pmb->loc, Globals::my_rank,
                                                  gid, Kokkos::Array<int, 3>{0, 0, 0}, 0,
                                                  0, 0, 0, pmb->block_coarsenings - 1); 
        }
      } else { 
        PARTHENON_REQUIRE(gmg_grids_.count(gmg_level + 1), "Must have a finer grid than this one if there is an internal block.");
        auto fine_grid = gmg_grids_.at(gmg_level + 1);
        if (fine_grid.block_coarsenings() == pmb->block_coarsenings) { 
          for (auto &d : pmb->loc.GetDaughters(ndim)) {
            int gid = forest.GetGid(d, fine_grid.block_coarsenings());
            if (gid >= 0) {
              int leaf_gid = forest.GetLeafGid(d);
              pmb->gmg_finer_neighbors.emplace_back(pmb->pmy_mesh, d, d, ranklist[leaf_gid],
                                                    gid, Kokkos::Array<int, 3>{0, 0, 0}, 0,
                                                    0, 0, 0, fine_grid.block_coarsenings());
            }
          }  
        } else { 
          PARTHENON_REQUIRE(fine_grid.block_coarsenings() == cur_grid.block_coarsenings() - 1, "Grids must be related by a single coarsening");
          int gid = forest.GetGid(pmb->loc, fine_grid.block_coarsenings());
          pmb->gmg_finer_neighbors.emplace_back(pmb->pmy_mesh, pmb->loc, pmb->loc, Globals::my_rank,
                                                  gid, Kokkos::Array<int, 3>{0, 0, 0}, 0,
                                                  0, 0, 0, fine_grid.block_coarsenings()); 
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
    } 
  }
}
} // namespace parthenon
