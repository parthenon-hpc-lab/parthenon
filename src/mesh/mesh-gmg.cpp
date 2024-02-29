//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2023 The Parthenon collaboration
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
#include "mesh/forest.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/meshblock_tree.hpp"
#include "parthenon_arrays.hpp"
#include "utils/bit_hacks.hpp"
#include "utils/buffer_utils.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

void Mesh::PopulateLeafLocationMap() {
  const int nbtot = ranklist.size();
  leaf_grid_locs.clear();
  for (int ib = 0; ib < nbtot; ++ib) {
    leaf_grid_locs[loclist[ib]] = std::make_pair(ib, ranklist[ib]);
  }
}

void Mesh::SetForestNeighbors(BlockList_t &block_list, int nbs,
                              const std::unordered_set<LogicalLocation> &newly_refined) {
  Indexer3D offsets({ndim > 0 ? -1 : 0, ndim > 0 ? 1 : 0},
                    {ndim > 1 ? -1 : 0, ndim > 1 ? 1 : 0},
                    {ndim > 2 ? -1 : 0, ndim > 2 ? 1 : 0});
  printf("Calling set neighbors\n");
  for (auto &pmb : block_list) {
    std::vector<NeighborBlock> all_neighbors;
    auto neighbors = forest.FindNeighbors(pmb->loc);

    // TODO(LFR): Remove the next three lines when done comparing to old results
    std::unordered_set<LogicalLocation> old_possible_neighbor_set;
    old_possible_neighbor_set.insert(pmb->loc);
    for (auto &n : neighbors)
      old_possible_neighbor_set.insert(n.global_loc);

    // Build NeighborBlocks for unique neighbors
    for (const auto &nloc : neighbors) {
      auto gid = forest.GetGid(nloc.global_loc);
      auto offsets = pmb->loc.GetSameLevelOffsetsForest(nloc.origin_loc);
      // TODO(LFR): Get the rank here correctly
      int rank = 0;
      auto f = pmb->loc.GetAthenaXXFaceOffsets(nloc.origin_loc, offsets[0], offsets[1],
                                               offsets[2]);
      all_neighbors.emplace_back(pmb->pmy_mesh, nloc.global_loc, rank, gid, offsets, f[0],
                                 f[1]);

      // Set neighbor block ownership
      auto &nb = all_neighbors.back();
      auto neighbor_neighbors = forest.FindNeighbors(nb.loc);

      // TODO(LFR): Remove the next six lines once done testing. This is only
      // to ensure compatibility with the old infrastructure which
      // didn't bother to set neighbor ownership correctly where it
      // is unused.
      std::vector<NeighborLocation> reduced_neighbor_neighbors;
      for (const auto &n : neighbor_neighbors) {
        if (old_possible_neighbor_set.count(n.global_loc)) {
          reduced_neighbor_neighbors.push_back(n);
        }
      }
      nb.ownership =
          DetermineOwnershipForest(nb.loc, reduced_neighbor_neighbors, newly_refined);
      nb.ownership.initialized = true;
    }

    // TODO(LFR): Remove these checks
    // Just check that we agree for now
    PARTHENON_REQUIRE(all_neighbors.size() == pmb->neighbors.size(),
                      "Didn't find the same number of neighbors.");
    for (auto &onb : pmb->neighbors) {
      bool found = false;
      for (auto &nb : all_neighbors)
        if (nb.loc == onb.loc) {
          PARTHENON_REQUIRE(nb.ni == onb.ni,
                            "Bad neighbor indices relative to old neighbor finding");
          PARTHENON_REQUIRE(nb.snb == onb.snb,
                            "Old neighbor finding and new neighbor finding simple "
                            "neighbor blocks don't agree.");
          PARTHENON_REQUIRE(
              nb.ownership == onb.ownership,
              "Old neighbor finding and new neighbor finding ownership don't agree.");
          found = true;
        }
      PARTHENON_REQUIRE(found, "Neighbor lists don't agree.");
    }

    // TODO(LFR): Update the neighbor list here
  }
}

void Mesh::SetSameLevelNeighbors(
    BlockList_t &block_list, const LogicalLocMap_t &loc_map, RootGridInfo root_grid,
    int nbs, bool gmg_neighbors, int composite_logical_level,
    const std::unordered_set<LogicalLocation> &newly_refined) {
  for (auto &pmb : block_list) {
    auto loc = pmb->loc;
    auto gid = pmb->gid;
    auto *neighbor_list = gmg_neighbors ? &(pmb->gmg_same_neighbors) : &(pmb->neighbors);
    if (gmg_neighbors && loc.level() == composite_logical_level - 1) {
      neighbor_list = &(pmb->gmg_composite_finer_neighbors);
    } else if (gmg_neighbors && loc.level() == composite_logical_level) {
      neighbor_list = &(pmb->gmg_same_neighbors);
    } else if (gmg_neighbors) {
      PARTHENON_FAIL("GMG grid was build incorrectly.");
    }

    *neighbor_list = {};

    auto possible_neighbors = loc.GetPossibleNeighbors(root_grid);
    for (auto &pos_neighbor_location : possible_neighbors) {
      if (gmg_neighbors && loc.level() == composite_logical_level - 1 &&
          loc.level() == pos_neighbor_location.level())
        continue;
      if (loc_map.count(pos_neighbor_location) > 0) {
        const auto &gid_rank = loc_map.at(pos_neighbor_location);
        auto offsets = loc.GetSameLevelOffsets(pos_neighbor_location, root_grid);
        // This inner loop is necessary in case a block pair has multiple neighbor
        // connections due to periodic boundaries
        for (auto ox1 : offsets[0]) {
          for (auto ox2 : offsets[1]) {
            for (auto ox3 : offsets[2]) {
              NeighborConnect nc;
              if (pos_neighbor_location.level() != loc.level()) {
                // Check that the two blocks are in fact neighbors in this offset
                // direction, since we only checked that they are actually neighbors
                // when they have both been derefined to the coarser of their levels
                // (this should only play a role in small meshes with periodic
                // bounday conditions)
                auto &fine_loc = pos_neighbor_location.level() > loc.level()
                                     ? pos_neighbor_location
                                     : loc;
                int mult = loc.level() - pos_neighbor_location.level();
                std::array<int, 3> ox{ox1, ox2, ox3};
                bool not_neighbor = false;
                for (int dir = 0; dir < 3; ++dir) {
                  if (ox[dir] != 0) {
                    // temp should be +1 if a block is to the right within its parent
                    // block and -1 if it is to the left.
                    const int temp =
                        2 * (fine_loc.l(dir) - 2 * (fine_loc.l(dir) >> 1)) - 1;
                    PARTHENON_DEBUG_REQUIRE(temp * temp == 1, "Bad Offset");
                    if (temp != mult * ox[dir]) not_neighbor = true;
                  }
                }
                if (not_neighbor) continue;
              }
              int connect_indicator = std::abs(ox1) + std::abs(ox2) + std::abs(ox3);
              if (connect_indicator == 0) continue;
              if (connect_indicator == 1) {
                nc = NeighborConnect::face;
              } else if (connect_indicator == 2) {
                nc = NeighborConnect::edge;
              } else if (connect_indicator == 3) {
                nc = NeighborConnect::corner;
              }
              auto f = loc.GetAthenaXXFaceOffsets(pos_neighbor_location, ox1, ox2, ox3,
                                                  root_grid);
              neighbor_list->emplace_back(
                  pmb->pmy_mesh, pos_neighbor_location, gid_rank.second, gid_rank.first,
                  gid_rank.first - nbs, std::array<int, 3>{ox1, ox2, ox3}, nc, 0, 0, f[0],
                  f[1]);
            }
          }
        }
      }
    }
    // Set neighbor block ownership
    std::unordered_set<LogicalLocation> allowed_neighbors;
    allowed_neighbors.insert(pmb->loc);
    for (auto &nb : *neighbor_list)
      allowed_neighbors.insert(nb.loc);
    for (auto &nb : *neighbor_list) {
      nb.ownership =
          DetermineOwnership(nb.loc, allowed_neighbors, root_grid, newly_refined);
      nb.ownership.initialized = true;
    }
  }
}

void Mesh::BuildGMGHierarchy(int nbs, ParameterInput *pin, ApplicationInput *app_in) {
  if (!multigrid) return;
  // Create GMG logical location lists, first just copy coarsest grid
  auto block_size_default = GetBlockSize();

  int gmg_level_offset = std::numeric_limits<int>::max();
  for (auto dir : {X1DIR, X2DIR, X3DIR}) {
    if (!mesh_size.symmetry(dir)) {
      int dir_allowed_levels =
          NumberOfBinaryTrailingZeros(block_size_default.nx(dir) * nrbx[dir - 1]);
      gmg_level_offset = std::min(dir_allowed_levels, gmg_level_offset);
    }
  }

  const int gmg_min_level = root_level - gmg_level_offset;
  gmg_min_logical_level_ = gmg_min_level;

  const int gmg_levels = current_level - gmg_min_level + 1;
  gmg_grid_locs = std::vector<LogicalLocMap_t>(gmg_levels);
  gmg_block_lists = std::vector<BlockList_t>(gmg_levels);

  // Create MeshData objects for GMG
  gmg_mesh_data = std::vector<DataCollection<MeshData<Real>>>(gmg_levels);
  for (auto &mdc : gmg_mesh_data)
    mdc.SetMeshPointer(this);

  // Add leaf grid locations to GMG grid levels
  int gmg_gid = 0;
  for (auto loc : loclist) {
    const int gmg_level = gmg_levels - 1 + loc.level() - current_level;
    gmg_grid_locs[gmg_level].insert(
        {loc, std::pair<int, int>(gmg_gid, ranklist[gmg_gid])});
    if (gmg_level < gmg_levels - 1) {
      gmg_grid_locs[gmg_level + 1].insert(
          {loc, std::pair<int, int>(gmg_gid, ranklist[gmg_gid])});
    }
    if (ranklist[gmg_gid] == Globals::my_rank) {
      const int lid = gmg_gid - nslist[Globals::my_rank];
      gmg_block_lists[gmg_level].push_back(block_list[lid]);
      if (gmg_level < gmg_levels - 1)
        gmg_block_lists[gmg_level + 1].push_back(block_list[lid]);
    }
    gmg_gid++;
  }

  // Fill in internal nodes for GMG grid levels from levels on finer GMG grid
  for (int gmg_level = gmg_levels - 2; gmg_level >= 0; --gmg_level) {
    int grid_logical_level = gmg_level - gmg_levels + 1 + current_level;
    for (auto &[loc, gid_rank] : gmg_grid_locs[gmg_level + 1]) {
      if (loc.level() == grid_logical_level + 1) {
        auto parent = loc.GetParent();
        if (parent.morton() == loc.morton()) {
          gmg_grid_locs[gmg_level].insert(
              {parent, std::make_pair(gmg_gid, gid_rank.second)});
          if (gid_rank.second == Globals::my_rank) {
            BoundaryFlag block_bcs[6];
            auto block_size = block_size_default;
            SetBlockSizeAndBoundaries(parent, block_size, block_bcs);
            gmg_block_lists[gmg_level].push_back(
                MeshBlock::Make(gmg_gid, -1, parent, block_size, block_bcs, this, pin,
                                app_in, packages, resolved_packages, gflag));
          }
          gmg_gid++;
        }
      }
    }
  }

  // Find same level neighbors on all GMG levels
  auto root_grid = this->GetRootGridInfo();
  for (int gmg_level = 0; gmg_level < gmg_levels; ++gmg_level) {
    int grid_logical_level = gmg_level - gmg_levels + 1 + current_level;
    SetSameLevelNeighbors(gmg_block_lists[gmg_level], gmg_grid_locs[gmg_level], root_grid,
                          nbs, true, grid_logical_level);
  }

  // Now find GMG coarser neighbor
  for (int gmg_level = 1; gmg_level < gmg_levels; ++gmg_level) {
    int grid_logical_level = gmg_level - gmg_levels + 1 + current_level;
    for (auto &pmb : gmg_block_lists[gmg_level]) {
      if (pmb->loc.level() != grid_logical_level) continue;
      auto parent_loc = pmb->loc.GetParent();
      auto loc = pmb->loc;
      auto gid = pmb->gid;
      auto rank = Globals::my_rank;
      if (gmg_grid_locs[gmg_level - 1].count(parent_loc) > 0) {
        loc = parent_loc;
        gid = gmg_grid_locs[gmg_level - 1][parent_loc].first;
        rank = gmg_grid_locs[gmg_level - 1][parent_loc].second;
      } else {
        PARTHENON_FAIL("There is something wrong with GMG block list.");
      }
      pmb->gmg_coarser_neighbors.emplace_back(pmb->pmy_mesh, loc, rank, gid, gid - nbs,
                                              std::array<int, 3>{0, 0, 0},
                                              NeighborConnect::none, 0, 0, 0, 0);
    }
  }

  // Now find finer GMG neighbors
  for (int gmg_level = 0; gmg_level < gmg_levels - 1; ++gmg_level) {
    int grid_logical_level = gmg_level - gmg_levels + 1 + current_level;
    for (auto &pmb : gmg_block_lists[gmg_level]) {
      if (pmb->loc.level() != grid_logical_level) continue;
      auto daughter_locs = pmb->loc.GetDaughters();
      for (auto &daughter_loc : daughter_locs) {
        if (gmg_grid_locs[gmg_level + 1].count(daughter_loc) > 0) {
          auto &gid_rank = gmg_grid_locs[gmg_level + 1][daughter_loc];
          pmb->gmg_finer_neighbors.emplace_back(
              pmb->pmy_mesh, daughter_loc, gid_rank.second, gid_rank.first,
              gid_rank.first - nbs, std::array<int, 3>{0, 0, 0}, NeighborConnect::none, 0,
              0, 0, 0);
        }
      }
    }
  }
}
} // namespace parthenon
