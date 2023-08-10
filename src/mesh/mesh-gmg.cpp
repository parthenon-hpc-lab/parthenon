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
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/meshblock_tree.hpp"
#include "parthenon_arrays.hpp"
#include "utils/buffer_utils.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

void SetSameLevelNeighbors(BlockList_t &block_list, const LogicalLocMap_t &loc_map,
                           RootGridInfo root_grid, int nbs) {
  for (auto &pmb : block_list) {
    auto loc = pmb->loc;
    auto gid = pmb->gid;
    pmb->gmg_same_neighbors = {};
    auto possible_neighbors = loc.GetPossibleNeighbors(root_grid);
    for (auto &pos_neighbor_location : possible_neighbors) {
      if (loc_map.count(pos_neighbor_location) > 0) {
        const auto &gid_rank = loc_map.at(pos_neighbor_location);
        auto offsets = loc.GetSameLevelOffsets(pos_neighbor_location, root_grid);
        // This inner loop is necessary in case a block pair has multiple neighbor
        // connections due to periodic boundaries
        for (auto ox1 : offsets[0]) {
          for (auto ox2 : offsets[1]) {
            for (auto ox3 : offsets[2]) {
              NeighborConnect nc;
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
              pmb->gmg_same_neighbors.emplace_back();
              pmb->gmg_same_neighbors.back().SetNeighbor(
                  pos_neighbor_location, gid_rank.second, pos_neighbor_location.level(),
                  gid_rank.first, gid_rank.first - nbs, ox1, ox2, ox3, nc, 0, 0, f[0],
                  f[1]);
            }
          }
        }
      }
    }
    // Set neighbor block ownership
    std::unordered_set<LogicalLocation> allowed_neighbors;
    allowed_neighbors.insert(pmb->loc);
    for (auto &nb : pmb->gmg_same_neighbors)
      allowed_neighbors.insert(nb.loc);
    for (auto &nb : pmb->gmg_same_neighbors) {
      nb.ownership = DetermineOwnership(nb.loc, allowed_neighbors, root_grid);
      nb.ownership.initialized = true;
    }
  }
}

void Mesh::BuildGMGHierarchy(int nbs, ParameterInput *pin, ApplicationInput *app_in) {
  // Create GMG logical location lists, first just copy coarsest grid
  const int gmg_levels = multigrid ? current_level : 0;
  gmg_grid_locs = std::vector<LogicalLocMap_t>(gmg_levels + 1);
  gmg_block_lists = std::vector<BlockList_t>(gmg_levels + 1);

  // Create MeshData objects for GMG
  gmg_mesh_data = std::vector<DataCollection<MeshData<Real>>>(gmg_levels + 1);
  for (auto &mdc : gmg_mesh_data)
    mdc.SetMeshPointer(this);

  // Build most refined GMG grid from already known grid
  int gmg_gid = 0;
  for (auto loc : loclist) {
    gmg_grid_locs[gmg_levels].insert(
        {loc, std::pair<int, int>(gmg_gid, ranklist[gmg_gid])});
    gmg_gid++;
  }

  // Most refined GMG grid block list is just the main block list for the Mesh
  gmg_block_lists[gmg_levels] = block_list;

  // Build meshes and blocklists for increasingly coarsened grids
  auto block_size = GetBlockSize();
  for (int gmg_level = gmg_levels - 1; gmg_level >= 0; --gmg_level) {
    // Determine mesh structure for this level on all ranks
    for (auto &[loc, gid_rank] : gmg_grid_locs[gmg_level + 1]) {
      auto &rank = gid_rank.second;
      if (loc.level() == gmg_level + 1) {
        auto parent = loc.GetParent();
        if (parent.morton() == loc.morton()) {
          gmg_grid_locs[gmg_level].insert({parent, std::make_pair(gmg_gid++, rank)});
        }
      } else {
        gmg_grid_locs[gmg_level].insert({loc, std::make_pair(gmg_gid++, rank)});
      }
    }

    // Create blocklist for this level on this rank
    for (auto &[loc, gid_rank] : gmg_grid_locs[gmg_level]) {
      if (gid_rank.second == Globals::my_rank) {
        BoundaryFlag block_bcs[6];
        SetBlockSizeAndBoundaries(loc, block_size, block_bcs);
        if (loc.level() < root_level) {
          // The results of SetBlockSize and Boundaries are wrong
          int root_fac = 1 << (root_level - loc.level());
          for (auto &dir : {X1DIR, X2DIR, X3DIR}) {
            Real deltax =
                (mesh_size.xmax(dir) - mesh_size.xmin(dir)) / nrbx[dir - 1] * root_fac;
            block_size.xmin(dir) = mesh_size.xmin(dir) + deltax * loc.l(dir - 1);
            block_size.xmax(dir) = block_size.xmin(dir) + deltax;
          }
        }
        gmg_block_lists[gmg_level].push_back(
            MeshBlock::Make(gid_rank.first, -1, loc, block_size, block_bcs, this, pin,
                            app_in, packages, resolved_packages, gflag));
      }
    }
  }

  // Find same level neighbors on all GMG levels
  auto root_grid = this->GetRootGridInfo();
  for (int gmg_level = 0; gmg_level <= gmg_levels; ++gmg_level) {
    SetSameLevelNeighbors(gmg_block_lists[gmg_level], gmg_grid_locs[gmg_level], root_grid,
                          nbs);
  }

  // Now find GMG coarser neighbor
  for (int gmg_level = 1; gmg_level <= gmg_levels; ++gmg_level) {
    for (auto &pmb : gmg_block_lists[gmg_level]) {
      auto parent_loc = pmb->loc.GetParent();
      auto loc = pmb->loc;
      auto gid = pmb->gid;
      auto rank = Globals::my_rank;
      if (gmg_grid_locs[gmg_level - 1].count(parent_loc) > 0) {
        loc = parent_loc;
        gid = gmg_grid_locs[gmg_level - 1][parent_loc].first;
        rank = gmg_grid_locs[gmg_level - 1][parent_loc].second;
      } else if (gmg_grid_locs[gmg_level - 1].count(loc) > 0) {
        gid = gmg_grid_locs[gmg_level - 1][loc].first;
        rank = gmg_grid_locs[gmg_level - 1][loc].second;
      } else {
        PARTHENON_FAIL("There is something wrong with GMG block list.");
      }
      pmb->gmg_coarser_neighbors.emplace_back();
      pmb->gmg_coarser_neighbors.back().SetNeighbor(
          loc, rank, loc.level(), gid, gid - nbs, 0, 0, 0, NeighborConnect::none, 0, 0);
    }
  }

  // Now find finer GMG neighbors
  for (int gmg_level = 0; gmg_level < gmg_levels; ++gmg_level) {
    for (auto &pmb : gmg_block_lists[gmg_level]) {
      auto daughter_locs = pmb->loc.GetDaughters();
      daughter_locs.push_back(pmb->loc); // It is also possible that this block itself is
                                         // present on the finer mesh
      for (auto &daughter_loc : daughter_locs) {
        if (gmg_grid_locs[gmg_level + 1].count(daughter_loc) > 0) {
          auto &gid_rank = gmg_grid_locs[gmg_level + 1][daughter_loc];
          pmb->gmg_finer_neighbors.emplace_back();
          pmb->gmg_finer_neighbors.back().SetNeighbor(
              daughter_loc, gid_rank.second, daughter_loc.level(), gid_rank.first,
              gid_rank.first - nbs, 0, 0, 0, NeighborConnect::none, 0, 0);
        }
      }
    }
  }
  // CheckNeighborFinding(block_list, "AMR LoadBalance");
}

void CheckNeighborFinding(BlockList_t &block_list, std::string call_site) {
  for (auto &pmb : block_list) {
    CheckNeighborFinding(pmb, call_site);
  }
}

void CheckNeighborFinding(std::shared_ptr<MeshBlock> &pmb, std::string call_site) {
  // Check each block one by one
  std::unordered_map<LogicalLocation, NeighborBlock> neighbs;
  printf("Checking neighbors at callsite %s.\n", call_site.c_str());
  bool fail = false;
  for (auto &nb : pmb->gmg_same_neighbors)
    neighbs[nb.loc] = nb;
  if (pmb->pbval->nneighbor != pmb->gmg_same_neighbors.size()) {
    printf("New algorithm found different number of neighbor blocks on %i (%i vs %li).\n",
           pmb->gid, pmb->pbval->nneighbor, pmb->gmg_same_neighbors.size());
    fail = true;
  }
  for (int nn = 0; nn < pmb->pbval->nneighbor; ++nn) {
    auto &nb = pmb->pbval->neighbor[nn];
    if (neighbs.count(nb.loc) > 0) {
      auto &nb2 = neighbs[nb.loc];
      if (nb.ni.ox1 == nb2.ni.ox1 && nb.ni.ox2 == nb2.ni.ox2 && nb.ni.ox3 == nb2.ni.ox3) {
      } else {
        printf("Bad offsets for block %i %s: %s ox1=%i ox2=%i %s ox1=%i ox2=%i\n",
               pmb->gid, pmb->loc.label().c_str(), nb.loc.label().c_str(), nb.ni.ox1,
               nb.ni.ox2, nb2.loc.label().c_str(), nb2.ni.ox1, nb2.ni.ox2);
        fail = true;
      }
      if (nb.ni.fi1 == nb2.ni.fi1 && nb.ni.fi2 == nb2.ni.fi2) {
      } else {
        printf("Bad face offsets for block %i %s: %s f1=%i f2=%i %s f1=%i f2=%i\n",
               pmb->gid, pmb->loc.label().c_str(), nb.loc.label().c_str(), nb.ni.fi1,
               nb.ni.fi2, nb2.loc.label().c_str(), nb2.ni.fi1, nb2.ni.fi2);
        fail = true;
      }

      if (nb.snb.gid == nb2.snb.gid && nb.snb.lid == nb2.snb.lid &&
          nb.snb.level == nb2.snb.level) {
      } else {
        printf("Bad compressed indexing for block %i %s: %s gid=%i lid=%i level=%i %s "
               "gid=%i lid=%i level=%i\n",
               pmb->gid, pmb->loc.label().c_str(), nb.loc.label().c_str(), nb.snb.gid,
               nb.snb.lid, nb.snb.level, nb2.loc.label().c_str(), nb2.snb.gid,
               nb2.snb.lid, nb.snb.level);
        fail = true;
      }

      if (nb.ni.type == nb2.ni.type) {
      } else {
        printf("Bad face id for block %i %s: %s fid=%i ox=(%i, %i, %i) %s fid=%i\n",
               pmb->gid, pmb->loc.label().c_str(), nb.loc.label().c_str(),
               static_cast<int>(nb.ni.type), nb.ni.ox1, nb.ni.ox2, nb.ni.ox3,
               nb2.loc.label().c_str(), static_cast<int>(nb2.ni.type));
        fail = true;
      }

    } else {
      printf("Block %i %s new neighbor list missing %s.\n", pmb->gid,
             pmb->loc.label().c_str(), nb.loc.label().c_str());
      fail = true;
    }
  }
  // if (fail) PARTHENON_FAIL("Bad neighbor list");
  //  printf("Finished checking neighbors for %i.\n", pmb->gid);
}

} // namespace parthenon
