//========================================================================================
// (C) (or copyright) 2023. Triad National Security, LLC. All rights reserved.
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
#ifndef MESH_FOREST_FOREST_HPP_
#define MESH_FOREST_FOREST_HPP_

#include <array>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "basic_types.hpp"
#include "defs.hpp"
#include "mesh/forest/tree.hpp"
#include "mesh/logical_location.hpp"
#include "utils/bit_hacks.hpp"
#include "utils/indexer.hpp"

namespace parthenon {
namespace forest {

class Forest {
  bool gids_resolved = false;

 public:
  std::vector<std::shared_ptr<Tree>> trees;
  int root_level;
  int forest_level;

  void AddTree(const std::shared_ptr<Tree>& in) { 
    trees.push_back(in);
  }

  int AddMeshBlock(const LogicalLocation &loc, bool enforce_proper_nesting = true) {
    gids_resolved = false;
    return trees[loc.tree()]->AddMeshBlock(loc, enforce_proper_nesting);
  }
  int Refine(const LogicalLocation &loc, bool enforce_proper_nesting = true) {
    gids_resolved = false;
    return trees[loc.tree()]->Refine(loc, enforce_proper_nesting);
  }
  int Derefine(const LogicalLocation &loc, bool enforce_proper_nesting = true) {
    gids_resolved = false;
    return trees[loc.tree()]->Derefine(loc, enforce_proper_nesting);
  }

  std::vector<LogicalLocation> GetMeshBlockListAndResolveGids();

  RegionSize GetBlockDomain(const LogicalLocation &loc) const {
    return trees[loc.tree()]->GetBlockDomain(loc);
  }

  std::array<BoundaryFlag, BOUNDARY_NFACES>
  GetBlockBCs(const LogicalLocation &loc) const {
    return trees[loc.tree()]->GetBlockBCs(loc);
  }

  std::vector<NeighborLocation> FindNeighbors(const LogicalLocation &loc, int ox1,
                                              int ox2, int ox3) const {
    return trees[loc.tree()]->FindNeighbors(loc, ox1, ox2, ox3);
  }

  std::vector<NeighborLocation> FindNeighbors(const LogicalLocation &loc) const {
    return trees[loc.tree()]->FindNeighbors(loc);
  }
  std::size_t CountMeshBlock() const {
    std::size_t count{0};
    for (auto &tree : trees)
      count += tree->CountMeshBlock();
    return count;
  }

  // TODO(LFR): Probably eventually remove this. This is only meaningful for simply
  // oriented grids
  LogicalLocation GetAthenaCompositeLocation(const LogicalLocation &loc) const {
    if (loc.tree() < 0)
      return loc; // This is already presumed to be an Athena++ tree location
    auto parent_loc = trees[loc.tree()]->athena_forest_loc;
    int composite_level = parent_loc.level() + loc.level();
    int lx1 = (parent_loc.lx1() << loc.level()) + loc.lx1();
    int lx2 = (parent_loc.lx2() << loc.level()) + loc.lx2();
    int lx3 = (parent_loc.lx3() << loc.level()) + loc.lx3();
    return LogicalLocation(composite_level, lx1, lx2, lx3);
  }

  LogicalLocation
  GetForestLocationFromAthenaCompositeLocation(const LogicalLocation &loc) const {
    if (loc.tree() >= 0)
      return loc; // This location is already associated with a tree in the Parthenon
                  // forest
    int macro_level = trees[0]->athena_forest_loc.level();
    auto forest_loc = loc.GetParent(loc.level() - macro_level);
    for (auto &t : trees) {
      if (t->athena_forest_loc == forest_loc) {
        return LogicalLocation(
            t->GetId(), loc.level() - macro_level,
            loc.lx1() - (forest_loc.lx1() << (loc.level() - macro_level)),
            loc.lx2() - (forest_loc.lx2() << (loc.level() - macro_level)),
            loc.lx3() - (forest_loc.lx3() << (loc.level() - macro_level)));
      }
    }
    PARTHENON_FAIL("Somehow didn't find a tree.");
    return LogicalLocation();
  }

  std::size_t CountTrees() const { return trees.size(); }

  std::int64_t GetGid(const LogicalLocation &loc) const {
    PARTHENON_REQUIRE(gids_resolved, "Asking for GID in invalid state.");
    return trees[loc.tree()]->GetGid(loc);
  }

  std::int64_t GetOldGid(const LogicalLocation &loc) const {
    PARTHENON_REQUIRE(gids_resolved, "Asking for GID in invalid state.");
    return trees[loc.tree()]->GetOldGid(loc);
  }

  // Build a logically hyper-rectangular forest that mimics the grid
  // setups available in Athena++
  static Forest AthenaXX(RegionSize mesh_size, RegionSize block_size,
                         std::array<BoundaryFlag, BOUNDARY_NFACES> mesh_bcs);
};

} // namespace forest
} // namespace parthenon

#endif // MESH_FOREST_FOREST_HPP_
