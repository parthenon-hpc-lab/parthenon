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
#include "mesh/forest/logical_location.hpp"
#include "mesh/forest/tree.hpp"
#include "utils/bit_hacks.hpp"
#include "utils/indexer.hpp"

namespace parthenon {
namespace forest {

class Forest {
  bool gids_resolved = false;
  std::map<std::int64_t, std::shared_ptr<Tree>> trees;

 public:
  int root_level;
  int forest_level;

  void AddTree(const std::shared_ptr<Tree> &in) {
    if (trees.count(in->GetId())) {
      PARTHENON_WARN("Adding tree to forest twice.");
    }
    trees[in->GetId()] = in;
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

  int count(const LogicalLocation &loc) const {
    if (trees.count(loc.tree()) > 0) {
      return trees.at(loc.tree())->count(loc);
    }
    return 0;
  }

  RegionSize GetBlockDomain(const LogicalLocation &loc) const {
    return trees.at(loc.tree())->GetBlockDomain(loc);
  }

  std::array<BoundaryFlag, BOUNDARY_NFACES>
  GetBlockBCs(const LogicalLocation &loc) const {
    return trees.at(loc.tree())->GetBlockBCs(loc);
  }

  std::vector<NeighborLocation> FindNeighbors(const LogicalLocation &loc, int ox1,
                                              int ox2, int ox3) const {
    return trees.at(loc.tree())->FindNeighbors(loc, ox1, ox2, ox3);
  }

  std::vector<NeighborLocation>
  FindNeighbors(const LogicalLocation &loc,
                GridIdentifier grid_id = GridIdentifier::leaf()) const {
    return trees.at(loc.tree())->FindNeighbors(loc, grid_id);
  }
  std::size_t CountMeshBlock() const {
    std::size_t count{0};
    for (auto &[id, tree] : trees)
      count += tree->CountMeshBlock();
    return count;
  }

  // TODO(LFR): Probably eventually remove this. This is only meaningful for simply
  // oriented grids
  LogicalLocation GetAthenaCompositeLocation(const LogicalLocation &loc) const {
    if (loc.tree() < 0)
      return loc; // This is already presumed to be an Athena++ tree location
    auto parent_loc = trees.at(loc.tree())->athena_forest_loc;
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
    int macro_level = (*trees.begin()).second->athena_forest_loc.level();
    auto forest_loc = loc.GetParent(loc.level() - macro_level);
    for (auto &[id, t] : trees) {
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
    return trees.at(loc.tree())->GetGid(loc);
  }
  std::int64_t GetLeafGid(const LogicalLocation &loc) const {
    PARTHENON_REQUIRE(gids_resolved, "Asking for GID in invalid state.");
    return trees.at(loc.tree())->GetLeafGid(loc);
  }

  std::int64_t GetOldGid(const LogicalLocation &loc) const {
    PARTHENON_REQUIRE(gids_resolved, "Asking for GID in invalid state.");
    return trees.at(loc.tree())->GetOldGid(loc);
  }

  // Build a logically hyper-rectangular forest that mimics the grid
  // setups available in Athena++
  static Forest AthenaXX(RegionSize mesh_size, RegionSize block_size,
                         std::array<BoundaryFlag, BOUNDARY_NFACES> mesh_bcs);
};

struct NeighborLocation {
  NeighborLocation(const LogicalLocation &g, const LogicalLocation &o)
      : global_loc(g), origin_loc(o) {}
  LogicalLocation global_loc; // Global location of neighboring block
  LogicalLocation
      origin_loc; // Logical location of neighboring block in index space of origin block
};

} // namespace forest
} // namespace parthenon

#endif // MESH_FOREST_FOREST_HPP_
