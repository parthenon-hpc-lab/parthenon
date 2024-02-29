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
#ifndef MESH_FOREST_HPP_
#define MESH_FOREST_HPP_

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
#include "mesh/logical_location.hpp"
#include "utils/bit_hacks.hpp"
#include "utils/indexer.hpp"

namespace parthenon {
namespace forest {
enum class Direction : uint { I = 0, J = 1, K = 2 };

struct RelativeOrientation {
  RelativeOrientation() : dir_connection{0, 1, 2}, dir_flip{false, false, false} {};

  void SetDirection(Direction origin, Direction neighbor, bool reversed = false) {
    dir_connection[static_cast<uint>(origin)] = static_cast<uint>(neighbor);
    dir_flip[static_cast<uint>(origin)] = reversed;
  }

  LogicalLocation Transform(const LogicalLocation &loc_in,
                            std::int64_t destination) const;
  LogicalLocation TransformBack(const LogicalLocation &loc_in, std::int64_t origin) const;

  bool use_offset = false;
  std::array<int, 3> offset;
  std::array<int, 3> dir_connection;
  std::array<bool, 3> dir_flip;
};

// We don't allow for periodic boundaries, since we can encode periodicity through
// connectivity in the forest
class Tree : public std::enable_shared_from_this<Tree> {
  // This allows us to ensure that Trees are only created as shared_ptrs
  struct private_t {};

 public:
  Tree(private_t, std::int64_t id, int ndim, int root_level,
       RegionSize domain = RegionSize());

  template <class... Ts>
  static std::shared_ptr<Tree> create(Ts &&...args) {
    auto ptree = std::make_shared<Tree>(private_t(), std::forward<Ts>(args)...);
    // Make the tree its own central neighbor to reduce code duplication
    RelativeOrientation orient;
    orient.use_offset = true;
    orient.offset = {0, 0, 0};
    ptree->neighbors[13].insert({ptree, orient});
    return ptree;
  }

  // Methods for modifying the tree
  int AddMeshBlock(const LogicalLocation &loc, bool enforce_proper_nesting = true);
  int Refine(const LogicalLocation &ref_loc, bool enforce_proper_nesting = true);
  int Derefine(const LogicalLocation &ref_loc, bool enforce_proper_nesting = true);

  // Methods for getting block properties
  std::vector<LogicalLocation> GetMeshBlockList() const;
  RegionSize GetBlockDomain(const LogicalLocation &loc) const;
  std::vector<NeighborLocation> FindNeighbors(const LogicalLocation &loc) const;
  std::vector<NeighborLocation> FindNeighbors(const LogicalLocation &loc, int ox1,
                                              int ox2, int ox3) const;

  std::vector<LogicalLocation>
  GetLocalLocationsFromNeighborLocation(const LogicalLocation &loc);

  std::size_t CountMeshBlock() const { return leaves.size(); }

  // Methods for building tree connectivity
  void AddNeighborTree(int location_idx, std::shared_ptr<Tree> neighbor_tree,
                       RelativeOrientation orient);

  std::uint64_t GetId() const { return my_id; }

  const std::unordered_map<LogicalLocation, std::uint64_t> &GetLeaves() const {
    return leaves;
  }

  void InsertGid(const LogicalLocation &loc, std::uint64_t gid) {
    PARTHENON_REQUIRE(leaves.count(loc) == 1,
                      "Trying to add gid for non-existent location.");
    leaves[loc] = gid;
  }

  std::uint64_t GetGid(const LogicalLocation &loc) const { return leaves.at(loc); }

  // TODO(LFR): Eventually remove this. 
  LogicalLocation athena_forest_loc;

 private:
  void FindNeighborsImpl(const LogicalLocation &loc, int ox1, int ox2, int ox3,
                         std::vector<NeighborLocation> *neighbor_locs) const;

  int ndim;
  const std::uint64_t my_id;
  std::unordered_map<LogicalLocation, std::uint64_t> leaves;
  std::unordered_set<LogicalLocation> internal_nodes;

  // This contains all of the neighbor information for this tree, for each of the
  // 3^3 possible neighbor connections. Since an edge or node connection can have
  // multiple neighbors generally, we keep a map at each neighbor location from
  // the tree sptr to the relative logical coordinate orientation of the neighbor
  // block.
  std::array<std::unordered_map<std::shared_ptr<Tree>, RelativeOrientation>, 27>
      neighbors;

  // Helper maps for going from tree ids to neighbor connections to those trees
  // as well as from tree id to the tree sptr. More or less inverts the neighbors
  // object above.
  std::unordered_map<std::uint64_t, std::set<int>> tid_to_connection_set;
  std::unordered_map<std::uint64_t, std::shared_ptr<Tree>> tid_to_tree_sptr;
  RegionSize domain;
};

class Forest {
  bool gids_resolved = false;

 public:
  std::vector<std::shared_ptr<Tree>> trees;

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
  
  // TODO(LFR): Probably eventually remove this. This is only meaningful for simply oriented grids
  LogicalLocation GetAthenaCompositeLocation(const LogicalLocation& loc) { 
    auto parent_loc = trees[loc.tree()]->athena_forest_loc; 
    int composite_level = parent_loc.level() + loc.level();
    int lx1 = (parent_loc.lx1() << loc.level()) + loc.lx1();
    int lx2 = (parent_loc.lx2() << loc.level()) + loc.lx2();
    int lx3 = (parent_loc.lx3() << loc.level()) + loc.lx3();
    return LogicalLocation(composite_level, lx1, lx2, lx3);
  }

  LogicalLocation GetForestLocationFromAthenaCompositeLocation(const LogicalLocation& loc) { 
    int macro_level = trees[0]->athena_forest_loc.level(); 
    auto forest_loc = loc.GetParent(loc.level() - macro_level);
    for (auto &t : trees) { 
      if (t->athena_forest_loc == forest_loc) { 
        return LogicalLocation(t->GetId(), loc.level() - macro_level, 
                               loc.lx1() - (forest_loc.lx1() << (loc.level() - macro_level)),
                               loc.lx2() - (forest_loc.lx2() << (loc.level() - macro_level)),
                               loc.lx3() - (forest_loc.lx3() << (loc.level() - macro_level)));
      }
    }
    PARTHENON_FAIL("Somehow didn't find a tree.");
    return LogicalLocation();
  }

  std::size_t CountTrees() const { return trees.size(); }

  std::uint64_t GetGid(const LogicalLocation &loc) const {
    PARTHENON_REQUIRE(gids_resolved, "Asking for GID in invalid state.");
    return trees[loc.tree()]->GetGid(loc);
  }

  // Build a logically hyper-rectangular forest that mimics the grid
  // setups available in Athena++
  static Forest AthenaXX(RegionSize mesh_size, RegionSize block_size,
                         std::array<bool, 3> periodic);
};

} // namespace forest
} // namespace parthenon

#endif // MESH_FOREST_HPP_
