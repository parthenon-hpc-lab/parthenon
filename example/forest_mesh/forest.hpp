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
#ifndef FOREST_HPP_
#define FOREST_HPP_

#include <array>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
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

  LogicalLocation Transform(const LogicalLocation &loc_in) const;

  int dir_connection[3];
  bool dir_flip[3];
};

using ForestLocation = std::pair<std::uint64_t, LogicalLocation>;

// We don't allow for periodic boundaries, since we can encode periodicity through
// connectivity in the forest
class Tree : public std::enable_shared_from_this<Tree> {
  // This allows us to ensure that Trees are only created as shared_ptrs
  struct private_t {};

 public:
  Tree(private_t, int ndim, int root_level, RegionSize domain = RegionSize());

  template <class... Ts>
  static std::shared_ptr<Tree> create(Ts &&...args) {
    auto ptree = std::make_shared<Tree>(private_t(), std::forward<Ts>(args)...);
    // Make the tree its own central neighbor to reduce code duplication
    ptree->neighbors[13].insert({ptree, RelativeOrientation()});
    return ptree;
  }

  // Methods for modifying the tree
  int AddMeshBlock(const LogicalLocation &loc, bool enforce_proper_nesting = true);
  int Refine(const LogicalLocation &ref_loc, bool enforce_proper_nesting = true);
  int Derefine(const LogicalLocation &ref_loc, bool enforce_proper_nesting = true);

  // Methods for getting block properties
  std::vector<ForestLocation> GetMeshBlockList() const;
  RegionSize GetBlockDomain(LogicalLocation loc) const;
  std::vector<ForestLocation> FindNeighbor(const LogicalLocation &loc, int ox1, int ox2,
                                           int ox3) const;
  std::size_t CountMeshBlock() const { return leaves.size(); }

  // Methods for building tree connectivity
  void AddNeighbor(int location_idx, std::shared_ptr<Tree> neighbor_tree,
                   RelativeOrientation orient) {
    neighbors[location_idx].insert({neighbor_tree, orient});
  }
  void SetId(std::uint64_t id) { my_id = id; }
  std::uint64_t GetId() { return my_id; }

  const std::unordered_set<LogicalLocation> &GetLeaves() const { return leaves; }

 private:
  int ndim;
  std::uint64_t my_id;
  std::unordered_set<LogicalLocation> leaves;
  std::unordered_set<LogicalLocation> internal_nodes;
  std::array<std::unordered_map<std::shared_ptr<Tree>, RelativeOrientation>, 27>
      neighbors;
  RegionSize domain;
};

class Forest {
 public:
  std::vector<std::shared_ptr<Tree>> trees;

  int AddMeshBlock(const ForestLocation &loc, bool enforce_proper_nesting = true) {
    return trees[loc.first]->AddMeshBlock(loc.second, enforce_proper_nesting);
  }
  int Refine(const ForestLocation &loc, bool enforce_proper_nesting = true) {
    return trees[loc.first]->Refine(loc.second, enforce_proper_nesting);
  }
  int Derefine(const ForestLocation &loc, bool enforce_proper_nesting = true) {
    return trees[loc.first]->Derefine(loc.second, enforce_proper_nesting);
  }

  std::vector<ForestLocation> GetMeshBlockList() const;
  RegionSize GetBlockDomain(const ForestLocation &loc) const {
    return trees[loc.first]->GetBlockDomain(loc.second);
  }
  std::vector<ForestLocation> FindNeighbor(const ForestLocation &loc, int ox1, int ox2,
                                           int ox3) const {
    return trees[loc.first]->FindNeighbor(loc.second, ox1, ox2, ox3);
  }
  std::size_t CountMeshBlock() const {
    std::size_t count;
    for (auto &tree : trees)
      count += tree->CountMeshBlock();
    return count;
  }

  std::size_t CountTrees() const { 
    return trees.size();
  }

  // Build a logically hyper-rectangular forest that mimics the grid 
  // setups available in Athena++
  static Forest AthenaXX(RegionSize mesh_size, RegionSize block_size,
                         std::array<bool, 3> periodic);
};

} // namespace forest
} // namespace parthenon

#endif // FOREST_HPP_
