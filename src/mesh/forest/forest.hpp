//========================================================================================
// (C) (or copyright) 2023-2024. Triad National Security, LLC. All rights reserved.
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
#include "mesh/forest/forest_topology.hpp"
#include "mesh/forest/logical_location.hpp"
#include "mesh/forest/tree.hpp"
#include "utils/bit_hacks.hpp"
#include "utils/indexer.hpp"

namespace parthenon {
class ApplicationInput;

namespace forest {

template <class ELEMENT>
struct ForestBC {
  ELEMENT element;
  BoundaryFlag bflag;
  std::optional<ELEMENT> periodicElement;
};

class Forest;
class ForestDefinition {
 protected:
  friend class Forest;
  std::vector<std::shared_ptr<Face>> faces;
  RegionSize block_size;
  std::vector<ForestBC<Edge>> bc_edges;
  std::vector<LogicalLocation> refinement_locations;
  std::vector<RegionSize> face_sizes;

 public:
  using ar3_t = std::array<Real, 3>;
  using ai3_t = std::array<int, 3>;
  void AddFace(std::size_t id, std::array<std::shared_ptr<Node>, 4> nodes_in,
               ar3_t xmin = {0.0, 0.0, 0.0}, ar3_t xmax = {1.0, 1.0, 1.0}) {
    faces.emplace_back(Face::create(id, nodes_in));
    face_sizes.emplace_back(xmin, xmax, ar3_t{1.0, 1.0, 1.0}, ai3_t{1, 1, 1});
  }

  void AddBC(Edge edge, BoundaryFlag bf = BoundaryFlag::user,
             std::optional<Edge> periodic_connection = {}) {
    if (bf == BoundaryFlag::periodic)
      PARTHENON_REQUIRE(periodic_connection,
                        "Must specify another edge for periodic boundary conditions.");
    bc_edges.emplace_back(ForestBC<Edge>{edge, bf, periodic_connection});
  }

  void AddInitialRefinement(const LogicalLocation &loc) {
    refinement_locations.push_back(loc);
  }

  void SetBlockSize(const RegionSize &bs) { block_size = bs; }
};

class Forest {
  bool gids_resolved = false;
  std::map<std::int64_t, std::shared_ptr<Tree>> trees;

 public:
  int root_level;
  std::optional<int> forest_level{};

  std::vector<std::shared_ptr<Tree>> GetTrees() {
    std::vector<std::shared_ptr<Tree>> trees_out;
    for (auto &[id, tree] : trees)
      trees_out.push_back(tree);
    return trees_out;
  }

  std::shared_ptr<Tree> &GetTreePtr(std::int64_t id) {
    PARTHENON_REQUIRE(trees.count(id) > 0, "Tree " + std::to_string(id) + " not found.");
    return trees[id];
  }

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

  void EnrollBndryFncts(ApplicationInput *app_in, const BValNames_t &names,
                        const BValNames_t &swarm_names,
                        const BValFuncArray_t &UserBoundaryFunctions_in,
                        const SBValFuncArray_t &UserSwarmBoundaryFunctions_in) {
    for (auto &[id, ptree] : trees)
      ptree->EnrollBndryFncts(app_in, names, swarm_names, UserBoundaryFunctions_in,
                              UserSwarmBoundaryFunctions_in);
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
  LogicalLocation GetLegacyTreeLocation(const LogicalLocation &loc) const;

  LogicalLocation
  GetForestLocationFromLegacyTreeLocation(const LogicalLocation &loc) const;

  std::size_t CountTrees() const { return trees.size(); }

  std::int64_t GetGid(const LogicalLocation &loc) const {
    PARTHENON_REQUIRE(gids_resolved, "Asking for GID in invalid state.");
    return trees.at(loc.tree())->GetGid(loc);
  }

  // Get the gid of the leaf block with the same Morton number
  // as loc (on the same tree)
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
  static Forest HyperRectangular(RegionSize mesh_size, RegionSize block_size,
                                 std::array<BoundaryFlag, BOUNDARY_NFACES> mesh_bcs);

  static Forest Make2D(ForestDefinition &forest_def);
};

} // namespace forest
} // namespace parthenon

#endif // MESH_FOREST_FOREST_HPP_
