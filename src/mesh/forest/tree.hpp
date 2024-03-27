//========================================================================================
// (C) (or copyright) 2024. Triad National Security, LLC. All rights reserved.
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
#ifndef MESH_FOREST_TREE_HPP_
#define MESH_FOREST_TREE_HPP_

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
#include "mesh/forest/cell_center_offsets.hpp"
#include "mesh/forest/logical_location.hpp"
#include "mesh/forest/relative_orientation.hpp"
#include "utils/bit_hacks.hpp"
#include "utils/indexer.hpp"

namespace parthenon {
namespace forest {

// We don't explicitly allow for periodic boundaries, since we can encode periodicity
// through connectivity in the forest
class Tree : public std::enable_shared_from_this<Tree> {
  // This allows us to ensure that Trees are only created as shared_ptrs
  struct private_t {};

 public:
  Tree(private_t, std::int64_t id, int ndim, int root_level,
       RegionSize domain = RegionSize(),
       std::array<BoundaryFlag, BOUNDARY_NFACES> bcs = {
           BoundaryFlag::block, BoundaryFlag::block, BoundaryFlag::block,
           BoundaryFlag::block, BoundaryFlag::block, BoundaryFlag::block});

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
  int count(const LogicalLocation &loc) const { return leaves.count(loc); }
  std::vector<LogicalLocation> GetSortedMeshBlockList() const;
  std::vector<LogicalLocation> GetSortedInternalNodeList() const;
  RegionSize GetBlockDomain(const LogicalLocation &loc) const;
  std::array<BoundaryFlag, BOUNDARY_NFACES> GetBlockBCs(const LogicalLocation &loc) const;
  std::vector<NeighborLocation>
  FindNeighbors(const LogicalLocation &loc,
                GridIdentifier grid_id = GridIdentifier::leaf()) const;
  std::vector<NeighborLocation> FindNeighbors(const LogicalLocation &loc, int ox1,
                                              int ox2, int ox3) const;
  std::size_t CountMeshBlock() const { return leaves.size(); }

  // Gid related methods 
  void InsertGid(const LogicalLocation &loc, std::int64_t gid);
  std::int64_t GetGid(const LogicalLocation &loc) const;
  std::int64_t GetOldGid(const LogicalLocation &loc) const;
  // Get the gid of the leaf block with the same Morton number 
  // as loc
  std::int64_t GetLeafGid(const LogicalLocation &loc) const;

  // Methods for building tree connectivity
  void AddNeighborTree(CellCentOffsets offset, std::shared_ptr<Tree> neighbor_tree,
                       RelativeOrientation orient);
  
  // Global id of the tree
  std::uint64_t GetId() const { return my_id; }

  // TODO(LFR): Eventually remove this.
  LogicalLocation athena_forest_loc;

 private:
  void FindNeighborsImpl(const LogicalLocation &loc, int ox1, int ox2, int ox3,
                         std::vector<NeighborLocation> *neighbor_locs,
                         GridIdentifier grid_type) const;

  int ndim;
  const std::uint64_t my_id;
  // Structure mapping location of block in this tree to current gid and previous gid
  using LocMap_t =
      std::unordered_map<LogicalLocation, std::pair<std::int64_t, std::int64_t>>;
  static std::pair<LogicalLocation, std::pair<std::int64_t, std::int64_t>>
  LocMapEntry(const LogicalLocation &loc, const int gid, const int gid_old) {
    return std::make_pair(loc, std::make_pair(gid, gid_old));
  }

  LocMap_t leaves;
  LocMap_t internal_nodes;

  // This contains all of the neighbor information for this tree, for each of the
  // 3^3 possible neighbor connections. Since an edge or node connection can have
  // multiple neighbors generally, we keep a map at each neighbor location from
  // the tree sptr to the relative logical coordinate orientation of the neighbor
  // block.
  std::array<std::unordered_map<std::shared_ptr<Tree>, RelativeOrientation>, 27>
      neighbors;

  std::array<BoundaryFlag, BOUNDARY_NFACES> boundary_conditions;

  RegionSize domain;
};
} // namespace forest
} // namespace parthenon

#endif // MESH_FOREST_TREE_HPP_
