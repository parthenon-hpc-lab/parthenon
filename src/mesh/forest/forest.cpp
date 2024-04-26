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

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stack>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "basic_types.hpp"
#include "defs.hpp"
#include "mesh/forest/forest.hpp"
#include "mesh/forest/logical_coordinate_transformation.hpp"
#include "mesh/forest/logical_location.hpp"
#include "mesh/forest/tree.hpp"
#include "utils/bit_hacks.hpp"
#include "utils/indexer.hpp"

namespace parthenon {
namespace forest {

std::vector<LogicalLocation> Forest::GetMeshBlockListAndResolveGids() {
  std::vector<LogicalLocation> mb_list;
  std::uint64_t gid{0};
  for (auto &[id, tree] : trees) {
    std::size_t start = mb_list.size();
    auto tree_mbs = tree->GetSortedMeshBlockList();
    mb_list.insert(mb_list.end(), std::make_move_iterator(tree_mbs.begin()),
                   std::make_move_iterator(tree_mbs.end()));
    std::size_t end = mb_list.size();
    for (int i = start; i < end; ++i)
      tree->InsertGid(mb_list[i], gid++);
  }

  // Assign gids to the internal nodes
  for (auto &[id, tree] : trees) {
    std::size_t start = mb_list.size();
    auto tree_int_locs = tree->GetSortedInternalNodeList();
    for (auto &loc : tree_int_locs)
      tree->InsertGid(loc, gid++);
  }

  // The index of blocks in this list corresponds to their gid
  gids_resolved = true;
  return mb_list;
}

Forest Forest::HyperRectangular(RegionSize mesh_size, RegionSize block_size,
                                std::array<BoundaryFlag, BOUNDARY_NFACES> mesh_bcs) {
  std::array<bool, 3> periodic{mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::periodic,
                               mesh_bcs[BoundaryFace::inner_x2] == BoundaryFlag::periodic,
                               mesh_bcs[BoundaryFace::inner_x3] ==
                                   BoundaryFlag::periodic};

  std::array<int, 3> nblock, ntree;
  int ndim = 0;
  int max_common_power2_divisor = std::numeric_limits<int>::max();
  for (auto dir : {X1DIR, X2DIR, X3DIR}) {
    if (mesh_size.symmetry(dir)) {
      nblock[dir - 1] = 1;
      continue;
    }
    // Add error checking
    ndim = dir;
    nblock[dir - 1] = mesh_size.nx(dir) / block_size.nx(dir);
    max_common_power2_divisor =
        std::min(max_common_power2_divisor, MaximumPowerOf2Divisor(nblock[dir - 1]));
  }
  int max_ntree = 0;
  for (auto dir : {X1DIR, X2DIR, X3DIR}) {
    if (mesh_size.symmetry(dir)) {
      ntree[dir - 1] = 1;
      continue;
    }
    ntree[dir - 1] = nblock[dir - 1] / max_common_power2_divisor;
    max_ntree = std::max(ntree[dir - 1], max_ntree);
  }

  auto ref_level = IntegerLog2Floor(max_common_power2_divisor);
  auto level = IntegerLog2Ceil(max_ntree);

  // Create the trees and the tree logical locations in the forest (which
  // works here since we assume the trees are layed out as a hyper rectangle)
  // so we can put them in z-order. This should insure that block gids are the
  // same as they were for an athena++ style base grid.
  Indexer3D idxer({0, ntree[0] - 1}, {0, ntree[1] - 1}, {0, ntree[2] - 1});

  std::map<LogicalLocation, std::pair<RegionSize, std::shared_ptr<Tree>>> ll_map;

  for (int n = 0; n < idxer.size(); ++n) {
    auto [ix1, ix2, ix3] = idxer(n);
    RegionSize tree_domain = block_size;
    auto LLCoordLeft = [](int idx, int npoints) {
      return static_cast<double>(idx) / npoints;
    };
    auto LLCoordRight = [](int idx, int npoints) {
      return static_cast<double>(idx + 1) / npoints;
    };
    tree_domain.xmin(X1DIR) =
        mesh_size.LogicalToActualPosition(LLCoordLeft(ix1, ntree[0]), X1DIR);
    tree_domain.xmax(X1DIR) =
        mesh_size.LogicalToActualPosition(LLCoordRight(ix1, ntree[0]), X1DIR);

    tree_domain.xmin(X2DIR) =
        mesh_size.LogicalToActualPosition(LLCoordLeft(ix2, ntree[1]), X2DIR);
    tree_domain.xmax(X2DIR) =
        mesh_size.LogicalToActualPosition(LLCoordRight(ix2, ntree[1]), X2DIR);

    tree_domain.xmin(X3DIR) =
        mesh_size.LogicalToActualPosition(LLCoordLeft(ix3, ntree[2]), X3DIR);
    tree_domain.xmax(X3DIR) =
        mesh_size.LogicalToActualPosition(LLCoordRight(ix3, ntree[2]), X3DIR);
    LogicalLocation loc(level, ix1, ix2, ix3);
    ll_map[loc] = std::make_pair(tree_domain, std::shared_ptr<Tree>());
  }

  // Initialize the trees in macro-morton order
  std::int64_t tid{0};
  for (auto &[loc, p] : ll_map) {
    auto tree_bcs = mesh_bcs;
    if (loc.lx1() != 0) tree_bcs[BoundaryFace::inner_x1] = BoundaryFlag::block;
    if (loc.lx1() != ntree[0] - 1) tree_bcs[BoundaryFace::outer_x1] = BoundaryFlag::block;
    if (loc.lx2() != 0) tree_bcs[BoundaryFace::inner_x2] = BoundaryFlag::block;
    if (loc.lx2() != ntree[1] - 1) tree_bcs[BoundaryFace::outer_x2] = BoundaryFlag::block;
    if (loc.lx3() != 0) tree_bcs[BoundaryFace::inner_x3] = BoundaryFlag::block;
    if (loc.lx3() != ntree[2] - 1) tree_bcs[BoundaryFace::outer_x3] = BoundaryFlag::block;
    p.second = Tree::create(tid++, ndim, ref_level, p.first, tree_bcs);
    p.second->athena_forest_loc = loc;
  }

  // Connect the trees to each other
  Indexer3D offsets({ndim > 0 ? -1 : 0, ndim > 0 ? 1 : 0},
                    {ndim > 1 ? -1 : 0, ndim > 1 ? 1 : 0},
                    {ndim > 2 ? -1 : 0, ndim > 2 ? 1 : 0});
  for (int n = 0; n < idxer.size(); ++n) {
    auto [ix1, ix2, ix3] = idxer(n);
    LogicalLocation loc(level, ix1, ix2, ix3);
    std::array<int, 3> ix{ix1, ix2, ix3};
    for (int o = 0; o < offsets.size(); ++o) {
      CellCentOffsets ox(offsets.GetIdxArray(o));
      std::array<int, 3> nx;
      bool add = true;
      for (int dir = 0; dir < 3; ++dir) {
        nx[dir] = ix[dir] + ox[dir];
        if (nx[dir] >= ntree[dir] || nx[dir] < 0) {
          if (periodic[dir]) {
            nx[dir] = (nx[dir] + ntree[dir]) % ntree[dir];
          } else {
            add = false;
          }
        }
      }
      if (add) {
        LogicalLocation nloc(level, nx[0], nx[1], nx[2]);
        LogicalCoordinateTransformation lcoord_trans;
        lcoord_trans.use_offset = true;
        lcoord_trans.offset = ox;
        ll_map[loc].second->AddNeighborTree(ox, ll_map[nloc].second, lcoord_trans);
      }
    }
  }

  // Sort trees by their logical location in the tree mesh
  Forest fout;
  fout.root_level = ref_level;
  fout.forest_level = level;
  for (auto &[loc, p] : ll_map)
    fout.AddTree(p.second);
  return fout;
}

Forest Forest::Make2D(std::vector<std::shared_ptr<Face>> faces, std::vector<ForestBC<Edge>> bc_edges) {
  // Set the topological connections of the faces 
  for (auto &face : faces) {
    face->SetNeighbors();
    for (int ix = 0; ix < 3; ++ix) {
      for (int iy = 0; iy < 3; ++iy) {
        for (auto &neighbor : face->neighbors[ix][iy]) {
          printf("face %li at offset (%i, %i) has neighbor %li.\n", face->my_id, ix - 1, iy - 1, neighbor->my_id);
        }
      }
    }
  }
  
  // Build tree boundary conditions
  for (auto &face : faces) {
    std::array<BoundaryFlag, BOUNDARY_NFACES> bcs{BoundaryFlag::block}; 
    for (auto &[loc, edge] : face->edges) { 
      for (auto &b : bc_edges) { 
        if (edge.RelativeOrientation(b.element)) {
          bcs[loc.GetBoundaryFace()] = b.bflag;
          // TODO(LFR): Add mechanism to deal with periodic boundaries 
          break;
        }
      } 
    } 
  }

  for (auto &face : faces) {
    for (auto side : {EdgeLoc::North, EdgeLoc::East, EdgeLoc::South, EdgeLoc::West}) {
      auto neighbors = FindEdgeNeighbors(face, side);
      for (auto &n : neighbors) {
        auto lcoord_trans = LogicalCoordinateTransformationFromSharedEdge2D(side, std::get<1>(n),
                                                                     std::get<2>(n));
        face->tree->AddNeighborTree(side.GetFaceIdx2D(), std::get<0>(n)->tree, lcoord_trans);
      }
    }
    // TODO(LFR): Need to find corner neighbors
  
  }

  Forest fout;
  fout.root_level = 0;
  fout.forest_level = 0;
  for (auto &face : faces)
    fout.AddTree(face->tree);
  return fout;
}

} // namespace forest
} // namespace parthenon
