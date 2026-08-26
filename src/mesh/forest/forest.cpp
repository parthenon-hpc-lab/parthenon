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

#include "application_input.hpp"
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
      // Symmetry directions have just a single zone for
      // both the mesh and for each block
      max_common_power2_divisor = 1;
      continue;
    }
    // Add error checking
    ndim = dir;
    nblock[dir - 1] = mesh_size.nx(dir) / block_size.nx(dir);
    PARTHENON_REQUIRE(mesh_size.nx(dir) % block_size.nx(dir) == 0,
                      "Block size is not evenly divisible into the base mesh size.");
    PARTHENON_REQUIRE(nblock[dir - 1] > 0,
                      "Must have a mesh that has a block size greater than one.");
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
    tree_domain.xmin(X1DIR) = mesh_size.SymmetrizedLogicalToActualPosition(
        LogicalLocation::IndexToSymmetrizedCoordinate(ix1, BlockLocation::Left, ntree[0]),
        X1DIR);
    tree_domain.xmax(X1DIR) = mesh_size.SymmetrizedLogicalToActualPosition(
        LogicalLocation::IndexToSymmetrizedCoordinate(ix1, BlockLocation::Right,
                                                      ntree[0]),
        X1DIR);

    tree_domain.xmin(X2DIR) = mesh_size.SymmetrizedLogicalToActualPosition(
        LogicalLocation::IndexToSymmetrizedCoordinate(ix2, BlockLocation::Left, ntree[1]),
        X2DIR);
    tree_domain.xmax(X2DIR) = mesh_size.SymmetrizedLogicalToActualPosition(
        LogicalLocation::IndexToSymmetrizedCoordinate(ix2, BlockLocation::Right,
                                                      ntree[1]),
        X2DIR);

    tree_domain.xmin(X3DIR) = mesh_size.SymmetrizedLogicalToActualPosition(
        LogicalLocation::IndexToSymmetrizedCoordinate(ix3, BlockLocation::Left, ntree[2]),
        X3DIR);
    tree_domain.xmax(X3DIR) = mesh_size.SymmetrizedLogicalToActualPosition(
        LogicalLocation::IndexToSymmetrizedCoordinate(ix3, BlockLocation::Right,
                                                      ntree[2]),
        X3DIR);
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

    auto &tree_domain = p.first;
    p.second = Tree::create(tid++, ndim, ref_level, tree_domain, tree_bcs);
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
      bool p = false;
      for (int dir = 0; dir < 3; ++dir) {
        nx[dir] = ix[dir] + ox[dir];
        if (nx[dir] >= ntree[dir] || nx[dir] < 0) {
          if (periodic[dir]) {
            nx[dir] = (nx[dir] + ntree[dir]) % ntree[dir];
            p = true;
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
        ll_map[loc].second->AddNeighborTree(ox, ll_map[nloc].second, lcoord_trans, p);
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

Forest Forest::Make2D(ForestDefinition &forest_def) {
  auto &faces = forest_def.faces;
  // Set the topological connections of the faces
  for (auto &face : faces) {
    face->SetNeighbors();
    face->SetEdgeCoordinateTransforms();
  }
  // Have to do this in a second sweep after setting edge transformations, since it relies
  // on composing edge coordinate transformations
  for (auto &face : faces)
    face->SetNodeCoordinateTransforms();

  using tree_bc_t = std::array<BoundaryFlag, BOUNDARY_NFACES>;
  std::unordered_map<int64_t, tree_bc_t> tree_bcs;
  for (auto &face : faces) {
    tree_bc_t bcs;

    // Set the boundaries that are shared with other trees
    if (face->HasNeighbor(-1, 0)) bcs[BoundaryFace::inner_x1] = BoundaryFlag::block;
    if (face->HasNeighbor(1, 0)) bcs[BoundaryFace::outer_x1] = BoundaryFlag::block;
    if (face->HasNeighbor(0, -1)) bcs[BoundaryFace::inner_x2] = BoundaryFlag::block;
    if (face->HasNeighbor(0, 1)) bcs[BoundaryFace::outer_x2] = BoundaryFlag::block;
    bcs[BoundaryFace::inner_x3] = BoundaryFlag::periodic;
    bcs[BoundaryFace::outer_x3] = BoundaryFlag::periodic;

    tree_bcs[face->GetId()] = bcs;
  }

  // Set the user specified boundary conditions, the order of this matters wrt the
  // BoundaryFlag::block setting
  for (auto &bc_edge : forest_def.bc_edges) {
    auto edge = bc_edge.element;
    for (auto &node : edge.nodes) {
      for (auto &w_face : node->associated_faces) {
        PARTHENON_REQUIRE(!w_face.expired(), "Invalid weak pointer to face.");
        auto face = w_face.lock();
        auto opt_offset = face->IsEdge(edge);
        if (opt_offset) {
          auto &bcs = tree_bcs[face->GetId()];
          if ((*opt_offset)[0] == Offset::Low) {
            bcs[BoundaryFace::inner_x1] = bc_edge.bflag;
          } else if ((*opt_offset)[0] == Offset::Up) {
            bcs[BoundaryFace::outer_x1] = bc_edge.bflag;
          } else if ((*opt_offset)[1] == Offset::Low) {
            bcs[BoundaryFace::inner_x2] = bc_edge.bflag;
          } else if ((*opt_offset)[1] == Offset::Up) {
            bcs[BoundaryFace::outer_x2] = bc_edge.bflag;
          }
        }
      }
    }
  }

  // Build the list of trees and set neighbors
  std::unordered_map<std::int64_t, std::shared_ptr<Tree>> trees;
  Real x_offset = 0.0;
  for (int f = 0; f < faces.size(); ++f) {
    const auto &face = faces[f];
    const auto &face_size = forest_def.face_sizes[f];
    RegionSize tree_domain = forest_def.block_size;
    // TODO(LFR): Fix this to do something not stupid
    tree_domain.xmin(X1DIR) = face_size.xmin(X1DIR);
    tree_domain.xmax(X1DIR) = face_size.xmax(X1DIR);
    tree_domain.xmin(X2DIR) = face_size.xmin(X2DIR);
    tree_domain.xmax(X2DIR) = face_size.xmax(X2DIR);
    tree_domain.xmin(X3DIR) = face_size.xmin(X3DIR);
    tree_domain.xmax(X3DIR) = face_size.xmax(X3DIR);
    auto &bcs = tree_bcs[face->GetId()];
    trees[face->GetId()] = Tree::create(face->GetId(), 2, 0, tree_domain,
                                        tree_bcs[face->GetId()], face->nodes);
    x_offset += 2.0;
  }

  for (const auto &face : faces) {
    for (int ox1 = -1; ox1 < 2; ++ox1) {
      for (int ox2 = -1; ox2 < 2; ++ox2) {
        for (auto &[neighbor, ct] : face->neighbors(ox1, ox2)) {
          PARTHENON_REQUIRE(!neighbor.expired(), "Invalid weak pointer to face.");
          trees[face->GetId()]->AddNeighborTree(
              CellCentOffsets(ox1, ox2, 0), trees[neighbor.lock()->GetId()], ct, false);
        }
      }
    }
  }

  Forest fout;
  fout.root_level = 0;
  fout.forest_level = 0;
  for (auto &[id, tree] : trees)
    fout.AddTree(tree);

  // Add requested refinement to base forest
  for (const auto &loc : forest_def.refinement_locations)
    fout.AddMeshBlock(loc);

  return fout;
}

// TODO(LFR): Probably eventually remove this. This is only meaningful for simply
// oriented grids
LogicalLocation Forest::GetLegacyTreeLocation(const LogicalLocation &loc) const {
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
Forest::GetForestLocationFromLegacyTreeLocation(const LogicalLocation &loc) const {
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

} // namespace forest
} // namespace parthenon
