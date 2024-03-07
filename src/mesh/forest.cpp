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
#include "mesh/forest.hpp"
#include "mesh/logical_location.hpp"
#include "utils/bit_hacks.hpp"
#include "utils/indexer.hpp"

namespace parthenon {
namespace forest {

LogicalLocation RelativeOrientation::Transform(const LogicalLocation &loc_in,
                                               std::int64_t destination) const {
  std::array<std::int64_t, 3> l_out;
  int nblock = 1LL << loc_in.level();
  for (int dir = 0; dir < 3; ++dir) {
    std::int64_t l_in = loc_in.l(dir);
    // First shift the logical location index back into the interior
    // of a bordering tree assuming they have the same coordinate
    // orientation
    // TODO(LFR): Probably remove the offset option and assume it is always true
    if (use_offset) {
      l_in -= offset[dir] * nblock;
    } else {
      l_in = (l_in + nblock) % nblock;
    }
    // Then permute (and possibly flip) the coordinate indices
    // to move to the logical coordinate system of the new tree
    if (dir_flip[dir]) {
      l_out[abs(dir_connection[dir])] = nblock - 1 - l_in;
    } else {
      l_out[abs(dir_connection[dir])] = l_in;
    }
  }
  return LogicalLocation(destination, loc_in.level(), l_out[0], l_out[1], l_out[2]);
}

LogicalLocation RelativeOrientation::TransformBack(const LogicalLocation &loc_in,
                                                   std::int64_t origin) const {
  std::array<std::int64_t, 3> l_out;
  int nblock = 1LL << loc_in.level();
  for (int dir = 0; dir < 3; ++dir) {
    std::int64_t l_in = loc_in.l(abs(dir_connection[dir]));

    // Then permute (and possibly flip) the coordinate indices
    // to move to the logical coordinate system of the new tree
    if (dir_flip[dir]) {
      l_out[dir] = nblock - 1 - l_in;
    } else {
      l_out[dir] = l_in;
    }

    // First shift the logical location index back into the interior
    // of a bordering tree assuming they have the same coordinate
    // orientation
    // TODO(LFR): Probably remove the offset option and assume it is always true
    if (use_offset) {
      l_out[dir] += offset[dir] * nblock;
    } else {
      l_out[dir] = (l_out[dir] + nblock) % nblock;
    }
  }
  return LogicalLocation(origin, loc_in.level(), l_out[0], l_out[1], l_out[2]);
}

Tree::Tree(Tree::private_t, std::int64_t id, int ndim, int root_level, RegionSize domain,
           std::array<BoundaryFlag, BOUNDARY_NFACES> bcs)
    : my_id(id), ndim(ndim), domain(domain), boundary_conditions(bcs) {
  // Add internal and leaf nodes of the initial tree
  for (int l = 0; l <= root_level; ++l) {
    for (int k = 0; k < (ndim > 2 ? (1LL << l) : 1); ++k) {
      for (int j = 0; j < (ndim > 1 ? (1LL << l) : 1); ++j) {
        for (int i = 0; i < (ndim > 0 ? (1LL << l) : 1); ++i) {
          if (l == root_level) {
            leaves.emplace(std::make_pair(LogicalLocation(my_id, l, i, j, k),
                                          std::make_pair(-1, -1)));
          } else {
            internal_nodes.emplace(my_id, l, i, j, k);
          }
        }
      }
    }
  }
}

int Tree::AddMeshBlock(const LogicalLocation &loc, bool enforce_proper_nesting) {
  PARTHENON_REQUIRE(
      loc.tree() == my_id,
      "Trying to add a meshblock to a tree with a LogicalLocation on a different tree.");
  if (internal_nodes.count(loc)) return -1;
  if (leaves.count(loc)) return 0;

  std::stack<LogicalLocation> refinement_locs;
  auto parent = loc.GetParent();
  for (int l = loc.level() - 1; l >= 0; --l) {
    refinement_locs.push(parent);
    if (leaves.count(parent)) break;
    parent = parent.GetParent();
  }

  int added = 0;
  while (!refinement_locs.empty()) {
    added += Refine(refinement_locs.top(), enforce_proper_nesting);
    refinement_locs.pop();
  }

  return added;
}

int Tree::Refine(const LogicalLocation &ref_loc, bool enforce_proper_nesting) {
  PARTHENON_REQUIRE(
      ref_loc.tree() == my_id,
      "Trying to refine a tree with a LogicalLocation on a different tree.");
  // Check that this is a valid refinement location
  if (!leaves.count(ref_loc)) return 0; // Can't refine a block that doesn't exist

  // Perform the refinement for this block
  std::vector<LogicalLocation> daughters = ref_loc.GetDaughters(ndim);
  auto gid_parent = leaves[ref_loc].first;
  leaves.erase(ref_loc);
  internal_nodes.insert(ref_loc);
  for (auto &d : daughters) {
    leaves.insert(std::make_pair(d, std::make_pair(gid_parent, -1)));
  }
  int nadded = daughters.size() - 1;

  if (enforce_proper_nesting) {
    LogicalLocation parent = ref_loc.GetParent();
    int ox1 = ref_loc.lx1() - (parent.lx1() << 1);
    int ox2 = ref_loc.lx2() - (parent.lx2() << 1);
    int ox3 = ref_loc.lx3() - (parent.lx3() << 1);

    for (int k = 0; k < (ndim > 2 ? 2 : 1); ++k) {
      for (int j = 0; j < (ndim > 1 ? 2 : 1); ++j) {
        for (int i = 0; i < (ndim > 0 ? 2 : 1); ++i) {
          LogicalLocation neigh = parent.GetSameLevelNeighbor(
              i + ox1 - 1, j + ox2 - (ndim > 1), k + ox3 - (ndim > 2));
          // Need to communicate this refinement action to possible neighboring tree(s)
          // and trigger refinement there
          int n_idx =
              neigh.NeighborTreeIndex(); // Note that this can point you back to this tree
          for (auto &[neighbor_tree, orientation] : neighbors[n_idx]) {
            nadded += neighbor_tree->Refine(
                orientation.Transform(neigh, neighbor_tree->GetId()));
          }
        }
      }
    }
  }
  return nadded;
}

std::vector<NeighborLocation> Tree::FindNeighbors(const LogicalLocation &loc) const {
  const Indexer3D offsets({ndim > 0 ? -1 : 0, ndim > 0 ? 1 : 0},
                          {ndim > 1 ? -1 : 0, ndim > 1 ? 1 : 0},
                          {ndim > 2 ? -1 : 0, ndim > 2 ? 1 : 0});
  std::vector<NeighborLocation> neighbor_locs;
  for (int o = 0; o < offsets.size(); ++o) {
    auto [ox1, ox2, ox3] = offsets(o);
    if (std::abs(ox1) + std::abs(ox2) + std::abs(ox3) == 0) continue;
    FindNeighborsImpl(loc, ox1, ox2, ox3, &neighbor_locs);
  }

  const int clev = loc.level() - 1;

  return neighbor_locs;
}

std::vector<NeighborLocation> Tree::FindNeighbors(const LogicalLocation &loc, int ox1,
                                                  int ox2, int ox3) const {
  std::vector<NeighborLocation> neighbor_locs;
  FindNeighborsImpl(loc, ox1, ox2, ox3, &neighbor_locs);
  return neighbor_locs;
}

void Tree::FindNeighborsImpl(const LogicalLocation &loc, int ox1, int ox2, int ox3,
                             std::vector<NeighborLocation> *neighbor_locs) const {
  PARTHENON_REQUIRE(
      loc.tree() == my_id,
      "Trying to find neighbors in a tree with a LogicalLocation on a different tree.");
  PARTHENON_REQUIRE(leaves.count(loc) == 1, "Location must be a leaf to find neighbors.");
  auto neigh = loc.GetSameLevelNeighbor(ox1, ox2, ox3);
  int n_idx = neigh.NeighborTreeIndex();
  for (auto &[neighbor_tree, orientation] : neighbors[n_idx]) {
    auto tneigh = orientation.Transform(neigh, neighbor_tree->GetId());
    auto tloc = orientation.Transform(loc, neighbor_tree->GetId());
    PARTHENON_REQUIRE(orientation.TransformBack(tloc, GetId()) == loc,
                      "Inverse transform not working.");
    if (neighbor_tree->leaves.count(tneigh)) {
      neighbor_locs->push_back({tneigh, orientation.TransformBack(tneigh, GetId())});
    } else if (neighbor_tree->internal_nodes.count(tneigh)) {
      auto daughters = tneigh.GetDaughters(neighbor_tree->ndim);
      for (auto &n : daughters) {
        if (tloc.IsNeighborForest(n))
          neighbor_locs->push_back({n, orientation.TransformBack(n, GetId())});
      }
    } else if (neighbor_tree->leaves.count(tneigh.GetParent())) {
      auto neighp = orientation.TransformBack(tneigh.GetParent(), GetId());
      // Since coarser neighbors can cover multiple elements of the origin block and
      // because our communication algorithm packs this extra data by hand, we do not wish
      // to duplicate coarser blocks in the neighbor list. Therefore, we only include the
      // coarse block in one offset position
      auto sl_offset = loc.GetSameLevelOffsetsForest(neighp);
      if (sl_offset[0] == ox1 && sl_offset[1] == ox2 && sl_offset[2] == ox3)
        neighbor_locs->push_back({tneigh.GetParent(), neighp});
    }
  }
}

int Tree::Derefine(const LogicalLocation &ref_loc, bool enforce_proper_nesting) {
  PARTHENON_REQUIRE(
      ref_loc.tree() == my_id,
      "Trying to derefine a tree with a LogicalLocation on a different tree.");

  // ref_loc is the block to be added and its daughters are the blocks to be removed
  std::vector<LogicalLocation> daughters = ref_loc.GetDaughters(ndim);

  // Check that we can actually de-refine
  for (LogicalLocation &d : daughters) {
    // Check that the daughters actually exist as leaf nodes
    if (!leaves.count(d)) return 0;

    // Check that removing these blocks doesn't break proper nesting, that just means that
    // any of the daughters same level neighbors can't be in the internal node list (which
    // would imply that the daughter abuts a finer block) Note: these loops check more
    // than is necessary, but as written are simpler than the minimal set
    if (enforce_proper_nesting) {
      const std::vector<int> active{-1, 0, 1};
      const std::vector<int> inactive{0};
      for (int k : (ndim > 2) ? active : inactive) {
        for (int j : (ndim > 1) ? active : inactive) {
          for (int i : (ndim > 0) ? active : inactive) {
            LogicalLocation neigh = d.GetSameLevelNeighbor(i, j, k);
            // Need to check that this derefinement doesn't break proper nesting with
            // a neighboring tree or this tree
            int n_idx = neigh.NeighborTreeIndex();
            for (auto &[neighbor_tree, orientation] : neighbors[n_idx]) {
              if (neighbor_tree->internal_nodes.count(
                      orientation.Transform(neigh, neighbor_tree->GetId())))
                return 0;
            }
          }
        }
      }
    }
  }

  // Derefinement is ok
  std::int64_t dgid = std::numeric_limits<std::int64_t>::max();
  for (auto &d : daughters) {
    auto node = leaves.extract(d);
    dgid = std::min(dgid, node.mapped().first);
  }
  internal_nodes.erase(ref_loc);
  leaves.insert(std::make_pair(ref_loc, std::make_pair(dgid, -1)));
  return daughters.size() - 1;
}

std::vector<LogicalLocation> Tree::GetMeshBlockList() const {
  std::vector<LogicalLocation> mb_list;
  mb_list.reserve(leaves.size());
  for (auto &[loc, gid] : leaves)
    mb_list.push_back(loc);
  std::sort(mb_list.begin(), mb_list.end(),
            [](const auto &a, const auto &b) { return a < b; });
  return mb_list;
}

RegionSize Tree::GetBlockDomain(const LogicalLocation &loc) const {
  PARTHENON_REQUIRE(loc.IsInTree(), "Probably there is a mistake...");
  RegionSize out = domain;
  for (auto dir : {X1DIR, X2DIR, X3DIR}) {
    if (!domain.symmetry(dir)) {
      if (loc.level() >= 0) {
        out.xmin(dir) =
            domain.LogicalToActualPosition(loc.LLCoord(dir, BlockLocation::Left), dir);
        out.xmax(dir) =
            domain.LogicalToActualPosition(loc.LLCoord(dir, BlockLocation::Right), dir);
      } else {
        // Negative logical levels correspond to reduced block sizes covering the entire
        // domain.
        auto reduction_fac = 1LL << (-loc.level());
        out.nx(dir) = domain.nx(dir) / reduction_fac;
        PARTHENON_REQUIRE(out.nx(dir) % reduction_fac == 0,
                          "Trying to go to too large of a negative level.");
      }
    }
    // If this is a translational symmetry direction, set the cell to cover the entire
    // tree in that direction.
  }
  return out;
}

std::array<BoundaryFlag, BOUNDARY_NFACES>
Tree::GetBlockBCs(const LogicalLocation &loc) const {
  PARTHENON_REQUIRE(loc.IsInTree(), "Probably there is a mistake...");
  std::array<BoundaryFlag, BOUNDARY_NFACES> block_bcs = boundary_conditions;
  const int nblock = 1 << std::max(loc.level(), 0);
  if (loc.lx1() != 0) block_bcs[BoundaryFace::inner_x1] = BoundaryFlag::block;
  if (loc.lx1() != nblock - 1) block_bcs[BoundaryFace::outer_x1] = BoundaryFlag::block;
  if (loc.lx2() != 0) block_bcs[BoundaryFace::inner_x2] = BoundaryFlag::block;
  if (loc.lx2() != nblock - 1) block_bcs[BoundaryFace::outer_x2] = BoundaryFlag::block;
  if (loc.lx3() != 0) block_bcs[BoundaryFace::inner_x3] = BoundaryFlag::block;
  if (loc.lx3() != nblock - 1) block_bcs[BoundaryFace::outer_x3] = BoundaryFlag::block;
  return block_bcs;
}

void Tree::AddNeighborTree(CellCentOffsets offset, std::shared_ptr<Tree> neighbor_tree,
                           RelativeOrientation orient) {
  int location_idx = offset.GetIdx();
  neighbors[location_idx].insert({neighbor_tree, orient});
  BoundaryFace fidx = offset.Face();
  if (fidx >= 0) boundary_conditions[fidx] = BoundaryFlag::block;
}

std::vector<LogicalLocation> Forest::GetMeshBlockListAndResolveGids() {
  std::vector<LogicalLocation> mb_list;
  std::uint64_t gid{0};
  for (auto &tree : trees) {
    std::size_t start = mb_list.size();
    auto tree_mbs = tree->GetMeshBlockList();
    mb_list.insert(mb_list.end(), std::make_move_iterator(tree_mbs.begin()),
                   std::make_move_iterator(tree_mbs.end()));
    std::size_t end = mb_list.size();
    for (int i = start; i < end; ++i)
      tree->InsertGid(mb_list[i], gid++);
  }
  // The index of blocks in this list corresponds to their gid
  gids_resolved = true;
  return mb_list;
}

Forest Forest::AthenaXX(RegionSize mesh_size, RegionSize block_size,
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
    auto &dmn = tree_domain;
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
        RelativeOrientation orient;
        orient.use_offset = true;
        orient.offset = ox;
        ll_map[loc].second->AddNeighborTree(ox, ll_map[nloc].second, orient);
      }
    }
  }

  // Sort trees by their logical location in the tree mesh
  Forest fout;
  fout.root_level = ref_level;
  fout.forest_level = level;
  for (auto &[loc, p] : ll_map)
    fout.trees.push_back(p.second);
  return fout;
}

} // namespace forest
} // namespace parthenon
