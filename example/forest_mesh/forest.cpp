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
#include "mesh/logical_location.hpp"
#include "utils/bit_hacks.hpp"
#include "utils/indexer.hpp"

#include "forest.hpp"

namespace parthenon {
namespace forest {

LogicalLocation RelativeOrientation::Transform(const LogicalLocation &loc_in) const {
  std::array<std::int64_t, 3> l_out;
  int nblock = 1LL << loc_in.level();
  for (int dir = 0; dir < 3; ++dir) {
    std::int64_t l_in = loc_in.l(dir);
    // First shift the logical location index back into the interior
    // of a bordering tree assuming they have the same coordinate
    // orientation
    l_in = (l_in + nblock) % nblock;
    // Then permute (and possibly flip) the coordinate indices
    // to move to the logical coordinate system of the new tree
    if (dir_flip[dir]) {
      l_out[abs(dir_connection[dir])] = nblock - 1 - l_in;
    } else {
      l_out[abs(dir_connection[dir])] = l_in;
    }
  }
  return LogicalLocation(loc_in.level(), l_out[0], l_out[1], l_out[2]);
}

Tree::Tree(Tree::private_t, int ndim, int root_level, RegionSize domain)
    : ndim(ndim), domain(domain) {
  // Add internal and leaf nodes of the initial tree
  for (int l = 0; l <= root_level; ++l) {
    for (int k = 0; k < (ndim > 2 ? (1LL << l) : 1); ++k) {
      for (int j = 0; j < (ndim > 1 ? (1LL << l) : 1); ++j) {
        for (int i = 0; i < (ndim > 0 ? (1LL << l) : 1); ++i) {
          if (l == root_level) {
            leaves.emplace(l, i, j, k);
          } else {
            internal_nodes.emplace(l, i, j, k);
          }
        }
      }
    }
  }
}

int Tree::AddMeshBlock(const LogicalLocation &loc, bool enforce_proper_nesting) {
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
  // Check that this is a valid refinement location
  if (!leaves.count(ref_loc)) return 0; // Can't refine a block that doesn't exist

  // Perform the refinement for this block
  std::vector<LogicalLocation> daughters = ref_loc.GetDaughters(ndim);
  leaves.erase(ref_loc);
  internal_nodes.insert(ref_loc);
  leaves.insert(daughters.begin(), daughters.end());
  int nadded = daughters.size();

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
            nadded += neighbor_tree->Refine(orientation.Transform(neigh));
          }
        }
      }
    }
  }
  return nadded;
}

std::vector<ForestLocation> Tree::FindNeighbor(const LogicalLocation &loc, int ox1,
                                               int ox2, int ox3) const {
  PARTHENON_REQUIRE(leaves.count(loc) == 1, "Location must be a leaf to find neighbors.");
  std::vector<ForestLocation> neighbor_locs;
  auto neigh = loc.GetSameLevelNeighbor(ox1, ox2, ox3);
  int n_idx = neigh.NeighborTreeIndex();
  for (auto &[neighbor_tree, orientation] : neighbors[n_idx]) {
    auto tneigh = orientation.Transform(neigh);
    if (neighbor_tree->leaves.count(tneigh)) {
      neighbor_locs.push_back(ForestLocation{neighbor_tree->GetId(), tneigh});
    } else if (neighbor_tree->internal_nodes.count(tneigh)) {
      auto daughters = tneigh.GetDaughters(neighbor_tree->ndim);
      for (auto &n : daughters)
        neighbor_locs.push_back(ForestLocation{neighbor_tree->GetId(), n});
    } else if (neighbor_tree->leaves.count(tneigh.GetParent())) {
      neighbor_locs.push_back(ForestLocation{neighbor_tree->GetId(), tneigh.GetParent()});
    }
  }
  return neighbor_locs;
}

int Tree::Derefine(const LogicalLocation &ref_loc, bool enforce_proper_nesting) {
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
              if (neighbor_tree->internal_nodes.count(orientation.Transform(neigh)))
                return 0;
            }
          }
        }
      }
    }
  }

  // Derefinement is ok
  for (auto &d : daughters)
    leaves.erase(d);
  internal_nodes.erase(ref_loc);
  leaves.insert(ref_loc);
  return daughters.size();
}

std::vector<ForestLocation> Tree::GetMeshBlockList() const {
  std::vector<ForestLocation> mb_list;
  mb_list.reserve(leaves.size());
  for (auto &loc : leaves)
    mb_list.push_back({my_id, loc});
  std::sort(mb_list.begin(), mb_list.end(),
            [](const auto &a, const auto &b) { return a.second < b.second; });
  return mb_list;
}

RegionSize Tree::GetBlockDomain(LogicalLocation loc) const {
  RegionSize out = domain;
  for (auto dir : {X1DIR, X2DIR, X3DIR}) {
    if (!domain.symmetry(dir)) {
      int n = 1 << loc.level();
      out.xmin(dir) = domain.LogicalToActualPosition((double)loc.l(dir - 1) / n, dir);
      out.xmax(dir) = domain.LogicalToActualPosition((double)(loc.l(dir - 1) + 1) / n, dir);
    }
    // If this is a translational symmetry direction, set the cell to cover the entire
    // tree in that direction.
  }
  return out;
}

std::vector<ForestLocation> Forest::GetMeshBlockList() const {
  std::vector<ForestLocation> mb_list;
  for (auto &tree : trees) {
    auto tree_mbs = tree->GetMeshBlockList();
    mb_list.insert(mb_list.end(), std::make_move_iterator(tree_mbs.begin()),
                   std::make_move_iterator(tree_mbs.end()));
  }
  // The index of blocks in this list corresponds to their gid
  return mb_list;
}

Forest Forest::AthenaXX(RegionSize mesh_size, RegionSize block_size,
                        std::array<bool, 3> periodic) {
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
    printf("dir: %i nblock: %i\n", dir, nblock[dir - 1]);
    max_common_power2_divisor =
        std::min(max_common_power2_divisor, MaximumPowerOf2Divisor(nblock[dir - 1]));
  }
  printf("max common divisor = %i\n", max_common_power2_divisor);
  int max_ntree = 0;
  for (auto dir : {X1DIR, X2DIR, X3DIR}) {
    if (mesh_size.symmetry(dir)) {
      ntree[dir - 1] = 1;
      continue;
    }
    ntree[dir - 1] = nblock[dir - 1] / max_common_power2_divisor;
    max_ntree = std::max(ntree[dir - 1], max_ntree);
  }

  printf("ntree = {%i, %i, %i}\n", ntree[0], ntree[1], ntree[2]);

  auto ref_level = IntegerLog2(max_common_power2_divisor);
  auto level = IntegerLog2(max_ntree);

  // Create the trees and the tree logical locations in the forest (which
  // works here since we assume the trees are layed out as a hyper rectangle)
  // so we can put them in z-order. This should insure that block gids are the
  // same as they were for an athena++ style base grid.
  Indexer3D idxer({0, ntree[0] - 1}, {0, ntree[1] - 1}, {0, ntree[2] - 1});
  using p_loc_tree_t = std::pair<LogicalLocation, std::shared_ptr<Tree>>;
  std::vector<p_loc_tree_t> loc_tree;
  for (int n = 0; n < idxer.size(); ++n) {
    auto [ix1, ix2, ix3] = idxer(n);
    RegionSize tree_domain = block_size;
    tree_domain.xmin(X1DIR) = mesh_size.LogicalToActualPosition((double)ix1 / ntree[0], X1DIR);
    tree_domain.xmax(X1DIR) =
        mesh_size.LogicalToActualPosition((double)(ix1 + 1) / ntree[0], X1DIR);

    tree_domain.xmin(X2DIR) = mesh_size.LogicalToActualPosition((double)ix2 / ntree[1], X2DIR);
    tree_domain.xmax(X2DIR) =
        mesh_size.LogicalToActualPosition((double)(ix2 + 1) / ntree[1], X2DIR);
    
    tree_domain.xmin(X3DIR) = mesh_size.LogicalToActualPosition((double)ix3 / ntree[2], X3DIR);
    tree_domain.xmax(X3DIR) =
        mesh_size.LogicalToActualPosition((double)(ix3 + 1) / ntree[2], X3DIR);
    loc_tree.emplace_back(p_loc_tree_t{LogicalLocation(level, ix1, ix2, ix3),
                                       Tree::create(ndim, ref_level, tree_domain)});
    auto &dmn = tree_domain;
    printf("[%i, %i, %i], %e, %e, %e, %e\n", ix1, ix2, ix3, 
        dmn.xmin(X1DIR), dmn.xmax(X1DIR),
        dmn.xmin(X2DIR), dmn.xmax(X2DIR));
  }

  // Connect the trees to each other
  Indexer3D offsets({ndim > 0 ? -1 : 0, ndim > 0 ? 1 : 0},
                    {ndim > 1 ? -1 : 0, ndim > 1 ? 1 : 0},
                    {ndim > 2 ? -1 : 0, ndim > 2 ? 1 : 0});
  for (int n = 0; n < idxer.size(); ++n) {
    auto [ix1, ix2, ix3] = idxer(n);
    std::array<int, 3> ix{ix1, ix2, ix3};
    for (int o = 0; o < offsets.size(); ++o) {
      auto [ox1, ox2, ox3] = offsets(o);
      std::array<int, 3> ox{ox1, ox2, ox3}, nx;
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
        int neigh_idx = nx[0] + ntree[0] * (nx[1] + ntree[1] * nx[2]);
        int loc_idx = (ox1 + 1) + 3 * (ox2 + 1) + 9 * (ox3 + 1);
        loc_tree[n].second->AddNeighbor(loc_idx, loc_tree[neigh_idx].second,
                                        RelativeOrientation());
      }
    }
  }
  // Sort trees by their logical location in the tree mesh
  Forest fout;
  std::sort(loc_tree.begin(), loc_tree.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });
  for (auto &[loc, tree] : loc_tree)
    fout.trees.push_back(tree);

  // Assign tree ids
  std::uint64_t id = 0;
  for (auto &tree : fout.trees)
    tree->SetId(id++);
  return fout;
}

} // namespace forest
} // namespace parthenon
