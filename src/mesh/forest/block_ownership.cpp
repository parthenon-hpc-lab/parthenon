//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2023 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2023. Triad National Security, LLC. All rights reserved.
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
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "mesh/forest/block_ownership.hpp"
#include "mesh/forest/logical_location.hpp"
#include "utils/error_checking.hpp"
#include "utils/morton_number.hpp"

namespace parthenon {

block_ownership_t
DetermineOwnershipForest(const LogicalLocation &main_block,
                         const std::vector<NeighborLocation> &allowed_neighbors,
                         const std::unordered_set<LogicalLocation> &newly_refined) {
  block_ownership_t main_owns;

  auto ownership_level = [&](const LogicalLocation &a) {
    // Newly-refined blocks are treated as higher-level than blocks at their
    // parent level, but lower-level than previously-refined blocks at their
    // current level.
    if (newly_refined.count(a)) return 2 * a.level() - 1;
    return 2 * a.level();
  };

  auto ownership_less_than = [ownership_level](const LogicalLocation &a,
                                               const LogicalLocation &b) {
    // Ownership is first determined by block with the highest level, then by maximum
    // (tree, Morton) number this is reversed in precedence from the normal comparators
    // where (tree, Morton) number takes precedence
    if (ownership_level(a) != ownership_level(b))
      return ownership_level(a) < ownership_level(b);
    if (a.tree() != b.tree()) return a.tree() < b.tree();
    return a.morton() < b.morton();
  };

  for (int ox1 : {-1, 0, 1}) {
    for (int ox2 : {-1, 0, 1}) {
      for (int ox3 : {-1, 0, 1}) {
        main_owns(ox1, ox2, ox3) = true;
        for (const auto &n : allowed_neighbors) {
          if (ownership_less_than(main_block, n.global_loc) &&
              main_block.IsNeighborOfTEForest(n.origin_loc, {ox1, ox2, ox3})) {
            main_owns(ox1, ox2, ox3) = false;
            break;
          }
        }
      }
    }
  }
  return main_owns;
}

// Given a topological element, ownership array of the sending block, and offset indices
// defining the location of an index region within the block (i.e. the ghost zones passed
// across the x-face or the ghost zones passed across the z-edge), return the index range
// masking array required for masking out unowned regions of the index space. ox? defines
// buffer location on the owner block
block_ownership_t
GetIndexRangeMaskFromOwnership(TopologicalElement el,
                               const block_ownership_t &sender_ownership, int ox1,
                               int ox2, int ox3) {
  using vp_t = std::vector<std::pair<int, int>>;

  // Transform general block ownership to element ownership over entire block. For
  // instance, x-faces only care about block ownership in the x-direction First index of
  // the pair is the element index and the second index is the block index that is copied
  // to that element index
  block_ownership_t element_ownership = sender_ownership;
  auto x1_idxs = TopologicalOffsetI(el) ? vp_t{{-1, -1}, {0, 0}, {1, 1}}
                                        : vp_t{{-1, 0}, {0, 0}, {1, 0}};
  auto x2_idxs = TopologicalOffsetJ(el) ? vp_t{{-1, -1}, {0, 0}, {1, 1}}
                                        : vp_t{{-1, 0}, {0, 0}, {1, 0}};
  auto x3_idxs = TopologicalOffsetK(el) ? vp_t{{-1, -1}, {0, 0}, {1, 1}}
                                        : vp_t{{-1, 0}, {0, 0}, {1, 0}};
  for (auto [iel, ibl] : x1_idxs) {
    for (auto [jel, jbl] : x2_idxs) {
      for (auto [kel, kbl] : x3_idxs) {
        element_ownership(iel, jel, kel) = sender_ownership(ibl, jbl, kbl);
      }
    }
  }

  // Now, the ownership status is correct for the entire interior index range of the
  // block, but the offsets ox? define a subset of these indices (e.g. one edge of the
  // interior). Therefore, we need to set the index ownership to true for edges of the
  // index range that are contained in the interior of the sending block
  if (ox1 != 0) {
    for (auto j : {-1, 0, 1}) {
      for (auto k : {-1, 0, 1}) {
        element_ownership(-ox1, j, k) = element_ownership(0, j, k);
      }
    }
  }
  if (ox2 != 0) {
    for (auto i : {-1, 0, 1}) {
      for (auto k : {-1, 0, 1}) {
        element_ownership(i, -ox2, k) = element_ownership(i, 0, k);
      }
    }
  }
  if (ox3 != 0) {
    for (auto i : {-1, 0, 1}) {
      for (auto j : {-1, 0, 1}) {
        element_ownership(i, j, -ox3) = element_ownership(i, j, 0);
      }
    }
  }

  return element_ownership;
}

} // namespace parthenon
