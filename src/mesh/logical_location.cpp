//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
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

#include "mesh/logical_location.hpp"
#include "utils/error_checking.hpp"
#include "utils/morton_number.hpp"

namespace parthenon {

bool LogicalLocation::IsContainedIn(const LogicalLocation &container) const {
  if (container.level() > level()) return false;
  bool is_contained = true;
  const int level_shift = level() - container.level();
  for (int i = 0; i < 3; ++i)
    is_contained = is_contained && (l(i) >> level_shift == container.l(i));
  return is_contained;
}

bool LogicalLocation::Contains(const LogicalLocation &containee) const {
  if (containee.level() < level_) return false;
  const std::int64_t shifted_lx1 = containee.lx1() >> (containee.level() - level());
  const std::int64_t shifted_lx2 = containee.lx2() >> (containee.level() - level());
  const std::int64_t shifted_lx3 = containee.lx3() >> (containee.level() - level());
  return (shifted_lx1 == lx1()) && (shifted_lx2 == lx2()) && (shifted_lx3 == lx3());
}

std::array<int, 3> LogicalLocation::GetOffset(const LogicalLocation &neighbor,
                                              const RootGridInfo &rg_info) const {
  std::array<int, 3> offset;
  const int level_diff_1 = std::max(neighbor.level() - level(), 0);
  const int level_diff_2 = std::max(level() - neighbor.level(), 0);
  const int n_per_root_block = 1 << (std::min(level(), neighbor.level()) - rg_info.level);
  for (int i = 0; i < 3; ++i) {
    offset[i] = (neighbor.l(i) >> level_diff_1) - (l(i) >> level_diff_2);
    if (rg_info.periodic[i]) {
      const int n_cells_level = std::max(n_per_root_block * rg_info.n[i], 1);
      if (std::abs(offset[i]) > (n_cells_level / 2)) {
        offset[i] %= n_cells_level;
        offset[i] += offset[i] > 0 ? -n_cells_level : n_cells_level;
      }
    }
  }

  return offset;
}

std::array<std::vector<int>, 3>
LogicalLocation::GetSameLevelOffsets(const LogicalLocation &neighbor,
                                     const RootGridInfo &rg_info) const {
  std::array<std::vector<int>, 3> offsets;
  const int level_diff_1 = std::max(neighbor.level() - level(), 0);
  const int level_diff_2 = std::max(level() - neighbor.level(), 0);
  const int n_per_root_block = 1 << std::max((std::min(level(), neighbor.level()) - rg_info.level), 0);
  for (int i = 0; i < 3; ++i) {
    const auto idxt = l(i) >> level_diff_2;
    const auto idxn = neighbor.l(i) >> level_diff_1;
    if (std::abs(idxn - idxt) <= 1) offsets[i].push_back(idxn - idxt);

    const int n_blocks_level = std::max(n_per_root_block * rg_info.n[i], 1);
    if (rg_info.periodic[i]) {
      if (std::abs(idxn - n_blocks_level - idxt) <= 1)
        offsets[i].push_back(idxn - n_blocks_level - idxt);
      if (std::abs(idxn + n_blocks_level - idxt) <= 1)
        offsets[i].push_back(idxn + n_blocks_level - idxt);
    }
  }

  return offsets;
}

template <bool TENeighbor>
bool LogicalLocation::NeighborFindingImpl(const LogicalLocation &in,
                                          const std::array<int, 3> &te_offset,
                                          const RootGridInfo &rg_info) const {
  if (in.level() >= level() && Contains(in)) return false;      // You share a volume
  if (in.level() < level() && in.Contains(*this)) return false; // You share a volume

  // We work on the finer level of in.level() and this->level()
  const int max_level = std::max(in.level(), level());
  const int level_shift_1 = max_level - level();
  const int level_shift_2 = max_level - in.level();
  const auto block_size_1 = 1 << level_shift_1;
  const auto block_size_2 = 1 << level_shift_2;

  // TODO(LFR): Think about what this should do when we are above the root level
  const int n_per_root_block = 1 << (max_level - rg_info.level);
  std::array<bool, 3> b;

  for (int i = 0; i < 3; ++i) {
    // Index range of daughters of this block on current level plus a one block halo on
    // either side
    auto low = (l(i) << level_shift_1) - 1;
    auto hi = low + block_size_1 + 1;
    // Indexing for topological offset calculation
    if constexpr (TENeighbor) {
      if (te_offset[i] == -1) {
        // Left side offset, so only two possible block indices are allowed in this
        // direction
        hi -= block_size_1;
      } else if (te_offset[i] == 0) {
        // No offset in this direction, so only interior
        low += 1;
        hi -= 1;
      } else {
        // Right side offset, so only two possible block indices are allowed in this
        // direction
        low += block_size_1;
      }
    }
    // Index range of daughters of possible neighbor block on current level
    const auto in_low = in.l(i) << level_shift_2;
    const auto in_hi = in_low + block_size_2 - 1;
    // Check if these two ranges overlap at all
    b[i] = in_hi >= low && in_low <= hi;
    if (rg_info.periodic[i]) {
      const int n_cells_level = std::max(n_per_root_block * rg_info.n[i], 1);
      b[i] = b[i] || (in_hi + n_cells_level >= low && in_low + n_cells_level <= hi);
      b[i] = b[i] || (in_hi - n_cells_level >= low && in_low - n_cells_level <= hi);
    }
  }

  return b[0] && b[1] && b[2];
}
template bool
LogicalLocation::NeighborFindingImpl<true>(const LogicalLocation &in,
                                           const std::array<int, 3> &te_offset,
                                           const RootGridInfo &rg_info) const;
template bool
LogicalLocation::NeighborFindingImpl<false>(const LogicalLocation &in,
                                            const std::array<int, 3> &te_offset,
                                            const RootGridInfo &rg_info) const;

std::vector<LogicalLocation> LogicalLocation::GetDaughters() const {
  std::vector<LogicalLocation> daughters;
  daughters.reserve(8);
  for (int i : {0, 1}) {
    for (int j : {0, 1}) {
      for (int k : {0, 1}) {
        daughters.push_back(GetDaughter(i, j, k));
      }
    }
  }
  return daughters;
}

std::unordered_set<LogicalLocation>
LogicalLocation::GetPossibleNeighbors(const RootGridInfo &rg_info) {
  const std::vector<int> irange{-1, 0, 1};
  const std::vector<int> jrange{-1, 0, 1};
  const std::vector<int> krange{-1, 0, 1};
  const std::vector<int> daughter_irange{0, 1};
  const std::vector<int> daughter_jrange{0, 1};
  const std::vector<int> daughter_krange{0, 1};

  return GetPossibleNeighborsImpl(irange, jrange, krange, daughter_irange,
                                  daughter_jrange, daughter_krange, rg_info);
}

std::unordered_set<LogicalLocation>
LogicalLocation::GetPossibleBlocksSurroundingTopologicalElement(
    int ox1, int ox2, int ox3, const RootGridInfo &rg_info) const {
  const auto irange =
      (std::abs(ox1) == 1) ? std::vector<int>{0, ox1} : std::vector<int>{0};
  const auto jrange =
      (std::abs(ox2) == 1) ? std::vector<int>{0, ox2} : std::vector<int>{0};
  const auto krange =
      (std::abs(ox3) == 1) ? std::vector<int>{0, ox3} : std::vector<int>{0};
  const auto daughter_irange =
      (std::abs(ox1) == 1) ? std::vector<int>{ox1 > 0} : std::vector<int>{0, 1};
  const auto daughter_jrange =
      (std::abs(ox2) == 1) ? std::vector<int>{ox2 > 0} : std::vector<int>{0, 1};
  const auto daughter_krange =
      (std::abs(ox3) == 1) ? std::vector<int>{ox3 > 0} : std::vector<int>{0, 1};

  return GetPossibleNeighborsImpl(irange, jrange, krange, daughter_irange,
                                  daughter_jrange, daughter_krange, rg_info);
}

std::unordered_set<LogicalLocation> LogicalLocation::GetPossibleNeighborsImpl(
    const std::vector<int> &irange, const std::vector<int> &jrange,
    const std::vector<int> &krange, const std::vector<int> &daughter_irange,
    const std::vector<int> &daughter_jrange, const std::vector<int> &daughter_krange,
    const RootGridInfo &rg_info) const {
  std::vector<LogicalLocation> locs;

  auto AddNeighbors = [&](const LogicalLocation &loc, bool include_parents) {
    const int n_per_root_block = 1 << std::max(loc.level() - rg_info.level, 0);
    const int down_shift = std::max(rg_info.level - loc.level(), 0); 
    int n1_cells_level = std::max(n_per_root_block * (rg_info.n[0] >> down_shift), 1);
    int n2_cells_level = std::max(n_per_root_block * (rg_info.n[1] >> down_shift), 1);
    int n3_cells_level = std::max(n_per_root_block * (rg_info.n[2] >> down_shift), 1);
    for (int i : irange) {
      for (int j : jrange) {
        for (int k : krange) {
          auto lx1 = loc.lx1() + i;
          auto lx2 = loc.lx2() + j;
          auto lx3 = loc.lx3() + k;
          // This should include blocks that are connected by periodic boundaries
          if (rg_info.periodic[0]) lx1 = (lx1 + n1_cells_level) % n1_cells_level;
          if (rg_info.periodic[1]) lx2 = (lx2 + n2_cells_level) % n2_cells_level;
          if (rg_info.periodic[2]) lx3 = (lx3 + n3_cells_level) % n3_cells_level;
          if (0 <= lx1 && lx1 < n1_cells_level && 0 <= lx2 && lx2 < n2_cells_level &&
              0 <= lx3 && lx3 < n3_cells_level) {
            if (loc.level() > level()) {
              const int s = loc.level() - level();
              if ((lx1 >> s) != this->lx1() || (lx2 >> s) != this->lx2() || (lx3 >> s) != this->lx3()) {
                locs.emplace_back(loc.level(), lx1, lx2, lx3);
              }
            } else { 
              locs.emplace_back(loc.level(), lx1, lx2, lx3);
            }
            if (include_parents) {
              auto parent = locs.back().GetParent();
              if (IsNeighbor(parent, rg_info)) locs.push_back(parent);
            }
          }
        }
      }
    }
  };

  // Find the same level and lower level blocks of this block
  AddNeighbors(*this, true);

  // Iterate over daughters of this block that share the same topological element
  for (int l : daughter_irange) {
    for (int m : daughter_jrange) {
      for (int n : daughter_krange) {
        AddNeighbors(GetDaughter(l, m, n), false);
      }
    }
  }
  // The above procedure likely duplicated some blocks, so put them in a set
  std::unordered_set<LogicalLocation> unique_locs;
  for (auto &loc : locs)
    unique_locs.emplace(std::move(loc));
  return unique_locs;
}

block_ownership_t
DetermineOwnership(const LogicalLocation &main_block,
                   const std::unordered_set<LogicalLocation> &allowed_neighbors,
                   const RootGridInfo &rg_info) {
  block_ownership_t main_owns;

  auto ownership_less_than = [](const LogicalLocation &a, const LogicalLocation &b) {
    // Ownership is first determined by block with the highest level, then by maximum
    // Morton number this is reversed in precedence from the normal comparators where
    // Morton number takes precedence
    if (a.level() == b.level()) return a.morton() < b.morton();
    return a.level() < b.level();
  };

  for (int ox1 : {-1, 0, 1}) {
    for (int ox2 : {-1, 0, 1}) {
      for (int ox3 : {-1, 0, 1}) {
        main_owns(ox1, ox2, ox3) = true;
        for (auto &n : allowed_neighbors) {
          if (ownership_less_than(main_block, n) &&
              main_block.IsNeighborOfTE(n, ox1, ox2, ox3, rg_info)) {
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
