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

#include "mesh/forest/logical_location.hpp"
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

std::array<int, 3>
LogicalLocation::GetSameLevelOffsetsForest(const LogicalLocation &neighbor) const {
  std::array<int, 3> offsets;
  const int level_shift_neigh = std::max(neighbor.level() - level(), 0);
  const int level_shift_me = std::max(level() - neighbor.level(), 0);
  for (int dir = 0; dir < 3; ++dir) {
    // coarsen locations to the same level
    offsets[dir] = (neighbor.l(dir) >> level_shift_neigh) - (l(dir) >> level_shift_me);
  }
  return offsets;
}

std::array<std::vector<int>, 3>
LogicalLocation::GetSameLevelOffsets(const LogicalLocation &neighbor,
                                     const RootGridInfo &rg_info) const {
  std::array<std::vector<int>, 3> offsets;
  const int level_diff_1 = std::max(neighbor.level() - level(), 0);
  const int level_diff_2 = std::max(level() - neighbor.level(), 0);
  const int n_per_root_block =
      1 << std::max((std::min(level(), neighbor.level()) - rg_info.level), 0);
  const int root_block_per_n =
      1 << std::max(rg_info.level - std::min(level(), neighbor.level()), 0);
  for (int i = 0; i < 3; ++i) {
    const auto idxt = l(i) >> level_diff_2;
    const auto idxn = neighbor.l(i) >> level_diff_1;
    if (std::abs(idxn - idxt) <= 1) offsets[i].push_back(idxn - idxt);

    int n_blocks_level = std::max(n_per_root_block * rg_info.n[i], 1);
    if (root_block_per_n > 1)
      n_blocks_level =
          rg_info.n[i] / root_block_per_n + (rg_info.n[i] % root_block_per_n != 0);
    if (rg_info.periodic[i]) {
      if (std::abs(idxn - n_blocks_level - idxt) <= 1)
        offsets[i].push_back(idxn - n_blocks_level - idxt);
      if (std::abs(idxn + n_blocks_level - idxt) <= 1)
        offsets[i].push_back(idxn + n_blocks_level - idxt);
    }
  }

  return offsets;
}

bool LogicalLocation::IsNeighborForest(const LogicalLocation &in) const {
  PARTHENON_REQUIRE(tree() == in.tree(),
                    "Trying to compare locations not in the same octree.");
  const int max_level = std::max(in.level(), level());
  const int level_shift_in = max_level - in.level();
  const int level_shift_this = max_level - level();
  const auto block_size_in = 1 << level_shift_in;
  const auto block_size_this = 1 << level_shift_this;

  bool neighbors = true;
  for (int dir = 0; dir < 3; ++dir) {
    auto low = (l(dir) << level_shift_this) - 1;
    auto hi = low + block_size_this + 1;

    auto low_in = (in.l(dir) << level_shift_in);
    auto hi_in = low_in + block_size_in - 1;
    neighbors = neighbors && !(hi < low_in || low > hi_in);
  }
  return neighbors;
}

bool LogicalLocation::IsNeighborOfTEForest(const LogicalLocation &in,
                                           const std::array<int, 3> &te_offset) const {
  PARTHENON_REQUIRE(tree() == in.tree(),
                    "Trying to compare locations not in the same octree.");
  const int max_level = std::max(in.level(), level());
  const int level_shift_in = max_level - in.level();
  const int level_shift_this = max_level - level();
  const auto block_size_in = 1 << level_shift_in;
  const auto block_size_this = 1 << level_shift_this;

  bool neighbors = true;
  for (int dir = 0; dir < 3; ++dir) {
    auto low = (l(dir) << level_shift_this);
    auto hi = low + block_size_this - 1;
    if (te_offset[dir] == -1) {
      low -= 1;
      hi = low + 1;
    } else if (te_offset[dir] == 1) {
      hi += 1;
      low = hi - 1;
    }

    auto low_in = (in.l(dir) << level_shift_in);
    auto hi_in = low_in + block_size_in - 1;
    neighbors = neighbors && !(hi < low_in || low > hi_in);
  }
  return neighbors;
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
  const int n_per_root_block = 1 << std::max(max_level - rg_info.level, 0);
  const int root_block_per_n = 1 << std::max(rg_info.level - max_level, 0);
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
      int n_cells_level = std::max(n_per_root_block * rg_info.n[i], 1);
      if (root_block_per_n > 1)
        n_cells_level =
            rg_info.n[i] / root_block_per_n + (rg_info.n[i] % root_block_per_n != 0);
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

std::vector<LogicalLocation> LogicalLocation::GetDaughters(int ndim) const {
  std::vector<LogicalLocation> daughters;
  if (level() < 0) {
    daughters.push_back(GetDaughter(0, 0, 0));
    return daughters;
  }

  const std::vector<int> active{0, 1};
  const std::vector<int> inactive{0};
  daughters.reserve(1LL << ndim);
  for (int i : active) {
    for (int j : ndim > 1 ? active : inactive) {
      for (int k : ndim > 2 ? active : inactive) {
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
    const int down_shift = 1 << std::max(rg_info.level - loc.level(), 0);
    // Account for the fact that the root grid may be overhanging into a partial block
    const int extra1 = (rg_info.n[0] % down_shift > 0);
    const int extra2 = (rg_info.n[1] % down_shift > 0);
    const int extra3 = (rg_info.n[2] % down_shift > 0);
    int n1_cells_level =
        std::max(n_per_root_block * (rg_info.n[0] / down_shift + extra1), 1);
    int n2_cells_level =
        std::max(n_per_root_block * (rg_info.n[1] / down_shift + extra2), 1);
    int n3_cells_level =
        std::max(n_per_root_block * (rg_info.n[2] / down_shift + extra3), 1);
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
              if ((lx1 >> s) != this->lx1() || (lx2 >> s) != this->lx2() ||
                  (lx3 >> s) != this->lx3()) {
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

} // namespace parthenon
