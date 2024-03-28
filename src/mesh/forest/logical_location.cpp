//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2024 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2024. Triad National Security, LLC. All rights reserved.
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
std::string LogicalLocation::label() const {
  return "([" + std::to_string(tree_idx_) + "] " + std::to_string(level_) + ": " +
         std::to_string(l_[0]) + ", " + std::to_string(l_[1]) + ", " +
         std::to_string(l_[2]) + ")";
}

bool LogicalLocation::IsInTree(int nghost) const {
  const int low = -nghost;
  const int up = (1LL << std::max(level(), 0)) + nghost;
  return (l_[0] >= low) && (l_[0] < up) && (l_[1] >= low) && (l_[1] < up) &&
         (l_[2] >= low) && (l_[2] < up);
}

int LogicalLocation::NeighborTreeIndex() const {
  auto up = 1LL << std::max(level(), 0);
  int i1 = (l_[0] >= 0) - (l_[0] < (1LL << up)) + 1;
  int i2 = (l_[1] >= 0) - (l_[1] < (1LL << up)) + 1;
  int i3 = (l_[2] >= 0) - (l_[2] < (1LL << up)) + 1;
  int idx = i1 + 3 * i2 + 9 * i3;
  PARTHENON_REQUIRE(idx >= 0 && idx < 27, "Bad index.");
  return idx;
}

Real LogicalLocation::LLCoord(CoordinateDirection dir, BlockLocation bloc) const {
  auto nblocks_tot = 1 << std::max(level(), 0);
  return (static_cast<Real>(l(dir - 1)) + 0.5 * static_cast<Real>(bloc)) /
         static_cast<Real>(nblocks_tot);
}

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
LogicalLocation::GetSameLevelOffsets(const LogicalLocation &neighbor) const {
  std::array<int, 3> offsets;
  const int level_shift_neigh = std::max(neighbor.level() - level(), 0);
  const int level_shift_me = std::max(level() - neighbor.level(), 0);
  for (int dir = 0; dir < 3; ++dir) {
    // coarsen locations to the same level
    offsets[dir] = (neighbor.l(dir) >> level_shift_neigh) - (l(dir) >> level_shift_me);
  }
  return offsets;
}

bool LogicalLocation::IsNeighbor(const LogicalLocation &in) const {
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

bool LogicalLocation::IsNeighborOfTE(const LogicalLocation &in,
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

std::array<int, 3>
LogicalLocation::GetAthenaXXFaceOffsets(const LogicalLocation &neighbor, int ox1, int ox2,
                                        int ox3) const {
  // The neighbor block struct should only use the first two, but we have three to allow
  // for this being a parent of neighbor, this should be checked for elsewhere
  std::array<int, 3> f{0, 0, 0};
  if (neighbor.level() == level() + 1) {
    int idx = 0;
    if (ox1 == 0) f[idx++] = neighbor.lx1() % 2;
    if (ox2 == 0) f[idx++] = neighbor.lx2() % 2;
    if (ox3 == 0) f[idx++] = neighbor.lx3() % 2;
  }
  return f;
}

} // namespace parthenon
