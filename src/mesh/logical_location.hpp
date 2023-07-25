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
#ifndef MESH_LOGICAL_LOCATION_HPP_
#define MESH_LOGICAL_LOCATION_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "utils/error_checking.hpp"
#include "utils/morton_number.hpp"

namespace parthenon {

struct RootGridInfo {
  int level;
  std::array<int, 3> n; 
  std::array<bool, 3> periodic;
  // Defaults to root grid of single block at the
  // coarsest level
  RootGridInfo()
      : level(0), n{1, 1, 1}, periodic{false, false, false} {}
  RootGridInfo(int level, int nx1, int nx2, int nx3, bool p1, bool p2, bool p3)
      : level(level), n{nx1, nx2, nx3}, periodic{p1, p2, p3} {}
};

//--------------------------------------------------------------------------------------
//! \struct LogicalLocation
//  \brief stores logical location and level of MeshBlock

class LogicalLocation { // aggregate and POD type
  // These values can exceed the range of std::int32_t even if the root grid has only a
  // single MeshBlock if >30 levels of AMR are used, since the corresponding max index =
  // 1*2^31 > INT_MAX = 2^31 -1 for most 32-bit signed integer type impelementations
  std::array<std::int64_t, 3> l_;
  MortonNumber morton_;
  int level_;

 public:
  LogicalLocation(int lev, std::int64_t l1, std::int64_t l2, std::int64_t l3)
      : l_{l1, l2, l3}, level_(lev), morton_(lev, l1, l2, l3) {}
  LogicalLocation() : LogicalLocation(0, 0, 0, 0) {}

  std::string label() const {
    return "(" + std::to_string(level_) + ": " + std::to_string(l_[0]) + ", " +
           std::to_string(l_[1]) + ", " + std::to_string(l_[2]) + ")";
  }
  const auto &l(int i) const {return l_[i];}
  const auto &lx1() const { return l_[0]; }
  const auto &lx2() const { return l_[1]; }
  const auto &lx3() const { return l_[2]; }
  const auto &level() const { return level_; }
  const auto &morton() const { return morton_; }

  bool IsContainedIn(const LogicalLocation &container) const {
    if (container.level() > level()) return false;
    bool is_contained = true; 
    const int level_shift = level() - container.level();
    for (int i = 0; i < 3; ++i) 
        is_contained = is_contained && (l(i) >> level_shift == container.l(i));
    return is_contained;
  }

  bool Contains(const LogicalLocation &containee) const {
    if (containee.level() < level_) return false;
    const std::int64_t shifted_lx1 = containee.lx1() >> (containee.level() - level());
    const std::int64_t shifted_lx2 = containee.lx2() >> (containee.level() - level());
    const std::int64_t shifted_lx3 = containee.lx3() >> (containee.level() - level());
    return (shifted_lx1 == lx1()) && (shifted_lx2 == lx2()) && (shifted_lx3 == lx3());
  }

  std::array<int, 3> GetOffset(const LogicalLocation &neighbor,
                               const RootGridInfo &rg_info = RootGridInfo()) const {
    std::array<int, 3> offset;
    const int level_diff_1 = std::max(neighbor.level() - level(), 0);
    const int level_diff_2 = std::max(level() - neighbor.level(), 0);
    const int n_per_root_block = 1
                                 << (std::min(level(), neighbor.level()) - rg_info.level); 
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

  // Being a neighbor implies that you share a face, edge, or node and don't share a
  // volume
  bool IsNeighbor(const LogicalLocation &in,
                  const RootGridInfo &rg_info = RootGridInfo()) const {
    if (in.level() >= level() && Contains(in)) return false; // You share a volume
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
    
    for (int i=0; i<3; ++i) {
      // Index range of daughters of this block on current level plus a one block halo on either side
      const auto low = (l(i) << level_shift_1) - 1; 
      const auto hi = (l(i) << level_shift_1) + block_size_1; 
      // Index range of daughters of possible neighbor block on current level
      const auto in_low = in.l(i) << level_shift_2; 
      const auto in_hi = (in.l(i) << level_shift_2) + block_size_2 - 1; 
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

  LogicalLocation
  GetSameLevelNeighbor(int ox1, int ox2, int ox3,
                       const RootGridInfo &rg_info = RootGridInfo()) const {
    return LogicalLocation(level(), lx1() + ox1, lx2() + ox2, lx3() + ox3);
  }

  LogicalLocation GetParent() const {
    if (level_ == 0) return *this;
    return LogicalLocation(level() - 1, lx1() >> 1, lx2() >> 1, lx3() >> 1);
  }

  std::vector<LogicalLocation> GetDaughters() const {
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

  LogicalLocation GetDaughter(int ox1, int ox2, int ox3) const {
    return LogicalLocation(level_ + 1, (lx1() << 1) + ox1, (lx2() << 1) + ox2,
                           (lx3() << 1) + ox3);
  }

  auto GetAthenaXXOffsets(const LogicalLocation &neighbor,
                          const RootGridInfo &rg_info = RootGridInfo()) {
    auto offsets = GetOffset(neighbor, rg_info);
    // The neighbor block struct should only use the first two, but we have three to allow
    // for this being a parent of neighbor, this should be checked for elsewhere
    std::array<int, 3> f{0, 0, 0};
    if (neighbor.level() == level() + 1) {
      int idx = 0;
      if (offsets[0] == 0) f[idx++] = neighbor.lx1() % 2;
      if (offsets[1] == 0) f[idx++] = neighbor.lx2() % 2;
      if (offsets[2] == 0) f[idx++] = neighbor.lx3() % 2;
    }
    return std::make_tuple(offsets, f);
  }

  std::set<LogicalLocation>
  GetPossibleNeighbors(const RootGridInfo &rg_info = RootGridInfo()) {
    const std::vector<int> irange{-1, 0, 1};
    const std::vector<int> jrange{-1, 0, 1};
    const std::vector<int> krange{-1, 0, 1};
    const std::vector<int> daughter_irange{0, 1};
    const std::vector<int> daughter_jrange{0, 1};
    const std::vector<int> daughter_krange{0, 1};

    return GetPossibleNeighborsImpl(irange, jrange, krange, daughter_irange,
                                    daughter_jrange, daughter_krange, rg_info);
  }

  std::set<LogicalLocation> GetPossibleBlocksSurroundingTopologicalElement(
      int ox1, int ox2, int ox3, const RootGridInfo &rg_info = RootGridInfo()) const {
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

  std::set<LogicalLocation> GetPossibleNeighborsImpl(
      const std::vector<int> &irange, const std::vector<int> &jrange,
      const std::vector<int> &krange, const std::vector<int> &daughter_irange,
      const std::vector<int> &daughter_jrange, const std::vector<int> &daughter_krange,
      const RootGridInfo &rg_info = RootGridInfo()) const {
    std::vector<LogicalLocation> locs;

    auto AddNeighbors = [&](const LogicalLocation &loc) {
      const int n_per_root_block = 1 << (loc.level() - rg_info.level);
      int n1_cells_level = std::max(n_per_root_block * rg_info.n[0], 1);
      int n2_cells_level = std::max(n_per_root_block * rg_info.n[1], 1);
      int n3_cells_level = std::max(n_per_root_block * rg_info.n[2], 1);
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
              locs.emplace_back(loc.level(), lx1, lx2, lx3);
              auto parent = locs.back().GetParent();
              if (IsNeighbor(parent, rg_info)) locs.push_back(parent);
            }
          }
        }
      }
    };

    // Find the same level and lower level blocks of this block
    AddNeighbors(*this);

    // Iterate over daughters of this block that share the same topological element
    for (int l : daughter_irange) {
      for (int m : daughter_jrange) {
        for (int n : daughter_krange) {
          AddNeighbors(GetDaughter(l, m, n));
        }
      }
    }
    // The above procedure likely duplicated some blocks, so put them in a set
    return std::set<LogicalLocation>(std::begin(locs), std::end(locs));
  }
};

inline bool operator<(const LogicalLocation &lhs, const LogicalLocation &rhs) {
  if (lhs.morton() == rhs.morton()) return lhs.level() < rhs.level();
  return lhs.morton() < rhs.morton();
}

inline bool operator>(const LogicalLocation &lhs, const LogicalLocation &rhs) {
  if (lhs.morton() == rhs.morton()) return lhs.level() > rhs.level();
  return lhs.morton() > rhs.morton();
}

inline bool operator==(const LogicalLocation &lhs, const LogicalLocation &rhs) {
  return ((lhs.level() == rhs.level()) && (lhs.lx1() == rhs.lx1()) &&
          (lhs.lx2() == rhs.lx2()) && (lhs.lx3() == rhs.lx3()));
}

struct block_ownership_t {
 public:
  KOKKOS_FORCEINLINE_FUNCTION
  const bool &operator()(int ox1, int ox2, int ox3) const {
    return ownership[ox1 + 1][ox2 + 1][ox3 + 1];
  }
  KOKKOS_FORCEINLINE_FUNCTION
  bool &operator()(int ox1, int ox2, int ox3) {
    return ownership[ox1 + 1][ox2 + 1][ox3 + 1];
  }

  KOKKOS_FORCEINLINE_FUNCTION
  block_ownership_t() : block_ownership_t(false) {}

  KOKKOS_FORCEINLINE_FUNCTION
  explicit block_ownership_t(bool value) : initialized(false) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          ownership[i][j][k] = value;
        }
      }
    }
  }

  bool initialized;

 private:
  bool ownership[3][3][3];
};

inline block_ownership_t
DetermineOwnership(const LogicalLocation &main_block,
                   const std::set<LogicalLocation> &allowed_neighbors,
                   const RootGridInfo &rg_info = RootGridInfo()) {
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
        auto possible_neighbors =
            main_block.GetPossibleBlocksSurroundingTopologicalElement(ox1, ox2, ox3,
                                                                      rg_info);

        std::vector<LogicalLocation> actual_neighbors;
        std::set_intersection(std::begin(allowed_neighbors), std::end(allowed_neighbors),
                              std::begin(possible_neighbors),
                              std::end(possible_neighbors),
                              std::back_inserter(actual_neighbors));

        if (actual_neighbors.size() == 0) {
          main_owns(ox1, ox2, ox3) = true;
        } else {
          auto max = std::max_element(std::begin(actual_neighbors),
                                      std::end(actual_neighbors), ownership_less_than);
          main_owns(ox1, ox2, ox3) =
              *max == main_block || ownership_less_than(*max, main_block);
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
inline auto GetIndexRangeMaskFromOwnership(TopologicalElement el,
                                           const block_ownership_t &sender_ownership,
                                           int ox1, int ox2, int ox3) {
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

template <>
struct std::hash<parthenon::LogicalLocation> {
  std::size_t operator()(const parthenon::LogicalLocation &key) const noexcept {
    // TODO(LFR): Think more carefully about what the best choice for this key is,
    // probably the least significant sizeof(size_t) * 8 bits of the morton number
    // with 3 * (level - 21) trailing bits removed.
    return key.morton().bits[0];
  }
};

#endif // MESH_LOGICAL_LOCATION_HPP_
