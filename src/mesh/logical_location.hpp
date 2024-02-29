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
#ifndef MESH_LOGICAL_LOCATION_HPP_
#define MESH_LOGICAL_LOCATION_HPP_

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

#include "basic_types.hpp"
#include "utils/error_checking.hpp"
#include "utils/morton_number.hpp"

namespace parthenon {
class LogicalLocation;
}

// This must be declared before an unordered_set of LogicalLocation is used
// below, but must be *implemented* after the class definition
template <>
struct std::hash<parthenon::LogicalLocation> {
  std::size_t operator()(const parthenon::LogicalLocation &key) const noexcept;
};

namespace parthenon {

struct RootGridInfo {
  int level;
  std::array<int, 3> n;
  std::array<bool, 3> periodic;
  // Defaults to root grid of single block at the
  // coarsest level
  RootGridInfo() : level(0), n{1, 1, 1}, periodic{false, false, false} {}
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
  std::int64_t tree_idx_;
  MortonNumber morton_;
  int level_;

 public:
  // No check is provided that the requested LogicalLocation is in the allowed
  // range of logical location in the requested level.
  LogicalLocation(int lev, std::int64_t l1, std::int64_t l2, std::int64_t l3)
      : l_{l1, l2, l3}, level_{lev}, tree_idx_{-1}, morton_(lev, l1, l2, l3) {}
  LogicalLocation(std::int64_t tree, int lev, std::int64_t l1, std::int64_t l2,
                  std::int64_t l3)
      : l_{l1, l2, l3}, level_{lev}, tree_idx_{tree}, morton_(lev, l1, l2, l3) {}
  LogicalLocation() : LogicalLocation(0, 0, 0, 0) {}

  std::string label() const {
    return "([" + std::to_string(tree_idx_) + "] " + std::to_string(level_) + ": " +
           std::to_string(l_[0]) + ", " + std::to_string(l_[1]) + ", " +
           std::to_string(l_[2]) + ")";
  }
  const auto &l(int i) const { return l_[i]; }
  const auto &lx1() const { return l_[0]; }
  const auto &lx2() const { return l_[1]; }
  const auto &lx3() const { return l_[2]; }
  const auto &level() const { return level_; }
  const auto &morton() const { return morton_; }
  const auto &tree() const { return tree_idx_; }

  // Check if this logical location is actually in the domain of the tree,
  // possibly including a ghost halo around the tree
  bool IsInTree(int nghost = 0) const {
    const int low = -nghost;
    const int up = 1LL << level() + nghost;
    return (l_[0] >= low) && (l_[0] < up) && (l_[1] >= low) && (l_[1] < up) &&
           (l_[2] >= low) && (l_[2] < up);
  }

  // Check if a LL is in the ghost halo of the tree it is associated with
  bool IsInHalo(int nghost) const { return IsInTree(nghost) && !IsInTree(0); }

  int NeighborTreeIndex() const {
    int i1 = (l_[0] >= 0) - (l_[0] < (1LL << level())) + 1;
    int i2 = (l_[1] >= 0) - (l_[1] < (1LL << level())) + 1;
    int i3 = (l_[2] >= 0) - (l_[2] < (1LL << level())) + 1;
    return i1 + 3 * i2 + 9 * i3;
  }

  // Returns the coordinate in the range [0, 1] of the left side of
  // a logical location in a given direction on refinement level level
  Real LLCoord(CoordinateDirection dir, BlockLocation bloc = BlockLocation::Left) const {
    auto nblocks_tot = 1 << std::max(level(), 0);
    return (static_cast<Real>(l(dir - 1)) + 0.5 * static_cast<Real>(bloc)) /
           static_cast<Real>(nblocks_tot);
  }

  bool IsContainedIn(const LogicalLocation &container) const;

  bool Contains(const LogicalLocation &containee) const;

  std::array<int, 3> GetOffset(const LogicalLocation &neighbor,
                               const RootGridInfo &rg_info = RootGridInfo()) const;
  std::array<int, 3> GetSameLevelOffsetsForest(const LogicalLocation &neighbor) const;
  std::array<std::vector<int>, 3> GetSameLevelOffsets(const LogicalLocation &neighbor,
                                                      const RootGridInfo &rg_info) const;
  // Being a neighbor implies that you share a face, edge, or node and don't share a
  // volume
  bool IsNeighbor(const LogicalLocation &in,
                  const RootGridInfo &rg_info = RootGridInfo()) const {
    return NeighborFindingImpl<false>(in, std::array<int, 3>(), rg_info);
  }

  bool IsNeighborForest(const LogicalLocation &in) const;
  bool IsNeighborOfTEForest(const LogicalLocation &in,
                            const std::array<int, 3> &te_offset) const;

  bool IsNeighborOfTE(const LogicalLocation &in, int ox1, int ox2, int ox3,
                      const RootGridInfo &rg_info = RootGridInfo()) const {
    return NeighborFindingImpl<true>(in, std::array<int, 3>{ox1, ox2, ox3}, rg_info);
  }

  LogicalLocation
  GetSameLevelNeighbor(int ox1, int ox2, int ox3,
                       const RootGridInfo &rg_info = RootGridInfo()) const {
    return LogicalLocation(tree(), level(), lx1() + ox1, lx2() + ox2, lx3() + ox3);
  }

  LogicalLocation GetParent(int nlevel = 1) const {
    if (level() - nlevel < 0) return LogicalLocation(tree(), level() - nlevel, 0, 0, 0);
    return LogicalLocation(tree(), level() - nlevel, lx1() >> nlevel, lx2() >> nlevel,
                           lx3() >> nlevel);
  }

  std::vector<LogicalLocation> GetDaughters(int ndim = 3) const;

  LogicalLocation GetDaughter(int ox1, int ox2, int ox3) const {
    if (level() < 0) return LogicalLocation(tree(), level() + 1, 0, 0, 0);
    return LogicalLocation(tree(), level() + 1, (lx1() << 1) + ox1, (lx2() << 1) + ox2,
                           (lx3() << 1) + ox3);
  }

  // LFR: This returns the face offsets of fine-coarse neighbor blocks as defined in
  // Athena++, which are stored in the NeighborBlock struct. I believe that these are
  // currently only required for flux correction and can eventually be removed when flux
  // correction is combined with boundary communication.
  auto GetAthenaXXFaceOffsets(const LogicalLocation &neighbor, int ox1, int ox2, int ox3,
                              const RootGridInfo &rg_info = RootGridInfo()) const {
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

  std::unordered_set<LogicalLocation>
  GetPossibleNeighbors(const RootGridInfo &rg_info = RootGridInfo());

  std::unordered_set<LogicalLocation> GetPossibleBlocksSurroundingTopologicalElement(
      int ox1, int ox2, int ox3, const RootGridInfo &rg_info = RootGridInfo()) const;

 private:
  template <bool TENeighbor>
  bool NeighborFindingImpl(const LogicalLocation &in, const std::array<int, 3> &te_offset,
                           const RootGridInfo &rg_info = RootGridInfo()) const;

  std::unordered_set<LogicalLocation> GetPossibleNeighborsImpl(
      const std::vector<int> &irange, const std::vector<int> &jrange,
      const std::vector<int> &krange, const std::vector<int> &daughter_irange,
      const std::vector<int> &daughter_jrange, const std::vector<int> &daughter_krange,
      const RootGridInfo &rg_info = RootGridInfo()) const;
};

inline bool operator<(const LogicalLocation &lhs, const LogicalLocation &rhs) {
  if (lhs.tree() != rhs.tree()) return lhs.tree() < rhs.tree();
  if (lhs.morton() != rhs.morton()) return lhs.morton() < rhs.morton();
  return lhs.level() < rhs.level();
}

inline bool operator>(const LogicalLocation &lhs, const LogicalLocation &rhs) {
  if (lhs.tree() != rhs.tree()) return lhs.tree() > rhs.tree();
  if (lhs.morton() != rhs.morton()) return lhs.morton() > rhs.morton();
  return lhs.level() > rhs.level();
}

inline bool operator==(const LogicalLocation &lhs, const LogicalLocation &rhs) {
  return ((lhs.level() == rhs.level()) && (lhs.lx1() == rhs.lx1()) &&
          (lhs.lx2() == rhs.lx2()) && (lhs.lx3() == rhs.lx3()) &&
          (lhs.tree() == rhs.tree()));
}

inline bool operator!=(const LogicalLocation &lhs, const LogicalLocation &rhs) {
  return !(lhs == rhs);
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

  bool operator==(const block_ownership_t &rhs) const {
    bool same = initialized == rhs.initialized;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          same = same && (ownership[i][j][k] == rhs.ownership[i][j][k]);
        }
      }
    }
    return same;
  }

 private:
  bool ownership[3][3][3];
};

struct NeighborLocation {
  LogicalLocation global_loc; // Global location of neighboring block
  LogicalLocation
      origin_loc; // Logical location of neighboring block in index space of origin block
};

block_ownership_t
DetermineOwnership(const LogicalLocation &main_block,
                   const std::unordered_set<LogicalLocation> &allowed_neighbors,
                   const RootGridInfo &rg_info = RootGridInfo(),
                   const std::unordered_set<LogicalLocation> &newly_refined = {});

block_ownership_t
DetermineOwnershipForest(const LogicalLocation &main_block,
                         const std::vector<NeighborLocation> &allowed_neighbors,
                         const std::unordered_set<LogicalLocation> &newly_refined = {});

// Given a topological element, ownership array of the sending block, and offset indices
// defining the location of an index region within the block (i.e. the ghost zones passed
// across the x-face or the ghost zones passed across the z-edge), return the index range
// masking array required for masking out unowned regions of the index space. ox? defines
// buffer location on the owner block
block_ownership_t
GetIndexRangeMaskFromOwnership(TopologicalElement el,
                               const block_ownership_t &sender_ownership, int ox1,
                               int ox2, int ox3);

} // namespace parthenon

inline std::size_t std::hash<parthenon::LogicalLocation>::operator()(
    const parthenon::LogicalLocation &key) const noexcept {
  // TODO(LFR): Think more carefully about what the best choice for this key is,
  // probably the least significant sizeof(size_t) * 8 bits of the morton number
  // with 3 * (level - 21) trailing bits removed.
  return key.morton().bits[0];
}

#endif // MESH_LOGICAL_LOCATION_HPP_
