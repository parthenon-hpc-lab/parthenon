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
#ifndef MESH_FOREST_LOGICAL_LOCATION_HPP_
#define MESH_FOREST_LOGICAL_LOCATION_HPP_

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
      : l_{l1, l2, l3}, level_{lev}, tree_idx_{-1}, morton_(std::max(lev, 0), l1, l2, l3) {}
  LogicalLocation(std::int64_t tree, int lev, std::int64_t l1, std::int64_t l2,
                  std::int64_t l3)
      : l_{l1, l2, l3}, level_{lev}, tree_idx_{tree}, morton_(std::max(lev, 0), l1, l2, l3) {}
  LogicalLocation() : LogicalLocation(0, 0, 0, 0) {}

  std::string label() const;
  const auto &l(int i) const { return l_[i]; }
  const auto &lx1() const { return l_[0]; }
  const auto &lx2() const { return l_[1]; }
  const auto &lx3() const { return l_[2]; }
  const auto &level() const { return level_; }
  const auto &morton() const { return morton_; }
  const auto &tree() const { return tree_idx_; }

  // Check if this logical location is actually in the domain of the tree,
  // possibly including a ghost block halo around the tree
  bool IsInTree(int nghost = 0) const;

  // Check if a LL is in the ghost block halo of the tree it is associated with
  bool IsInHalo(int nghost) const { return IsInTree(nghost) && !IsInTree(0); }

  int NeighborTreeIndex() const;

  // Returns the coordinate in the range [0, 1] of the left side of
  // a logical location in a given direction on refinement level level
  Real LLCoord(CoordinateDirection dir, BlockLocation bloc = BlockLocation::Left) const;

  bool IsContainedIn(const LogicalLocation &container) const;

  bool Contains(const LogicalLocation &containee) const;

  std::array<int, 3> GetSameLevelOffsets(const LogicalLocation &neighbor) const;

  // Being a neighbor implies that you share a face, edge, or node and don't share a
  // volume
  bool IsNeighbor(const LogicalLocation &in) const;
  bool IsNeighborOfTE(const LogicalLocation &in,
                      const std::array<int, 3> &te_offset) const;

  LogicalLocation GetSameLevelNeighbor(int ox1, int ox2, int ox3) const {
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
  std::array<int, 3> GetAthenaXXFaceOffsets(const LogicalLocation &neighbor, int ox1,
                                            int ox2, int ox3) const;
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

struct NeighborLocation {
  NeighborLocation(const LogicalLocation &g, const LogicalLocation &o)
      : global_loc(g), origin_loc(o) {}
  LogicalLocation global_loc; // Global location of neighboring block
  LogicalLocation
      origin_loc; // Logical location of neighboring block in index space of origin block
};

} // namespace parthenon

// Inject hash function for LogicalLocation into the std namespace
template <>
struct std::hash<parthenon::LogicalLocation> {
  std::size_t operator()(const parthenon::LogicalLocation &key) const noexcept {
    // TODO(LFR): Think more carefully about what the best choice for this key is,
    // probably the least significant sizeof(size_t) * 8 bits of the morton number
    // with 3 * (level - 21) trailing bits removed.
    return key.morton().bits[0];
  }
};

#endif // MESH_FOREST_LOGICAL_LOCATION_HPP_
