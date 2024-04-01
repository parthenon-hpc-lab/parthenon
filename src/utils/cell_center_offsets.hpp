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
#ifndef UTILS_CELL_CENTER_OFFSETS_HPP_
#define UTILS_CELL_CENTER_OFFSETS_HPP_

#include <array>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "basic_types.hpp"
#include "defs.hpp"
#include "mesh/forest/logical_location.hpp"
#include "utils/bit_hacks.hpp"
#include "utils/indexer.hpp"

namespace parthenon {
enum class Direction : uint { I = 0, J = 1, K = 2 };

// CellCentOffsets defines the position of a topological element
// within a cell or a neighboring cell via offsets from the cell center. The center of
// cell is defined by zero offsets in each direction. The faces have
// a Offset::Low or Offset::Up in one direction and Offset::Middle
// in the others, etc. The topological position of an element in the
// is turned into an unsigned index 0...27 via
// (x_offset + 1) + 3 * (y_offset + 1) + 9 * (z_offset + 1)

// TODO(LFR): Consider switching this to C-style enum within a namespace to avoid
// static_cast
enum class Offset : int { Low = -1, Middle = 0, Up = 1 };
inline int operator+(Offset a, int b) { return static_cast<int>(a) + b; }
inline int operator+(int b, Offset a) { return static_cast<int>(a) + b; }

struct CellCentOffsets {
  std::array<Offset, 3> u;

  explicit CellCentOffsets(const std::array<int, 3> &in)
      : u{static_cast<Offset>(in[0]), static_cast<Offset>(in[1]),
          static_cast<Offset>(in[2])} {}

  CellCentOffsets(int ox1, int ox2, int ox3)
      : u{static_cast<Offset>(ox1), static_cast<Offset>(ox2), static_cast<Offset>(ox3)} {}

  Offset &operator[](int idx) { return u[idx]; }

  operator std::array<int, 3>() const {
    return {static_cast<int>(u[0]), static_cast<int>(u[1]), static_cast<int>(u[2])};
  }

  BoundaryFace Face() const {
    if (!IsFace()) return BoundaryFace::undef;
    for (int dir = 0; dir < 3; ++dir) {
      if (static_cast<int>(u[dir]))
        return static_cast<BoundaryFace>((1 + static_cast<int>(u[dir])) / 2 + 2 * dir);
    }
    PARTHENON_FAIL("Shouldn't get here.");
    return BoundaryFace::undef;
  }

  // Get the logical directions that are tangent to this element
  // (in cyclic order, XY, YZ, ZX, XYZ)
  std::vector<Direction> GetTangentDirections() const {
    std::vector<Direction> dirs;
    Direction missed;
    for (auto dir : {Direction::I, Direction::J, Direction::K}) {
      uint dir_idx = static_cast<uint>(dir);
      if (!static_cast<int>(
              u[dir_idx])) { // This direction has no offset, so must be tangent direction
        dirs.push_back(dir);
      } else {
        missed = dir;
      }
    }
    if (dirs.size() == 2 && missed == Direction::J) {
      dirs = {Direction::K, Direction::I}; // Make sure we are in cyclic order
    }
    return dirs;
  }

  // Get the logical directions that are normal to this element
  // (in cyclic order, XY, YZ, ZX, XYZ) along with the offset of the
  // element in that direction from the cell center.
  std::vector<std::pair<Direction, Offset>> GetNormals() const {
    std::vector<std::pair<Direction, Offset>> dirs;
    Direction missed;
    for (auto dir : {Direction::I, Direction::J, Direction::K}) {
      uint dir_idx = static_cast<uint>(dir);
      if (static_cast<int>(u[dir_idx])) {
        dirs.push_back({dir, u[dir_idx]});
      } else {
        missed = dir;
      }
    }
    if (dirs.size() == 2 && missed == Direction::J) {
      dirs = {dirs[1], dirs[0]}; // Make sure we are in cyclic order
    }
    return dirs;
  }

  bool IsNode() const {
    return 3 == abs(static_cast<int>(u[0])) + abs(static_cast<int>(u[1])) +
                    abs(static_cast<int>(u[2]));
  }

  bool IsEdge() const {
    return 2 == abs(static_cast<int>(u[0])) + abs(static_cast<int>(u[1])) +
                    abs(static_cast<int>(u[2]));
  }

  bool IsFace() const {
    return 1 == abs(static_cast<int>(u[0])) + abs(static_cast<int>(u[1])) +
                    abs(static_cast<int>(u[2]));
  }

  bool IsCell() const {
    return 0 == abs(static_cast<int>(u[0])) + abs(static_cast<int>(u[1])) +
                    abs(static_cast<int>(u[2]));
  }

  int GetIdx() const {
    return (static_cast<int>(u[0]) + 1) + 3 * (static_cast<int>(u[1]) + 1) +
           9 * (static_cast<int>(u[2]) + 1);
  }
};
} // namespace parthenon

#endif // UTILS_CELL_CENTER_OFFSETS_HPP_
