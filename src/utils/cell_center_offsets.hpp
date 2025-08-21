//========================================================================================
// (C) (or copyright) 2023-2024. Triad National Security, LLC. All rights reserved.
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
#include "utils/bit_hacks.hpp"
#include "utils/indexer.hpp"

namespace parthenon {
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
inline Offset operator-(Offset in) { return static_cast<Offset>(-static_cast<int>(in)); }

struct CellCentOffsets {
  Kokkos::Array<Offset, 3> u;

  CellCentOffsets() = default;

  explicit CellCentOffsets(const Kokkos::Array<int, 3> &in)
      : u{static_cast<Offset>(in[0]), static_cast<Offset>(in[1]),
          static_cast<Offset>(in[2])} {}

  constexpr CellCentOffsets(int ox1, int ox2, int ox3)
      : u{static_cast<Offset>(ox1), static_cast<Offset>(ox2), static_cast<Offset>(ox3)} {}

  Offset &operator[](int idx) { return u[idx]; }
  const Offset &operator[](int idx) const { return u[idx]; }
  int operator()(CoordinateDirection dir) const { return static_cast<int>(u[dir - 1]); }

  operator std::array<int, 3>() const {
    return {static_cast<int>(u[0]), static_cast<int>(u[1]), static_cast<int>(u[2])};
  }

  BoundaryFace Face() const;

  // Get the logical directions that are tangent to this element
  // (in cyclic order, XY, YZ, ZX, XYZ)
  std::vector<CoordinateDirection> GetTangentDirections() const;

  // Get the logical directions that are normal to this element
  // (in cyclic order, XY, YZ, ZX, XYZ) along with the offset of the
  // element in that direction from the cell center.
  std::vector<std::pair<CoordinateDirection, Offset>> GetNormals() const;

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

  int GetReverseIdx() const {
    return (1 - static_cast<int>(u[0])) + 3 * (1 - static_cast<int>(u[1])) +
           9 * (1 - static_cast<int>(u[2]));
  }
};

template <class... Args>
CellCentOffsets AverageOffsets(Args &&...args) {
  return CellCentOffsets((static_cast<int>(args[0]) + ...) / sizeof...(args),
                         (static_cast<int>(args[1]) + ...) / sizeof...(args),
                         (static_cast<int>(args[2]) + ...) / sizeof...(args));
}
} // namespace parthenon

#endif // UTILS_CELL_CENTER_OFFSETS_HPP_
