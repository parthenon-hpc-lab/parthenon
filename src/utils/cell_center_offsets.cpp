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

#include <vector>

#include "utils/cell_center_offsets.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

BoundaryFace CellCentOffsets::Face() const {
  if (!IsFace()) return BoundaryFace::undef;
  for (int dir = 0; dir < 3; ++dir) {
    if (static_cast<int>(u[dir]))
      return static_cast<BoundaryFace>((1 + static_cast<int>(u[dir])) / 2 + 2 * dir);
  }
  PARTHENON_FAIL("Shouldn't get here.");
  return BoundaryFace::undef;
}

std::vector<CoordinateDirection> CellCentOffsets::GetTangentDirections() const {
  std::vector<CoordinateDirection> dirs;
  CoordinateDirection missed;
  for (auto dir : {X1DIR, X2DIR, X3DIR}) {
    uint dir_idx = static_cast<uint>(dir);
    if (!static_cast<int>(u[dir_idx - 1])) { // This direction has no offset, so must be
                                             // tangent direction
      dirs.push_back(dir);
    } else {
      missed = dir;
    }
  }
  if (dirs.size() == 2 && missed == X2DIR) {
    dirs = {X3DIR, X1DIR}; // Make sure we are in cyclic order
  }
  return dirs;
}

std::vector<std::pair<CoordinateDirection, Offset>> CellCentOffsets::GetNormals() const {
  std::vector<std::pair<CoordinateDirection, Offset>> dirs;
  CoordinateDirection missed;
  for (auto dir : {X1DIR, X2DIR, X3DIR}) {
    uint dir_idx = dir - 1;
    if (static_cast<int>(u[dir_idx])) {
      dirs.push_back({dir, u[dir_idx]});
    } else {
      missed = dir;
    }
  }
  if (dirs.size() == 2 && missed == X2DIR) {
    dirs = {dirs[1], dirs[0]}; // Make sure we are in cyclic order
  }
  return dirs;
}

} // namespace parthenon
