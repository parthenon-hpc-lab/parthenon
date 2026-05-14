//========================================================================================
// (C) (or copyright) 2024. Triad National Security, LLC. All rights reserved.
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
#ifndef MESH_FOREST_LOGICAL_COORDINATE_TRANSFORMATION_HPP_
#define MESH_FOREST_LOGICAL_COORDINATE_TRANSFORMATION_HPP_

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
#include "utils/cell_center_offsets.hpp"
#include "utils/indexer.hpp"

namespace parthenon {
namespace forest {

struct LogicalCoordinateTransformation {
  KOKKOS_INLINE_FUNCTION
  LogicalCoordinateTransformation()
      : dir_connection{0, 1, 2}, dir_connection_inverse{0, 1, 2},
        dir_flip{false, false, false}, offset{0, 0, 0} {};

  void SetDirection(CoordinateDirection origin, CoordinateDirection neighbor,
                    bool reversed = false) {
    dir_connection[origin - 1] = neighbor - 1;
    dir_connection_inverse[neighbor - 1] = origin - 1;
    dir_flip[origin - 1] = reversed;
  }

  LogicalLocation Transform(const LogicalLocation &loc_in,
                            std::int64_t destination) const;
  LogicalLocation InverseTransform(const LogicalLocation &loc_in,
                                   std::int64_t origin) const;
  CellCentOffsets Transform(CellCentOffsets in) const;

  KOKKOS_INLINE_FUNCTION
  std::tuple<TopologicalElement, Real> Transform(TopologicalElement el) const {
    int iel = static_cast<int>(el);
    Real fac = 1.0;
    if (iel >= 3 && iel < 9) {
      int dir = iel % 3;
      iel = (iel / 3) * 3 + abs(dir_connection[dir]);
      fac = dir_flip[dir] ? -1.0 : 1.0;
    }
    return {static_cast<TopologicalElement>(iel), fac};
  }

  KOKKOS_INLINE_FUNCTION
  std::tuple<TopologicalElement, Real> InverseTransform(TopologicalElement el) const {
    int iel = static_cast<int>(el);
    Real fac = 1.0;
    if (iel >= 3 && iel < 9) {
      const int dir = iel % 3;
      const int outdir = abs(dir_connection_inverse[dir]);
      iel = (iel / 3) * 3 + outdir;
      fac = dir_flip[outdir] ? -1.0 : 1.0;
    }
    return {static_cast<TopologicalElement>(iel), fac};
  }

  KOKKOS_INLINE_FUNCTION
  std::array<int, 3> Transform(std::array<int, 3> ijk) const {
    std::array<int, 3> ijk_out;
    for (int dir = 0; dir < 3; ++dir) {
      const int outdir = abs(dir_connection[dir]);
      ijk_out[outdir] = dir_flip[dir] ? ncell - 1 - ijk[dir] : ijk[dir];
    }
    return ijk_out;
  }

  KOKKOS_INLINE_FUNCTION
  std::array<int, 3> InverseTransform(std::array<int, 3> ijk) const {
    std::array<int, 3> ijk_out;
    for (int dir = 0; dir < 3; ++dir) {
      const int indir = abs(dir_connection[dir]);
      ijk_out[dir] = dir_flip[dir] ? ncell - 1 - ijk[indir] : ijk[indir];
    }
    return ijk_out;
  }

  bool use_offset = false;
  std::array<int, 3> offset;
  std::array<int, 3> dir_connection, dir_connection_inverse;
  std::array<bool, 3> dir_flip;
  int ncell;
};

LogicalCoordinateTransformation
ComposeTransformations(const LogicalCoordinateTransformation &first,
                       const LogicalCoordinateTransformation &second);

struct NeighborLocation {
  NeighborLocation(const LogicalLocation &g, const LogicalLocation &o,
                   const LogicalCoordinateTransformation &lcoord_trans)
      : global_loc(g), origin_loc(o), lcoord_trans(lcoord_trans) {}
  LogicalLocation global_loc; // Global location of neighboring block
  LogicalLocation
      origin_loc; // Logical location of neighboring block in index space of origin block
  LogicalCoordinateTransformation lcoord_trans;
};

} // namespace forest
} // namespace parthenon

#endif // MESH_FOREST_LOGICAL_COORDINATE_TRANSFORMATION_HPP_
