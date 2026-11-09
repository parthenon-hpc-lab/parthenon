//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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
// This file was made in part with generative AI.

#ifndef MESH_FOREST_FOREST_HEX_HPP_
#define MESH_FOREST_FOREST_HEX_HPP_

#include <array>
#include <cstdint>
#include <vector>

#include "basic_types.hpp"
#include "defs.hpp"
#include "mesh/forest/logical_coordinate_transformation.hpp"
#include "utils/cell_center_offsets.hpp"

namespace parthenon {
namespace forest {

// Topology of a 3D forest of hexahedral trees given by their corner node ids.  The
// corner n = i + 2 j + 4 k of a tree is at logical position (i, j, k) in {0, 1}^3, so the
// logical frame of the tree is fixed by the order of its nodes.
struct Hex {
  std::int64_t id;
  std::array<std::int64_t, 8> nodes;
};

// A boundary face of the forest, given by its four node ids (in any order)
struct HexBC {
  std::array<std::int64_t, 4> nodes;
  BoundaryFlag bflag;
};

struct HexNeighbor {
  int index; // index of the neighboring hex
  CellCentOffsets offset;
  LogicalCoordinateTransformation ct;
};

// The neighbors of every hex (indexed as hexes), with the transformations from its
// logical frame to theirs.  Hexes sharing a face (4 nodes), an edge (2) or a node (1)
// are face, edge and node neighbors; face transformations follow from the shared
// nodes, and edge and node transformations are composed through face neighbors.
std::vector<std::vector<HexNeighbor>> FindHexNeighbors(const std::vector<Hex> &hexes);

// The boundary flags of every hex: block for faces shared with another hex, else the
// flag of the matching HexBC (each unshared face must have one)
std::vector<std::array<BoundaryFlag, BOUNDARY_NFACES>>
GetHexBCs(const std::vector<Hex> &hexes,
          const std::vector<std::vector<HexNeighbor>> &neighbors,
          const std::vector<HexBC> &bcs);

} // namespace forest
} // namespace parthenon

#endif // MESH_FOREST_FOREST_HEX_HPP_
