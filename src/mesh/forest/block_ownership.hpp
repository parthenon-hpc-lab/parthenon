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
#ifndef MESH_FOREST_BLOCK_OWNERSHIP_HPP_
#define MESH_FOREST_BLOCK_OWNERSHIP_HPP_

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
#include "mesh/forest/logical_coordinate_transformation.hpp"
#include "mesh/forest/logical_location.hpp"
#include "utils/error_checking.hpp"
#include "utils/indexer.hpp"
#include "utils/morton_number.hpp"

namespace parthenon {

block_ownership_t
DetermineOwnership(const LogicalLocation &main_block,
                   const std::vector<forest::NeighborLocation> &allowed_neighbors,
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

#endif // MESH_FOREST_BLOCK_OWNERSHIP_HPP_
