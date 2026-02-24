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
#ifndef MESH_MESH_NEIGHBORS_HPP_
#define MESH_MESH_NEIGHBORS_HPP_

#include <memory>
#include <unordered_set>
#include <vector>

#include "basic_types.hpp"
#include "mesh/forest/logical_location.hpp"

namespace parthenon {

// Forward declarations
class Mesh;
class MeshBlock;

// Define BlockList_t to avoid circular dependencies
using BlockList_t = std::vector<std::shared_ptr<MeshBlock>>;

// Sets the neighbors for a list of MeshBlocks
// If newly_refined is empty, ownership of shared elements will be determined
// purely by block gid
void SetMeshBlockNeighbors(Mesh* pmesh, GridIdentifier grid_id, BlockList_t &block_list,
                      const std::vector<int> &ranklist,
                      const std::unordered_set<LogicalLocation> &newly_refined = {});

} // namespace parthenon

#endif // MESH_MESH_NEIGHBORS_HPP_