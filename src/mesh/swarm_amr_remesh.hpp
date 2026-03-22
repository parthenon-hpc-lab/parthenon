//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2026 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
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

// This file was made in part with generative AI

#ifndef MESH_SWARM_AMR_REMESH_HPP_
#define MESH_SWARM_AMR_REMESH_HPP_

#include <memory>
#include <vector>

#include "mesh/mesh.hpp"

namespace parthenon {

// This bundles the AMR/load-balance bookkeeping needed to map swarm ownership from the
// old leaf mesh onto the new one. Keeping the remesh helper interface in terms of one
// purpose-built context is easier to read at the call site than threading several mesh
// internals through the function signature directly.
struct SwarmRemeshContext {
  int old_start_gid;
  int old_end_gid;
  const std::vector<int> &old_to_new_gid;
  const std::vector<LogicalLocation> &old_locs;
  const std::vector<LogicalLocation> &new_locs;
  const std::vector<int> &old_ranks;
  const std::vector<int> &new_ranks;
};

// Remap all swarm records from the pre-remesh leaf mesh onto the post-remesh leaf mesh.
// The implementation preserves complete particle records and chooses the destination leaf
// by geometric ownership in physical space.
void RemeshSwarms(const std::shared_ptr<StateDescriptor> &resolved_packages,
                  const BlockList_t &old_block_list, Mesh *pmesh,
                  const SwarmRemeshContext &context);

// AMR remeshing can leave surviving MeshBlockData stages with stale cached swarm pack
// views even when the owning particle population has changed underneath them. Clear both
// MeshBlockData- and MeshData-level caches so the next access rebuilds from the
// post-remesh swarm layout.
void ClearSwarmCachesAfterRemesh(Mesh *pmesh, const BlockList_t &block_list);

} // namespace parthenon

#endif // MESH_SWARM_AMR_REMESH_HPP_
