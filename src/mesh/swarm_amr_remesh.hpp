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

#ifndef MESH_SWARM_AMR_REMESH_HPP_
#define MESH_SWARM_AMR_REMESH_HPP_

// This file was made in part with generative AI.

#include <memory>
#include <vector>

#include "mesh/mesh.hpp"
#include "mesh/mesh_layout.hpp"

namespace parthenon {

struct SwarmRemeshContext {
  SwarmRemeshContext(int old_gid_first_in, int old_gid_last_in,
                     const MeshLayout &old_layout_in, const MeshLayout &new_layout_in)
      : old_gid_first(old_gid_first_in), old_gid_last(old_gid_last_in),
        old_layout(old_layout_in), new_layout(new_layout_in) {}

  // The new block at the same location as the old block, its parent, or (for a refined
  // block) its first daughter
  int NewGid(const int old_gid) const {
    return new_layout.GetOverlappingGids(OldLoc(old_gid))[0];
  }
  const LogicalLocation &OldLoc(const int old_gid) const {
    return old_layout.Loc(old_gid);
  }
  const LogicalLocation &NewLoc(const int new_gid) const {
    return new_layout.Loc(new_gid);
  }
  const LogicalLocation &NewLocFromOld(const int old_gid) const {
    return NewLoc(NewGid(old_gid));
  }
  int OldRank(const int old_gid) const { return old_layout.Rank(old_gid); }
  int NewRank(const int new_gid) const { return new_layout.Rank(new_gid); }
  int NewRankFromOld(const int old_gid) const { return NewRank(NewGid(old_gid)); }
  int NumOldBlocks() const { return old_layout.NumBlocks(); }
  int NumNewBlocks() const { return new_layout.NumBlocks(); }

  int old_gid_first;
  int old_gid_last;

 private:
  const MeshLayout &old_layout;
  const MeshLayout &new_layout;
};

void RemeshSwarms(const std::shared_ptr<StateDescriptor> &resolved_packages,
                  const BlockList_t &old_block_list, Mesh *pmesh,
                  const SwarmRemeshContext &context);

void ClearSwarmCachesAfterRemesh(Mesh *pmesh, const BlockList_t &block_list);

} // namespace parthenon

#endif // MESH_SWARM_AMR_REMESH_HPP_
