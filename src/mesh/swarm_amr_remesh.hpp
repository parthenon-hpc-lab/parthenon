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

namespace parthenon {

struct SwarmRemeshContext {
  SwarmRemeshContext(int old_gid_first_in, int old_gid_last_in,
                     const std::vector<int> &old_to_new_gid_in,
                     const std::vector<LogicalLocation> &old_locs_in,
                     const std::vector<LogicalLocation> &new_locs_in,
                     const std::vector<int> &old_ranks_in,
                     const std::vector<int> &new_ranks_in)
      : old_gid_first(old_gid_first_in), old_gid_last(old_gid_last_in),
        old_to_new_gid(old_to_new_gid_in), old_locs(old_locs_in), new_locs(new_locs_in),
        old_ranks(old_ranks_in), new_ranks(new_ranks_in) {}

  int NewGid(const int old_gid) const { return old_to_new_gid[old_gid]; }
  const LogicalLocation &OldLoc(const int old_gid) const { return old_locs[old_gid]; }
  const LogicalLocation &NewLoc(const int new_gid) const { return new_locs[new_gid]; }
  const LogicalLocation &NewLocFromOld(const int old_gid) const {
    return new_locs[NewGid(old_gid)];
  }
  int OldRank(const int old_gid) const { return old_ranks[old_gid]; }
  int NewRank(const int new_gid) const { return new_ranks[new_gid]; }
  int NewRankFromOld(const int old_gid) const { return new_ranks[NewGid(old_gid)]; }
  int NumOldBlocks() const { return old_locs.size(); }
  int NumNewBlocks() const { return new_locs.size(); }

  int old_gid_first;
  int old_gid_last;

 private:
  const std::vector<int> &old_to_new_gid;
  const std::vector<LogicalLocation> &old_locs;
  const std::vector<LogicalLocation> &new_locs;
  const std::vector<int> &old_ranks;
  const std::vector<int> &new_ranks;
};

void RemeshSwarms(const std::shared_ptr<StateDescriptor> &resolved_packages,
                  const BlockList_t &old_block_list, Mesh *pmesh,
                  const SwarmRemeshContext &context);

void ClearSwarmCachesAfterRemesh(Mesh *pmesh, const BlockList_t &block_list);

} // namespace parthenon

#endif // MESH_SWARM_AMR_REMESH_HPP_
