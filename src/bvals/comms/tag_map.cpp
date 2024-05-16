//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2022 The Parthenon collaboration
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

#include "tag_map.hpp"
#include "bnd_info.hpp"
#include "bvals_utils.hpp"
#include "utils/loop_utils.hpp"

namespace parthenon {

using namespace loops;
using namespace loops::shorthands;

TagMap::rank_pair_t TagMap::MakeChannelPair(const MeshBlock *pmb,
                                            const NeighborBlock &nb) {
  const int location_idx_me = nb.offsets.GetIdx();
  const int location_idx_nb = nb.offsets.GetReverseIdx();
  BlockGeometricElementId bgei_me{pmb->gid, location_idx_me};
  BlockGeometricElementId bgei_nb{nb.gid, location_idx_nb};
  return UnorderedPair<BlockGeometricElementId>(bgei_me, bgei_nb);
}
template <BoundaryType BOUND>
void TagMap::AddMeshDataToMap(std::shared_ptr<MeshData<Real>> &md) {
  for (int block = 0; block < md->NumBlocks(); ++block) {
    auto &rc = md->GetBlockData(block);
    auto pmb = rc->GetBlockPointer();
    // type_t var = []{...}() pattern defines and uses a lambda that
    // returns  to reduce initializations of var
    auto *neighbors = [&pmb, &md] {
      if constexpr (BOUND == BoundaryType::gmg_restrict_send)
        return &(pmb->gmg_coarser_neighbors);
      if constexpr (BOUND == BoundaryType::gmg_restrict_recv)
        return &(pmb->gmg_finer_neighbors);
      if constexpr (BOUND == BoundaryType::gmg_prolongate_send)
        return &(pmb->gmg_finer_neighbors);
      if constexpr (BOUND == BoundaryType::gmg_prolongate_recv)
        return &(pmb->gmg_coarser_neighbors);
      if constexpr (BOUND == BoundaryType::gmg_prolongate_recv)
        return &(pmb->gmg_coarser_neighbors);
      if constexpr (BOUND == BoundaryType::gmg_same)
        return pmb->loc.level() == md->grid.logical_level
                   ? &(pmb->gmg_same_neighbors)
                   : &(pmb->gmg_composite_finer_neighbors);
      return &(pmb->neighbors);
    }();
    for (auto &nb : *neighbors) {
      const int other_rank = nb.rank;
      if (map_.count(other_rank) < 1) map_[other_rank] = rank_pair_map_t();
      auto &pair_map = map_[other_rank];
      // Add channel key with an invalid tag
      pair_map[MakeChannelPair(pmb, nb)] = -1;
      if (swarm_map_.count(other_rank) < 1) swarm_map_[other_rank] = rank_pair_map_t();
      auto &swarm_pair_map = swarm_map_[other_rank];
      // Add channel key with an invalid tag
      swarm_pair_map[MakeChannelPair(pmb, nb)] = -1;
    }
  }
}
template void
TagMap::AddMeshDataToMap<BoundaryType::any>(std::shared_ptr<MeshData<Real>> &md);
template void
TagMap::AddMeshDataToMap<BoundaryType::gmg_same>(std::shared_ptr<MeshData<Real>> &md);
template void TagMap::AddMeshDataToMap<BoundaryType::gmg_prolongate_send>(
    std::shared_ptr<MeshData<Real>> &md);
template void TagMap::AddMeshDataToMap<BoundaryType::gmg_restrict_send>(
    std::shared_ptr<MeshData<Real>> &md);
template void TagMap::AddMeshDataToMap<BoundaryType::gmg_prolongate_recv>(
    std::shared_ptr<MeshData<Real>> &md);
template void TagMap::AddMeshDataToMap<BoundaryType::gmg_restrict_recv>(
    std::shared_ptr<MeshData<Real>> &md);

void TagMap::ResolveMap() {
#ifdef MPI_PARALLEL
  int flag;
  void *max_tag; // largest supported MPI tag value
  PARTHENON_MPI_CHECK(MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &max_tag, &flag));
  if (!flag) {
    PARTHENON_FAIL("MPI error, cannot query largest supported MPI tag value.");
  }
#endif
  auto it = map_.begin();
  auto swarm_it = swarm_map_.begin();
  while (it != map_.end() && swarm_it != swarm_map_.end()) {
    auto &pair_map = it->second;
    int idx = 0;
    std::for_each(pair_map.begin(), pair_map.end(),
                  [&idx](auto &pair) { pair.second = idx++; });
    auto &swarm_pair_map = swarm_it->second;
    std::for_each(swarm_pair_map.begin(), swarm_pair_map.end(),
                  [&idx](auto &swarm_pair) { swarm_pair.second = idx++; });
#ifdef MPI_PARALLEL
    if (idx > (*reinterpret_cast<int *>(max_tag)) && it->first != Globals::my_rank)
      PARTHENON_FAIL("Number of tags exceeds the maximum allowed by this MPI version.");
#endif
    ++it;
    ++swarm_it;
  }
}

int TagMap::GetTag(const MeshBlock *pmb, const NeighborBlock &nb) {
  const int other_rank = nb.rank;
  auto &pair_map = map_[other_rank];
  auto cpair = MakeChannelPair(pmb, nb);
  PARTHENON_REQUIRE(pair_map.count(cpair) == 1,
                    "Trying to get tag for key that hasn't been entered.\n");
  return pair_map[cpair];
}

int TagMap::GetSwarmTag(const MeshBlock *pmb, const NeighborBlock &nb) {
  const int other_rank = nb.rank;
  auto &swarm_pair_map = swarm_map_[other_rank];
  auto cpair = MakeChannelPair(pmb, nb);
  PARTHENON_REQUIRE(swarm_pair_map.count(cpair) == 1,
                    "Trying to get tag for key that hasn't been entered.\n");
  return swarm_pair_map[cpair];
}

} // namespace parthenon
