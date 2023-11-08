//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2022 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2023. Triad National Security, LLC. All rights reserved.
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
  const int location_idx_me = (1 + nb.ni.ox1) + 3 * (1 + nb.ni.ox2 + 3 * (1 + nb.ni.ox3));
  const int location_idx_nb = (1 - nb.ni.ox1) + 3 * (1 - nb.ni.ox2 + 3 * (1 - nb.ni.ox3));
  BlockGeometricElementId bgei_me{pmb->gid, location_idx_me};
  BlockGeometricElementId bgei_nb{nb.snb.gid, location_idx_nb};
  return UnorderedPair<BlockGeometricElementId>(bgei_me, bgei_nb);
}
template <BoundaryType BOUND>
void TagMap::AddMeshDataToMap(std::shared_ptr<MeshData<Real>> &md) {
  ForEachBoundary<BOUND>(md, [&](auto pmb, sp_mbd_t rc, nb_t &nb, const sp_cv_t v) {
    const int other_rank = nb.snb.rank;
    if (map_.count(other_rank) < 1) map_[other_rank] = rank_pair_map_t();
    auto &pair_map = map_[other_rank];
    // Add channel key with an invalid tag
    pair_map[MakeChannelPair(pmb, nb)] = -1;
  });
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
  for (auto it = map_.begin(); it != map_.end(); ++it) {
    auto &pair_map = it->second;
    int idx = 0;
    std::for_each(pair_map.begin(), pair_map.end(),
                  [&idx](auto &pair) { pair.second = idx++; });
    if (idx > 32767)
      PARTHENON_FAIL("Number of tags exceeds the maximum allowed by the MPI standard.");
  }
}

int TagMap::GetTag(const MeshBlock *pmb, const NeighborBlock &nb) {
  const int other_rank = nb.snb.rank;
  auto &pair_map = map_[other_rank];
  return pair_map[MakeChannelPair(pmb, nb)];
}

} // namespace parthenon
