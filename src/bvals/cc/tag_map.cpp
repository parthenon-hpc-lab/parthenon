//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
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
#include "bvals_utils.hpp"

namespace parthenon {

using namespace cell_centered_bvars::impl;

TagMap::rank_pair_t TagMap::MakeChannelPair(const std::shared_ptr<MeshBlock> &pmb,
                                            const NeighborBlock &nb) {
  const int location_idx_me = (1 + nb.ni.ox1) + 3 * (1 + nb.ni.ox2 + 3 * (1 + nb.ni.ox3));
  const int location_idx_nb = (1 - nb.ni.ox1) + 3 * (1 - nb.ni.ox2 + 3 * (1 - nb.ni.ox3));
  BlockGeometricElementId bgei_me{pmb->gid, location_idx_me};
  BlockGeometricElementId bgei_nb{nb.snb.gid, location_idx_nb};
  return UnorderedPair<BlockGeometricElementId>(bgei_me, bgei_nb);
}

void TagMap::AddMeshDataToMap(std::shared_ptr<MeshData<Real>> &md) {
  ForEachBoundary(md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb, const sp_cv_t v) {
    const int other_rank = nb.snb.rank;
    if (map_.count(other_rank) < 1) map_[other_rank] = rank_pair_map_t();
    auto &pair_map = map_[other_rank];

    pair_map[MakeChannelPair(pmb, nb)] = 0;
  });
}

void TagMap::ResolveMap() {
  for (auto it = map_.begin(); it != map_.end(); ++it) {
    auto &pair_map = it->second;
    int idx = 0;
    std::for_each(pair_map.begin(), pair_map.end(),
                  [&idx](auto &pair) { pair.second = idx++; });
  }
}

int TagMap::GetTag(const std::shared_ptr<MeshBlock> &pmb, const NeighborBlock &nb) {
  const int other_rank = nb.snb.rank;
  auto &pair_map = map_[other_rank];
  return pair_map[MakeChannelPair(pmb, nb)];
}

} // namespace parthenon
