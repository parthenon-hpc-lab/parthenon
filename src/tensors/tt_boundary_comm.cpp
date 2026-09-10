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

#include "tensors/tt_boundary_comm.hpp"

#include <memory>
#include <vector>

#include "bvals/comms/bvals_utils.hpp"
#include "bvals/comms/calc_indices.hpp"
#include "bvals/neighbor_block.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "tensors/tt_boundary_cache.hpp"
#include "tensors/tt_comm_channel.hpp"
#include "utils/error_checking.hpp"
#include "utils/loop_utils.hpp"

namespace parthenon {

namespace {

bool IsIdentityTransform(const forest::LogicalCoordinateTransformation &t) {
  const forest::LogicalCoordinateTransformation id;
  return t.dir_connection == id.dir_connection && t.dir_flip == id.dir_flip;
}

} // namespace

void BuildTTBoundaryCache(std::shared_ptr<MeshTTData> &md, TTBoundaryCache *cache) {
  using namespace loops;
  Mesh *pmesh = md->GetMeshPointer();
  const bool ml = pmesh->multilevel;
  const int bound_buffer_id = 0; // single TT comm channel set for now

  // First pass: count boundaries so the flat device array can be sized once.
  int nbound = 0;
  ForEachBoundary<BoundaryType::any>(
      md, [&](auto /*pmb*/, auto /*rc*/, const NeighborBlock & /*nb*/, auto /*v*/) {
        ++nbound;
      });

  cache->clear();
  cache->bnd_info = TTBndInfoArr_t(ViewOfViewAlloc("tt_bnd_info"), nbound);
  cache->bnd_info_h = create_view_of_view_mirror(cache->bnd_info);
  cache->channels.reserve(nbound);

  // Whole-block (entire, incl. ghosts) spatial extents for flattening (k, j, i) to the
  // spatial-core index. Mesh-wide (every block shares the same cell shape), so derived
  // once from any block in the partition.
  if (md->NumBlocks() > 0) {
    const auto shapes =
        CalcIndexShapes(BlockInfo(md->GetBlockData(0)->GetBlockPointer()).block_size, ml);
    const IndexShape &cb = shapes[0];
    cache->ni = cb.ncellsi(IndexDomain::entire);
    cache->nj = cb.ncellsj(IndexDomain::entire);
  }

  int ibound = 0;
  ForEachBoundary<BoundaryType::any>(
      md, [&](auto pmb, auto /*rc*/, const NeighborBlock &nb, auto v) {
        PARTHENON_REQUIRE(
            IsIdentityTransform(nb.lcoord_trans),
            "TT boundary comm currently supports identity-transform boundaries only.");

        BlockInfo binfo(pmb);
        auto [other, rev] = ReverseNeighbor(binfo, nb);

        TTBndInfo info;
        // Sender's interior cells destined for the neighbor, and the neighbor's ghost
        // cells that receive them -- both on the whole-block index space.
        info.send = CalcIndices(nb, binfo, ml, v, TopologicalElement::CC,
                                IndexRangeType::BoundaryInteriorSend, false);
        info.recv = CalcIndices(rev, other, ml, v, TopologicalElement::CC,
                                IndexRangeType::BoundaryExteriorRecv, false);
        info.lcoord_trans = nb.lcoord_trans;

        // Ensure a channel exists for this boundary and record it by index (pointer stable
        // within an epoch, like the regular buf_vec).
        auto key = SendKey(pmb, nb, v, BoundaryType::any, bound_buffer_id);
        info.channel_idx = static_cast<int>(cache->channels.size());
        cache->channels.push_back(&pmesh->tt_comm_map[key]);

        cache->bnd_info_h(ibound) = info;
        ++ibound;
      });

  Kokkos::deep_copy(cache->bnd_info, cache->bnd_info_h);
  cache->epoch = pmesh->tt_comm_map.GetCurrentEpoch();
}

} // namespace parthenon
