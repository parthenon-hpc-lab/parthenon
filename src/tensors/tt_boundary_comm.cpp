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

#include "basic_types.hpp"
#include "bvals/comms/bvals_utils.hpp"
#include "bvals/comms/calc_indices.hpp"
#include "bvals/neighbor_block.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "tensors/tt_boundary_cache.hpp"
#include "tensors/tt_comm_channel.hpp"
#include "tensors/tt_operations.hpp"
#include "tensors/tt_pack.hpp"
#include "utils/error_checking.hpp"
#include "utils/loop_utils.hpp"

namespace parthenon {

namespace {

bool IsIdentityTransform(const forest::LogicalCoordinateTransformation &t) {
  const forest::LogicalCoordinateTransformation id;
  return t.dir_connection == id.dir_connection && t.dir_flip == id.dir_flip;
}

} // namespace

namespace {

int CountBoundaries(std::shared_ptr<MeshTTData> &md) {
  int nbound = 0;
  loops::ForEachBoundary<BoundaryType::any>(
      md, [&](auto, auto, const NeighborBlock &, auto) { ++nbound; });
  return nbound;
}

} // namespace

// Build the send-side boundary cache: walk every boundary, record its send channel (keyed
// by SendKey) and the send/recv index boxes needed to build the addend, packed into a flat
// device array for one batched launch. The set side needs no cache (it looks up its
// channel inline by ReceiveKey), so only this send cache exists.
void BuildTTBoundaryCache(std::shared_ptr<MeshTTData> &md, TTBoundaryCache *cache) {
  using namespace loops;
  Mesh *pmesh = md->GetMeshPointer();
  const bool ml = pmesh->multilevel;
  const int id = 0; // single TT comm channel set for now

  const int nbound = CountBoundaries(md);
  cache->clear();
  cache->channels.reserve(nbound);
  cache->bnd_info = TTBndInfoArr_t(ViewOfViewAlloc("tt_bnd_info"), nbound);
  cache->bnd_info_h = create_view_of_view_mirror(cache->bnd_info);

  int ibound = 0;
  ForEachBoundary<BoundaryType::any>(
      md, [&](auto pmb, auto /*rc*/, const NeighborBlock &nb, auto v) {
        PARTHENON_REQUIRE(
            IsIdentityTransform(nb.lcoord_trans),
            "TT boundary comm currently supports identity-transform boundaries only.");

        cache->channels.push_back(
            &pmesh->tt_comm_map[SendKey(pmb, nb, v, BoundaryType::any, id)]);

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
        cache->bnd_info_h(ibound) = info;
        ++ibound;
      });

  Kokkos::deep_copy(cache->bnd_info, cache->bnd_info_h);
  cache->epoch = pmesh->tt_comm_map.GetCurrentEpoch();
}

std::vector<std::shared_ptr<tensor2::TensorTrain>>
BuildBoundaryTensors(std::shared_ptr<MeshTTData> &md, const TTBoundaryCache &cache) {
  using namespace loops;
  using train_t = tensor2::TensorTrain;

  std::vector<train_t *> src;
  std::vector<std::shared_ptr<train_t>> out;
  ForEachBoundary<BoundaryType::any>(
      md, [&](auto /*pmb*/, auto /*rc*/, const NeighborBlock & /*nb*/, auto v) {
        src.push_back(&v->train());
        out.push_back(std::make_shared<train_t>(v->train().DeepCopy()));
      });
  const int nbound = static_cast<int>(out.size());
  if (nbound == 0) return out;
  PARTHENON_DEBUG_REQUIRE(nbound == static_cast<int>(cache.bnd_info_h.extent(0)),
                          "Boundary walk and cache disagree on boundary count.");

  using HostPack = tensor2::TensorTrainHostPackT<DefaultTTraits>;
  auto pack_src = HostPack::FromPointers(src).MakeDevicePack();
  auto pack_out = HostPack::FromSharedPtrs(out).MakeDevicePack();
  auto bnd_info = cache.bnd_info;

  // One launch over all boundaries: zero the addend's spatial core, then place each sender
  // interior cell into the corresponding receiver ghost cell (send-cell e -> recv-cell e),
  // for every rank column of the spatial core. Mirrors MakeNeighborTensors, but the shift
  // is the cached index maps rather than a hardcoded offset.
  constexpr int unused_scratch_size = 0;
  constexpr int unused_scratch_level = 1;
  parthenon::par_for_outer(
      PARTHENON_AUTO_LABEL, unused_scratch_size, unused_scratch_level, 0, nbound - 1,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int e) {
        auto &sc_src = pack_src(e, 0, 0);
        auto &sc_out = pack_out(e, 0, 0);
        const auto &idxer = pack_out.indexer(0);
        const auto &send = bnd_info(e).send;
        const auto &recv = bnd_info(e).recv;
        const int ncell = static_cast<int>(send.size());
        for (int r = 0; r < sc_out.RR(); ++r) {
          parthenon::par_for_inner(member, 0, sc_out.DD() - 1,
                                   [&](const int idx) { sc_out(0, idx, r) = 0.0; });
          member.team_barrier();
          parthenon::par_for_inner(member, 0, ncell - 1, [&](const int c) {
            const auto [ts, us, vs, ks, js, is] = send(c);
            const auto [tr, ur, vr, kr, jr, ir] = recv(c);
            const int src_idx = idxer.GetFlatIdx(ts, us, vs, ks, js, is);
            const int dst_idx = idxer.GetFlatIdx(tr, ur, vr, kr, jr, ir);
            sc_out(0, dst_idx, r) = sc_src(0, src_idx, r);
          });
          member.team_barrier();
        }
      });
  return out;
}

void TTSend(std::shared_ptr<MeshTTData> &md, TTBoundaryCache &cache, Real eps) {
  auto addends = BuildBoundaryTensors(md, cache);
  const int nbound = static_cast<int>(addends.size());
  if (nbound == 0) return;

  // Round each addend to keep ranks bounded before shipping.
  auto pack = tensor2::TensorTrainHostPackT<DefaultTTraits>::FromSharedPtrs(addends);
  tensor2::RoundGramSVD(pack, eps);

  // Deposit each addend into its send channel.
  for (int e = 0; e < nbound; ++e)
    cache.channels[e]->Send(addends[e]);
}

bool TTReceive(std::shared_ptr<MeshTTData> &md) {
  using namespace loops;
  Mesh *pmesh = md->GetMeshPointer();
  const int id = 0;
  bool all = true;
  ForEachBoundary<BoundaryType::any>(
      md, [&](auto pmb, auto /*rc*/, const NeighborBlock &nb, auto v) {
        auto &chan = pmesh->tt_comm_map[ReceiveKey(pmb, nb, v, BoundaryType::any, id)];
        all = chan.TryReceive() && all;
      });
  return all;
}

void TTSetBounds(std::shared_ptr<MeshTTData> &md, Real eps) {
  using namespace loops;
  using train_t = tensor2::TensorTrain;
  Mesh *pmesh = md->GetMeshPointer();
  const int id = 0;

  // Sum each received addend into its destination block's train, looking the receive
  // channel up inline by ReceiveKey (no cache needed on the set side -- the payload is
  // already on this block's index space, so there is no index math). Re-fetching via
  // rc->Get keeps a running sum as multiple neighbors contribute to the same field, then
  // the channel is staled for the next round.
  ForEachBoundary<BoundaryType::any>(
      md, [&](auto pmb, auto rc, const NeighborBlock &nb, auto v) {
        const auto name = v->label();
        auto &chan = pmesh->tt_comm_map[ReceiveKey(pmb, nb, v, BoundaryType::any, id)];
        auto cur = rc->Get(name);
        auto addend = chan.Get();
        std::vector<train_t> a{cur->train()};
        std::vector<train_t> b{*addend};
        auto summed = tensor2::NonDestructiveSum(a, b);
        cur->set_train(std::move(summed[0]));
        chan.Stale();
      });

  // Round every block's fields once now that all addends are summed in.
  tensor2::RoundGramSVD(md.get(), eps);
}

} // namespace parthenon
