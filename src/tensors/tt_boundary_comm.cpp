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

// This file was made in part with generative AI.

#include "tensors/tt_boundary_comm.hpp"

#include <memory>
#include <string>
#include <vector>

#include "basic_types.hpp"
#include "bvals/comms/bvals_utils.hpp"
#include "bvals/comms/calc_indices.hpp"
#include "bvals/neighbor_block.hpp"
#include "globals.hpp"
#include "interface/metadata.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "tensors/tt_boundary_cache.hpp"
#include "tensors/tt_comm_channel.hpp"
#include "tensors/tt_operations.hpp"
#include "tensors/tt_pack.hpp"
#include "tensors/tt_serialize.hpp"
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

// Look up an existing receive channel. operator[] would silently create a fresh stale
// channel that no sender writes to, and TTReceive would then wait forever.
TTCommChannel &GetReceiveChannel(Mesh *pmesh, const Mesh::channel_key_t &key) {
  PARTHENON_REQUIRE(pmesh->tt_comm_map.count(key) > 0,
                    "No TT comm channel for this receive key. Was BuildTTBoundaryCache "
                    "called after the last (re)mesh?");
  return pmesh->tt_comm_map.at(key);
}

// Return the send channel for boundary (pmb, nb) of field v, creating it if needed. A
// same-rank neighbor shares one local channel between both ends (its SendKey is the
// receiver's ReceiveKey). A remote neighbor gets a remote send channel here and a
// remote receive channel for the reverse direction, so the receive side can find it by
// ReceiveKey. Remote channels use the regular-field geometric tag on the field's TT
// communicator, so both ranks agree on tags without exchanging anything.
template <class Var>
TTCommChannel &GetOrCreateChannels(Mesh *pmesh, MeshBlock *pmb, const NeighborBlock &nb,
                                   const Var &v, const int id) {
  auto &map = pmesh->tt_comm_map;
  const auto skey = SendKey(pmb, nb, v, BoundaryType::any, id);
  if (nb.rank == Globals::my_rank) return map[skey];
#ifdef MPI_PARALLEL
  const int tag = pmesh->tag_map.GetTag(pmb, nb, id);
  const auto comm = pmesh->GetMPIComm(TTCommLabel(v->label()));
  const auto rkey = ReceiveKey(pmb, nb, v, BoundaryType::any, id);
  if (map.count(rkey) == 0)
    map[rkey] = TTCommChannel(tag, nb.rank, Globals::my_rank, comm);
  if (map.count(skey) == 0)
    map[skey] = TTCommChannel(tag, Globals::my_rank, nb.rank, comm);
  return map.at(skey);
#else
  PARTHENON_FAIL("Remote TT neighbor without MPI.");
  return map[skey];
#endif
}

} // namespace

// Build the send-side boundary cache: walk every boundary, record its send channel (keyed
// by SendKey, creating local or remote channels as needed) and the send/recv index boxes
// needed to build the addend, packed into a flat device array for one batched launch. The
// set side needs no cache (it looks up its channel inline by ReceiveKey), so only this
// send cache exists.
TaskStatus BuildTTBoundaryCache(std::shared_ptr<MeshTTData> &md) {
  using namespace loops;
  Mesh *pmesh = md->GetMeshPointer();
  const int id = 0; // single TT comm channel set for now
  TTBoundaryCache *cache = &md->GetBoundaryCache();
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
        cache->channels.push_back(&GetOrCreateChannels(pmesh, pmb, nb, v, id));

        BlockInfo binfo(pmb);
        auto [other, rev] = ReverseNeighbor(binfo, nb);
        TTBndInfo info;
        // Sender's interior cells destined for the neighbor, and the neighbor's ghost
        // cells that receive them -- both on the whole-block index space.
        const TopologicalElement te = TopologicalElement::CC; // TODO(LFR): Fix this
        info.send =
            CalcIndices(nb, binfo, v, te, IndexRangeType::BoundaryInteriorSend, false);
        info.recv =
            CalcIndices(rev, other, v, te, IndexRangeType::BoundaryExteriorRecv, false);
        info.lcoord_trans = nb.lcoord_trans;
        if (NeighborIsCoarser(binfo, nb)) {
          info.btype = BoundaryRelation::f2c;
          info.prores =
              CalcIndices(nb, binfo, v, te, IndexRangeType::BoundaryInteriorSend, true);
        } else if (NeighborIsFiner(binfo, nb)) {
          info.btype = BoundaryRelation::c2f;
          info.prores =
              CalcIndices(rev, other, v, te, IndexRangeType::BoundaryExteriorRecv, true);
        } else {
          info.btype = BoundaryRelation::same;
        }
        info.cfmap = CoarseFineMap::FromNDim(pmesh->ndim, Globals::nghost);
        cache->bnd_info_h(ibound) = info;
        ++ibound;
      });

  Kokkos::deep_copy(cache->bnd_info, cache->bnd_info_h);
  cache->epoch = pmesh->tt_comm_map.GetCurrentEpoch();

  return TaskStatus::complete;
}

std::vector<std::shared_ptr<tensor::TensorTrain>>
BuildBoundaryTensors(std::shared_ptr<MeshTTData> &md, const TTBoundaryCache &cache) {
  using namespace loops;
  using train_t = tensor::TensorTrain;
  using HostPack = tensor::TensorTrainHostPackT<tensor::DefaultTTraits>;

  // Build all the sets of trains for stages of boundary communication
  std::vector<train_t *> src;
  std::vector<std::shared_ptr<train_t>> out;
  std::vector<std::shared_ptr<train_t>> coarse;
  ForEachBoundary<BoundaryType::any>(
      md, [&](auto pmb, auto /*rc*/, const NeighborBlock &nb, auto v) {
        src.push_back(&v->train());
        out.push_back(std::make_shared<train_t>(v->train().DeepCopy()));
        // Create a coarse version of the spatial cores for each boundary
        TTFieldMetadata metadata({}, v->metadata());
        const int ranks = v->train().GetCoreHost(0).RR();
        coarse.push_back(std::make_shared<train_t>(metadata.CoreIndexers(pmb, true),
                                                   std::vector<int>{}, 1, ranks));
      });

  const int nbound = static_cast<int>(out.size());
  PARTHENON_DEBUG_REQUIRE(nbound == static_cast<int>(cache.bnd_info_h.extent(0)),
                          "Boundary walk and cache disagree on boundary count.");
  if (nbound == 0) return out;

  auto pack_src = HostPack(src).MakeDevicePack();
  auto pack_coarse = HostPack(coarse).MakeDevicePack();
  auto pack_out = HostPack(out).MakeDevicePack();
  auto bnd_info = cache.bnd_info;

  // Restrict to coarse trains where required
  {
    constexpr int unused_scratch_size = 0;
    constexpr int unused_scratch_level = 1;
    parthenon::par_for_outer(
        PARTHENON_AUTO_LABEL, unused_scratch_size, unused_scratch_level, 0, nbound - 1,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int e) {
          const auto &info = bnd_info(e);
          if (info.btype == BoundaryRelation::f2c) {
            auto &sc_src = pack_src(e, 0, 0);
            auto &sc_out = pack_coarse(e, 0, 0);
            const auto &idxerp = info.prores;
            const auto &idxerf = pack_src.indexer(0);
            const auto &idxerc = pack_coarse.indexer(0);
            const auto &cfmap = info.cfmap;
            const Real winv = 1.0 / cfmap.NumFine();
            for (int r = 0; r < sc_out.RR(); ++r) {
              parthenon::par_for_inner(member, 0, sc_out.DD() - 1,
                                       [&](const int idx) { sc_out(0, idx, r) = 0.0; });
              member.team_barrier();
              parthenon::par_for_inner(member, 0, idxerp.size() - 1, [&](const int idx) {
                const auto [t, u, v, kc, jc, ic] = idxerp(idx);
                const auto coarse_idx = idxerc.GetFlatIdx(t, u, v, kc, jc, ic);
                Real sum = 0.0;
                cfmap.ForEachFine(kc, jc, ic, [&](int kf, int jf, int iff) {
                  sum += sc_src(0, idxerf.GetFlatIdx(t, u, v, kf, jf, iff), r);
                });
                sc_out(0, coarse_idx, r) = winv * sum;
              });
              member.team_barrier();
            }
          }
        });
  }

  // Move cells from the sender index space to the receiver index space
  constexpr int unused_scratch_size = 0;
  constexpr int unused_scratch_level = 1;
  parthenon::par_for_outer(
      PARTHENON_AUTO_LABEL, unused_scratch_size, unused_scratch_level, 0, nbound - 1,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int e) {
        auto &info = bnd_info(e);
        auto &sc_src = (info.btype == BoundaryRelation::f2c) ? pack_coarse(e, 0, 0)
                                                             : pack_src(e, 0, 0);
        auto &sc_out = (info.btype == BoundaryRelation::c2f) ? pack_coarse(e, 0, 0)
                                                             : pack_out(e, 0, 0);
        auto &idxer_src = (info.btype == BoundaryRelation::f2c) ? pack_coarse.indexer(0)
                                                                : pack_src.indexer(0);
        auto &idxer_out = (info.btype == BoundaryRelation::c2f) ? pack_coarse.indexer(0)
                                                                : pack_out.indexer(0);
        const auto &send = info.send;
        const auto &recv = info.recv;
        const int ncell = static_cast<int>(send.size());
        for (int r = 0; r < sc_out.RR(); ++r) {
          parthenon::par_for_inner(member, 0, sc_out.DD() - 1,
                                   [&](const int idx) { sc_out(0, idx, r) = 0.0; });
          member.team_barrier();
          parthenon::par_for_inner(member, 0, ncell - 1, [&](const int c) {
            const auto [ts, us, vs, ks, js, is] = send(c);
            const auto [tr, ur, vr, kr, jr, ir] = recv(c);
            const int src_idx = idxer_src.GetFlatIdx(ts, us, vs, ks, js, is);
            const int dst_idx = idxer_out.GetFlatIdx(tr, ur, vr, kr, jr, ir);
            sc_out(0, dst_idx, r) = sc_src(0, src_idx, r);
          });
          member.team_barrier();
        }
      });

  // Prolongate on c2f blocks
  {
    constexpr int unused_scratch_size = 0;
    constexpr int unused_scratch_level = 1;
    parthenon::par_for_outer(
        PARTHENON_AUTO_LABEL, unused_scratch_size, unused_scratch_level, 0, nbound - 1,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int e) {
          const auto &info = bnd_info(e);
          if (info.btype == BoundaryRelation::c2f) {
            auto &sc_src = pack_coarse(e, 0, 0);
            auto &sc_out = pack_out(e, 0, 0);
            const auto &idxerp = info.prores;
            const auto &idxerf = pack_out.indexer(0);
            const auto &idxerc = pack_coarse.indexer(0);
            for (int r = 0; r < sc_out.RR(); ++r) {
              parthenon::par_for_inner(member, 0, idxerf.size() - 1,
                                       [&](const int idx) { sc_out(0, idx, r) = 0.0; });
              member.team_barrier();

              parthenon::par_for_inner(member, 0, idxerp.size() - 1, [&](const int idx) {
                const auto [t, u, v, kc, jc, ic] = idxerp(idx);
                const auto coarse_idx = idxerc.GetFlatIdx(t, u, v, kc, jc, ic);
                const Real cval = sc_src(0, coarse_idx, r);
                info.cfmap.ForEachFine(kc, jc, ic, [&](int kf, int jf, int iff) {
                  sc_out(0, idxerf.GetFlatIdx(t, u, v, kf, jf, iff), r) = cval;
                });
              });
              member.team_barrier();
            }
          }
        });
  }

  return out;
}

TaskStatus TTZeroBoundaries(std::shared_ptr<MeshTTData> &md) {
  if (md->NumBlocks() == 0) return TaskStatus::complete;

  // TODO(LFR): Interior bounds assume cell-centered fields, as in BuildTTBoundaryCache.
  const IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  const IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  const IndexRange kb = md->GetBoundsK(IndexDomain::interior);

  // Only FillGhost fields take part in boundary comm (see ForEachBoundary).
  std::vector<std::string> names;
  for (const auto &name : md->FieldNames())
    if (md->GetBlockData(0)->Get(name)->IsSet(Metadata::FillGhost)) names.push_back(name);
  if (names.empty()) return TaskStatus::complete;
  auto pack = tensor::TensorTrainHostPack(*md, names).MakeDevicePack();

  // One team per (block, field); each team reads its own spatial core's size and
  // indexer, so nothing here assumes the fields share a shape.
  constexpr int unused_scratch_size = 0;
  constexpr int unused_scratch_level = 1;
  parthenon::par_for_outer(
      PARTHENON_AUTO_LABEL, unused_scratch_size, unused_scratch_level, 0,
      pack.GetNBlocks() - 1, 0, pack.GetNVars() - 1,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int v) {
        auto &sc = pack(b, v, 0);
        const auto &idxer = pack.indexer(0);
        const int nrr = sc.RR();
        parthenon::par_for_inner(member, 0, sc.DD() - 1, [&](const int idx) {
          const auto [t, u, w, k, j, i] = idxer(idx);
          const bool interior =
              i >= ib.s && i <= ib.e && j >= jb.s && j <= jb.e && k >= kb.s && k <= kb.e;
          if (interior) return;
          for (int r = 0; r < nrr; ++r)
            sc(0, idx, r) = 0.0;
        });
      });
  return TaskStatus::complete;
}

TaskStatus TTSend(std::shared_ptr<MeshTTData> &md, Real eps) {
  using train_t = tensor::TensorTrain;
  Mesh *pmesh = md->GetMeshPointer();
  TTBoundaryCache &cache = md->GetBoundaryCache();
  // The channel map is cleared (and its epoch bumped) on remesh.
  if (cache.epoch != pmesh->tt_comm_map.GetCurrentEpoch()) BuildTTBoundaryCache(md);

  // Wait until every channel is writable: local channels once the receiver has staled
  // them, remote ones once the previous send has completed.
  for (auto *chan : cache.channels)
    if (!chan->IsAvailableForWrite()) return TaskStatus::incomplete;

  auto addends = BuildBoundaryTensors(md, cache);
  const int nbound = static_cast<int>(addends.size());
  if (nbound == 0) return TaskStatus::complete;

  // Round each addend to keep ranks bounded before shipping.
  tensor::TensorTrainHostPack pack(addends);
  tensor::RoundGramSVD(pack, eps);

  // Same-rank addends are handed over directly. Remote addends are serialized, all in
  // one batch, into their channels' send buffers and then sent.
  std::vector<int> remote;
  std::vector<std::size_t> sizes;
  std::vector<const train_t *> trains;
  std::vector<tensor::SerialTarget<TTCommChannel::buf_t>> targets;
  for (int e = 0; e < nbound; ++e) {
    auto *chan = cache.channels[e];
    if (chan->IsLocal()) {
      chan->Send(addends[e]);
      continue;
    }
    const std::size_t size = tensor::SerializedSize(*addends[e]);
    remote.push_back(e);
    sizes.push_back(size);
    trains.push_back(addends[e].get());
    targets.push_back({chan->PrepareSendBuffer(size), 0});
  }
  if (remote.empty()) return TaskStatus::complete;

  // SerializeTrains fences, so the buffers are complete before MPI reads them.
  tensor::SerializeTrains(trains, targets);
  for (std::size_t i = 0; i < remote.size(); ++i)
    cache.channels[remote[i]]->SendBuffer(sizes[i]);

  return TaskStatus::complete;
}

TaskStatus TTReceive(std::shared_ptr<MeshTTData> &md) {
  using namespace loops;
  Mesh *pmesh = md->GetMeshPointer();
  const int id = 0;
  // Receive channels for remote neighbors are created with the cache.
  if (md->GetBoundaryCache().epoch != pmesh->tt_comm_map.GetCurrentEpoch())
    BuildTTBoundaryCache(md);
  bool all = true;
  ForEachBoundary<BoundaryType::any>(
      md, [&](auto pmb, auto /*rc*/, const NeighborBlock &nb, auto v) {
        const auto key = ReceiveKey(pmb, nb, v, BoundaryType::any, id);
        auto &chan = GetReceiveChannel(pmesh, key);
        all = chan.TryReceive() && all;
      });
  if (all) return TaskStatus::complete;
  return TaskStatus::incomplete;
}

TaskStatus TTSetBounds(std::shared_ptr<MeshTTData> &md, Real eps) {
  using namespace loops;
  using train_t = tensor::TensorTrain;
  Mesh *pmesh = md->GetMeshPointer();
  const int id = 0;

  // Rebuild the addends that arrived from other ranks, all in one batch. They are
  // deserialized onto the receiving field's per-core layout.
  {
    std::vector<TTCommChannel *> chans;
    std::vector<std::shared_ptr<train_t>> trains;
    std::vector<train_t *> outs;
    std::vector<std::vector<Indexer6D>> indexers;
    std::vector<tensor::SerialTarget<TTCommChannel::buf_t>> sources;
    ForEachBoundary<BoundaryType::any>(
        md, [&](auto pmb, auto /*rc*/, const NeighborBlock &nb, auto v) {
          auto &chan =
              GetReceiveChannel(pmesh, ReceiveKey(pmb, nb, v, BoundaryType::any, id));
          if (chan.IsLocal()) return;
          PARTHENON_REQUIRE(chan.GetState() == BufferState::received,
                            "TTSetBounds called before TTReceive completed.");
          chans.push_back(&chan);
          trains.push_back(std::make_shared<train_t>());
          outs.push_back(trains.back().get());
          indexers.push_back(v->train().GetCoreIndexers());
          sources.push_back({chan.ReceiveBuffer(), 0});
        });
    tensor::DeserializeTrains(sources, indexers, outs);
    for (std::size_t i = 0; i < chans.size(); ++i)
      chans[i]->SetTrain(trains[i]);
  }

  // Sum each received addend into its destination block's train, looking the receive
  // channel up inline by ReceiveKey (no cache needed on the set side -- the payload is
  // already on this block's index space, so there is no index math). Re-fetching via
  // rc->Get keeps a running sum as multiple neighbors contribute to the same field, then
  // the channel is staled for the next round.
  ForEachBoundary<BoundaryType::any>(
      md, [&](auto pmb, auto rc, const NeighborBlock &nb, auto v) {
        const auto name = v->label();
        const auto key = ReceiveKey(pmb, nb, v, BoundaryType::any, id);
        auto &chan = GetReceiveChannel(pmesh, key);
        PARTHENON_REQUIRE(chan.GetState() == BufferState::received,
                          "TTSetBounds called before TTReceive completed.");
        auto cur = rc->Get(name);
        // The channel holds the only reference to the addend (TTSend's local vector is
        // gone), so DestructiveSum may take its fibers.
        auto addend = chan.Get();
        std::vector<train_t *> a{&cur->train()};
        std::vector<train_t *> b{addend.get()};
        tensor::TensorTrainHostPack pa(a), pb(b);
        tensor::DestructiveSum(pa, pb);
        chan.Stale();
      });

  // Round every block's fields once now that all addends are summed in.
  tensor::TensorTrainHostPack pack(*md);
  tensor::RoundGramSVD(pack, eps);
  return TaskStatus::complete;
}

} // namespace parthenon
