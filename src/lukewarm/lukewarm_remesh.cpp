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

#include "lukewarm/lukewarm_remesh.hpp"

#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "globals.hpp"
#include "interface/mesh_data.hpp"
#include "kokkos_abstraction.hpp"
#include "lukewarm/lukewarm.hpp"
#include "lukewarm/lukewarm_comms.hpp"
#include "lukewarm/lukewarm_container.hpp"
#include "lukewarm/lukewarm_geometry.hpp"
#include "lukewarm/lukewarm_relocate.hpp"
#include "mesh/mesh.hpp"
#include "pack/default_names.hpp"
#include "utils/error_checking.hpp"
#include "utils/instrument.hpp"

namespace parthenon {

namespace {
// Head every particle of lukewarm to the new block of the daughter of its block it lies
// in, new_gids(b, daughter), through channel routes(b, daughter) of comms
void Route(Lukewarm &lukewarm, const LukewarmGeometry &geometry,
           const ParArray2DRaw<int> &new_gids, const ParArray2DRaw<int> &routes,
           LukewarmComms &comms) {
  const int n = lukewarm.NumParticles();
  comms.Reserve(n);
  comms.nleaving = n;
  auto counts = comms.counts;
  Kokkos::deep_copy(counts, 0);
  auto leaving = comms.leaving;
  auto destinations = comms.destinations;
  auto real = lukewarm.GetData<Real>();
  auto integer = lukewarm.GetData<int>();
  const int x = lukewarm.Get<Real>(lukewarm::x::name()).GetOffset();
  const int y = lukewarm.Get<Real>(lukewarm::y::name()).GetOffset();
  const int z = lukewarm.Get<Real>(lukewarm::z::name()).GetOffset();
  constexpr int block = Lukewarm::block_row;
  par_for(
      PARTHENON_AUTO_LABEL, 0, n - 1, KOKKOS_LAMBDA(const int i) {
        const int b = integer(block, i);
        const int daughter = geometry.Daughter(b, real(x, i), real(y, i), real(z, i));
        // Until it is delivered, a particle's block is the gid of its new block
        integer(block, i) = new_gids(b, daughter);
        destinations(i) = routes(b, daughter);
        leaving(i) = i;
        Kokkos::atomic_add(&counts(routes(b, daughter)), 1);
      });
  Kokkos::deep_copy(comms.counts_h, counts);
}
} // namespace

DetachedLukewarms DetachLukewarms(Mesh *pmesh, const std::vector<LogicalLocation> &locs,
                                  const std::vector<int> &ranks) {
  PARTHENON_INSTRUMENT
  DetachedLukewarms detached;
  if (pmesh->resolved_packages->AllLukewarms().empty()) return detached;

  // Every particle is then in the block its block row names
  for (const auto &[name, metadata] : pmesh->resolved_packages->AllLukewarms()) {
    RelocateLukewarm(pmesh, name);
  }

  detached.locs = locs;
  detached.ranks = ranks;
  detached.owners = GatherLukewarmOwners(pmesh);
  for (auto &partition : pmesh->GetDefaultBlockPartitions()) {
    auto &md = pmesh->mesh_data.Add("base", partition);
    std::vector<int> gids;
    for (int b = 0; b < md->NumBlocks(); ++b) {
      gids.push_back(md->GetBlockData(b)->GetBlockPointer()->gid);
    }
    auto container = md->ReleaseLukewarms();
    // Their channels go with the old MeshData
    for (const auto &[name, lukewarm] : container->GetLukewarms()) {
      container->GetComms(name) = LukewarmComms();
    }
    detached.containers.push_back(std::move(container));
    detached.gids.push_back(std::move(gids));
  }
  return detached;
}

void RemeshLukewarms(Mesh *pmesh, DetachedLukewarms &&detached,
                     const std::vector<int> &oldtonew,
                     const std::vector<LogicalLocation> &locs,
                     const std::vector<int> &ranks) {
  PARTHENON_INSTRUMENT
  if (pmesh->resolved_packages->AllLukewarms().empty()) return;
  const int my_rank = Globals::my_rank;
  const auto owners = GatherLukewarmOwners(pmesh);
  const int nold = detached.locs.size();

  // The new block holding each daughter of every old block, numbered as by
  // LukewarmGeometry::Daughter(). It is one block for all of them unless the old block
  // was refined.
  std::vector<std::array<int, 8>> targets(nold);
  const int ndaughters = 1 << pmesh->ndim;
  for (int on = 0; on < nold; ++on) {
    const int nn = oldtonew[on];
    const auto &loc = detached.locs[on];
    if (locs[nn].level() <= loc.level()) {
      targets[on].fill(nn);
      continue;
    }
    // The daughters are the consecutive new blocks starting at nn
    targets[on].fill(-1);
    int found = 0;
    for (int l = 0; l < ndaughters; ++l) {
      for (int d = 0; d < 8; ++d) {
        if (locs[nn + l] == loc.GetDaughter(d & 1, (d >> 1) & 1, (d >> 2) & 1)) {
          targets[on][d] = nn + l;
          found += 1;
        }
      }
    }
    PARTHENON_REQUIRE_THROWS(found == ndaughters,
                             "Block " + std::to_string(on) +
                                 " was refined into blocks that are not its daughters");
  }

  // A channel from each old MeshData to every new MeshData its particles may go to. Each
  // rank finds every channel, and keeps those with an end on it.
  std::map<std::pair<int, int>, LukewarmChannel> all;
  for (int on = 0; on < nold; ++on) {
    for (const int nn : targets[on]) {
      if (nn < 0) continue;
      const int sender = detached.owners[on], receiver = owners[nn];
      all.emplace(std::make_pair(sender, receiver),
                  LukewarmChannel{sender, receiver, detached.ranks[on], ranks[nn]});
    }
  }
  std::vector<LukewarmChannel> channels;
  // The receivers of each old MeshData of this rank and the senders of each new one,
  // which come out sorted
  std::map<int, std::vector<int>> receivers, senders;
  for (const auto &[key, channel] : all) {
    if (channel.send_rank == my_rank)
      receivers[channel.sender].push_back(channel.receiver);
    if (channel.recv_rank == my_rank) senders[channel.receiver].push_back(channel.sender);
    if (channel.send_rank == my_rank || channel.recv_rank == my_rank) {
      channels.push_back(channel);
    }
  }

  // Where each particle of an old MeshData goes, by its block and daughter
  const int nsenders = detached.containers.size();
  std::vector<ParArray2DRaw<int>> new_gids(nsenders), routes(nsenders);
  for (int p = 0; p < nsenders; ++p) {
    const auto &gids = detached.gids[p];
    const auto &to = receivers[detached.containers[p]->GetId()];
    new_gids[p] = ParArray2DRaw<int>("RemeshLukewarms::new_gids", gids.size(), 8);
    routes[p] = ParArray2DRaw<int>("RemeshLukewarms::routes", gids.size(), 8);
    auto new_gids_h = Kokkos::create_mirror_view(new_gids[p]);
    auto routes_h = Kokkos::create_mirror_view(routes[p]);
    for (int b = 0; b < gids.size(); ++b) {
      for (int d = 0; d < 8; ++d) {
        const int nn = targets[gids[b]][d];
        new_gids_h(b, d) = nn;
        routes_h(b, d) =
            nn < 0 ? -1 : std::lower_bound(to.begin(), to.end(), owners[nn]) - to.begin();
      }
    }
    Kokkos::deep_copy(new_gids[p], new_gids_h);
    Kokkos::deep_copy(routes[p], routes_h);
  }

  std::vector<MeshData<Real> *> mds;
  for (auto &partition : pmesh->GetDefaultBlockPartitions()) {
    mds.push_back(pmesh->mesh_data.Add("base", partition).get());
  }
  for (const auto &[name, metadata] : pmesh->resolved_packages->AllLukewarms()) {
    // Buffers wait on their messages when destroyed
    lukewarm_comm_map_t buffers;
    AddLukewarmBuffers(pmesh, name, channels, buffers);

    std::vector<LukewarmComms> receives;
    for (auto *md : mds) {
      const int id = md->GetLukewarmContainer().GetId();
      std::vector<LukewarmBuffers *> receive;
      for (const int sender : senders[id]) {
        receive.push_back(&buffers.at({sender, id, name}));
      }
      receives.emplace_back(std::vector<LukewarmBuffers *>(), receive);
      StartReceiveLukewarmParticles(receives.back());
    }

    for (int p = 0; p < nsenders; ++p) {
      auto &container = *detached.containers[p];
      const int id = container.GetId();
      std::vector<LukewarmBuffers *> send;
      for (const int receiver : receivers[id]) {
        send.push_back(&buffers.at({id, receiver, name}));
      }
      LukewarmComms comms(send, {});
      auto &lukewarm = container.GetLukewarm(name);
      Route(lukewarm, container.GetGeometry(), new_gids[p], routes[p], comms);
      // The buffers are new, so they are free to send
      PARTHENON_REQUIRE_THROWS(SendLukewarmParticles(lukewarm, comms),
                               "Lukewarm buffers are busy during a remesh");
    }

    std::vector<bool> received(mds.size(), false);
    for (int nreceived = 0; nreceived < mds.size();) {
      for (int p = 0; p < mds.size(); ++p) {
        if (received[p]) continue;
        auto &container = mds[p]->GetLukewarmContainer();
        received[p] = ReceiveLukewarmParticles(container.GetLukewarm(name), receives[p],
                                               container.GetGeometry());
        nreceived += received[p];
      }
    }
  }
}

} // namespace parthenon
