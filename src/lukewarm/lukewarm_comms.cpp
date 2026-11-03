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

#include "lukewarm/lukewarm_comms.hpp"

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "globals.hpp"
#include "interface/mesh_data.hpp"
#include "lukewarm/lukewarm_container.hpp"
#include "lukewarm/lukewarm_geometry.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "pack/default_names.hpp"
#include "parthenon_mpi.hpp"
#include "utils/error_checking.hpp"
#include "utils/instrument.hpp"

namespace parthenon {

namespace {
// Hands out buffers of at least the size asked for. It keeps the largest one so far, so a
// buffer dropped after an empty message comes back without a new allocation.
template <typename T>
std::function<BufArray1D<T>(int)> BufferResource(const std::string &label) {
  auto spare = std::make_shared<BufArray1D<T>>();
  return [label, spare](const int size) {
    if (spare->extent_int(0) < size) {
      *spare = BufArray1D<T>(label, std::max(size, 2 * spare->extent_int(0)));
    }
    return *spare;
  };
}

// Messages of a channel get consecutive tags, one per field type
LukewarmBuffers MakeBuffers(const int channel, const int send_rank, const int recv_rank,
                            mpi_comm_t comm, const std::string &label) {
  LukewarmBuffers buffers;
  LukewarmTypes::IterateTypes([&](auto t) {
    using T = decltype(t);
    constexpr int idx = LukewarmTypes::GetIdx<T>();
    const int tag = LukewarmTypes::n_types * channel + idx;
    // Message sizes are not known ahead of time, so receivers probe for them
    buffers.Get<T>() = LukewarmBuffers::buffer_t<T>(
        tag, send_rank, recv_rank, comm,
        BufferResource<T>(label + "::buffer" + std::to_string(idx)), true);
  });
  return buffers;
}
} // namespace

LukewarmComms::LukewarmComms(const std::vector<LukewarmBuffers *> &send,
                             const std::vector<LukewarmBuffers *> &receive)
    : send(send), receive(receive), counts("LukewarmComms::counts", send.size()),
      send_messages_("LukewarmComms::send_messages", send.size()),
      receive_messages_("LukewarmComms::receive_messages", receive.size()),
      offsets_("LukewarmComms::offsets", receive.size() + 1) {
  counts_h = Kokkos::create_mirror_view(counts);
  send_messages_h_ = Kokkos::create_mirror_view(send_messages_);
  receive_messages_h_ = Kokkos::create_mirror_view(receive_messages_);
  offsets_h_ = Kokkos::create_mirror_view(offsets_);
}

void LukewarmComms::Reserve(const int n) {
  if (destinations.extent_int(0) >= n) return;
  // Grown like particle storage, so that a growing lukewarm reallocates rarely
  const int size = std::max(n, 2 * destinations.extent_int(0));
  leaving = ParArray1DRaw<int>(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "LukewarmComms::leaving"), size);
  destinations = ParArray1DRaw<int>(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "LukewarmComms::destinations"),
      size);
}

std::vector<int> GatherLukewarmOwners(Mesh *pmesh) {
  std::vector<int> owners(pmesh->nbtotal, -1);
  for (auto &partition : pmesh->GetDefaultBlockPartitions()) {
    auto &md = pmesh->mesh_data.Add("base", partition);
    const int id = md->GetLukewarmContainer().GetId();
    for (int b = 0; b < md->NumBlocks(); ++b) {
      owners[md->GetBlockData(b)->GetBlockPointer()->gid] = id;
    }
  }
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, owners.data(), owners.size(), MPI_INT,
                                    MPI_MAX, MPI_COMM_WORLD));
#endif
  return owners;
}

void AddLukewarmBuffers(Mesh *pmesh, const std::string &name,
                        const std::vector<LukewarmChannel> &channels,
                        lukewarm_comm_map_t &map) {
  // Number the channels between each pair of ranks by sender and receiver
  std::map<std::pair<int, int>, std::set<std::pair<int, int>>> between;
  for (const auto &c : channels) {
    if (c.send_rank == c.recv_rank) continue;
    between[{c.send_rank, c.recv_rank}].insert({c.sender, c.receiver});
  }
  mpi_comm_t comm{};
#ifdef MPI_PARALLEL
  comm = pmesh->GetMPIComm(LukewarmCommLabel(name));
  int flag;
  void *max_tag;
  PARTHENON_MPI_CHECK(MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &max_tag, &flag));
  PARTHENON_REQUIRE_THROWS(flag, "Cannot query the largest MPI tag");
  for (const auto &[ranks, set] : between) {
    PARTHENON_REQUIRE_THROWS(LukewarmTypes::n_types * set.size() <=
                                 static_cast<std::size_t>(*static_cast<int *>(max_tag)),
                             "Lukewarms need more MPI tags than are available");
  }
#endif

  for (const auto &c : channels) {
    const auto key = std::make_tuple(c.sender, c.receiver, name);
    if (map.count(key) > 0) continue;
    int channel = 0;
    if (c.send_rank != c.recv_rank) {
      const auto &set = between.at({c.send_rank, c.recv_rank});
      channel = std::distance(set.begin(), set.find({c.sender, c.receiver}));
    }
    const std::string label = "Lukewarm::" + name + "::" + std::to_string(c.sender) +
                              "->" + std::to_string(c.receiver);
    map.emplace(key, MakeBuffers(channel, c.send_rank, c.recv_rank, comm, label));
  }
}

void BuildLukewarmComms(Mesh *pmesh) {
  PARTHENON_INSTRUMENT
  const int my_rank = Globals::my_rank;
  std::vector<MeshData<Real> *> mds;
  for (auto &partition : pmesh->GetDefaultBlockPartitions()) {
    mds.push_back(pmesh->mesh_data.Add("base", partition).get());
  }

  // Every MeshData's id is the smallest gid of its blocks, and every rank learns which
  // MeshData owns each block
  for (auto *md : mds) {
    int id = pmesh->nbtotal;
    for (int b = 0; b < md->NumBlocks(); ++b) {
      id = std::min(id, md->GetBlockData(b)->GetBlockPointer()->gid);
    }
    md->GetLukewarmContainer().SetNeighbors(id, {});
  }
  const auto owners = GatherLukewarmOwners(pmesh);

  // Block neighbors are mutual, so a MeshData is a neighbor of each of its neighbors and
  // both ends of every channel know about it
  std::vector<LukewarmChannel> channels;
  for (auto *md : mds) {
    auto &container = md->GetLukewarmContainer();
    const int id = container.GetId();
    std::map<int, int> rank_of;
    for (int b = 0; b < md->NumBlocks(); ++b) {
      for (const auto &nb : md->GetBlockData(b)->GetBlockPointer()->GetNeighbors()) {
        const int owner = owners.at(nb.gid);
        PARTHENON_REQUIRE_THROWS(owner >= 0, "Block " + std::to_string(nb.gid) +
                                                 " has no lukewarm MeshData");
        if (owner != id) rank_of[owner] = nb.rank;
      }
    }
    std::vector<LukewarmNeighbor> neighbors;
    std::vector<int> neighbor_ids;
    for (const auto &[nb_id, rank] : rank_of) {
      neighbors.push_back({nb_id, rank});
      neighbor_ids.push_back(nb_id);
      channels.push_back({id, nb_id, my_rank, rank});
      channels.push_back({nb_id, id, rank, my_rank});
    }
    container.SetNeighbors(id, neighbors);
    container.SetGeometry(LukewarmGeometry(md, owners, neighbor_ids));
  }

  // Buffers wait on their messages when destroyed
  auto &map = pmesh->lukewarm_comm_map;
  map.clear();
  for (const auto &[name, metadata] : pmesh->resolved_packages->AllLukewarms()) {
    AddLukewarmBuffers(pmesh, name, channels, map);
    for (auto *md : mds) {
      auto &container = md->GetLukewarmContainer();
      const int id = container.GetId();
      std::vector<LukewarmBuffers *> send, receive;
      for (const auto &nb : container.GetNeighbors()) {
        send.push_back(&map.at({id, nb.id, name}));
        receive.push_back(&map.at({nb.id, id, name}));
      }
      container.GetComms(name) = LukewarmComms(send, receive);
    }
  }
}

namespace {
// Copy particle i into slot k of a message of count particles, and back
template <typename T>
KOKKOS_INLINE_FUNCTION void Pack(const ParArray2DRaw<T> &data, const int i, T *message,
                                 const int k, const int count) {
  for (int c = 0; c < data.extent_int(0); ++c) {
    message[c * count + k] = data(c, i);
  }
}
template <typename T>
KOKKOS_INLINE_FUNCTION void Unpack(const T *message, const int k, const int count,
                                   const ParArray2DRaw<T> &data, const int i) {
  for (int c = 0; c < data.extent_int(0); ++c) {
    data(c, i) = message[c * count + k];
  }
}

template <class F>
void ForEachBuffer(const std::vector<LukewarmBuffers *> &buffers, const F &func) {
  for (auto *b : buffers) {
    LukewarmTypes::IterateTypes([&](auto t) { func(b->Get<decltype(t)>()); });
  }
}
} // namespace

bool SendLukewarmParticles(Lukewarm &lukewarm, LukewarmComms &comms) {
  PARTHENON_INSTRUMENT
  bool available = true;
  ForEachBuffer(comms.send,
                [&](auto &buf) { available = available && buf.IsAvailableForWrite(); });
  if (!available) return false;

  if (comms.nleaving == 0) {
    ForEachBuffer(comms.send, [](auto &buf) { buf.SendNull(); });
    return true;
  }

  const int nchannels = comms.send.size();
  const auto &counts_h = comms.counts_h;
  auto messages_h = comms.send_messages_h_;
  for (int nb = 0; nb < nchannels; ++nb) {
    LukewarmTypes::IterateTypes([&](auto t) {
      using T = decltype(t);
      const int size = counts_h(nb) * lukewarm.NumComponents<T>();
      auto &buf = comms.send[nb]->Get<T>();
      if (size > 0) buf.Allocate(size);
      messages_h(nb).template Get<T>() = size > 0 ? buf.buffer().data() : nullptr;
    });
  }
  auto messages = comms.send_messages_;
  Kokkos::deep_copy(messages, messages_h);

  // Each team packs the particles going through one channel, in the order they are
  // stored
  auto real = lukewarm.GetData<Real>();
  auto integer = lukewarm.GetData<int>();
  auto uint64 = lukewarm.GetData<std::uint64_t>();
  auto leaving = comms.leaving;
  auto destinations = comms.destinations;
  auto counts = comms.counts;
  const int nleaving = comms.nleaving;
  Kokkos::parallel_for(
      PARTHENON_AUTO_LABEL, Kokkos::TeamPolicy<>(DevExecSpace(), nchannels, Kokkos::AUTO),
      KOKKOS_LAMBDA(team_mbr_t member) {
        const int nb = member.league_rank();
        const int count = counts(nb);
        if (count == 0) return;
        const LukewarmMessages message = messages(nb);
        Kokkos::parallel_scan(Kokkos::TeamThreadRange(member, nleaving),
                              [&](const int k, int &slot, const bool final) {
                                const int i = leaving(k);
                                if (destinations(i) != nb) return;
                                if (final) {
                                  Pack(real, i, message.real, slot, count);
                                  Pack(integer, i, message.integer, slot, count);
                                  Pack(uint64, i, message.uint64, slot, count);
                                }
                                slot += 1;
                              });
      });
  auto marks = lukewarm.GetRemovalMarks();
  par_for(
      PARTHENON_AUTO_LABEL, 0, nleaving - 1,
      KOKKOS_LAMBDA(const int k) { marks(leaving(k)) = 1; });
#ifdef MPI_PARALLEL
  // Messages go straight from device memory, so the packing has to be done first
  Kokkos::fence();
#endif

  for (int nb = 0; nb < nchannels; ++nb) {
    LukewarmTypes::IterateTypes([&](auto t) {
      using T = decltype(t);
      const int size = counts_h(nb) * lukewarm.NumComponents<T>();
      auto &buf = comms.send[nb]->Get<T>();
      if (size > 0) {
        buf.Send(false, size);
      } else {
        buf.SendNull();
      }
    });
  }

  lukewarm.RemoveMarked();
  comms.nleaving = 0;
  return true;
}

void StartReceiveLukewarmParticles(LukewarmComms &comms) {
  ForEachBuffer(comms.receive, [](auto &buf) { buf.TryStartReceive(); });
}

bool ReceiveLukewarmParticles(Lukewarm &lukewarm, LukewarmComms &comms,
                              const LukewarmGeometry &geometry) {
  PARTHENON_INSTRUMENT
  bool received = true;
  // Try every buffer, rather than stopping at the first that is not ready, so that all
  // of them make progress
  ForEachBuffer(comms.receive,
                [&](auto &buf) { received = buf.TryReceive() && received; });
  if (!received) return false;

  const int nchannels = comms.receive.size();
  auto messages_h = comms.receive_messages_h_;
  auto offsets_h = comms.offsets_h_;
  offsets_h(0) = 0;
  for (int nb = 0; nb < nchannels; ++nb) {
    // Every message of a channel holds the same particles
    int count = -1;
    LukewarmTypes::IterateTypes([&](auto t) {
      using T = decltype(t);
      const int ncomp = lukewarm.NumComponents<T>();
      auto &buf = comms.receive[nb]->Get<T>();
      const int size = buf.GetMessageSize();
      messages_h(nb).template Get<T>() = size > 0 ? buf.buffer().data() : nullptr;
      if (ncomp == 0) return;
      PARTHENON_REQUIRE_THROWS(size % ncomp == 0 && (count < 0 || size / ncomp == count),
                               "Lukewarm " + lukewarm.label() +
                                   " received a malformed message");
      count = size / ncomp;
    });
    offsets_h(nb + 1) = offsets_h(nb) + std::max(count, 0);
  }
  const int total = offsets_h(nchannels);
  comms.received = IndexRange{0, -1};

  if (total > 0) {
    auto messages = comms.receive_messages_;
    auto offsets = comms.offsets_;
    Kokkos::deep_copy(messages, messages_h);
    Kokkos::deep_copy(offsets, offsets_h);
    comms.received = lukewarm.Append(total);
    const int start = comms.received.s;
    auto real = lukewarm.GetData<Real>();
    auto integer = lukewarm.GetData<int>();
    auto uint64 = lukewarm.GetData<std::uint64_t>();
    constexpr int block = Lukewarm::block_row;
    Kokkos::parallel_for(
        PARTHENON_AUTO_LABEL,
        Kokkos::TeamPolicy<>(DevExecSpace(), nchannels, Kokkos::AUTO),
        KOKKOS_LAMBDA(team_mbr_t member) {
          const int nb = member.league_rank();
          const int count = offsets(nb + 1) - offsets(nb);
          const LukewarmMessages message = messages(nb);
          Kokkos::parallel_for(Kokkos::TeamThreadRange(member, count), [&](const int k) {
            const int i = start + offsets(nb) + k;
            Unpack(message.real, k, count, real, i);
            Unpack(message.integer, k, count, integer, i);
            Unpack(message.uint64, k, count, uint64, i);
            integer(block, i) = geometry.LocalIndex(integer(block, i));
          });
        });
    // The buffers are written by the next messages as soon as they are staled
    Kokkos::fence();
  }

  ForEachBuffer(comms.receive, [](auto &buf) { buf.Stale(); });
  return true;
}

} // namespace parthenon
