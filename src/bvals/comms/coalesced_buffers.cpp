//========================================================================================
// (C) (or copyright) 2024. Triad National Security, LLC. All rights reserved.
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// AI was used to assist with coalesced buffer kernel optimization.
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
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "basic_types.hpp"
#include "bvals/comms/bnd_id.hpp"
#include "bvals/comms/bvals_utils.hpp"
#include "bvals/comms/coalesced_buffers.hpp"
#include "bvals/neighbor_block.hpp"
#include "coordinates/coordinates.hpp"
#include "interface/mesh_data.hpp"
#include "interface/variable.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "utils/communication_buffer.hpp"

namespace parthenon {
namespace {

int CoalescedTeamsPerBuffer(const Mesh *pmesh, int buffers) {
  static const bool enabled = []() {
    const char *value = std::getenv("RIOT_COALESCED_TEAMS");
    return value != nullptr && std::strcmp(value, "1") == 0;
  }();
  if (!enabled || buffers == 0) return 1;
  return std::max(1, (pmesh->minimum_number_of_teams_for_boundary_kernel + buffers - 1) /
                         buffers);
}

int CoalescedCopyChunk() {
  static const int chunk = []() {
    const char *value = std::getenv("RIOT_COALESCED_CHUNK");
    if (value == nullptr) return 0;
    char *end = nullptr;
    const long n = std::strtol(value, &end, 10);
    PARTHENON_REQUIRE(end != value && *end == '\0' &&
                          (n == 0 || (n >= 256 && n <= 1048576)),
                      "RIOT_COALESCED_CHUNK must be 0 or 256..1048576 elements");
    return static_cast<int>(n);
  }();
  return chunk;
}

bool DenseStaticComms() {
  static const bool enabled = []() {
    const char *value = std::getenv("RIOT_DENSE_STATIC_COMMS");
    return value != nullptr && std::strcmp(value, "1") == 0;
  }();
  return enabled;
}

// Each team owns a disjoint interval of the combined buffer. Find the first
// source/destination boundary by offset, then copy intersections with the tile.
// No new metadata cache or allocation lifetime is introduced.
template <bool Pack>
void CoalescedChunkCopy(const ParArray1DRaw<BndId> &bids, std::size_t total,
                        const DevExecSpace &exec) {
  const std::size_t chunk = CoalescedCopyChunk();
  if (total == 0) return;
  const int count = static_cast<int>(bids.size());
  Kokkos::parallel_for(
      Pack ? "RIOT::ChunkPack" : "RIOT::ChunkUnpack",
      Kokkos::TeamPolicy<DevExecSpace>(exec, (total + chunk - 1) / chunk, 256),
      KOKKOS_LAMBDA(team_mbr_t team) {
        std::size_t begin = static_cast<std::size_t>(team.league_rank()) * chunk;
        const std::size_t end = std::min(begin + chunk, total);
        int lo = 0;
        int hi = count;
        while (lo < hi) {
          const int mid = lo + (hi - lo) / 2;
          if (static_cast<std::size_t>(bids[mid].start_idx()) <= begin)
            lo = mid + 1;
          else
            hi = mid;
        }
        int b = lo - 1;
        while (begin < end && b < count) {
          if (bids[b].buf_allocated) {
            const std::size_t start = bids[b].start_idx();
            const std::size_t stop = start + bids[b].size();
            const std::size_t piece_end = std::min(end, stop);
            Real *combined = &(bids[b].coalesced_buf(start));
            Real *individual = &(bids[b].buf(0));
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange<>(team, begin - start, piece_end - start),
                [&](const std::size_t i) {
                  if constexpr (Pack)
                    combined[i] = individual[i];
                  else
                    individual[i] = combined[i];
                });
            begin = piece_end;
          }
          ++b;
        }
      });

  const char *verify = std::getenv("RIOT_VERIFY_COPIES");
  if (verify != nullptr && std::strcmp(verify, "1") == 0) {
    int errors = 0;
    Kokkos::parallel_reduce(
        "RIOT::VerifyCoalescedCopy",
        Kokkos::TeamPolicy<DevExecSpace>(exec, count, Kokkos::AUTO),
        KOKKOS_LAMBDA(team_mbr_t team, int &bad) {
          const int b = team.league_rank();
          int local = 0;
          if (bids[b].buf_allocated) {
            Real *combined = &(bids[b].coalesced_buf(bids[b].start_idx()));
            Real *individual = &(bids[b].buf(0));
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange<>(team, bids[b].size()),
                [&](const int i, int &n) { n += combined[i] != individual[i]; }, local);
          }
          Kokkos::single(Kokkos::PerTeam(team), [&]() { bad += local; });
        },
        errors);
    PARTHENON_REQUIRE(errors == 0, "Coalesced copy element comparison failed");
  }
}

template <bool Pack>
void CoalescedCopy(const ParArray1DRaw<BndId> &bids, std::size_t total, const Mesh *pmesh,
                   const DevExecSpace &exec) {
  if (CoalescedCopyChunk() != 0) {
    CoalescedChunkCopy<Pack>(bids, total, exec);
    return;
  }

  const int nteams_per_buffer =
      CoalescedTeamsPerBuffer(pmesh, static_cast<int>(bids.size()));
  if (nteams_per_buffer == 1) {
    Kokkos::parallel_for(
        PARTHENON_AUTO_LABEL,
        Kokkos::TeamPolicy<DevExecSpace>(exec, bids.size(), Kokkos::AUTO),
        KOKKOS_LAMBDA(team_mbr_t team) {
          const int b = team.league_rank();
          if (bids[b].buf_allocated) {
            Real *combined = &(bids[b].coalesced_buf(bids[b].start_idx()));
            Real *individual = &(bids[b].buf(0));
            Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team, bids[b].size()),
                                 [&](const int i) {
                                   if constexpr (Pack)
                                     combined[i] = individual[i];
                                   else
                                     individual[i] = combined[i];
                                 });
          }
        });
    return;
  }

  Kokkos::parallel_for(
      PARTHENON_AUTO_LABEL,
      Kokkos::TeamPolicy<DevExecSpace>(exec, bids.size() * nteams_per_buffer,
                                       Kokkos::AUTO),
      KOKKOS_LAMBDA(team_mbr_t team) {
        const int b = team.league_rank() / nteams_per_buffer;
        const int bteam = team.league_rank() % nteams_per_buffer;
        if (bids[b].buf_allocated) {
          const std::size_t size = bids[b].size();
          Real *combined = &(bids[b].coalesced_buf(bids[b].start_idx()));
          Real *individual = &(bids[b].buf(0));
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange<>(team, size * bteam / nteams_per_buffer,
                                        size * (bteam + 1) / nteams_per_buffer),
              [&](const int i) {
                if constexpr (Pack)
                  combined[i] = individual[i];
                else
                  individual[i] = combined[i];
              });
        }
      });
}

} // namespace

//----------------------------------------------------------------------------------------
CoalescedBuffer::CoalescedBuffer(bool sender, int partition, int other_rank,
                                 BoundaryType b_type, mpi_comm_t comm, Mesh *pmesh)
    : sender(sender), partition(partition), other_rank(other_rank), b_type(b_type),
      comm(comm), pmesh(pmesh), current_size(0),
      coalesced_comm_buffer(
          2 * partition, sender ? Globals::my_rank : other_rank,
          sender ? other_rank : Globals::my_rank, comm,
          [](int size) { return buf_t("Combined Buffer", 2 * size); }, true),
      sparse_status_buffer(
          2 * partition + 1, sender ? Globals::my_rank : other_rank,
          sender ? other_rank : Globals::my_rank, comm,
          [](int size) { return std::vector<int>(size); }, true) {}

//----------------------------------------------------------------------------------------
ParArray1DRaw<BndId> &CoalescedBuffer::GetBndIdsOnDevice(const std::set<Uid_t> &vars,
                                                         int *pcomb_size) {
  const auto &var_set = vars.size() == 0 ? all_vars : vars;
  int nbnd_id{0};
  std::size_t comb_size{0};
  for (auto uid : var_set) {
    // Skip this variable if it is not communicated in this BoundaryType
    if (coalesced_info_buf.count(uid) == 0) continue;
    nbnd_id += coalesced_info_buf.at(uid).size();
    for (auto &[bnd_id, pvbbuf] : coalesced_info_buf.at(uid)) {
      auto buf_state = pvbbuf->GetState();
      if ((buf_state == BufferState::sending) || (buf_state == BufferState::received))
        comb_size += bnd_id.size();
    }
  }
  if (pcomb_size != nullptr) *pcomb_size = comb_size;

  bool updated = false;
  if (comb_size > coalesced_comm_buffer.buffer().size()) {
    PARTHENON_REQUIRE(
        sender, "Something bad is going on if we are doing this on a receiving buffer.");
    coalesced_comm_buffer.Allocate(comb_size);
    updated = true;
  }

  if (bnd_ids_device_map.count(var_set) == 0)
    bnd_ids_device_map.emplace(std::make_pair(
        var_set, ParArray1DRaw<BndId>(parthenon::ViewOfViewAlloc("bnd_id"), nbnd_id)));
  auto &bnd_ids_device = bnd_ids_device_map.at(var_set);
  if (bnd_ids_host_map.count(var_set) == 0)
    bnd_ids_host_map.emplace(
        std::make_pair(var_set, create_view_of_view_mirror(bnd_ids_device)));
  auto &bnd_ids_host = bnd_ids_host_map.at(var_set);

  if (nbnd_id != bnd_ids_device.size()) {
    bnd_ids_device = ParArray1DRaw<BndId>(parthenon::ViewOfViewAlloc("bnd_id"), nbnd_id);
    bnd_ids_host = create_view_of_view_mirror(bnd_ids_device);
    updated = true;
  }

  int idx{0};
  std::size_t c_buf_idx{0}; // Index at which v-b buffer starts in combined buffer
  for (auto uid : var_set) {
    // Skip this variable if it is not communicated in this BoundaryType
    if (coalesced_info_buf.count(uid) == 0) continue;
    for (auto &[bnd_id, pvbbuf] : coalesced_info_buf.at(uid)) {
      auto &bid_h = bnd_ids_host[idx];
      auto buf_state = pvbbuf->GetState();
      PARTHENON_REQUIRE(buf_state != BufferState::stale,
                        "Trying to work with a stale buffer.");

      const bool alloc =
          (buf_state == BufferState::sending) || (buf_state == BufferState::received);

      // Test if this boundary has changed
      if (!bid_h.SameBVChannel(bnd_id) || (bid_h.buf_allocated != alloc) ||
          (bid_h.start_idx() != c_buf_idx) ||
          !UsingSameResource(bid_h.buf, pvbbuf->buffer()) ||
          bid_h.coalesced_buf.data() != coalesced_comm_buffer.buffer().data()) {
        updated = true;
        bid_h = bnd_id;
        bid_h.buf_allocated = alloc;
        bid_h.start_idx() = c_buf_idx;
        bid_h.coalesced_buf = coalesced_comm_buffer.buffer();
        if (bid_h.buf_allocated) bid_h.buf = pvbbuf->buffer();
      }
      if (bid_h.buf_allocated) c_buf_idx += bid_h.size();
      idx++;
    }
  }
  if (updated) Kokkos::deep_copy(bnd_ids_device, bnd_ids_host);
  return bnd_ids_device;
}

//----------------------------------------------------------------------------------------
void CoalescedBuffer::PackAndSend(const std::set<Uid_t> &vars) {
  PARTHENON_REQUIRE(coalesced_comm_buffer.IsAvailableForWrite(),
                    "Trying to write to a buffer that is in use.");
  int comb_size;
  auto &bids = GetBndIdsOnDevice(vars, &comb_size);
  CoalescedCopy<true>(bids, comb_size, pmesh, DevExecSpace());
#ifdef MPI_PARALLEL
  Kokkos::fence();
#endif
  coalesced_comm_buffer.Send(false, comb_size);

  // Send the sparse null info as well
  const bool dense_static = DenseStaticComms() && bids.size() > 0;
  if (bids.size() != sparse_status_buffer.buffer().size()) {
    sparse_status_buffer.Allocate(bids.size());
  }

  const auto &var_set = vars.size() == 0 ? all_vars : vars;
  auto &stat = sparse_status_buffer.buffer();
  int idx{0};
  for (auto uid : var_set) {
    // Skip this variable if it is not communicated in this BoundaryType
    if (coalesced_info_buf.count(uid) == 0) continue;
    for (auto &[bnd_id, pvbbuf] : coalesced_info_buf.at(uid)) {
      const auto state = pvbbuf->GetState();
      PARTHENON_REQUIRE(state == BufferState::sending ||
                            state == BufferState::sending_null,
                        "Bad state.");
      const int send_type = (state == BufferState::sending);
      PARTHENON_REQUIRE(!dense_static || send_type == 1,
                        "Dense static communication requires all channels active");
      stat[idx] = send_type;
      ++idx;
    }
  }
  if (!dense_static) sparse_status_buffer.Send();

  // Information in these send buffers is no longer required
  for (auto uid : var_set) {
    // Skip this variable if it is not communicated in this BoundaryType
    if (coalesced_info_buf.count(uid) == 0) continue;
    for (auto &[bnd_id, pvbbuf] : coalesced_info_buf.at(uid)) {
      pvbbuf->Stale();
    }
  }
}

//----------------------------------------------------------------------------------------
bool CoalescedBuffer::TryReceiveAndUnpack(const std::set<Uid_t> &vars) {
  if ((sparse_status_buffer.GetState() == BufferState::received) &&
      (coalesced_comm_buffer.GetState() == BufferState::received))
    return true;

  const auto &var_set = vars.size() == 0 ? all_vars : vars;
  // Make sure the var-boundary buffers are available to write to
  int nbuf{0};
  for (auto uid : var_set) {
    // Skip this variable if it is not communicated in this BoundaryType
    if (coalesced_info_buf.count(uid) == 0) continue;
    for (auto &[bnd_id, pvbbuf] : coalesced_info_buf.at(uid)) {
      if (pvbbuf->GetState() != BufferState::stale) return false;
      nbuf++;
    }
  }

  // Dense communication has no sparse-null information to transmit. The sender verifies
  // that every selected channel is active, so synthesize the all-active status locally.
  if (DenseStaticComms() && nbuf > 0) {
    sparse_status_buffer.Allocate(nbuf);
    auto &dense_status = sparse_status_buffer.buffer();
    std::fill(dense_status.begin(), dense_status.end(), 1);
    sparse_status_buffer.SetReceived();
  }
  auto received_sparse = sparse_status_buffer.TryReceive();
  auto received = coalesced_comm_buffer.TryReceive();
  if (!received || !received_sparse) return false;

  // Allocate and free buffers as required
  int idx{0};
  auto &stat = sparse_status_buffer.buffer();
  for (auto uid : var_set) {
    // Skip this variable if it is not communicated in this BoundaryType
    if (coalesced_info_buf.count(uid) == 0) continue;
    for (auto &[bnd_id, pvbbuf] : coalesced_info_buf.at(uid)) {
      if (stat[idx] == 1) {
        pvbbuf->SetReceived();
        if (!pvbbuf->IsActive()) pvbbuf->Allocate();
      } else {
        pvbbuf->SetReceivedNull();
        if (pvbbuf->IsActive()) pvbbuf->Free();
      }
      idx++;
    }
  }

  int comb_size;
  auto &bids = GetBndIdsOnDevice(vars, &comb_size);
  CoalescedCopy<false>(bids, comb_size, pmesh, DevExecSpace());
  coalesced_comm_buffer.Stale();
  sparse_status_buffer.Stale();

  return true;
}

//----------------------------------------------------------------------------------------
void CoalescedBuffer::InitializeNCCLBuffers() {
#ifdef ENABLE_NCCL
  if (current_size > 0) coalesced_comm_buffer.Allocate(current_size);
  nccl_status_buffer = BufArray1D<int>(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "NCCL sparse status"),
      TotalBuffers());
  nccl_status_host = Kokkos::create_mirror_view(nccl_status_buffer);
#else
  PARTHENON_FAIL("NCCL buffers requested in a build without NCCL support.");
#endif
}

//----------------------------------------------------------------------------------------
bool CoalescedBuffer::SetNCCLDirectBuffer(var_buf_t *buffer, BndInfo *bnd_info) {
  for (auto &[uid, entries] : coalesced_info_buf) {
    for (auto &[bnd_id, boundary_buffer] : entries) {
      if (boundary_buffer != buffer) continue;
      const std::size_t begin = bnd_id.start_idx();
      const std::size_t end = begin + bnd_id.size();
      PARTHENON_REQUIRE(end <= coalesced_comm_buffer.buffer().size(),
                        "Direct NCCL boundary buffer exceeds its coalesced allocation.");
      bnd_info->direct_buf = coalesced_comm_buffer.buffer().data() + begin;
      return true;
    }
  }
  return false;
}

//----------------------------------------------------------------------------------------
void CoalescedBuffer::PrepareNCCL(const std::set<Uid_t> &vars, const DevExecSpace &exec) {
#ifdef ENABLE_NCCL
  PARTHENON_REQUIRE(sender, "Only sender buffers can be prepared for NCCL.");
  PARTHENON_REQUIRE(!nccl_prepared, "NCCL buffer was prepared twice.");

  nccl_vars = vars;
  int idx{0};
  for (auto &[uid, entries] : coalesced_info_buf) {
    const bool selected = vars.empty() || vars.count(uid) > 0;
    for (auto &[bnd_id, pvbbuf] : entries) {
      const auto state = pvbbuf->GetState();
      nccl_status_host(idx++) = selected && state == BufferState::sending ? 1 : 0;
    }
  }

  DevExecSpace default_exec;
  nccl_pack_ready.Record(default_exec.cuda_stream());
  nccl_pack_ready.Wait(exec.cuda_stream());
  Kokkos::deep_copy(exec, nccl_status_buffer, nccl_status_host);

  nccl_prepared = true;
#else
  PARTHENON_FAIL("NCCL preparation requested in a build without NCCL support.");
#endif
}

//----------------------------------------------------------------------------------------
void CoalescedBuffer::ApplyNCCLStatusAndUnpack(const std::set<Uid_t> &vars,
                                               const DevExecSpace &exec) {
  (void)exec;
#ifdef ENABLE_NCCL
  PARTHENON_REQUIRE(!sender, "Only receiver buffers can be unpacked from NCCL.");

  int idx{0};
  for (auto &[uid, entries] : coalesced_info_buf) {
    const bool selected = vars.empty() || vars.count(uid) > 0;
    for (auto &[bnd_id, pvbbuf] : entries) {
      if (selected) {
        if (nccl_status_host(idx) == 1) {
          pvbbuf->SetReceived();
          if (!pvbbuf->IsActive()) pvbbuf->Allocate();
        } else {
          pvbbuf->SetReceivedNull();
          if (pvbbuf->IsActive()) pvbbuf->Free();
        }
      }
      ++idx;
    }
  }

#else
  PARTHENON_FAIL("NCCL unpack requested in a build without NCCL support.");
#endif
}

//----------------------------------------------------------------------------------------
void CoalescedBuffer::FinishNCCLSend() {
#ifdef ENABLE_NCCL
  for (auto &[uid, entries] : coalesced_info_buf) {
    if (!nccl_vars.empty() && nccl_vars.count(uid) == 0) continue;
    for (auto &[bnd_id, pvbbuf] : entries)
      pvbbuf->Stale();
  }
  nccl_vars.clear();
  nccl_prepared = false;
#endif
}

//----------------------------------------------------------------------------------------
void CoalescedBuffer::AddVarBoundary(BndId &bnd_id) {
  auto key = GetChannelKey(bnd_id);
  PARTHENON_REQUIRE(pmesh->boundary_comm_map.count(key), "Buffer doesn't exist.");
  var_buf_t *pbuf = &(pmesh->boundary_comm_map.at(key));
  if (bnd_id.start_idx() < 0) bnd_id.start_idx() = current_size;
  coalesced_info_buf[bnd_id.var_id()].push_back(std::make_pair(bnd_id, pbuf));
  current_size = std::max(current_size, bnd_id.start_idx() + bnd_id.size());
  all_vars.insert(bnd_id.var_id());
}

void CoalescedBuffer::AddVarBoundary(MeshBlock *pmb, const NeighborBlock &nb,
                                     const std::shared_ptr<Variable<Real>> &var) {
  // Store both the variable-boundary buffer information and a pointer to the v-b buffer
  // itself associated with var ids
  BndId bnd_id = BndId::GetSend(pmb, nb, var, b_type, partition, -1);
  AddVarBoundary(bnd_id);
}

//----------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------
CoalescedBuffersRank::CoalescedBuffersRank(int o_rank, BoundaryType b_type, bool send,
                                           mpi_comm_t comm, Mesh *pmesh)
    : other_rank(o_rank), b_type(b_type), sender(send), buffers_built(false), comm(comm),
      pmesh(pmesh) {

  int tag = static_cast<int>(GetAssociatedSender(b_type));
  if (sender) {
    message = com_buf_t(tag, Globals::my_rank, other_rank, comm,
                        [](int size) { return std::vector<int>(size); });
  } else {
    message = com_buf_t(
        tag, other_rank, Globals::my_rank, comm,
        [](int size) { return std::vector<int>(size); }, true);
  }
  PARTHENON_REQUIRE(other_rank != Globals::my_rank, "Should only build for other ranks.");
}

//----------------------------------------------------------------------------------------
void CoalescedBuffersRank::AddSendBuffer(int partition, MeshBlock *pmb,
                                         const NeighborBlock &nb,
                                         const std::shared_ptr<Variable<Real>> &var) {
  if (coalesced_bufs.count(partition) == 0)
    coalesced_bufs.emplace(
        std::make_pair(partition, CoalescedBuffer(true, partition, other_rank, b_type,
                                                  comm, pmb->pmy_mesh)));

  auto &coal_buf = coalesced_bufs.at(partition);
  coal_buf.AddVarBoundary(pmb, nb, var);
}

//----------------------------------------------------------------------------------------
bool CoalescedBuffersRank::TryReceiveBufInfo() {
  PARTHENON_REQUIRE(!sender, "Trying to receive on a combined sender.");
  if (buffers_built) return buffers_built;

  bool received = message.TryReceive();
  if (received) {
    auto &mess_buf = message.buffer();
    int npartitions = mess_buf[0];
    // Unpack into per combined buffer information
    int idx{nglobal};

    for (int p = 0; p < npartitions; ++p) {
      const int partition = mess_buf[idx++];
      const int nbuf = mess_buf[idx++];
      const int total_size = mess_buf[idx++];

      // Create the new partition
      coalesced_bufs.emplace(std::make_pair(
          partition, CoalescedBuffer(false, partition, other_rank, b_type, comm, pmesh)));
      auto &coal_buf = coalesced_bufs.at(partition);

      for (int b = 0; b < nbuf; ++b) {
        BndId bnd_id(&(mess_buf[idx]));
        coal_buf.AddVarBoundary(bnd_id);
        idx += BndId::NDAT;
      }
    }
    message.Stale();

    buffers_built = true;
    return true;
  }
  return false;
}

//----------------------------------------------------------------------------------------
void CoalescedBuffersRank::ResolveAndSendBufInfo() {
  // First calculate the total size of the message
  int total_buffers{0};
  for (auto &[partition, coalesced_buf] : coalesced_bufs)
    total_buffers += coalesced_buf.TotalBuffers();
  std::size_t total_partitions = coalesced_bufs.size();

  std::size_t mesg_size =
      nglobal + nper_part * total_partitions + BndId::NDAT * total_buffers;
  message.Allocate(mesg_size);

  auto &mess_buf = message.buffer();
  mess_buf[0] = total_partitions;

  // Pack the data
  int idx{nglobal};
  for (auto &[partition, coalesced_buf] : coalesced_bufs) {
    mess_buf[idx++] = partition;                    // Used as the comm tag
    mess_buf[idx++] = coalesced_buf.TotalBuffers(); // Number of buffers
    mess_buf[idx++] =
        coalesced_buf.current_size; // combined size of buffers (now probably unused)
    for (auto &[uid, v] : coalesced_buf.coalesced_info_buf) {
      for (auto &[bnd_id, pbvbuf] : v) {
        bnd_id.Serialize(&(mess_buf[idx]));
        idx += BndId::NDAT;
      }
    }
  }

  message.Send();

  buffers_built = true;
}

//----------------------------------------------------------------------------------------
void CoalescedBuffersRank::PackAndSend(MeshData<Real> *pmd) {
  PARTHENON_REQUIRE(buffers_built,
                    "Trying to send combined buffers before they have been built");
  if (coalesced_bufs.count(pmd->partition)) {
    coalesced_bufs.at(pmd->partition).PackAndSend(pmd->GetUids());
  } else {
    PARTHENON_FAIL("Trying to do coalesced communication using a partition that is "
                   "unregistered (i.e. not a default partition).");
  }

  return;
}

//----------------------------------------------------------------------------------------
bool CoalescedBuffersRank::IsAvailableForWrite(MeshData<Real> *pmd) {
  PARTHENON_REQUIRE(sender, "Shouldn't be checking this on non-sender.");
  if (coalesced_bufs.count(pmd->partition) == 0) return true;
  return coalesced_bufs.at(pmd->partition).IsAvailableForWrite();
}

//----------------------------------------------------------------------------------------
bool CoalescedBuffersRank::TryReceiveAndUnpack(MeshData<Real> *pmd, int partition) {
  PARTHENON_REQUIRE(buffers_built,
                    "Trying to recv combined buffers before they have been built");
  PARTHENON_REQUIRE(coalesced_bufs.count(partition) > 0,
                    "Trying to receive on a non-existent combined receive buffer.");
  return coalesced_bufs.at(partition).TryReceiveAndUnpack(pmd->GetUids());
}

//----------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------
CoalescedComms::CoalescedComms(Mesh *pmesh) : pmesh(pmesh) {
  // TODO(LFR): Switch to a different communicator for each BoundaryType pair
  for (auto b_type :
       {BoundaryType::any, BoundaryType::flxcor_send, BoundaryType::gmg_same,
        BoundaryType::gmg_restrict_send, BoundaryType::gmg_prolongate_send}) {
    auto &comm = comms[b_type];
#ifdef MPI_PARALLEL
    PARTHENON_MPI_CHECK(MPI_Comm_dup(MPI_COMM_WORLD, &comm));
#else
    comm = 0;
#endif
  }

  if (pmesh->UseNCCLCoalescedComms()) {
#ifdef ENABLE_NCCL
    nccl_exec = SpaceInstance<DevExecSpace>::create();
#else
    PARTHENON_FAIL(
        "coalesced_comm_backend=nccl requested without PARTHENON_ENABLE_NCCL.");
#endif
  }
}

//----------------------------------------------------------------------------------------
CoalescedComms::~CoalescedComms() {
#ifdef ENABLE_NCCL
  if (pmesh->UseNCCLCoalescedComms()) {
    const auto stream = nccl_exec.cuda_stream();
    nccl_exec.fence();
    for (auto &[b_type, state] : nccl_states)
      PARTHENON_NCCL_CHECK(ncclCommDestroy(state->comm));
    nccl_states.clear();
    // Release Kokkos' reference before destroying the externally-created stream.
    // Otherwise the Kokkos::Cuda destructor tries to synchronize a destroyed stream.
    nccl_exec = DevExecSpace();
    PARTHENON_CUDA_EVENT_CHECK(cudaStreamDestroy(stream));
  }
#endif
#ifdef MPI_PARALLEL
  for (auto &[b_type, comm] : comms)
    PARTHENON_MPI_CHECK(MPI_Comm_free(&comm));
#endif
}

//----------------------------------------------------------------------------------------
void CoalescedComms::clear() {
#ifdef ENABLE_NCCL
  if (!nccl_states.empty()) {
    nccl_exec.fence();
    for (auto &[b_type, state] : nccl_states) {
      state->in_flight = false;
      state->unpack_recorded = false;
      state->vars.clear();
    }
  }
#endif
  bool can_delete;
  int iter{0};
  const int max_iters = 1e8;
  do {
    can_delete = true;
    for (auto &[p, cbr] : coalesced_send_buffers) {
      can_delete = cbr.message.IsAvailableForWrite() && can_delete;
      for (auto &[r, cbrp] : cbr.coalesced_bufs) {
        can_delete = cbrp.IsAvailableForWrite() && can_delete;
      }
    }
    iter++;
  } while (!can_delete && iter < max_iters);
  if (iter >= max_iters) PARTHENON_FAIL("Waited too long to clear CoalescedComms.");

  coalesced_send_buffers.clear();
  coalesced_recv_buffers.clear();
}

//----------------------------------------------------------------------------------------
void CoalescedComms::AddSendBuffer(int partition, MeshBlock *pmb, const NeighborBlock &nb,
                                   const std::shared_ptr<Variable<Real>> &var,
                                   BoundaryType b_type) {
  if (coalesced_send_buffers.count({nb.rank, b_type}) == 0)
    coalesced_send_buffers.emplace(
        std::make_pair(std::make_pair(nb.rank, b_type),
                       CoalescedBuffersRank(nb.rank, b_type, true,
                                            comms[GetAssociatedSender(b_type)], pmesh)));
  coalesced_send_buffers.at({nb.rank, b_type}).AddSendBuffer(partition, pmb, nb, var);
}

//----------------------------------------------------------------------------------------
void CoalescedComms::AddRecvBuffer(MeshBlock *pmb, const NeighborBlock &nb,
                                   const std::shared_ptr<Variable<Real>>,
                                   BoundaryType b_type) {
  // We don't actually know enough here to register this particular buffer, but we do
  // know that it's existence implies that we need to receive a message from the
  // neighbor block rank eventually telling us the details
  if (coalesced_recv_buffers.count({nb.rank, b_type}) == 0)
    coalesced_recv_buffers.emplace(
        std::make_pair(std::make_pair(nb.rank, b_type),
                       CoalescedBuffersRank(nb.rank, b_type, false,
                                            comms[GetAssociatedSender(b_type)], pmesh)));
}

//----------------------------------------------------------------------------------------
void CoalescedComms::ResolveAndSendSendBuffers() {
  for (auto &[id, buf] : coalesced_send_buffers)
    buf.ResolveAndSendBufInfo();
}

//----------------------------------------------------------------------------------------
void CoalescedComms::ReceiveBufferInfo() {
  constexpr std::int64_t max_it = 1e10;
  std::vector<bool> received(coalesced_recv_buffers.size(), false);
  bool all_received;
  std::int64_t receive_iters = 0;
  do {
    all_received = true;
    for (auto &[id, buf] : coalesced_recv_buffers)
      all_received = buf.TryReceiveBufInfo() && all_received;
    receive_iters++;
  } while (!all_received && receive_iters < max_it);
  PARTHENON_REQUIRE(
      receive_iters < max_it,
      "Too many iterations waiting to receive boundary communication buffers.");
}

//----------------------------------------------------------------------------------------
bool CoalescedComms::IsAvailableForWrite(MeshData<Real> *pmd, BoundaryType b_type) {
  if (UsesNCCL(b_type)) {
#ifdef ENABLE_NCCL
    const auto canonical = GetAssociatedSender(b_type);
    return !nccl_states.at(canonical)->in_flight;
#endif
  }
  bool available{true};
  for (int rank = 0; rank < Globals::nranks; ++rank) {
    if (coalesced_send_buffers.count({rank, b_type})) {
      available =
          available && coalesced_send_buffers.at({rank, b_type}).IsAvailableForWrite(pmd);
    }
  }
  return available;
}

//----------------------------------------------------------------------------------------
bool CoalescedComms::UsesNCCL(BoundaryType b_type) const {
  if (!pmesh->UseNCCLCoalescedComms()) return false;
  const auto canonical = GetAssociatedSender(b_type);
  // Restriction/prolongation GMG exchanges use custom task graphs that are not yet
  // region-coordinated. Keep those exchanges on the tagged MPI path.
  const bool supported = canonical == BoundaryType::any ||
                         canonical == BoundaryType::flxcor_send ||
                         canonical == BoundaryType::gmg_same;
#ifdef ENABLE_NCCL
  return supported && nccl_states.count(canonical) > 0;
#else
  return supported;
#endif
}

//----------------------------------------------------------------------------------------
void CoalescedComms::SetNCCLDirectBuffers(BvarsSubCache_t *cache, BoundaryType b_type,
                                          bool sender) {
  auto &buffers = sender ? coalesced_send_buffers : coalesced_recv_buffers;
  for (std::size_t i = 0; i < cache->buf_vec.size(); ++i) {
    for (auto &[key, rank_buffers] : buffers) {
      if (key.second != b_type) continue;
      bool found = false;
      for (auto &[partition, buffer] : rank_buffers.coalesced_bufs) {
        if (buffer.SetNCCLDirectBuffer(cache->buf_vec[i], &cache->bnd_info_h(i))) {
          found = true;
          break;
        }
      }
      if (found) break;
    }
  }
  Kokkos::deep_copy(cache->bnd_info, cache->bnd_info_h);
}

//----------------------------------------------------------------------------------------
void CoalescedComms::InitializeNCCLBuffers() {
  if (!pmesh->UseNCCLCoalescedComms()) return;
#ifdef ENABLE_NCCL
  // Finish the MPI metadata handshake before starting NCCL proxy threads. In
  // particular, an active NCCL communicator can starve progress of the nonblocking
  // metadata messages when several ranks share one host.
  for (auto b_type :
       {BoundaryType::any, BoundaryType::flxcor_send, BoundaryType::gmg_same}) {
    if (nccl_states.count(b_type) > 0) continue;

    int active = 0;
    for (const auto &entry : coalesced_send_buffers)
      active = active || GetAssociatedSender(entry.first.second) == b_type;
    for (const auto &entry : coalesced_recv_buffers)
      active = active || GetAssociatedSender(entry.first.second) == b_type;
    PARTHENON_MPI_CHECK(
        MPI_Allreduce(MPI_IN_PLACE, &active, 1, MPI_INT, MPI_MAX, comms.at(b_type)));
    if (active == 0) continue;

    auto state = std::make_unique<NCCLExchangeState>();
    ncclUniqueId id;
    if (Globals::my_rank == 0) PARTHENON_NCCL_CHECK(ncclGetUniqueId(&id));
    PARTHENON_MPI_CHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, comms.at(b_type)));
    PARTHENON_NCCL_CHECK(
        ncclCommInitRank(&state->comm, Globals::nranks, id, Globals::my_rank));
    nccl_states.emplace(b_type, std::move(state));
  }
#endif
  for (auto &[key, rank_buffers] : coalesced_send_buffers)
    for (auto &[partition, buffer] : rank_buffers.coalesced_bufs)
      buffer.InitializeNCCLBuffers();
  for (auto &[key, rank_buffers] : coalesced_recv_buffers)
    for (auto &[partition, buffer] : rank_buffers.coalesced_bufs)
      buffer.InitializeNCCLBuffers();
}

//----------------------------------------------------------------------------------------
void CoalescedComms::PrepareNCCLSend(MeshData<Real> *pmd, BoundaryType b_type) {
  PARTHENON_REQUIRE(UsesNCCL(b_type), "Preparing NCCL on an unsupported boundary type.");
#ifdef ENABLE_NCCL
  auto &state = *nccl_states.at(GetAssociatedSender(b_type));
  const auto &vars = pmd->GetUids();
  if (!state.vars.empty() && !vars.empty())
    PARTHENON_REQUIRE(
        state.vars == vars,
        "All partitions in an NCCL exchange must select the same variables.");
  if (!vars.empty()) state.vars = vars;

  for (int rank = 0; rank < Globals::nranks; ++rank) {
    auto it = coalesced_send_buffers.find({rank, b_type});
    if (it == coalesced_send_buffers.end()) continue;
    auto &rank_buffers = it->second;
    auto buffer = rank_buffers.coalesced_bufs.find(pmd->partition);
    if (buffer != rank_buffers.coalesced_bufs.end())
      buffer->second.PrepareNCCL(vars, nccl_exec);
  }
#endif
}

//----------------------------------------------------------------------------------------
TaskStatus CoalescedComms::LaunchNCCLExchange(BoundaryType send_type,
                                              BoundaryType recv_type) {
  PARTHENON_REQUIRE(UsesNCCL(send_type), "Launching an unsupported NCCL exchange.");
#ifdef ENABLE_NCCL
  auto &state = *nccl_states.at(GetAssociatedSender(send_type));
  PARTHENON_REQUIRE(!state.in_flight, "NCCL exchange is already in flight.");
  const auto real_type = std::is_same_v<Real, float> ? ncclFloat32 : ncclFloat64;

  PARTHENON_NCCL_CHECK(ncclGroupStart());
  for (int rank = 0; rank < Globals::nranks; ++rank) {
    auto it = coalesced_recv_buffers.find({rank, recv_type});
    if (it == coalesced_recv_buffers.end()) continue;
    for (auto &[partition, buffer] : it->second.coalesced_bufs) {
      PARTHENON_NCCL_CHECK(ncclRecv(buffer.coalesced_comm_buffer.buffer().data(),
                                    buffer.current_size, real_type, rank, state.comm,
                                    nccl_exec.cuda_stream()));
      PARTHENON_NCCL_CHECK(ncclRecv(buffer.nccl_status_buffer.data(),
                                    buffer.TotalBuffers(), ncclInt32, rank, state.comm,
                                    nccl_exec.cuda_stream()));
    }
  }
  for (int rank = 0; rank < Globals::nranks; ++rank) {
    auto it = coalesced_send_buffers.find({rank, send_type});
    if (it == coalesced_send_buffers.end()) continue;
    for (auto &[partition, buffer] : it->second.coalesced_bufs) {
      PARTHENON_REQUIRE(buffer.nccl_prepared,
                        "NCCL send buffer was not prepared before exchange.");
      PARTHENON_NCCL_CHECK(ncclSend(buffer.coalesced_comm_buffer.buffer().data(),
                                    buffer.current_size, real_type, rank, state.comm,
                                    nccl_exec.cuda_stream()));
      PARTHENON_NCCL_CHECK(ncclSend(buffer.nccl_status_buffer.data(),
                                    buffer.TotalBuffers(), ncclInt32, rank, state.comm,
                                    nccl_exec.cuda_stream()));
    }
  }
  PARTHENON_NCCL_CHECK(ncclGroupEnd());

  for (int rank = 0; rank < Globals::nranks; ++rank) {
    auto it = coalesced_recv_buffers.find({rank, recv_type});
    if (it == coalesced_recv_buffers.end()) continue;
    for (auto &[partition, buffer] : it->second.coalesced_bufs)
      Kokkos::deep_copy(nccl_exec, buffer.nccl_status_host, buffer.nccl_status_buffer);
  }
  state.transfer_complete.Record(nccl_exec.cuda_stream());
  state.in_flight = true;
  state.unpack_recorded = false;
  return TaskStatus::complete;
#else
  return TaskStatus::fail;
#endif
}

//----------------------------------------------------------------------------------------
TaskStatus CoalescedComms::PollNCCLExchange(BoundaryType send_type) {
#ifdef ENABLE_NCCL
  auto &state = *nccl_states.at(GetAssociatedSender(send_type));
  if (!state.in_flight) return TaskStatus::complete;
  return state.transfer_complete.Query() ? TaskStatus::complete : TaskStatus::incomplete;
#else
  return TaskStatus::fail;
#endif
}

//----------------------------------------------------------------------------------------
TaskStatus CoalescedComms::FinalizeNCCLExchange(BoundaryType send_type,
                                                BoundaryType recv_type) {
#ifdef ENABLE_NCCL
  auto &state = *nccl_states.at(GetAssociatedSender(send_type));
  PARTHENON_REQUIRE(state.in_flight && state.transfer_complete.Query(),
                    "Finalizing NCCL before transfer completion.");

  for (int rank = 0; rank < Globals::nranks; ++rank) {
    auto send = coalesced_send_buffers.find({rank, send_type});
    if (send != coalesced_send_buffers.end())
      for (auto &[partition, buffer] : send->second.coalesced_bufs)
        buffer.FinishNCCLSend();

    auto recv = coalesced_recv_buffers.find({rank, recv_type});
    if (recv != coalesced_recv_buffers.end())
      for (auto &[partition, buffer] : recv->second.coalesced_bufs)
        buffer.ApplyNCCLStatusAndUnpack(state.vars, nccl_exec);
  }
  state.unpack_complete.Record(nccl_exec.cuda_stream());
  state.unpack_recorded = true;
  state.in_flight = false;
  state.vars.clear();
  return TaskStatus::complete;
#else
  return TaskStatus::fail;
#endif
}

//----------------------------------------------------------------------------------------
void CoalescedComms::WaitForNCCLUnpack(BoundaryType send_type, const DevExecSpace &exec) {
#ifdef ENABLE_NCCL
  auto &state = *nccl_states.at(GetAssociatedSender(send_type));
  if (state.unpack_recorded) state.unpack_complete.Wait(exec.cuda_stream());
#endif
}

//----------------------------------------------------------------------------------------
void CoalescedComms::PackAndSend(MeshData<Real> *pmd, BoundaryType b_type) {
  for (int rank = 0; rank < Globals::nranks; ++rank) {
    if (coalesced_send_buffers.count({rank, b_type})) {
      coalesced_send_buffers.at({rank, b_type}).PackAndSend(pmd);
    }
  }
}

//----------------------------------------------------------------------------------------
bool CoalescedComms::TryReceiveAny(MeshData<Real> *pmd, BoundaryType b_type) {
#ifdef MPI_PARALLEL
  bool all_received = true;
  for (int rank = 0; rank < Globals::nranks; ++rank) {
    if (coalesced_recv_buffers.count({rank, b_type})) {
      auto &coal_bufs = coalesced_recv_buffers.at({rank, b_type});
      for (auto &[partition, coal_buf] : coal_bufs.coalesced_bufs) {
        bool received = coal_buf.TryReceiveAndUnpack(pmd->GetUids());
        all_received = all_received && received;
      }
    }
  }
  return all_received;
#else
  // There are no coalesced comms without MPI, so all buffers
  // are automatically received
  return true;
#endif
}
} // namespace parthenon
