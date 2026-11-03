//========================================================================================
// (C) (or copyright) 2024-2026. Triad National Security, LLC. All rights reserved.
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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <set>
#include <string>
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
#include "utils/comms_trace.hpp"
#include "utils/communication_buffer.hpp"

namespace parthenon {

namespace {
// All comm lanes of a sender BoundaryType share a communicator, so combine the partition
// tag and the lane into a unique MPI tag for the data message (the sparse status message
// uses this tag + 1)
int CoalescedDataTag(Mesh *pmesh, BoundaryType b_type, int partition_tag, int lane) {
  const int nlanes = pmesh->GetNumberOfCommLanes(GetAssociatedSender(b_type));
  return 2 * (nlanes * partition_tag + lane);
}

// Copy between the var-boundary buffers and the coalesced buffer, splitting each buffer
// among enough teams that the kernel launches at least
// minimum_number_of_teams_for_boundary_kernel teams
template <bool TO_COALESCED>
void CopyCoalesced(Mesh *pmesh, ParArray1DRaw<BndId> &bids) {
  const int nbound = bids.size();
  if (nbound == 0) return;
  const int nteams_per_buffer = GetNteamsPerBoundaryBuffer(pmesh, nbound);
  Kokkos::parallel_for(
      PARTHENON_AUTO_LABEL,
      Kokkos::TeamPolicy<>(parthenon::DevExecSpace(), nbound * nteams_per_buffer,
                           Kokkos::AUTO),
      KOKKOS_LAMBDA(parthenon::team_mbr_t team_member) {
        const int b = team_member.league_rank() / nteams_per_buffer;
        const int bteam = team_member.league_rank() % nteams_per_buffer;
        if (bids[b].buf_allocated) {
          const SplitFlatIndexRangeAmongTeams split(nteams_per_buffer, 1,
                                                    bids[b].size());
          Real *com_buf = &(bids[b].coalesced_buf(bids[b].start_idx()));
          Real *buf = &(bids[b].buf(0));
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange<>(team_member, split.GetStart(bteam),
                                        split.GetEnd(bteam)),
              [&](const int idx) {
                if constexpr (TO_COALESCED) {
                  com_buf[idx] = buf[idx];
                } else {
                  buf[idx] = com_buf[idx];
                }
              });
        }
      });
}
} // namespace

//----------------------------------------------------------------------------------------
CoalescedBuffer::CoalescedBuffer(bool sender, int partition_tag, int lane, int other_rank,
                                 BoundaryType b_type, mpi_comm_t comm, Mesh *pmesh)
    : sender(sender), partition_tag(partition_tag), lane(lane), other_rank(other_rank),
      b_type(b_type), comm(comm), pmesh(pmesh), current_size(0),
      coalesced_comm_buffer(
          CoalescedDataTag(pmesh, b_type, partition_tag, lane),
          sender ? Globals::my_rank : other_rank, sender ? other_rank : Globals::my_rank,
          comm, [](int size) { return buf_t("Combined Buffer", 2 * size); }, true),
      sparse_status_buffer(
          CoalescedDataTag(pmesh, b_type, partition_tag, lane) + 1,
          sender ? Globals::my_rank : other_rank, sender ? other_rank : Globals::my_rank,
          comm, [](int size) { return std::vector<int>(size); }, true) {}

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
void CoalescedBuffer::TraceOpen(double t0) {
  trace_rec = comms_trace::ExchangeRecord();
  trace_rec.cycle = comms_trace::CurrentCycle();
  trace_rec.build = comms_trace::CurrentBuild();
  trace_rec.in_step = comms_trace::InStep();
  trace_rec.exchange = trace_nexchange++;
  trace_rec.is_send = sender;
  trace_rec.peer = other_rank;
  // Record the sender type on both sides so that sends and receives can be matched
  trace_rec.btype = static_cast<int>(GetAssociatedSender(b_type));
  trace_rec.lane = lane;
  trace_rec.partition_tag = partition_tag;
  trace_rec.t[0] = t0;
  trace_last_poll = t0;
  trace_open = true;
}

void CoalescedBuffer::TraceClose() {
  auto *stored = comms_trace::RecordExchange(trace_rec);
  if (sender) trace_send_rec = stored;
  trace_open = false;
}

//----------------------------------------------------------------------------------------
void CoalescedBuffer::PackAndSend(const std::set<Uid_t> &vars) {
  // The completion of the previous send was never observed
  trace_send_rec = nullptr;
  PARTHENON_REQUIRE(coalesced_comm_buffer.IsAvailableForWrite(),
                    "Trying to write to a buffer that is in use.");
  if (comms_trace::Active()) TraceOpen(comms_trace::Now());
  int comb_size;
  auto &bids = GetBndIdsOnDevice(vars, &comb_size);
  CopyCoalesced<true>(pmesh, bids);
#ifdef MPI_PARALLEL
  const double t_fence = comms_trace::Now();
  Kokkos::fence();
  const double t_fence_done = comms_trace::Now();
  comms_trace::AddFenceTime(t_fence_done - t_fence);
  if (trace_open) trace_rec.t[1] = t_fence_done;
#endif
  coalesced_comm_buffer.Send(false, comb_size);
  if (trace_open) {
    trace_rec.t[2] = comms_trace::Now();
    trace_rec.bytes = static_cast<std::int64_t>(comb_size) * sizeof(Real);
    trace_rec.nbuf = bids.size();
  }

  // Send the sparse null info as well
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
      int send_type = (state == BufferState::sending);
      stat[idx] = send_type;
      ++idx;
    }
  }
  sparse_status_buffer.Send();
  if (trace_open) {
    trace_rec.t[3] = comms_trace::Now();
    TraceClose();
  }

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

  if (trace_open || comms_trace::Active()) {
    const double now = comms_trace::Now();
    if (!trace_open) {
      TraceOpen(now);
    } else {
      trace_rec.max_poll_gap = std::max(trace_rec.max_poll_gap, now - trace_last_poll);
      trace_last_poll = now;
    }
    trace_rec.npolls++;
  }

  auto received_sparse = sparse_status_buffer.TryReceive();
  auto received = coalesced_comm_buffer.TryReceive();
  if (trace_open) {
    const double now = comms_trace::Now();
    auto &t = trace_rec.t;
    if (t[1] < 0.0 && (received_sparse || sparse_status_buffer.ReceiveStarted()))
      t[1] = now;
    if (t[2] < 0.0 && (received || coalesced_comm_buffer.ReceiveStarted())) t[2] = now;
    if (t[3] < 0.0 && received_sparse) t[3] = now;
    if (t[4] < 0.0 && received) t[4] = now;
  }
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
  CopyCoalesced<false>(pmesh, bids);
  coalesced_comm_buffer.Stale();
  sparse_status_buffer.Stale();
  if (trace_open) {
    trace_rec.t[5] = comms_trace::Now();
    trace_rec.bytes = static_cast<std::int64_t>(comb_size) * sizeof(Real);
    trace_rec.nbuf = bids.size();
    TraceClose();
  }

  return true;
}

//----------------------------------------------------------------------------------------
void CoalescedBuffer::AddVarBoundary(BndId &bnd_id) {
  auto key = GetChannelKey(bnd_id);
  PARTHENON_REQUIRE(pmesh->boundary_comm_map.count(key), "Buffer doesn't exist.");
  var_buf_t *pbuf = &(pmesh->boundary_comm_map.at(key));
  coalesced_info_buf[bnd_id.var_id()].push_back(std::make_pair(bnd_id, pbuf));
  current_size += bnd_id.size(); // This will be the maximum size of communication since
                                 // it includes all variables
  all_vars.insert(bnd_id.var_id());
}

void CoalescedBuffer::AddVarBoundary(MeshBlock *pmb, const NeighborBlock &nb,
                                     const std::shared_ptr<Variable<Real>> &var) {
  // Store both the variable-boundary buffer information and a pointer to the v-b buffer
  // itself associated with var ids
  BndId bnd_id = BndId::GetSend(pmb, nb, var, b_type, partition_tag, -1, lane);
  AddVarBoundary(bnd_id);
}

//----------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------
CoalescedBuffersRank::CoalescedBuffersRank(int o_rank, BoundaryType b_type, int lane,
                                           bool send, mpi_comm_t comm, Mesh *pmesh)
    : other_rank(o_rank), b_type(b_type), lane(lane), sender(send), buffers_built(false),
      comm(comm), pmesh(pmesh) {

  // There is a separate communicator for each sender BoundaryType, so the comm lane is
  // sufficient to make the info message tag unique
  int tag = lane;
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
void CoalescedBuffersRank::AddSendBuffer(int partition_tag, MeshBlock *pmb,
                                         const NeighborBlock &nb,
                                         const std::shared_ptr<Variable<Real>> &var) {
  if (coalesced_bufs.count(partition_tag) == 0)
    coalesced_bufs.emplace(std::make_pair(
        partition_tag, CoalescedBuffer(true, partition_tag, lane, other_rank, b_type,
                                       comm, pmb->pmy_mesh)));

  auto &coal_buf = coalesced_bufs.at(partition_tag);
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
      const int partition_tag = mess_buf[idx++];
      const int nbuf = mess_buf[idx++];
      const int total_size = mess_buf[idx++];

      // Create the new partition
      coalesced_bufs.emplace(std::make_pair(
          partition_tag,
          CoalescedBuffer(false, partition_tag, lane, other_rank, b_type, comm, pmesh)));
      auto &coal_buf = coalesced_bufs.at(partition_tag);

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
  for (auto &[partition_tag, coalesced_buf] : coalesced_bufs)
    total_buffers += coalesced_buf.TotalBuffers();
  std::size_t total_partitions = coalesced_bufs.size();

  std::size_t mesg_size =
      nglobal + nper_part * total_partitions + BndId::NDAT * total_buffers;
  message.Allocate(mesg_size);

  auto &mess_buf = message.buffer();
  mess_buf[0] = total_partitions;

  // Pack the data
  int idx{nglobal};
  for (auto &[partition_tag, coalesced_buf] : coalesced_bufs) {
    mess_buf[idx++] = partition_tag;                // Used as the comm tag
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
  // A registered partition may have no boundaries shared with other_rank, in which case
  // there is nothing to send
  const int partition_tag = pmesh->pcoalesced_comms->GetPartitionTag(pmd);
  if (coalesced_bufs.count(partition_tag))
    coalesced_bufs.at(partition_tag).PackAndSend(pmd->GetUids());
}

//----------------------------------------------------------------------------------------
bool CoalescedBuffersRank::IsAvailableForWrite(MeshData<Real> *pmd) {
  PARTHENON_REQUIRE(sender, "Shouldn't be checking this on non-sender.");
  const int partition_tag = pmesh->pcoalesced_comms->GetPartitionTag(pmd);
  if (coalesced_bufs.count(partition_tag) == 0) return true;
  return coalesced_bufs.at(partition_tag).IsAvailableForWrite();
}

//----------------------------------------------------------------------------------------
bool CoalescedBuffersRank::TryReceiveAndUnpack(MeshData<Real> *pmd, int partition_tag) {
  PARTHENON_REQUIRE(buffers_built,
                    "Trying to recv combined buffers before they have been built");
  PARTHENON_REQUIRE(coalesced_bufs.count(partition_tag) > 0,
                    "Trying to receive on a non-existent combined receive buffer.");
  return coalesced_bufs.at(partition_tag).TryReceiveAndUnpack(pmd->GetUids());
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
}

//----------------------------------------------------------------------------------------
CoalescedComms::~CoalescedComms() {
#ifdef MPI_PARALLEL
  for (auto &[b_type, comm] : comms)
    PARTHENON_MPI_CHECK(MPI_Comm_free(&comm));
#endif
}

//----------------------------------------------------------------------------------------
void CoalescedComms::clear() {
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
  partition_tags.clear();
}

//----------------------------------------------------------------------------------------
void CoalescedComms::RegisterPartition(const BlockListPartition *partition) {
  if (partition_tags.count(partition) == 0) {
    const int tag = partition_tags.size();
    partition_tags[partition] = tag;
  }
}

//----------------------------------------------------------------------------------------
int CoalescedComms::GetPartitionTag(MeshData<Real> *pmd) const {
  const auto it = partition_tags.find(pmd->partition.get());
  if (it == partition_tags.end())
    PARTHENON_FAIL("Trying to do coalesced communication using a partition that is "
                   "unregistered (i.e. not a default or multigrid partition).");
  return it->second;
}

//----------------------------------------------------------------------------------------
void CoalescedComms::AddSendBuffer(const BlockListPartition *partition, MeshBlock *pmb,
                                   const NeighborBlock &nb,
                                   const std::shared_ptr<Variable<Real>> &var,
                                   BoundaryType b_type, int lane) {
  PARTHENON_REQUIRE(partition_tags.count(partition),
                    "Partition must be registered before adding buffers.");
  const rank_key_t key{nb.rank, b_type, lane};
  if (coalesced_send_buffers.count(key) == 0)
    coalesced_send_buffers.emplace(std::make_pair(
        key, CoalescedBuffersRank(nb.rank, b_type, lane, true,
                                  comms[GetAssociatedSender(b_type)], pmesh)));
  coalesced_send_buffers.at(key).AddSendBuffer(partition_tags.at(partition), pmb, nb,
                                               var);
}

//----------------------------------------------------------------------------------------
void CoalescedComms::AddRecvBuffer(MeshBlock *pmb, const NeighborBlock &nb,
                                   const std::shared_ptr<Variable<Real>>,
                                   BoundaryType b_type, int lane) {
  // We don't actually know enough here to register this particular buffer, but we do
  // know that it's existence implies that we need to receive a message from the
  // neighbor block rank eventually telling us the details
  const rank_key_t key{nb.rank, b_type, lane};
  if (coalesced_recv_buffers.count(key) == 0)
    coalesced_recv_buffers.emplace(std::make_pair(
        key, CoalescedBuffersRank(nb.rank, b_type, lane, false,
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
void CoalescedComms::PrintCensus() {
  const int build = comms_trace::CurrentBuild();

  // One row per coalesced message: src, dst, b_type, lane, partition_tag, nbuf, size
  // (in Reals, assuming every variable is communicated and allocated)
  constexpr int NR = 7;
  std::vector<int> local;
  for (auto &[key, cbr] : coalesced_send_buffers) {
    for (auto &[ptag, cb] : cbr.coalesced_bufs) {
      local.insert(local.end(),
                   {Globals::my_rank, cb.other_rank, static_cast<int>(cb.b_type), cb.lane,
                    ptag, static_cast<int>(cb.TotalBuffers()), cb.current_size});
    }
  }
  const int nranks = Globals::nranks;
  std::vector<int> all = local;
#ifdef MPI_PARALLEL
  int nlocal = local.size();
  std::vector<int> counts(nranks), displs(nranks);
  PARTHENON_MPI_CHECK(
      MPI_Gather(&nlocal, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD));
  if (Globals::my_rank == 0) {
    int tot = 0;
    for (int r = 0; r < nranks; ++r) {
      displs[r] = tot;
      tot += counts[r];
    }
    all.resize(tot);
  }
  PARTHENON_MPI_CHECK(MPI_Gatherv(local.data(), nlocal, MPI_INT, all.data(),
                                  counts.data(), displs.data(), MPI_INT, 0,
                                  MPI_COMM_WORLD));
#endif
  if (Globals::my_rank != 0) return;

  auto btype_name = [](int b) {
    switch (static_cast<BoundaryType>(b)) {
    case BoundaryType::local:
      return "local";
    case BoundaryType::nonlocal:
      return "nonlocal";
    case BoundaryType::any:
      return "any";
    case BoundaryType::flxcor_send:
      return "flxcor_send";
    case BoundaryType::flxcor_recv:
      return "flxcor_recv";
    case BoundaryType::gmg_same:
      return "gmg_same";
    case BoundaryType::gmg_restrict_send:
      return "gmg_restrict_send";
    case BoundaryType::gmg_restrict_recv:
      return "gmg_restrict_recv";
    case BoundaryType::gmg_prolongate_send:
      return "gmg_prolongate_send";
    case BoundaryType::gmg_prolongate_recv:
      return "gmg_prolongate_recv";
    }
    return "unknown";
  };
  const double kib = sizeof(Real) / 1024.0;
  const int nmsg = all.size() / NR;
  auto row = [&](int m) { return &all[NR * m]; };
  auto on_node = [](const int *r) {
    return comms_trace::NodeOfRank(r[0]) == comms_trace::NodeOfRank(r[1]);
  };

  std::printf("\n==== Coalesced comms census (build %d): %d ranks on %d nodes, %d "
              "messages per exchange of all variables ====\n",
              build, nranks, comms_trace::NumNodes(), nmsg);
  std::printf("Each message is accompanied by a status message of nbuf ints. Sizes assume "
              "every variable is communicated.\n");
  for (int n = 0; n < comms_trace::NumNodes(); ++n) {
    std::printf("  node %d:", n);
    std::string host;
    for (int r = 0; r < nranks; ++r) {
      if (comms_trace::NodeOfRank(r) == n) {
        std::printf(" %d", r);
        host = comms_trace::HostOfRank(r);
      }
    }
    std::printf("  (%s)\n", host.c_str());
  }

  // Per rank, per boundary type totals
  struct Tot {
    int nmsg[2]{0, 0};
    double kib[2]{0.0, 0.0};
    int nbuf[2]{0, 0};
    double min_kib{1e300}, max_kib{0.0};
  };
  std::map<std::pair<int, int>, Tot> tots;
  for (int m = 0; m < nmsg; ++m) {
    const int *r = row(m);
    auto &t = tots[{r[0], r[2]}];
    const int off = on_node(r) ? 0 : 1;
    const double sz = r[6] * kib;
    t.nmsg[off]++;
    t.kib[off] += sz;
    t.nbuf[off] += r[5];
    t.min_kib = std::min(t.min_kib, sz);
    t.max_kib = std::max(t.max_kib, sz);
  }
  std::printf("\n%6s %-20s | %8s %10s %8s | %8s %10s %8s | %10s %10s\n", "rank", "btype",
              "on_msgs", "on_KiB", "on_nbuf", "off_msgs", "off_KiB", "off_nbuf",
              "min_KiB", "max_KiB");
  for (const auto &[k, t] : tots) {
    std::printf("%6d %-20s | %8d %10.1f %8d | %8d %10.1f %8d | %10.2f %10.2f\n", k.first,
                btype_name(k.second), t.nmsg[0], t.kib[0], t.nbuf[0], t.nmsg[1],
                t.kib[1], t.nbuf[1], t.min_kib, t.max_kib);
  }

  // Histogram of message sizes in power of two KiB buckets
  std::map<int, std::array<int, 2>> hist;
  for (int m = 0; m < nmsg; ++m) {
    const int *r = row(m);
    const double sz = r[6] * kib;
    int b = 0;
    while (std::pow(2.0, b) < sz)
      b++;
    if (sz < 1.0) b = 0;
    hist[b][on_node(r) ? 0 : 1]++;
  }
  std::printf("\nMessage size histogram (KiB bucket upper bound)\n%12s %8s %8s\n",
              "<= KiB", "on_node", "off_node");
  for (const auto &[b, h] : hist)
    std::printf("%12.0f %8d %8d\n", std::pow(2.0, b), h[0], h[1]);

  if (comms_trace::CensusPrintMessages()) {
    std::printf("\n%6s %6s %4s %-20s %4s %6s %6s %10s\n", "src", "dst", "node",
                "btype", "lane", "ptag", "nbuf", "KiB");
    for (int m = 0; m < nmsg; ++m) {
      const int *r = row(m);
      std::printf("%6d %6d %4s %-20s %4d %6d %6d %10.2f\n", r[0], r[1],
                  on_node(r) ? "on" : "off", btype_name(r[2]), r[3], r[4], r[5],
                  r[6] * kib);
    }
  }
  std::printf("==== End census ====\n\n");
  std::fflush(stdout);

  const std::string fname =
      comms_trace::Prefix() + ".census." + std::to_string(build) + ".csv";
  FILE *f = std::fopen(fname.c_str(), "w");
  if (f != nullptr) {
    std::fprintf(f, "src,dst,src_node,dst_node,btype,lane,partition_tag,nbuf,bytes\n");
    for (int m = 0; m < nmsg; ++m) {
      const int *r = row(m);
      std::fprintf(f, "%d,%d,%d,%d,%s,%d,%d,%d,%lld\n", r[0], r[1],
                   comms_trace::NodeOfRank(r[0]), comms_trace::NodeOfRank(r[1]),
                   btype_name(r[2]), r[3], r[4], r[5],
                   static_cast<long long>(r[6]) * static_cast<long long>(sizeof(Real)));
    }
    std::fclose(f);
  }
}

//----------------------------------------------------------------------------------------
bool CoalescedComms::IsAvailableForWrite(MeshData<Real> *pmd, BoundaryType b_type) {
  const int lane = pmd->GetCommLane(b_type);
  bool available{true};
  for (int rank = 0; rank < Globals::nranks; ++rank) {
    const rank_key_t key{rank, b_type, lane};
    if (coalesced_send_buffers.count(key)) {
      available = available && coalesced_send_buffers.at(key).IsAvailableForWrite(pmd);
    }
  }
  return available;
}

//----------------------------------------------------------------------------------------
void CoalescedComms::PackAndSend(MeshData<Real> *pmd, BoundaryType b_type) {
  const int lane = pmd->GetCommLane(b_type);
  for (int rank = 0; rank < Globals::nranks; ++rank) {
    const rank_key_t key{rank, b_type, lane};
    if (coalesced_send_buffers.count(key)) {
      coalesced_send_buffers.at(key).PackAndSend(pmd);
    }
  }
}

//----------------------------------------------------------------------------------------
bool CoalescedComms::TryReceiveAny(MeshData<Real> *pmd, BoundaryType b_type) {
#ifdef MPI_PARALLEL
  const int lane = pmd->GetCommLane(b_type);
  bool all_received = true;
  for (int rank = 0; rank < Globals::nranks; ++rank) {
    const rank_key_t key{rank, b_type, lane};
    if (coalesced_recv_buffers.count(key)) {
      auto &coal_bufs = coalesced_recv_buffers.at(key);
      for (auto &[partition_tag, coal_buf] : coal_bufs.coalesced_bufs) {
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
