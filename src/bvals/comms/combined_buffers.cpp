//========================================================================================
// (C) (or copyright) 2024. Triad National Security, LLC. All rights reserved.
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
#include <cstdio>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "basic_types.hpp"
#include "bvals/comms/bvals_utils.hpp"
#include "bvals/comms/combined_buffers.hpp"
#include "bvals/neighbor_block.hpp"
#include "coordinates/coordinates.hpp"
#include "interface/mesh_data.hpp"
#include "interface/variable.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "utils/communication_buffer.hpp"

namespace parthenon {

//----------------------------------------------------------------------------------------
void CombinedBuffersRankPartition::AllocateCombinedBuffer() {
  int send_rank = sender ? Globals::my_rank : other_rank;
  int recv_rank = sender ? other_rank : Globals::my_rank;
  combined_comm_buffer = CommBuffer<buf_t>(2 * partition, send_rank, recv_rank, comm_);
  combined_comm_buffer.ConstructBuffer("combined send buffer",
                                       current_size + 1); // Actually allocate the thing
  sparse_status_buffer =
      CommBuffer<std::vector<int>>(2 * partition + 1, send_rank, recv_rank, comm_);
  sparse_status_buffer.ConstructBuffer(current_size + 1);
  // PARTHENON_REQUIRE(current_size > 0, "Are we bigger than zero?");
  //  Point the BndId objects to the combined buffer
  for (auto uid : all_vars) {
    for (auto &[bnd_id, pvbbuf] : combined_info_buf.at(uid)) {
      bnd_id.combined_buf = combined_comm_buffer.buffer();
    }
  }
}

//----------------------------------------------------------------------------------------
ParArray1D<BndId> &
CombinedBuffersRankPartition::GetBndIdsOnDevice(const std::set<Uid_t> &vars) {
  int nbnd_id{0};
  const auto &var_set = vars.size() == 0 ? all_vars : vars;
  for (auto uid : var_set)
    nbnd_id += combined_info_buf.at(uid).size();

  bool updated = false;
  if (nbnd_id != bnd_ids_device.size()) {
    bnd_ids_device = ParArray1D<BndId>("bnd_id", nbnd_id);
    bnd_ids_host = Kokkos::create_mirror_view(bnd_ids_device);
    updated = true;
  }

  int idx{0};
  int c_buf_idx{0}; // Index at which v-b buffer starts in combined buffer
  for (auto uid : var_set) {
    for (auto &[bnd_id, pvbbuf] : combined_info_buf.at(uid)) {
      auto &bid_h = bnd_ids_host[idx];
      auto buf_state = pvbbuf->GetState();
      PARTHENON_REQUIRE(buf_state != BufferState::stale,
                        "Trying to work with a stale buffer.");

      const bool alloc =
          (buf_state == BufferState::sending) || (buf_state == BufferState::received);
      // Test if this boundary has changed
      if (!bid_h.SameBVChannel(bnd_id) || (bid_h.buf_allocated != alloc) ||
          (bid_h.start_idx() != c_buf_idx) ||
          !UsingSameResource(bid_h.buf, pvbbuf->buffer())) {
        updated = true;
        bid_h = bnd_id;
        bid_h.buf_allocated = alloc;
        bid_h.start_idx() = c_buf_idx;
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
void CombinedBuffersRankPartition::PackAndSend(const std::set<Uid_t> &vars) {
  PARTHENON_REQUIRE(combined_comm_buffer.IsAvailableForWrite(),
                    "Trying to write to a buffer that is in use.");
  auto &bids = GetBndIdsOnDevice(vars);
  Kokkos::parallel_for(
      PARTHENON_AUTO_LABEL,
      Kokkos::TeamPolicy<>(parthenon::DevExecSpace(), bids.size(), Kokkos::AUTO),
      KOKKOS_LAMBDA(parthenon::team_mbr_t team_member) {
        const int b = team_member.league_rank();
        if (bids[b].buf_allocated) {
          const int buf_size = bids[b].size();
          Real *com_buf = &(bids[b].combined_buf(bids[b].start_idx()));
          Real *buf = &(bids[b].buf(0));
          Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team_member, buf_size),
                               [&](const int idx) { com_buf[idx] = buf[idx]; });
        }
      });
#ifdef MPI_PARALLEL
  Kokkos::fence();
#endif
  combined_comm_buffer.Send();

  // Send the sparse null info as well
  if (bids.size() != sparse_status_buffer.buffer().size()) {
    sparse_status_buffer.ConstructBuffer(bids.size());
  }

  const auto &var_set = vars.size() == 0 ? all_vars : vars;
  auto &stat = sparse_status_buffer.buffer();
  int idx{0};
  for (auto uid : var_set) {
    for (auto &[bnd_id, pvbbuf] : combined_info_buf.at(uid)) {
      stat[idx] = (pvbbuf->GetState() == BufferState::sending);
      ++idx;
    }
  }
  sparse_status_buffer.Send();

  // Information in these send buffers is no longer required
  for (auto uid : var_set) {
    for (auto &[bnd_id, pvbbuf] : combined_info_buf.at(uid)) {
      pvbbuf->Stale();
    }
  }
}

//----------------------------------------------------------------------------------------
bool CombinedBuffersRankPartition::TryReceiveAndUnpack(mpi_message_t *message,
                                                       const std::set<Uid_t> &vars) {
  const auto &var_set = vars.size() == 0 ? all_vars : vars;
  // Make sure the var-boundary buffers are available to write to
  int nbuf{0};
  for (auto uid : var_set) {
    for (auto &[bnd_id, pvbbuf] : combined_info_buf.at(uid)) {
      if (pvbbuf->GetState() != BufferState::stale) return false;
      nbuf++;
    }
  }

  if (nbuf != sparse_status_buffer.buffer().size()) {
    sparse_status_buffer.ConstructBuffer(nbuf);
  }
  auto received_sparse = sparse_status_buffer.TryReceive();
  auto received = combined_comm_buffer.TryReceive(message);
  if (!received || !received_sparse) return false;

  // Allocate and free buffers as required
  int idx{0};
  auto &stat = sparse_status_buffer.buffer();
  for (auto uid : var_set) {
    for (auto &[bnd_id, pvbbuf] : combined_info_buf.at(uid)) {
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

  auto &bids = GetBndIdsOnDevice(vars);
  Kokkos::parallel_for(
      PARTHENON_AUTO_LABEL,
      Kokkos::TeamPolicy<>(parthenon::DevExecSpace(), bids.size(), Kokkos::AUTO),
      KOKKOS_LAMBDA(parthenon::team_mbr_t team_member) {
        const int b = team_member.league_rank();
        if (bids[b].buf_allocated) {
          const int buf_size = bids[b].size();
          Real *com_buf = &(bids[b].combined_buf(bids[b].start_idx()));
          Real *buf = &(bids[b].buf(0));
          Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team_member, buf_size),
                               [&](const int idx) { buf[idx] = com_buf[idx]; });
        }
      });
  combined_comm_buffer.Stale();
  sparse_status_buffer.Stale();

  return true;
}

//----------------------------------------------------------------------------------------
void CombinedBuffersRankPartition::AddVarBoundary(BndId &bnd_id) {
  auto key = GetChannelKey(bnd_id);
  PARTHENON_REQUIRE(pmesh->boundary_comm_map.count(key), "Buffer doesn't exist.");
  var_buf_t *pbuf = &(pmesh->boundary_comm_map.at(key));
  combined_info_buf[bnd_id.var_id()].push_back(std::make_pair(bnd_id, pbuf));
  current_size += bnd_id.size(); // This will be the maximum size of communication since
                                 // it includes all variables
  all_vars.insert(bnd_id.var_id());
}

void CombinedBuffersRankPartition::AddVarBoundary(
    MeshBlock *pmb, const NeighborBlock &nb, const std::shared_ptr<Variable<Real>> &var) {
  // Store both the variable-boundary buffer information and a pointer to the v-b buffer
  // itself associated with var ids
  BndId bnd_id = BndId::GetSend(pmb, nb, var, b_type, partition, -1);
  AddVarBoundary(bnd_id);
}

//----------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------
CombinedBuffersRank::CombinedBuffersRank(int o_rank, BoundaryType b_type, bool send,
                                         mpi_comm_t comm, Mesh *pmesh)
    : other_rank(o_rank), b_type(b_type), sender(send), buffers_built(false), comm_(comm),
      pmesh(pmesh) {

  int tag = 1234 + static_cast<int>(GetAssociatedSender(b_type));
  if (sender) {
    message = com_buf_t(tag, Globals::my_rank, other_rank, comm_,
                        [](int size) { return std::vector<int>(size); });
  } else {
    message = com_buf_t(
        tag, other_rank, Globals::my_rank, comm_,
        [](int size) { return std::vector<int>(size); }, true);
  }
  PARTHENON_REQUIRE(other_rank != Globals::my_rank, "Should only build for other ranks.");
}

//----------------------------------------------------------------------------------------
void CombinedBuffersRank::AddSendBuffer(int partition, MeshBlock *pmb,
                                        const NeighborBlock &nb,
                                        const std::shared_ptr<Variable<Real>> &var) {
  if (combined_bufs.count(partition) == 0)
    combined_bufs.emplace(std::make_pair(
        partition, CombinedBuffersRankPartition(true, partition, other_rank, b_type,
                                                comm_, pmb->pmy_mesh)));

  auto &comb_buf = combined_bufs.at(partition);
  comb_buf.AddVarBoundary(pmb, nb, var);
}

//----------------------------------------------------------------------------------------
bool CombinedBuffersRank::TryReceiveBufInfo() {
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
      combined_bufs.emplace(std::make_pair(
          partition, CombinedBuffersRankPartition(false, partition, other_rank, b_type,
                                                  comm_, pmesh)));
      auto &comb_buf = combined_bufs.at(partition);

      for (int b = 0; b < nbuf; ++b) {
        BndId bnd_id(&(mess_buf[idx]));
        comb_buf.AddVarBoundary(bnd_id);
        idx += BndId::NDAT;
      }
    }
    message.Stale();

    for (auto &[partition, com_buf] : combined_bufs) {
      com_buf.AllocateCombinedBuffer();
    }

    buffers_built = true;
    return true;
  }
  return false;
}

//----------------------------------------------------------------------------------------
void CombinedBuffersRank::ResolveSendBuffersAndSendInfo() {
  // First calculate the total size of the message
  int total_buffers{0};
  for (auto &[partition, combined_buf] : combined_bufs)
    total_buffers += combined_buf.TotalBuffers();
  int total_partitions = combined_bufs.size();

  int mesg_size = nglobal + nper_part * total_partitions + BndId::NDAT * total_buffers;
  message.Allocate(mesg_size);

  auto &mess_buf = message.buffer();
  mess_buf[0] = total_partitions;

  // Pack the data
  int idx{nglobal};
  for (auto &[partition, combined_buf] : combined_bufs) {
    mess_buf[idx++] = partition;                   // Used as the comm tag
    mess_buf[idx++] = combined_buf.TotalBuffers(); // Number of buffers
    mess_buf[idx++] =
        combined_buf.current_size; // combined size of buffers (now probably unused)
    for (auto &[uid, v] : combined_buf.combined_info_buf) {
      for (auto &[bnd_id, pbvbuf] : v) {
        bnd_id.Serialize(&(mess_buf[idx]));
        idx += BndId::NDAT;
      }
    }
  }

  message.Send();

  for (auto &[partition, com_buf] : combined_bufs)
    com_buf.AllocateCombinedBuffer();

  buffers_built = true;
}

//----------------------------------------------------------------------------------------
void CombinedBuffersRank::PackAndSend(MeshData<Real> *pmd) {
  PARTHENON_REQUIRE(buffers_built,
                    "Trying to send combined buffers before they have been built");
  if (combined_bufs.count(pmd->partition)) {
    combined_bufs.at(pmd->partition).PackAndSend(pmd->GetUids());
  }

  return;
}

//----------------------------------------------------------------------------------------
bool CombinedBuffersRank::IsAvailableForWrite(MeshData<Real> *pmd) {
  PARTHENON_REQUIRE(sender, "Shouldn't be checking this on non-sender.");
  if (combined_bufs.count(pmd->partition) == 0) return true;
  return combined_bufs.at(pmd->partition).IsAvailableForWrite();
}

//----------------------------------------------------------------------------------------
bool CombinedBuffersRank::TryReceiveAndUnpack(MeshData<Real> *pmd, int partition,
                                              mpi_message_t *message) {
  PARTHENON_REQUIRE(buffers_built,
                    "Trying to recv combined buffers before they have been built");
  PARTHENON_REQUIRE(combined_bufs.count(partition) > 0,
                    "Trying to receive on a non-existent combined receive buffer.");
  return combined_bufs.at(partition).TryReceiveAndUnpack(message, pmd->GetUids());
}

//----------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------
void CombinedBuffers::AddSendBuffer(int partition, MeshBlock *pmb,
                                    const NeighborBlock &nb,
                                    const std::shared_ptr<Variable<Real>> &var,
                                    BoundaryType b_type) {
  if (combined_send_buffers.count({nb.rank, b_type}) == 0)
    combined_send_buffers.emplace(
        std::make_pair(std::make_pair(nb.rank, b_type),
                       CombinedBuffersRank(nb.rank, b_type, true,
                                           comms_[GetAssociatedSender(b_type)], pmesh)));
  combined_send_buffers.at({nb.rank, b_type}).AddSendBuffer(partition, pmb, nb, var);
}

void CombinedBuffers::AddRecvBuffer(MeshBlock *pmb, const NeighborBlock &nb,
                                    const std::shared_ptr<Variable<Real>>,
                                    BoundaryType b_type) {
  // We don't actually know enough here to register this particular buffer, but we do
  // know that it's existence implies that we need to receive a message from the
  // neighbor block rank eventually telling us the details
  if (combined_recv_buffers.count({nb.rank, b_type}) == 0)
    combined_recv_buffers.emplace(
        std::make_pair(std::make_pair(nb.rank, b_type),
                       CombinedBuffersRank(nb.rank, b_type, false,
                                           comms_[GetAssociatedSender(b_type)], pmesh)));
}

void CombinedBuffers::ResolveAndSendSendBuffers() {
  for (auto &[id, buf] : combined_send_buffers)
    buf.ResolveSendBuffersAndSendInfo();
}

void CombinedBuffers::ReceiveBufferInfo() {
  constexpr std::int64_t max_it = 1e10;
  std::vector<bool> received(combined_recv_buffers.size(), false);
  bool all_received;
  std::int64_t receive_iters = 0;
  do {
    all_received = true;
    for (auto &[id, buf] : combined_recv_buffers)
      all_received = buf.TryReceiveBufInfo() && all_received;
    receive_iters++;
  } while (!all_received && receive_iters < max_it);
  PARTHENON_REQUIRE(
      receive_iters < max_it,
      "Too many iterations waiting to receive boundary communication buffers.");
}

bool CombinedBuffers::IsAvailableForWrite(MeshData<Real> *pmd, BoundaryType b_type) {
  bool available{true};
  for (int rank = 0; rank < Globals::nranks; ++rank) {
    if (combined_send_buffers.count({rank, b_type})) {
      available =
          available && combined_send_buffers.at({rank, b_type}).IsAvailableForWrite(pmd);
    }
  }
  return available;
}

void CombinedBuffers::PackAndSend(MeshData<Real> *pmd, BoundaryType b_type) {
  for (int rank = 0; rank < Globals::nranks; ++rank) {
    if (combined_send_buffers.count({rank, b_type})) {
      combined_send_buffers.at({rank, b_type}).PackAndSend(pmd);
    }
  }
}

void CombinedBuffers::TryReceiveAny(MeshData<Real> *pmd, BoundaryType b_type) {
#ifdef MPI_PARALLEL
  // This was an attempt at another method for receiving, it seemed to work
  // but was subject to the same problems as the Iprobe based code
  if (pmesh->receive_type == "old") {
    for (int rank = 0; rank < Globals::nranks; ++rank) {
      if (combined_recv_buffers.count({rank, b_type})) {
        auto &comb_bufs = combined_recv_buffers.at({rank, b_type});
        for (auto &[partition, comb_buf] : comb_bufs.combined_bufs) {
          comb_buf.TryReceiveAndUnpack(nullptr, pmd->GetUids());
        }
      }
    }
  } else if (pmesh->receive_type == "iprobe") {
    MPI_Status status;
    int flag;
    do {
      mpi_message_t message;
      MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comms_[GetAssociatedSender(b_type)], &flag,
                 &status);
      if (flag) {
        const int rank = status.MPI_SOURCE;
        const int partition = status.MPI_TAG;
        bool finished = combined_recv_buffers.at({rank, b_type})
                            .TryReceiveAndUnpack(pmd, partition, nullptr);
        if (!finished)
          processing_messages.insert(
              std::make_pair(std::pair<int, int>{rank, partition}, message));
      }
    } while (flag);

    // Process in-flight messages
    std::vector<std::pair<int, int>> finished_messages;
    for (auto &[p, message] : processing_messages) {
      int rank = p.first;
      int partition = p.second;
      bool finished = combined_recv_buffers.at({rank, b_type})
                          .TryReceiveAndUnpack(pmd, partition, nullptr);
      if (finished) finished_messages.push_back({rank, partition});
    }

    for (auto &m : finished_messages)
      processing_messages.erase(m);
  } else if (pmesh->receive_type == "improbe") {
    MPI_Status status;
    int flag;
    do {
      mpi_message_t message;
      MPI_Improbe(MPI_ANY_SOURCE, MPI_ANY_TAG, comms_[GetAssociatedSender(b_type)], &flag,
                  &message, &status);
      if (flag) {
        const int rank = status.MPI_SOURCE;
        const int partition = status.MPI_TAG;
        bool finished = combined_recv_buffers.at({rank, b_type})
                            .TryReceiveAndUnpack(pmd, partition, &message);
        if (!finished)
          processing_messages.insert(
              std::make_pair(std::pair<int, int>{rank, partition}, message));
      }
    } while (flag);

    // Process in-flight messages
    std::vector<std::pair<int, int>> finished_messages;
    for (auto &[p, message] : processing_messages) {
      int rank = p.first;
      int partition = p.second;
      bool finished = combined_recv_buffers.at({rank, b_type})
                          .TryReceiveAndUnpack(pmd, partition, &message);
      if (finished) finished_messages.push_back({rank, partition});
    }

    for (auto &m : finished_messages)
      processing_messages.erase(m);
  } else {
    PARTHENON_FAIL("Unknown receiving strategy.");
  }
#endif
}
} // namespace parthenon
