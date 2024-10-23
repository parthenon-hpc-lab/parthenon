//========================================================================================
// (C) (or copyright) 2020-2024. Triad National Security, LLC. All rights reserved.
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
#include "interface/variable.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "utils/communication_buffer.hpp"

namespace parthenon {

CombinedBuffersRank::CombinedBuffersRank(int o_rank, BoundaryType b_type, bool send)
    : other_rank(o_rank), b_type(b_type), sender(send), buffers_built(false) {

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

void CombinedBuffersRank::AddSendBuffer(int partition, MeshBlock *pmb,
                                        const NeighborBlock &nb,
                                        const std::shared_ptr<Variable<Real>> &var,
                                        BoundaryType b_type) {
  if (current_size.count(partition) == 0) current_size[partition] = 0;
  auto &cur_size = current_size[partition];
  combined_info[partition].push_back(
      BndId::GetSend(pmb, nb, var, b_type, partition, cur_size));
  cur_size += combined_info[partition].back().size();
}

bool CombinedBuffersRank::TryReceiveBufInfo(Mesh *pmesh) {
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
      combined_buffers[partition] =
          CommBuffer<buf_t>(913 + partition, other_rank, Globals::my_rank, comm_);
      combined_buffers[partition].ConstructBuffer("combined recv buffer", total_size);
      auto &cr_info = combined_info[partition];
      auto &bufs = buffers[partition];
      for (int b = 0; b < nbuf; ++b) {
        cr_info.emplace_back(&(mess_buf[idx]));
        auto &buf = cr_info.back();
        // Store the buffer
        PARTHENON_REQUIRE(pmesh->boundary_comm_map.count(GetChannelKey(buf)),
                          "Buffer doesn't exist.");
        buf.buf = pmesh->boundary_comm_map[GetChannelKey(buf)];
        bufs.push_back(&(pmesh->boundary_comm_map[GetChannelKey(buf)]));
        buf.combined_buf = combined_buffers[partition].buffer();
        idx += BndId::NDAT;
      }
    }
    message.Stale();

    // Get the BndId objects on device
    for (auto &[partition, buf_vec] : combined_info) {
      combined_info_device[partition] = ParArray1D<BndId>("bnd_id", buf_vec.size());
      auto ci_host = Kokkos::create_mirror_view(combined_info_device[partition]);
      for (int i = 0; i < ci_host.size(); ++i)
        ci_host[i] = buf_vec[i];
      Kokkos::deep_copy(combined_info_device[partition], ci_host);
    }

    buffers_built = true;
    return true;
  }
  return false;
}

void CombinedBuffersRank::ResolveSendBuffersAndSendInfo(Mesh *pmesh) {
  // First calculate the total size of the message
  int total_buffers{0};
  for (auto &[partition, buf_struct_vec] : combined_info)
    total_buffers += buf_struct_vec.size();
  int total_partitions = combined_info.size();

  int mesg_size = nglobal + nper_part * total_partitions + BndId::NDAT * total_buffers;
  message.Allocate(mesg_size);

  auto &mess_buf = message.buffer();
  mess_buf[0] = total_partitions;

  // Pack the data
  int idx{nglobal};
  for (auto &[partition, buf_struct_vec] : combined_info) {
    mess_buf[idx++] = partition;               // Used as the comm tag
    mess_buf[idx++] = buf_struct_vec.size();   // Number of buffers
    mess_buf[idx++] = current_size[partition]; // combined size of buffers
    auto &bufs = buffers[partition];
    for (auto &buf_struct : buf_struct_vec) {
      buf_struct.Serialize(&(mess_buf[idx]));
      PARTHENON_REQUIRE(pmesh->boundary_comm_map.count(GetChannelKey(buf_struct)),
                        "Buffer doesn't exist.");

      bufs.push_back(&(pmesh->boundary_comm_map[GetChannelKey(buf_struct)]));
      idx += BndId::NDAT;
    }
  }

  message.Send();

  // Allocate the combined buffers
  for (auto &[partition, size] : current_size) {
    combined_buffers[partition] =
        CommBuffer<buf_t>(913 + partition, Globals::my_rank, other_rank, comm_);
    combined_buffers[partition].ConstructBuffer("combined send buffer", size);
  }

  // Point the BndId objects to the combined buffers
  for (auto &[partition, buf_struct_vec] : combined_info) {
    for (auto &buf_struct : buf_struct_vec) {
      buf_struct.combined_buf = combined_buffers[partition].buffer();
    }
  }

  buffers_built = true;
}

void CombinedBuffersRank::RepointBuffers(Mesh *pmesh, int partition) {
  if (combined_info.count(partition) == 0) return;
  // Pull out the buffers and point them to the buf_struct
  auto &buf_struct_vec = combined_info[partition];
  for (auto &buf_struct : buf_struct_vec) {
    buf_struct.buf = pmesh->boundary_comm_map[GetChannelKey(buf_struct)];
  }

  // Get the BndId objects on device
  combined_info_device[partition] = ParArray1D<BndId>("bnd_id", buf_struct_vec.size());
  auto ci_host = Kokkos::create_mirror_view(combined_info_device[partition]);
  for (int i = 0; i < ci_host.size(); ++i)
    ci_host[i] = buf_struct_vec[i];
  Kokkos::deep_copy(combined_info_device[partition], ci_host);
}

void CombinedBuffersRank::PackAndSend(int partition) {
  PARTHENON_REQUIRE(buffers_built,
                    "Trying to send combined buffers before they have been built");
  if (combined_info_device.count(partition) == 0) return; // There is nothing to send here
  auto &comb_info = combined_info_device[partition];
  Kokkos::parallel_for(
      PARTHENON_AUTO_LABEL,
      Kokkos::TeamPolicy<>(parthenon::DevExecSpace(), combined_info[partition].size(),
                           Kokkos::AUTO),
      KOKKOS_LAMBDA(parthenon::team_mbr_t team_member) {
        const int b = team_member.league_rank();
        const int buf_size = comb_info[b].size();
        Real *com_buf = &(comb_info[b].combined_buf(comb_info[b].start_idx()));
        Real *buf = &(comb_info[b].buf(0));
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team_member, buf_size),
                             [&](const int idx) { com_buf[idx] = buf[idx]; });
      });
#ifdef MPI_PARALLEL
  Kokkos::fence();
#endif
  combined_buffers[partition].Send();
  // Information in these send buffers is no longer required
  for (auto &buf : buffers[partition])
    buf->Stale();
}

bool CombinedBuffersRank::AllReceived() {
  bool all_received{true};
  for (auto &[partition, buf] : combined_buffers) {
    bool received = buf.GetState() == BufferState::received;
    all_received = all_received && received;
  }
  return all_received;
}

void CombinedBuffersRank::StaleAllReceives() {
  for (auto &[partition, buf] : combined_buffers) {
    buf.Stale();
  }
}

bool CombinedBuffersRank::IsAvailableForWrite(int partition) {
  if (combined_buffers.count(partition) == 0) return true;
  return combined_buffers[partition].IsAvailableForWrite();
}

bool CombinedBuffersRank::TryReceiveAndUnpack(Mesh *pmesh, int partition) {
  PARTHENON_REQUIRE(buffers_built,
                    "Trying to recv combined buffers before they have been built");
  PARTHENON_REQUIRE(combined_buffers.count(partition) > 0,
                    "Trying to receive on a non-existent combined receive buffer.");
  auto received = combined_buffers[partition].TryReceive();
  if (!received) return false;

  // TODO(LFR): Fix this so it works in the more general case
  bool all_allocated{true};
  for (auto &buf : buffers[partition]) {
    if (!buf->IsActive()) {
      all_allocated = false;
      buf->Allocate();
    }
  }
  if (!all_allocated) {
    RepointBuffers(pmesh, partition);
  }
  auto &comb_info = combined_info_device[partition];
  Kokkos::parallel_for(
      PARTHENON_AUTO_LABEL,
      Kokkos::TeamPolicy<>(parthenon::DevExecSpace(), combined_info[partition].size(),
                           Kokkos::AUTO),
      KOKKOS_LAMBDA(parthenon::team_mbr_t team_member) {
        const int b = team_member.league_rank();
        const int buf_size = comb_info[b].size();
        Real *com_buf = &(comb_info[b].combined_buf(comb_info[b].start_idx()));
        Real *buf = &(comb_info[b].buf(0));
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team_member, buf_size),
                             [&](const int idx) { buf[idx] = com_buf[idx]; });
      });
  combined_buffers[partition].Stale();
  for (auto &buf : buffers[partition])
    buf->SetReceived();
  return true;
}

void CombinedBuffersRank::CompareReceivedBuffers(int partition) {
  if (Globals::my_rank != 0) return; // don't crush us with output
  PARTHENON_REQUIRE(buffers_built,
                    "Trying to recv combined buffers before they have been built")
  if (combined_info_device.count(partition) == 0) return;
  auto &comb_info = combined_info_device[partition];
  Kokkos::parallel_for(
      PARTHENON_AUTO_LABEL,
      Kokkos::TeamPolicy<>(parthenon::DevExecSpace(), combined_info[partition].size(),
                           Kokkos::AUTO),
      KOKKOS_LAMBDA(parthenon::team_mbr_t team_member) {
        const int b = team_member.league_rank();
        const int buf_size = comb_info[b].size();
        Real *com_buf = &(comb_info[b].combined_buf(comb_info[b].start_idx()));
        Real *buf = &(comb_info[b].buf(0));
        printf("Buffer [%i] start = %i size = %i\n", b, comb_info[b].start_idx(),
               buf_size);
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team_member, buf_size),
                             [&](const int idx) {
                               if (buf[idx] != com_buf[idx])
                                 printf("  [%i] %e %e\n", idx, buf[idx], com_buf[idx]);
                             });
      });
}

void CombinedBuffers::AddSendBuffer(int partition, MeshBlock *pmb,
                                    const NeighborBlock &nb,
                                    const std::shared_ptr<Variable<Real>> &var,
                                    BoundaryType b_type) {
  if (combined_send_buffers.count({nb.rank, b_type}) == 0)
    combined_send_buffers[{nb.rank, b_type}] = CombinedBuffersRank(nb.rank, b_type, true);
  combined_send_buffers[{nb.rank, b_type}].AddSendBuffer(partition, pmb, nb, var, b_type);
}

void CombinedBuffers::AddRecvBuffer(MeshBlock *pmb, const NeighborBlock &nb,
                                    const std::shared_ptr<Variable<Real>>,
                                    BoundaryType b_type) {
  // We don't actually know enough here to register this particular buffer, but we do
  // know that it's existence implies that we need to receive a message from the
  // neighbor block rank eventually telling us the details
  if (combined_recv_buffers.count({nb.rank, b_type}) == 0)
    combined_recv_buffers[{nb.rank, b_type}] =
        CombinedBuffersRank(nb.rank, b_type, false);
}

void CombinedBuffers::ResolveAndSendSendBuffers(Mesh *pmesh) {
  for (auto &[id, buf] : combined_send_buffers)
    buf.ResolveSendBuffersAndSendInfo(pmesh);
}

void CombinedBuffers::ReceiveBufferInfo(Mesh *pmesh) {
  constexpr std::int64_t max_it = 1e10;
  std::vector<bool> received(combined_recv_buffers.size(), false);
  bool all_received;
  std::int64_t receive_iters = 0;
  do {
    all_received = true;
    for (auto &[id, buf] : combined_recv_buffers)
      all_received = buf.TryReceiveBufInfo(pmesh) && all_received;
    receive_iters++;
  } while (!all_received && receive_iters < max_it);
  PARTHENON_REQUIRE(
      receive_iters < max_it,
      "Too many iterations waiting to receive boundary communication buffers.");
}

bool CombinedBuffers::IsAvailableForWrite(int partition, BoundaryType b_type) {
  bool available{true};
  for (int rank = 0; rank < Globals::nranks; ++rank) {
    if (combined_send_buffers.count({rank, b_type})) {
      available = available &&
                  combined_send_buffers[{rank, b_type}].IsAvailableForWrite(partition);
    }
  }
  return available;
}

void CombinedBuffers::PackAndSend(int partition, BoundaryType b_type) {
  for (int rank = 0; rank < Globals::nranks; ++rank) {
    if (combined_send_buffers.count({rank, b_type})) {
      combined_send_buffers[{rank, b_type}].PackAndSend(partition);
    }
  }
}

void CombinedBuffers::RepointSendBuffers(Mesh *pmesh, int partition,
                                         BoundaryType b_type) {
  for (int rank = 0; rank < Globals::nranks; ++rank) {
    if (combined_send_buffers.count({rank, b_type}))
      combined_send_buffers[{rank, b_type}].RepointBuffers(pmesh, partition);
  }
}

void CombinedBuffers::RepointRecvBuffers(Mesh *pmesh, int partition,
                                         BoundaryType b_type) {
  for (int rank = 0; rank < Globals::nranks; ++rank) {
    if (combined_recv_buffers.count({rank, b_type}))
      combined_recv_buffers[{rank, b_type}].RepointBuffers(pmesh, partition);
  }
}

void CombinedBuffers::TryReceiveAny(Mesh *pmesh, BoundaryType b_type) {
#ifdef MPI_PARALLEL
  MPI_Status status;
  int flag;
  do {
    // TODO(LFR): Switch to a different communicator for each BoundaryType
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
    if (flag) {
      const int rank = status.MPI_SOURCE;
      const int partition = status.MPI_TAG - 913;
      bool finished =
          combined_recv_buffers[{rank, b_type}].TryReceiveAndUnpack(pmesh, partition);
      if (!finished) processing_messages.insert({rank, partition});
    }
  } while (flag);

  // Process in flight messages
  std::set<std::pair<int, int>> finished_messages;
  for (auto &[rank, partition] : processing_messages) {
    bool finished =
        combined_recv_buffers[{rank, b_type}].TryReceiveAndUnpack(pmesh, partition);
    if (finished) finished_messages.insert({rank, partition});
  }

  for (auto &m : finished_messages)
    processing_messages.erase(m);

#endif
}

bool CombinedBuffers::AllReceived(BoundaryType b_type) {
  bool all_received{true};
  for (auto &[tag, bufs] : combined_recv_buffers) {
    if (std::get<1>(tag) == b_type) {
      all_received = all_received && bufs.AllReceived();
    }
  }
  return all_received;
}

void CombinedBuffers::StaleAllReceives(BoundaryType b_type) {
  for (auto &[tag, bufs] : combined_recv_buffers) {
    if (std::get<1>(tag) == b_type) {
      bufs.StaleAllReceives();
    }
  }
}

void CombinedBuffers::CompareReceivedBuffers(BoundaryType b_type) {
  for (auto &[tag, bufs] : combined_recv_buffers) {
    if (std::get<1>(tag) == b_type) {
      bufs.CompareReceivedBuffers(0);
    }
  }
}

} // namespace parthenon
