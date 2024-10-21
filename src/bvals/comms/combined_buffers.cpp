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
#include <map>
#include <memory>
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
          CommBuffer<buf_t>(partition, other_rank, Globals::my_rank, comm_);
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
        bufs.push_back(pmesh->boundary_comm_map[GetChannelKey(buf)]);
        buf.pcombined_buf = &(combined_buffers[partition].buffer());
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
      buf_struct.buf = pmesh->boundary_comm_map[GetChannelKey(buf_struct)];
      bufs.push_back(pmesh->boundary_comm_map[GetChannelKey(buf_struct)]);
      idx += BndId::NDAT;
    }
  }

  message.Send();

  // Allocate the combined buffers and point the BndId objects to them
  int total_size{0};
  for (auto &[partition, size] : current_size)
    total_size += size;

  int current_position{0};
  for (auto &[partition, size] : current_size) {
    combined_buffers[partition] =
        CommBuffer<buf_t>(partition, Globals::my_rank, other_rank, comm_);
    combined_buffers[partition].ConstructBuffer("combined send buffer", total_size);
    current_position += size;
  }

  for (auto &[partition, buf_struct_vec] : combined_info) {
    for (auto &buf_struct : buf_struct_vec) {
      buf_struct.pcombined_buf = &(combined_buffers[partition].buffer());
    }
  }

  // Get the BndId objects on device
  for (auto &[partition, buf_vec] : combined_info) {
    combined_info_device[partition] = ParArray1D<BndId>("bnd_id", buf_vec.size());
    auto ci_host = Kokkos::create_mirror_view(combined_info_device[partition]);
    for (int i = 0; i < ci_host.size(); ++i)
      ci_host[i] = buf_vec[i];
    Kokkos::deep_copy(combined_info_device[partition], ci_host);
  }

  buffers_built = true;
}

void CombinedBuffersRank::PackAndSend(int partition) {
  PARTHENON_REQUIRE(buffers_built, "Trying to send combined buffers before they have been built")
  auto &comb_info = combined_info_device[partition];
  Kokkos::parallel_for(
      PARTHENON_AUTO_LABEL,
      Kokkos::TeamPolicy<>(parthenon::DevExecSpace(), combined_info[partition].size(), Kokkos::AUTO),
      KOKKOS_LAMBDA(parthenon::team_mbr_t team_member) {
        const int b = team_member.league_rank();
        const int buf_size = comb_info[b].size();
        Real *com_buf = &((*comb_info[b].pcombined_buf)(comb_info[b].start_idx()));
        Real *buf = &(comb_info[b].buf(0));
        Kokkos::parallel_for(
              Kokkos::TeamThreadRange<>(team_member, buf_size),
              [&](const int idx) {
                com_buf[idx] = buf[idx];
              });
      });
#ifdef MPI_PARALLEL
  Kokkos::fence();
#endif
  combined_buffers[partition].Send();
  // Information in these send buffers is no longer required
  for (auto &buf : buffers[partition])
    buf.Stale();
}

bool CombinedBuffersRank::TryReceiveAndUnpack(int partition) {
  PARTHENON_REQUIRE(buffers_built, "Trying to recv combined buffers before they have been built")
  auto &comb_info = combined_info_device[partition];
  auto received = combined_buffers[partition].TryReceive();
  if (!received) return false;
  Kokkos::parallel_for(
      PARTHENON_AUTO_LABEL,
      Kokkos::TeamPolicy<>(parthenon::DevExecSpace(), combined_info[partition].size(), Kokkos::AUTO),
      KOKKOS_LAMBDA(parthenon::team_mbr_t team_member) {
        const int b = team_member.league_rank();
        const int buf_size = comb_info[b].size();
        Real *com_buf = &((*comb_info[b].pcombined_buf)(comb_info[b].start_idx()));
        Real *buf = &(comb_info[b].buf(0));
        Kokkos::parallel_for(
              Kokkos::TeamThreadRange<>(team_member, buf_size),
              [&](const int idx) {
                buf[idx] = com_buf[idx];
              });
      });
  combined_buffers[partition].Stale();
  for (auto &buf : buffers[partition])
    buf.SetReceived();
  return true;
}

} // namespace parthenon
