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
    : other_rank(o_rank), sender(send), buffers_built(false) {
  if (sender) {
    message =
        com_buf_t(1234, Globals::my_rank, other_rank, MPI_COMM_WORLD, [](int size) {
          PARTHENON_FAIL("Comms should not be allocating sender.");
          return std::vector<int>(size);
        });
  } else {
    message = com_buf_t(1234, other_rank, Globals::my_rank, MPI_COMM_WORLD,
                        [](int size) { return std::vector<int>(size); });
  }
  PARTHENON_REQUIRE(other_rank != Globals::my_rank,
                    "Should only build for other ranks.");
}

void CombinedBuffersRank::AddSendBuffer(int partition, MeshBlock *pmb, const NeighborBlock &nb,
                   const std::shared_ptr<Variable<Real>> &var, BoundaryType b_type) {
  if (current_size.count(partition) == 0) current_size[partition] = 0;
  auto &cur_size = current_size[partition];
  combined_info[partition].push_back(
      BndId::GetSend(pmb, nb, var, b_type, partition, cur_size));
  cur_size += combined_info[partition].back().size();
}

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
      combined_buffers[partition] = buf_t("combined recv buffer", total_size);
      auto &cr_info = combined_info[partition];
      for (int b = 0; b < nbuf; ++b) {
        cr_info.emplace_back(&(mess_buf[idx]));
        idx += BndId::NDAT;
      }
    }
    message.Stale();
    buffers_built = true;
    return true;
  }
  return false;
}

void CombinedBuffersRank::ResolveSendBuffersAndSendInfo() {
  // First calculate the total size of the message
  int total_buffers{0};
  for (auto &[partition, buf_struct_vec] : combined_info)
    total_buffers += buf_struct_vec.size();
  int total_partitions = combined_info.size();

  auto &mess_buf = message.buffer();
  int mesg_size = nglobal + nper_part * total_partitions + BndId::NDAT * total_buffers;
  mess_buf.resize(mesg_size);

  mess_buf[0] = total_partitions;

  // Pack the data
  int idx{nglobal};
  for (auto &[partition, buf_struct_vec] : combined_info) {
    mess_buf[idx++] = partition;               // Used as the comm tag
    mess_buf[idx++] = buf_struct_vec.size();   // Number of buffers
    mess_buf[idx++] = current_size[partition]; // combined size of buffers
    for (auto &buf_struct : buf_struct_vec) {
      buf_struct.Serialize(&(mess_buf[idx]));
      idx += BndId::NDAT;
    }
  }

  message.Send();

  // Allocate the combined buffers
  int total_size{0};
  for (auto &[partition, size] : current_size)
    total_size += size;

  buf_t alloc_at_once("shared combined buffer", total_size);
  int current_position{0};
  for (auto &[partition, size] : current_size) {
    combined_buffers[partition] =
        buf_t(alloc_at_once, std::make_pair(current_position, current_position + size));
    current_position += size;
  }
  buffers_built = true;
}
} // namespace parthenon
