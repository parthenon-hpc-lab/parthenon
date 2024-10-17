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

#ifndef BVALS_COMMS_COMBINED_BUFFERS_HPP_
#define BVALS_COMMS_COMBINED_BUFFERS_HPP_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "basic_types.hpp"
#include "bvals/comms/bvals_utils.hpp"
#include "bvals/neighbor_block.hpp"
#include "coordinates/coordinates.hpp"
#include "interface/variable.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "utils/communication_buffer.hpp"

namespace parthenon {
// Structure containing the information required for sending coalesced
// messages between ranks
struct CombinedBuffersRank {
  using coalesced_message_structure_t = std::vector<BndId>;
  using buf_t = BufArray1D<Real>;

  // Rank that these buffers communicate with
  int other_rank;

  // map from partion id to coalesced message structure for communication
  // from this rank to other_rank
  bool send_buffers_built{false};
  std::map<int, coalesced_message_structure_t> combined_send_info;
  std::map<int, BufArray1D<Real>> combined_send_buffers;

  // map from neighbor partition id to coalesced message structures that
  // this rank can receive from other_rank. We will use the partition id
  // as the mpi tag
  bool recv_buffers_built{false};
  std::map<int, coalesced_message_structure_t> combined_recv_info;
  std::map<int, BufArray1D<Real>> combined_recv_buffers;

  static constexpr int nglobal{1};
  static constexpr int nper_part{3};
  static constexpr int nper_buf{5};

  void AddSendBuffer(int partition, MeshBlock *pmb, const NeighborBlock &nb,
                     const std::shared_ptr<Variable<Real>> &var, BoundaryType b_type) {
    auto bnd = BndInfo::GetSendBndInfo(pmb, nb, var, b_type, nullptr);
    bnd.id.partition = partition;
    combined_send_info[partition].push_back(bnd.id);
  }

  void TryReceiveBufInfo() {
    if (recv_buffers_built) return;

    int received{1};
    int mesg_size{100};
    std::vector<int> message(mesg_size);
    if (received) {
      int npartitions = message[0];
      // Current starting buffer index
      int bidx{nglobal + nper_part * npartitions};
      // Current starting partition index
      int pidx{nglobal};
      for (int p = 0; p < npartitions; ++p) {
        const int partition = message[pidx++];
        const int nbuf = message[pidx++];
        const int total_size = message[pidx++];
        combined_recv_buffers[partition] = buf_t("combined recv buffer", total_size);
        auto &cr_info = combined_recv_info[partition];
        for (int b = 0; b < nbuf; ++b) {
          BndId id;
          id.partition = partition;
          id.tag = message[bidx++];
          id.var_id = message[bidx++];
          id.extra_id = message[bidx++];
          id.size = message[bidx++];
          id.start_idx = message[bidx++];
          id.rank_send = other_rank;
          id.rank_recv = Globals::my_rank;
          id.bound_type = static_cast<BoundaryType>(id.extra_id);
          cr_info.push_back(id);
        }
      }
      recv_buffers_built = true;
    }
  }

  void ResolveSendBuffersAndSendInfo() {
    // First calculate the total size of the message
    int total_buffers{0};
    for (auto &[partition, buf_struct_vec] : combined_send_info)
      total_buffers += buf_struct_vec.size();
    int total_partitions = combined_send_info.size();

    int mesg_size = nglobal + nper_part * total_partitions + nper_buf * total_buffers;
    std::vector<int> message(mesg_size);

    message[0] = total_partitions;

    // First store the number of buffers in each partition
    int p{0};
    for (auto &[partition, buf_struct_vec] : combined_send_info) {
      message[nglobal + nper_part * p] = partition; // Used as the comm tag
      message[nglobal + nper_part * p + 1] = buf_struct_vec.size(); // Number of buffers
      p++;
    }

    // Now store the buffer information for each partition,
    // the total size of the message associated with each
    // partition
    int b{0};
    p = 0;
    const int start = nglobal + nper_part * total_partitions;
    std::map<int, int> combined_buf_size;
    for (auto &[partition, buf_struct_vec] : combined_send_info) {
      int total_size{0};
      for (auto &buf_struct : buf_struct_vec) {
        buf_struct.start_idx = total_size;
        message[start + 4 * b + 0] = buf_struct.tag;
        message[start + 4 * b + 1] = buf_struct.var_id;
        message[start + 4 * b + 2] = buf_struct.extra_id;
        message[start + 4 * b + 3] = buf_struct.size;
        message[start + 4 * b + 4] = buf_struct.start_idx;
        total_size += buf_struct.size;
        b++;
      }
      combined_buf_size[partition] = total_size;
      message[nglobal + nper_part * p + 2] = total_size; // Size of combined buffer
      ++p;
    }

    // Send message to other rank
    // TODO(LFR): Actually do this send

    // Allocate the combined buffers
    int total_size{0};
    for (auto &[partition, size] : combined_buf_size)
      total_size += size;

    buf_t alloc_at_once("shared combined buffer", total_size);
    int current_position{0};
    for (auto &[partition, size] : combined_buf_size) {
      combined_send_buffers[partition] =
          buf_t(alloc_at_once, std::make_pair(current_position, current_position + size));
      current_position += size;
    }
    send_buffers_built = true;
  }

  bool ReceiveFinished() { return true; }
};

struct CombinedBuffers {
  // Combined buffers for each rank
  std::map<std::pair<int, BoundaryType>, CombinedBuffersRank> combined_buffers;
  void AddSendBuffer(int partition, MeshBlock *pmb, const NeighborBlock &nb,
                     const std::shared_ptr<Variable<Real>> &var, BoundaryType b_type) {
    combined_buffers[{nb.rank, b_type}].AddSendBuffer(partition, pmb, nb, var, b_type);
  }
};

} // namespace parthenon

#endif // BVALS_COMMS_COMBINED_BUFFERS_HPP_
