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
  // partition id of the sender will be the mpi tag we use
  bool buffers_built{false};
  std::map<int, coalesced_message_structure_t> combined_info;
  std::map<int, BufArray1D<Real>> combined_buffers;
  std::map<int, int> current_size;

  static constexpr int nglobal{1};
  static constexpr int nper_part{3};

  using com_buf_t = CommBuffer<std::vector<int>>;
  com_buf_t message;

  bool sender{true};
  CombinedBuffersRank() = default;
  CombinedBuffersRank(int o_rank, BoundaryType b_type, bool send);

  void AddSendBuffer(int partition, MeshBlock *pmb, const NeighborBlock &nb,
                     const std::shared_ptr<Variable<Real>> &var, BoundaryType b_type);

  bool TryReceiveBufInfo();

  void ResolveSendBuffersAndSendInfo();
};

struct CombinedBuffers {
  // Combined buffers for each rank
  std::map<std::pair<int, BoundaryType>, CombinedBuffersRank> combined_send_buffers;
  std::map<std::pair<int, BoundaryType>, CombinedBuffersRank> combined_recv_buffers;

  void clear() {
    // TODO(LFR): Need to be careful here that the asynchronous send buffers are finished
    combined_send_buffers.clear();
    combined_recv_buffers.clear();
  }

  void AddSendBuffer(int partition, MeshBlock *pmb, const NeighborBlock &nb,
                     const std::shared_ptr<Variable<Real>> &var, BoundaryType b_type) {
    if (combined_send_buffers.count({nb.rank, b_type}) == 0)
      combined_send_buffers[{nb.rank, b_type}] =
          CombinedBuffersRank(nb.rank, b_type, true);
    combined_send_buffers[{nb.rank, b_type}].AddSendBuffer(partition, pmb, nb, var,
                                                           b_type);
  }

  void AddRecvBuffer(MeshBlock *pmb, const NeighborBlock &nb,
                     const std::shared_ptr<Variable<Real>>, BoundaryType b_type) {
    // We don't actually know enough here to register this particular buffer, but we do
    // know that it's existence implies that we need to receive a message from the
    // neighbor block rank eventually telling us the details
    if (combined_recv_buffers.count({nb.rank, b_type}) == 0)
      combined_recv_buffers[{nb.rank, b_type}] =
          CombinedBuffersRank(nb.rank, b_type, false);
  }

  void ResolveAndSendSendBuffers() {
    for (auto &[id, buf] : combined_send_buffers)
      buf.ResolveSendBuffersAndSendInfo();
  }

  void ReceiveBufferInfo() {
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
};

} // namespace parthenon

#endif // BVALS_COMMS_COMBINED_BUFFERS_HPP_
