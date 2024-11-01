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

#ifndef BVALS_COMMS_COMBINED_BUFFERS_HPP_
#define BVALS_COMMS_COMBINED_BUFFERS_HPP_

#include <map>
#include <memory>
#include <set>
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

struct CombinedBuffersRankPartition {
  using buf_t = BufArray1D<Real>;

  // Rank that these buffers communicate with
  BoundaryType b_type;
  int other_rank;
  int partition;
  mpi_comm_t comm_;
  Mesh *pmesh;
  bool sender;

  using var_buf_t = CommBuffer<buf_pool_t<Real>::owner_t>;
  std::map<Uid_t, std::vector<std::pair<BndId, var_buf_t *>>> combined_info_buf;
  ParArray1D<BndId> bnd_ids_device;
  CommBuffer<buf_t> combined_comm_buffer;
  int current_size;

  CombinedBuffersRankPartition(bool sender, int partition, int other_rank,
                               BoundaryType b_type, mpi_comm_t comm, Mesh *pmesh)
      : sender(sender), partition(partition), other_rank(other_rank), b_type(b_type),
        comm_(comm), pmesh(pmesh), current_size(0) {}

  int TotalBuffers() const {
    int total_buffers{0};
    for (const auto &[uid, v] : combined_info_buf)
      total_buffers += v.size();
    return total_buffers;
  }

  void AddVarBoundary(BndId &bnd_id);

  void AddVarBoundary(MeshBlock *pmb, const NeighborBlock &nb,
                      const std::shared_ptr<Variable<Real>> &var);

  void AllocateCombinedBuffer();

  bool IsAvailableForWrite() { return combined_comm_buffer.IsAvailableForWrite(); }

  // TODO(LFR): Functions below should take a list of uids
  void RebuildBndIdsOnDevice();

  void PackAndSend();

  bool TryReceiveAndUnpack(mpi_message_t *message);
};

struct CombinedBuffersRank {
  using coalesced_message_structure_t = std::vector<BndId>;
  using buf_t = BufArray1D<Real>;

  // Rank that these buffers communicate with
  BoundaryType b_type;
  int other_rank;

  // map from partion id to coalesced message structure for communication
  // partition id of the sender will be the mpi tag we use
  bool buffers_built{false};

  std::map<int, CombinedBuffersRankPartition> combined_bufs;

  static constexpr int nglobal{1};
  static constexpr int nper_part{3};

  using com_buf_t = CommBuffer<std::vector<int>>;
  com_buf_t message;

  mpi_comm_t comm_;
  Mesh *pmesh;
  bool sender{true};

  explicit CombinedBuffersRank(int o_rank, BoundaryType b_type, bool send,
                               mpi_comm_t comm, Mesh *pmesh);

  void AddSendBuffer(int partition, MeshBlock *pmb, const NeighborBlock &nb,
                     const std::shared_ptr<Variable<Real>> &var);

  bool TryReceiveBufInfo();

  void ResolveSendBuffersAndSendInfo();

  void PackAndSend(int partition);

  bool TryReceiveAndUnpack(int partition, mpi_message_t *message);

  void RepointBuffers(int partition);

  bool IsAvailableForWrite(int partition);
};

struct CombinedBuffers {
  // Combined buffers for each rank
  std::map<std::pair<int, BoundaryType>, CombinedBuffersRank> combined_send_buffers;
  std::map<std::pair<int, BoundaryType>, CombinedBuffersRank> combined_recv_buffers;

  std::map<std::pair<int, int>, mpi_message_t> processing_messages;

  std::map<BoundaryType, mpi_comm_t> comms_;

  Mesh *pmesh;

  explicit CombinedBuffers(Mesh *pmesh) : pmesh(pmesh) {
    // TODO(LFR): Switch to a different communicator for each BoundaryType pair
    for (auto b_type :
         {BoundaryType::any, BoundaryType::flxcor_send, BoundaryType::gmg_same,
          BoundaryType::gmg_restrict_send, BoundaryType::gmg_prolongate_send}) {
      auto &comm = comms_[b_type];
#ifdef MPI_PARALLEL
      PARTHENON_MPI_CHECK(MPI_Comm_dup(MPI_COMM_WORLD, &comm));
#else
      comm = 0;
#endif
    }
  }

  ~CombinedBuffers() {
#ifdef MPI_PARALLEL
    for (auto &[b_type, comm] : comms_)
      PARTHENON_MPI_CHECK(MPI_Comm_free(&comm));
#endif
  }

  void clear() {
    // TODO(LFR): Need to be more careful here that the asynchronous send buffers are
    // finished
    combined_send_buffers.clear();
    combined_recv_buffers.clear();
    processing_messages.clear();
  }

  void AddSendBuffer(int partition, MeshBlock *pmb, const NeighborBlock &nb,
                     const std::shared_ptr<Variable<Real>> &var, BoundaryType b_type);

  void AddRecvBuffer(MeshBlock *pmb, const NeighborBlock &nb,
                     const std::shared_ptr<Variable<Real>>, BoundaryType b_type);

  void ResolveAndSendSendBuffers();

  void ReceiveBufferInfo();

  void PackAndSend(int partition, BoundaryType b_type);

  void RepointSendBuffers(int partition, BoundaryType b_type);

  void RepointRecvBuffers(int partition, BoundaryType b_type);

  void TryReceiveAny(BoundaryType b_type);

  bool IsAvailableForWrite(int partition, BoundaryType b_type);
};

} // namespace parthenon

#endif // BVALS_COMMS_COMBINED_BUFFERS_HPP_
