//========================================================================================
// (C) (or copyright) 2024. Triad National Security, LLC. All rights reserved.
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// AI was used to assist with NCCL coalesced-buffer state and interfaces.
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

#ifndef BVALS_COMMS_COALESCED_BUFFERS_HPP_
#define BVALS_COMMS_COALESCED_BUFFERS_HPP_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "basic_types.hpp"
#include "bvals/comms/bnd_id.hpp"
#include "bvals/comms/bvals_utils.hpp"
#include "bvals/comms/nccl_utils.hpp"
#include "bvals/neighbor_block.hpp"
#include "config.hpp"
#include "coordinates/coordinates.hpp"
#include "interface/variable.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "utils/communication_buffer.hpp"

namespace parthenon {

// Structure containing the information required for sending coalesced
// messages between ranks
struct CoalescedBuffer {
  using buf_t = BufArray1D<Real>;

  // Rank that these buffers communicate with
  BoundaryType b_type;
  int other_rank;
  int partition;
  mpi_comm_t comm;
  Mesh *pmesh;
  bool sender;

  using var_buf_t = CommBuffer<buf_pool_t<Real>::owner_t>;
  std::map<Uid_t, std::vector<std::pair<BndId, var_buf_t *>>> coalesced_info_buf;
  std::set<Uid_t> all_vars;
  std::unordered_map<std::set<Uid_t>, ParArray1DRaw<BndId>> bnd_ids_device_map;
  std::unordered_map<std::set<Uid_t>, ParArray1DRaw<BndId>::host_mirror_type>
      bnd_ids_host_map;
  CommBuffer<buf_t> coalesced_comm_buffer;
  CommBuffer<std::vector<int>> sparse_status_buffer;
  int current_size;

#ifdef ENABLE_NCCL
  BufArray1D<int> nccl_status_buffer;
  typename BufArray1D<int>::host_mirror_type nccl_status_host;
  std::set<Uid_t> nccl_vars;
  bool nccl_prepared{false};
  CudaEvent nccl_pack_ready;
#endif

  CoalescedBuffer(bool sender, int partition, int other_rank, BoundaryType b_type,
                  mpi_comm_t comm, Mesh *pmesh);

  std::size_t TotalBuffers() const {
    std::size_t total_buffers{0};
    for (const auto &[uid, v] : coalesced_info_buf)
      total_buffers += v.size();
    return total_buffers;
  }

  void AddVarBoundary(BndId &bnd_id);

  void AddVarBoundary(MeshBlock *pmb, const NeighborBlock &nb,
                      const std::shared_ptr<Variable<Real>> &var);

  bool IsAvailableForWrite() {
    return sparse_status_buffer.IsAvailableForWrite() &&
           coalesced_comm_buffer.IsAvailableForWrite();
  }

  ParArray1DRaw<BndId> &GetBndIdsOnDevice(const std::set<Uid_t> &vars,
                                          int *pcomb_size = nullptr);

  void PackAndSend(const std::set<Uid_t> &vars);

  bool TryReceiveAndUnpack(const std::set<Uid_t> &vars);

  void InitializeNCCLBuffers();
  void PrepareNCCL(const std::set<Uid_t> &vars, const DevExecSpace &exec);
  void ApplyNCCLStatusAndUnpack(const std::set<Uid_t> &vars, const DevExecSpace &exec);
  void FinishNCCLSend();
};

struct CoalescedBuffersRank {
  using coalesced_message_structure_t = std::vector<BndId>;
  using buf_t = BufArray1D<Real>;

  // Rank that these buffers communicate with
  BoundaryType b_type;
  int other_rank;

  // map from partion id to coalesced message structure for communication
  // partition id of the sender will be the mpi tag we use
  bool buffers_built{false};

  // Map from MeshData partition id to the coalesced buffer associated
  // with that partition
  std::map<int, CoalescedBuffer> coalesced_bufs;

  static constexpr int nglobal{1};
  static constexpr int nper_part{3};

  using com_buf_t = CommBuffer<std::vector<int>>;
  com_buf_t message;

  mpi_comm_t comm;
  Mesh *pmesh;
  bool sender{true};

  explicit CoalescedBuffersRank(int o_rank, BoundaryType b_type, bool send,
                                mpi_comm_t comm, Mesh *pmesh);

  void AddSendBuffer(int partition, MeshBlock *pmb, const NeighborBlock &nb,
                     const std::shared_ptr<Variable<Real>> &var);

  bool TryReceiveBufInfo();

  void ResolveAndSendBufInfo();

  void PackAndSend(MeshData<Real> *pmd);

  bool TryReceiveAndUnpack(MeshData<Real> *pmd, int partition);

  bool IsAvailableForWrite(MeshData<Real> *pmd);
};

struct CoalescedComms {
  // Combined buffers for each rank
  std::map<std::pair<int, BoundaryType>, CoalescedBuffersRank> coalesced_send_buffers;
  std::map<std::pair<int, BoundaryType>, CoalescedBuffersRank> coalesced_recv_buffers;

  std::map<BoundaryType, mpi_comm_t> comms;

  Mesh *pmesh;

#ifdef ENABLE_NCCL
  struct NCCLExchangeState {
    ncclComm_t comm{nullptr};
    CudaEvent transfer_complete;
    CudaEvent unpack_complete;
    bool in_flight{false};
    bool unpack_recorded{false};
    std::set<Uid_t> vars;
  };

  DevExecSpace nccl_exec;
  std::map<BoundaryType, std::unique_ptr<NCCLExchangeState>> nccl_states;
#endif

  explicit CoalescedComms(Mesh *pmesh);

  ~CoalescedComms();

  void clear();

  void AddSendBuffer(int partition, MeshBlock *pmb, const NeighborBlock &nb,
                     const std::shared_ptr<Variable<Real>> &var, BoundaryType b_type);

  void AddRecvBuffer(MeshBlock *pmb, const NeighborBlock &nb,
                     const std::shared_ptr<Variable<Real>>, BoundaryType b_type);

  void ResolveAndSendSendBuffers();

  void ReceiveBufferInfo();

  void PackAndSend(MeshData<Real> *pmd, BoundaryType b_type);

  bool TryReceiveAny(MeshData<Real> *pmd, BoundaryType b_type);

  bool IsAvailableForWrite(MeshData<Real> *pmd, BoundaryType b_type);

  bool UsesNCCL(BoundaryType b_type) const;
  void InitializeNCCLBuffers();
  void PrepareNCCLSend(MeshData<Real> *pmd, BoundaryType b_type);
  TaskStatus LaunchNCCLExchange(BoundaryType send_type, BoundaryType recv_type);
  TaskStatus PollNCCLExchange(BoundaryType send_type);
  TaskStatus FinalizeNCCLExchange(BoundaryType send_type, BoundaryType recv_type);
  void WaitForNCCLUnpack(BoundaryType send_type, const DevExecSpace &exec);
};

} // namespace parthenon

#endif // BVALS_COMMS_COALESCED_BUFFERS_HPP_
