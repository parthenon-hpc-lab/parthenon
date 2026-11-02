//========================================================================================
// (C) (or copyright) 2024. Triad National Security, LLC. All rights reserved.
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// AI was used to assist with NVSHMEM coalesced-buffer interface changes.
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

#include <cstddef>
#include <cstdint>
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
#include "bvals/neighbor_block.hpp"
#include "coordinates/coordinates.hpp"
#include "config.hpp"
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

#ifdef PARTHENON_ENABLE_NVSHMEM
  bool use_nvshmem{false};
  std::size_t local_data_offset{0};
  std::size_t remote_data_offset{0};
  std::size_t local_completion_index{0};
  std::size_t remote_completion_index{0};
  std::size_t local_ack_index{0};
  std::size_t remote_ack_index{0};
  std::uint64_t send_epoch{0};
  std::uint64_t receive_epoch{0};
  Real *local_symmetric_data{nullptr};
  Real *remote_symmetric_data{nullptr};
  std::uint64_t *local_signals{nullptr};
  std::uint64_t *remote_completion_signal{nullptr};
  std::uint64_t *remote_ack_signal{nullptr};

  void ConfigureNvshmemSender(Real *local_data, std::uint64_t *signals,
                              std::size_t remote_offset,
                              std::size_t remote_completion);
  void ConfigureNvshmemReceiver(Real *local_data, std::uint64_t *signals,
                                std::size_t local_offset,
                                std::size_t local_completion);
  void WarnMpiFallback(const char *operation) const;
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

  bool IsAvailableForWrite();

  ParArray1DRaw<BndId> &GetBndIdsOnDevice(const std::set<Uid_t> &vars,
                                          int *pcomb_size = nullptr,
                                          bool use_combined_buffer = true);

  void PackAndSend(const std::set<Uid_t> &vars);

  bool TryReceiveAndUnpack(const std::set<Uid_t> &vars);
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
#ifdef PARTHENON_ENABLE_NVSHMEM
  static constexpr int nper_part{4};
#else
  static constexpr int nper_part{3};
#endif

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

#ifdef PARTHENON_ENABLE_NVSHMEM
  Real *symmetric_data{nullptr};
  std::uint64_t *symmetric_signals{nullptr};
  std::size_t symmetric_data_size{0};
  std::size_t symmetric_signal_count{0};
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

#ifdef PARTHENON_ENABLE_NVSHMEM
  void PrepareNvshmemMetadata();
  void SetupNvshmemArena();
  void ReleaseNvshmemArena();
#endif

  void PackAndSend(MeshData<Real> *pmd, BoundaryType b_type);

  bool TryReceiveAny(MeshData<Real> *pmd, BoundaryType b_type);

  bool IsAvailableForWrite(MeshData<Real> *pmd, BoundaryType b_type);
};

} // namespace parthenon

#endif // BVALS_COMMS_COALESCED_BUFFERS_HPP_
