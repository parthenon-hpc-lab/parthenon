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

struct BufferStructure {
  // These first five variables should be enough information to
  // uniquely identify the buffer
  int tag;    // Tag defining communication channel between blocks
              // (which subsumes send_gid, recv_gid, location_on_block)
              // within a given MPI rank pair
  int var_id; // We use an int for the Uid_t since we will be sending via MPI
  int extra_id;
  int rank_send; // MPI rank of sender
  int rank_recv; // MPI rank of receiver

  // Other information that could be useful to sending messages
  int size;                 // Size of the buffer
  Mesh::channel_key_t key;  // Actual key
  bool currently_allocated; // Current allocation status of the buffer
  int partition;            // Partition of sender
  BoundaryType btype;       // Type of boundary this was registered for

  static BufferStructure Send(int partition, const MeshBlock *const pmb,
                              const NeighborBlock &nb,
                              const std::shared_ptr<Variable<Real>> &var,
                              BoundaryType b_type) {
    BufferStructure out;
    out.tag = pmb->pmy_mesh->tag_map.GetTag(pmb, nb);
    out.var_id = var->GetUniqueID();
    out.extra_id = static_cast<int>(b_type);
    out.rank_send = Globals::my_rank;
    out.rank_recv = nb.rank;

    out.key = SendKey(pmb, nb, var, b_type);
    out.size = GetBufferSize(pmb, nb, var);
    out.currently_allocated = true;
    out.partition = partition;
    out.btype = b_type;
    return out;
  }
};

// Structure containing the information required for sending coalesced
// messages between ranks
struct CombinedBuffersRank {
  using coalesced_message_structure_t = std::vector<BufferStructure>;

  // Rank that these buffers communicate with
  int other_rank;

  // map from partion id to coalesced message structure for communication
  // from this rank to other_rank
  std::map<int, coalesced_message_structure_t> combined_send_info;
  std::map<int, ParArray1D<Real>> combined_send_buffers;

  // map from neighbor partition id to coalesced message structures that
  // this rank can receive from other_rank. We will use the partition id
  // as the mpi tag
  std::map<int, coalesced_message_structure_t> combined_recv_info;
  std::map<int, ParArray1D<Real>> combined_recv_buffers;

  void AddSendBuffer(int partition, const MeshBlock *const &pmb, const NeighborBlock &nb,
                     const std::shared_ptr<Variable<Real>> &var, BoundaryType b_type) {
    combined_send_info[partition].push_back(
        BufferStructure::Send(partition, pmb, nb, var, b_type));
  }
};

struct CombinedBuffers {
  // Combined buffers for each rank
  std::vector<CombinedBuffersRank> combined_buffers;
  void AddSendBuffer(int partition, const MeshBlock *const pmb, const NeighborBlock &nb,
                     const std::shared_ptr<Variable<Real>> &var, BoundaryType b_type) {
    combined_buffers[nb.rank].AddSendBuffer(partition, pmb, nb, var, b_type);
  }
};

} // namespace parthenon

#endif // BVALS_COMMS_COMBINED_BUFFERS_HPP_
