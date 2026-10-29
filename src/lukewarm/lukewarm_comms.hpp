//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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
#ifndef LUKEWARM_LUKEWARM_COMMS_HPP_
#define LUKEWARM_LUKEWARM_COMMS_HPP_

// This file was made in part with generative AI.

#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "basic_types.hpp"
#include "kokkos_abstraction.hpp"
#include "lukewarm/lukewarm.hpp"
#include "lukewarm/lukewarm_geometry.hpp"
#include "utils/communication_buffer.hpp"

namespace parthenon {

class Mesh;

// Lukewarm particles move between MeshData, not between blocks. Every MeshData that owns
// lukewarms has a channel to each MeshData with a block next to one of its own, on this
// rank or another, and each channel carries one message per field type each time
// particles are exchanged. Particles are packed on device, component by component, and
// the messages go straight from device memory, as they do for fields.

// One buffer per field type, each holding every component of every particle in the
// message: component c of particle k is at c * (number of particles) + k
struct LukewarmBuffers {
  template <typename T>
  using buffer_t = CommBuffer<BufArray1D<T>>;

  std::tuple<buffer_t<Real>, buffer_t<int>, buffer_t<std::uint64_t>> buffers;

  template <typename T>
  buffer_t<T> &Get() {
    return std::get<LukewarmTypes::GetIdx<T>()>(buffers);
  }
};

// Buffers of every channel on this rank, by the ids of the sending and receiving
// MeshData and the name of the lukewarm. Both ends of a channel within a rank share one
// buffer.
using lukewarm_comm_map_t = std::map<std::tuple<int, int, std::string>, LukewarmBuffers>;

// A MeshData with a block next to a block of this one
struct LukewarmNeighbor {
  // The id of a MeshData that owns lukewarms is the smallest gid of its blocks
  int id;
  int rank;
};

// A channel from MeshData `sender` on rank send_rank to MeshData `receiver` on rank
// recv_rank
struct LukewarmChannel {
  int sender, receiver;
  int send_rank, recv_rank;
};

// Where the messages of one channel are, one per field type, or null for an empty one
struct LukewarmMessages {
  Real *real = nullptr;
  int *integer = nullptr;
  std::uint64_t *uint64 = nullptr;

  template <typename T>
  KOKKOS_FORCEINLINE_FUNCTION T *&Get() {
    if constexpr (std::is_same_v<T, Real>) {
      return real;
    } else if constexpr (std::is_same_v<T, int>) {
      return integer;
    } else {
      return uint64;
    }
  }
};

// The channels one lukewarm on one MeshData sends and receives particles through, and
// the device arrays it packs and unpacks them with, which are kept from one exchange to
// the next
class LukewarmComms {
 public:
  LukewarmComms() = default;
  LukewarmComms(const std::vector<LukewarmBuffers *> &send,
                const std::vector<LukewarmBuffers *> &receive);

  std::vector<LukewarmBuffers *> send, receive;

  // Particles waiting to be sent: leaving(k) for k < nleaving, in increasing order, goes
  // through channel destinations(leaving(k)) of send, and counts holds how many go
  // through each. Set by LocateLukewarm(), or by anything else that sends particles.
  int nleaving = 0;
  ParArray1DRaw<int> leaving;
  ParArray1DRaw<int> destinations;
  ParArray1DRaw<int> counts;
  ParArray1DRaw<int>::host_mirror_type counts_h;
  // Make room in leaving and destinations for n particles
  void Reserve(const int n);

  // Particles appended by the last ReceiveLukewarmParticles(), cleared by
  // LocateLukewarm()
  IndexRange received{0, -1};

 private:
  friend bool SendLukewarmParticles(Lukewarm &lukewarm, LukewarmComms &comms);
  friend bool ReceiveLukewarmParticles(Lukewarm &lukewarm, LukewarmComms &comms,
                                       const LukewarmGeometry &geometry);
  ParArray1DRaw<LukewarmMessages> send_messages_, receive_messages_;
  ParArray1DRaw<LukewarmMessages>::host_mirror_type send_messages_h_, receive_messages_h_;
  // Where the particles of each receive channel go among the appended ones
  ParArray1DRaw<int> offsets_;
  ParArray1DRaw<int>::host_mirror_type offsets_h_;
};

// Name of the MPI communicator of lukewarm `name` in Mesh::GetMPIComm()
inline std::string LukewarmCommLabel(const std::string &name) {
  return "lukewarm/" + name;
}

// The id of the MeshData owning each block, by gid, on every rank. This is collective
// over all ranks.
std::vector<int> GatherLukewarmOwners(Mesh *pmesh);

// Add buffers for the channels of lukewarm `name` to map, which has to be given every
// channel with an end on this rank. The channels between two ranks get tags in the
// order of their senders and receivers, the same on both ranks, and a channel within
// this rank gets one buffer for both of its ends.
void AddLukewarmBuffers(Mesh *pmesh, const std::string &name,
                        const std::vector<LukewarmChannel> &channels,
                        lukewarm_comm_map_t &map);

// Give every lukewarm MeshData its geometry, its neighbors, and the buffers of its
// channels, replacing any it had. This is collective over all ranks.
void BuildLukewarmComms(Mesh *pmesh);

// The device side of every exchange of particles between MeshData. Messages go straight
// from and to device memory, as they do for fields.

// Send every particle waiting in comms, an empty message to each channel no particle
// goes through, and remove the particles sent from lukewarm. False, with nothing done,
// while the last messages are still in flight.
bool SendLukewarmParticles(Lukewarm &lukewarm, LukewarmComms &comms);
// Start receiving on every receive channel of comms
void StartReceiveLukewarmParticles(LukewarmComms &comms);
// False until a message has arrived on every receive channel of comms. Then append their
// particles to lukewarm and turn the gid each carries as its block into its local index
// in geometry.
bool ReceiveLukewarmParticles(Lukewarm &lukewarm, LukewarmComms &comms,
                              const LukewarmGeometry &geometry);

} // namespace parthenon

#endif // LUKEWARM_LUKEWARM_COMMS_HPP_
