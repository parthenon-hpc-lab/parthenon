//========================================================================================
// (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
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
#ifndef INTERFACE_SWARM_HPP_
#define INTERFACE_SWARM_HPP_

///
/// A swarm contains all particles of a particular species
/// Date: August 21, 2019

#include <array>
#include <cstdint>
#include <list>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "basic_types.hpp"
#include "bvals/bvals_interfaces.hpp"
#include "globals.hpp" // my_rank
#include "metadata.hpp"
#include "parthenon_arrays.hpp"
#include "parthenon_mpi.hpp"
#include "swarm_boundaries.hpp"
#include "swarm_device_context.hpp"
#include "variable.hpp"
#include "variable_pack.hpp"

namespace parthenon {

struct BoundaryDeviceContext {
  ParticleBound *bounds[6];
};

class MeshBlock;

enum class PARTICLE_STATUS { UNALLOCATED, ALIVE, DEAD };

class Swarm {
 private:
  static const int IntVec = 0;
  static const int RealVec = 1;

  template <class T>
  static constexpr int getType() {
    if (std::is_same<T, int>::value) {
      return IntVec;
    }
    return RealVec;
  }

 public:
  Swarm(const std::string &label, const Metadata &metadata, const int nmax_pool_in = 3);

  ~Swarm() = default;

  /// Returns shared pointer to a block
  std::shared_ptr<MeshBlock> GetBlockPointer() const {
    if (pmy_block.expired()) {
      PARTHENON_THROW("Invalid pointer to MeshBlock!");
    }
    return pmy_block.lock();
  }

  SwarmDeviceContext GetDeviceContext() const;

  void AllocateBoundaries();

  // Set the pointer to the mesh block for this swarm
  void SetBlockPointer(std::weak_ptr<MeshBlock> pmb) { pmy_block = pmb; }

  /// Make a new Swarm based on an existing one
  std::shared_ptr<Swarm> AllocateCopy(const bool alloc_separate_fluxes_and_bvar = false,
                                      MeshBlock *pmb = nullptr);

  /// Add variable of given type to swarm
  template <class T>
  void Add_(const std::string &label);

  /// Add variable to swarm
  void Add(const std::string &label, const Metadata &metadata);

  /// Add multiple variables with common metadata to swarm
  void Add(const std::vector<std::string> &labelVector, const Metadata &metadata);

  /// Remote a variable from swarm
  void Remove(const std::string &label);

  /// Set a custom boundary condition
  void SetBoundary(
      const int n,
      std::unique_ptr<ParticleBound, parthenon::DeviceDeleter<parthenon::DevMemSpace>>
          bc) {
    bounds_uptrs[n] = std::move(bc);
    bounds_d.bounds[n] = bounds_uptrs[n].get();
  }

  /// Get particle variable
  template <class T>
  ParticleVariable<T> &Get(const std::string &label) {
    return *std::get<getType<T>()>(Maps_).at(label);
  }

  /// Assign label for swarm
  void setLabel(const std::string &label) { label_ = label; }

  /// retrieve label for swarm
  std::string label() const { return label_; }

  /// retrieve metadata for swarm
  const Metadata metadata() const { return m_; }

  /// Assign info for swarm
  void setInfo(const std::string &info) { info_ = info; }

  /// return information string
  std::string info() const { return info_; }

  /// Expand pool size geometrically as necessary
  void increasePoolMax() { setPoolMax(2 * nmax_pool_); }

  /// Set max pool size
  void setPoolMax(const int nmax_pool);

  /// Check whether metadata bit is set
  bool IsSet(const MetadataFlag bit) const { return m_.IsSet(bit); }

  /// Get the last index of active particles
  int GetMaxActiveIndex() const { return max_active_index_; }

  /// Get number of active particles
  int GetNumActive() const { return num_active_; }

  /// Get mask variable
  auto GetMask() const { return mask_; }

  /// Get the quality of the data layout. 1 is perfectly organized, < 1
  /// indicates gaps in the list.
  Real GetPackingEfficiency() const { return num_active_ / (max_active_index_ + 1); }

  /// Remove particles marked for removal and update internal indexing
  void RemoveMarkedParticles();

  /// Open up memory for new empty particles, return a mask to these particles
  ParArrayND<bool> AddEmptyParticles(const int num_to_add, ParArrayND<int> &new_indices);

  /// Defragment the list by moving active particles so they are contiguous in
  /// memory
  void Defrag();

  // used in case of swarm boundary communication
  void SetupPersistentMPI();
  std::shared_ptr<BoundarySwarm> vbswarm;
  bool mpiStatus;
  void AllocateComms(std::weak_ptr<MeshBlock> wpmb);

  // This is the particle data size for indexing boundary data buffers, for which
  // integers are cast as Reals.
  int GetParticleDataSize() {
    return std::get<0>(Vectors_).size() + std::get<1>(Vectors_).size();
  }

  void Send(BoundaryCommSubset phase);

  bool Receive(BoundaryCommSubset phase);

  void ResetCommunication();

  bool FinalizeCommunicationIterative();

  template <class T>
  SwarmVariablePack<T> PackAllVariables(PackIndexMap &vmap);

  template <class T>
  SwarmVariablePack<T> PackVariables(const std::vector<std::string> &name,
                                     PackIndexMap &vmap);

  // Temporarily public
  int num_particles_sent_;
  bool finished_transport;

  // Class to store raw pointers to boundary conditions on device. Copy locally for
  // compute kernel capture.
  BoundaryDeviceContext bounds_d;

  void LoadBuffers_(const int max_indices_size);
  void UnloadBuffers_();

  void ApplyBoundaries_(const int nparticles, ParArrayND<int> indices);

  std::unique_ptr<ParticleBound, DeviceDeleter<parthenon::DevMemSpace>> bounds_uptrs[6];

 private:
  template <class T>
  vpack_types::SwarmVarList<T> MakeVarListAll_();

  void SetNeighborIndices1D_();
  void SetNeighborIndices2D_();
  void SetNeighborIndices3D_();

  int CountParticlesToSend_();
  void CountReceivedParticles_();
  void UpdateNeighborBufferReceiveIndices_(ParArrayND<int> &neighbor_index,
                                           ParArrayND<int> &buffer_index);

  int debug = 0;
  std::weak_ptr<MeshBlock> pmy_block;

  int nmax_pool_;
  int max_active_index_ = 0;
  int num_active_ = 0;
  Metadata m_;
  std::string label_;
  std::string info_;
  std::shared_ptr<ParArrayND<PARTICLE_STATUS>> pstatus_;
  std::tuple<ParticleVariableVector<int>, ParticleVariableVector<Real>> Vectors_;

  std::tuple<MapToParticle<int>, MapToParticle<Real>> Maps_;

  std::list<int> free_indices_;
  ParticleVariable<bool> mask_;
  ParticleVariable<bool> marked_for_removal_;
  ParticleVariable<int> neighbor_send_index_; // -1 means no send
  ParArrayND<int> neighborIndices_; // Indexing of vbvar's neighbor array. -1 for same.
                                    // k,j indices unused in 3D&2D, 2D, respectively
  ParArrayND<int> blockIndex_; // Neighbor index for each particle. -1 for current block.

  constexpr static int this_block_ = -1;
  constexpr static int unset_index_ = -1;

  ParArrayND<int> num_particles_to_send_;
  ParArrayND<int> particle_indices_to_send_;

  std::vector<int> neighbor_received_particles_;
  int total_received_particles_;

  ParArrayND<int> neighbor_buffer_index_; // Map from neighbor index to neighbor bufid
};

template <class T>
inline vpack_types::SwarmVarList<T> Swarm::MakeVarListAll_() {
  int size = 0;
  vpack_types::SwarmVarList<T> vars;
  auto variables = std::get<getType<T>()>(Vectors_);
  for (auto it = variables.rbegin(); it != variables.rend(); ++it) {
    auto v = *it;
    vars.push_front(v);
    size++;
  }
  return vars;
}

template <class T>
inline SwarmVariablePack<T> Swarm::PackVariables(const std::vector<std::string> &names,
                                                 PackIndexMap &vmap) {
  vpack_types::SwarmVarList<T> vars = MakeVarListAll_<T>();
  auto pack = MakeSwarmPack<T>(vars, &vmap);
  SwarmPackIndxPair<T> value;
  value.pack = pack;
  value.map = vmap;
  return pack;
}

template <class T>
inline SwarmVariablePack<T> Swarm::PackAllVariables(PackIndexMap &vmap) {
  std::vector<std::string> names;
  names.reserve(std::get<getType<T>()>(Vectors_).size());
  for (const auto &v : std::get<getType<T>()>(Vectors_)) {
    names.push_back(v->label());
  }
  return PackVariables<T>(names, vmap);
}

template <class T>
inline void Swarm::Add_(const std::string &label) {
  ParticleVariable<T> pvar(label, nmax_pool_, m_);
  auto var = std::make_shared<ParticleVariable<T>>(pvar);

  std::get<getType<T>()>(Vectors_).push_back(var);
  std::get<getType<T>()>(Maps_)[label] = var;
}

using SP_Swarm = std::shared_ptr<Swarm>;
using SwarmVector = std::vector<SP_Swarm>;
using SwarmMap = std::unordered_map<std::string, SP_Swarm>;

} // namespace parthenon

#endif // INTERFACE_SWARM_HPP_
