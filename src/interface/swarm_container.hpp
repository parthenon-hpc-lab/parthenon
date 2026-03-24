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
#ifndef INTERFACE_SWARM_CONTAINER_HPP_
#define INTERFACE_SWARM_CONTAINER_HPP_

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "globals.hpp"
#include "swarm.hpp"

namespace parthenon {
///
/// Interface to underlying infrastructure for particle data declaration and
/// access.
/// Date: August 22, 2019
///
///
/// The SwarmContainer class is a container for the swarms of particles that
/// make up the simulation.
///
/// The container class will provide the following methods:
///

class MeshBlock;
template <typename T>
class MeshData;

class MeshNewParticlesContext {
 public:
  KOKKOS_DEFAULTED_FUNCTION
  MeshNewParticlesContext() = default;

  KOKKOS_FUNCTION
  MeshNewParticlesContext(const ParArray1D<NewParticlesContext> &block_contexts,
                          const ParArray1D<int> &flat_index_map, const int nblocks,
                          const int max_flat_index)
      : block_contexts_(block_contexts), flat_index_map_(flat_index_map),
        nblocks_(nblocks), max_flat_index_(max_flat_index) {}

  KOKKOS_INLINE_FUNCTION
  int GetNBlocks() const { return nblocks_; }

  KOKKOS_INLINE_FUNCTION
  int GetMaxFlatIndex() const { return max_flat_index_; }

  KOKKOS_INLINE_FUNCTION
  auto GetBlockParticleIndices(const int idx) const {
    PARTHENON_DEBUG_REQUIRE(idx >= 0 && idx <= max_flat_index_,
                            "Requested new-particle flat index out of bounds!");
    int b = 0;
    int r = nblocks_;
    while (r - b > 1) {
      const int c = static_cast<int>(0.5 * (b + r));
      if (flat_index_map_(c) > idx) {
        r = c;
      } else {
        b = c;
      }
    }
    return std::make_tuple(b, idx - flat_index_map_(b));
  }

  KOKKOS_INLINE_FUNCTION
  int GetNewParticleIndex(const int b, const int n) const {
    return block_contexts_(b).GetNewParticleIndex(n);
  }

 private:
  ParArray1D<NewParticlesContext> block_contexts_;
  ParArray1D<int> flat_index_map_;
  int nblocks_ = 0;
  int max_flat_index_ = -1;
};

class SwarmContainer {
 public:
  //-----------------
  // Public Methods
  //-----------------
  // Constructor does nothing
  SwarmContainer() = default;
  explicit SwarmContainer(const std::string &name) : swarm_name_(name) {}

  /// Returns a shared pointer to a block
  std::shared_ptr<MeshBlock> GetBlockPointer() {
    if (pmy_block.expired()) {
      PARTHENON_THROW("Invalid pointer to MeshBlock!");
    }
    return pmy_block.lock();
  }

  /// Set the pointer to the mesh block for this swarm container
  void SetBlockPointer(std::weak_ptr<MeshBlock> pmb) { pmy_block = pmb; }
  void SetBlockPointer(const std::shared_ptr<SwarmContainer> &other) {
    pmy_block = other->GetBlockPointer();
  }

  // TODO(BRR) also add Add() functions for setting single int, real, string
  // values?

  ///
  /// Allocate and add a variable<T> to the container
  ///
  /// This function will eventually look at the metadata flags to
  /// identify the size of the first dimension based on the
  /// topological location.  Dimensions will be taken from the metadata.
  ///
  /// @param label the name of the variable
  /// @param metadata the metadata associated with the variable
  ///
  void Add(const std::string &label, const Metadata &metadata);

  ///
  /// Allocate and add a variable<T> to the container
  ///
  /// This function will eventually look at the metadata flags to
  /// identify the size of the first dimension based on the
  /// topological location.  Dimensions will be taken from the metadata.
  ///
  /// @param labelArray the array of names of variables
  /// @param metadata the metadata associated with the variable
  ///
  void Add(const std::vector<std::string> &labelArray, const Metadata &metadata);

  void Initialize(const std::shared_ptr<StateDescriptor> resolved_packages,
                  const std::shared_ptr<MeshBlock> pmb);

  void Add(std::shared_ptr<Swarm> swarm) {
    swarmVector_.push_back(swarm);
    swarmMap_[swarm->label()] = swarm;
    UpdateMetadataMap_(swarm);
  }

  bool Contains(const std::string &label) const { return swarmMap_.count(label); }

  ///
  /// Get a swarm from the container
  /// @param label the name of the swarm
  /// @return the Swarm if found or throw exception
  std::shared_ptr<Swarm> &Get(const std::string &label) {
    if (swarmMap_.count(label) == 0) {
      throw std::invalid_argument(std::string("\n") + std::string(label) +
                                  std::string(" swarm not found in Get()\n"));
    }
    return swarmMap_[label];
  }

  std::shared_ptr<Swarm> &Get(const int index) { return swarmVector_[index]; }

  int Index(const std::string &label) const {
    for (int i = 0; i < swarmVector_.size(); i++) {
      if (!swarmVector_[i]->label().compare(label)) return i;
    }
    return -1;
  }

  const SwarmVector &GetSwarmVector() const { return swarmVector_; }
  const SwarmMap &GetSwarmMap() const { return swarmMap_; }

  ///
  /// Remove a variable from the container or throw exception if not
  /// found.
  /// @param label the name of the variable to be deleted
  /// TODO(JMM): Should we support this operation?
  void Remove(const std::string &label);

  // Temporary functions till we implement a *real* iterator

  /// Print list of labels in container
  void Print() const;

  // return number of stored arrays
  int Size() const { return swarmVector_.size(); }

  // Element accessor functions
  std::vector<std::shared_ptr<Swarm>> &allSwarms() { return swarmVector_; }

  // Return swarms meeting some conditions
  SwarmSet GetSwarmsByFlag(const Metadata::FlagCollection &flags);

  // Defragmentation task
  TaskStatus Defrag(double min_occupancy);
  TaskStatus DefragAll();

  // Sort-by-cell task
  TaskStatus SortParticlesByCell();

  // Communication routines
  void SetupPersistentMPI();
  [[deprecated("Not yet implemented")]] void SetBoundaries();
  [[deprecated("Not yet implemented")]] void SendBoundaryBuffers();
  [[deprecated("Not yet implemented")]] void ReceiveAndSetBoundariesWithWait();
  [[deprecated("Not yet implemented")]] bool ReceiveBoundaryBuffers();
  TaskStatus StartCommunication(BoundaryCommSubset phase);
  TaskStatus Send(BoundaryCommSubset phase);
  TaskStatus Receive(BoundaryCommSubset phase);
  TaskStatus ResetCommunication();
  TaskStatus FinalizeCommunicationIterative();
  [[deprecated("Not yet implemented")]] void ClearBoundary(BoundaryCommSubset phase);

  bool operator==(const SwarmContainer &cmp);

 private:
  void UpdateMetadataMap_(std::shared_ptr<Swarm> swarm) {
    for (const auto &flag : swarm->metadata().Flags()) {
      swarmMetadataMap_[flag].insert(swarm);
    }
  }

  std::string swarm_name_;
  int debug = 0;
  std::weak_ptr<MeshBlock> pmy_block;

  SwarmVector swarmVector_ = {};
  SwarmMap swarmMap_ = {};
  SwarmMetadataMap swarmMetadataMap_ = {};
};

TaskStatus ResetSwarmCommunication(std::shared_ptr<MeshData<Real>> &md);
TaskStatus SendSwarms(std::shared_ptr<MeshData<Real>> &md);
TaskStatus ReceiveSwarms(std::shared_ptr<MeshData<Real>> &md);
TaskStatus RemoveMarkedParticles(std::shared_ptr<MeshData<Real>> &md,
                                 const std::string &swarm_name);
TaskStatus RemoveMarkedParticles(MeshData<Real> *md, const std::string &swarm_name);
TaskStatus DefragSwarms(std::shared_ptr<MeshData<Real>> &md, double min_occupancy);
TaskStatus DefragAllSwarms(std::shared_ptr<MeshData<Real>> &md);
MeshNewParticlesContext AddEmptyParticles(std::shared_ptr<MeshData<Real>> &md,
                                          const std::string &swarm_name,
                                          const ParArray1D<int> &num_to_add);
MeshNewParticlesContext AddEmptyParticles(MeshData<Real> *md,
                                          const std::string &swarm_name,
                                          const ParArray1D<int> &num_to_add);

} // namespace parthenon
#endif // INTERFACE_SWARM_CONTAINER_HPP_
