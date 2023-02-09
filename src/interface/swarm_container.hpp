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
#ifndef INTERFACE_SWARM_CONTAINER_HPP_
#define INTERFACE_SWARM_CONTAINER_HPP_

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

class SwarmContainer {
 public:
  //-----------------
  // Public Methods
  //-----------------
  // Constructor does nothing
  SwarmContainer() {}

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

  void Add(std::shared_ptr<Swarm> swarm) {
    swarmVector_.push_back(swarm);
    swarmMap_[swarm->label()] = swarm;
    UpdateMetadataMap_(swarm);
  }

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

  void AllocateBoundaries();

  const SwarmVector &GetSwarmVector() const { return swarmVector_; }
  const SwarmMap &GetSwarmMap() const { return swarmMap_; }

  ///
  /// Remove a variable from the container or throw exception if not
  /// found.
  /// @param label the name of the variable to be deleted
  void Remove(const std::string &label);

  // Temporary functions till we implement a *real* iterator

  /// Print list of labels in container
  void Print() const;

  // return number of stored arrays
  int Size() const { return swarmVector_.size(); }

  // Element accessor functions
  std::vector<std::shared_ptr<Swarm>> &allSwarms() { return swarmVector_; }

  // Return swarms 

  // Defragmentation task
  TaskStatus Defrag(double min_occupancy);
  TaskStatus DefragAll() {
    return Defrag(1.0);
  }

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

  bool operator==(const SwarmContainer &cmp) {
    // Test that labels of swarms are the same
    std::vector<std::string> my_keys(swarmMap_.size());
    auto &cmpMap = cmp.GetSwarmMap();
    std::vector<std::string> cmp_keys(cmpMap.size());
    size_t i = 0;
    for (auto &s : swarmMap_) {
      my_keys[i] = s.first;
      i++;
    }
    i = 0;
    for (auto &s : cmpMap) {
      cmp_keys[i] = s.first;
      i++;
    }
    return my_keys == cmp_keys;
  }

 private:
  void UpdateMetadataMap_(std::shared_ptr<Swarm> swarm) {
    // for (const auto &flag : swarm->metadata().Flags()) {
    //   swarmMetadataMap_[flag].push_back(swarm);
    // }
  }

  int debug = 0;
  std::weak_ptr<MeshBlock> pmy_block;

  SwarmVector swarmVector_ = {};
  SwarmMap swarmMap_ = {};
  SwarmMetadataMap swarmMetadataMap_ = {};
};

} // namespace parthenon
#endif // INTERFACE_SWARM_CONTAINER_HPP_
