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
#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>

#include "globals.hpp" // my_rank
#include "mesh/mesh.hpp"
#include "swarm_container.hpp"

namespace parthenon {

void SwarmContainer::Add(const std::vector<std::string> &labelArray,
                         const Metadata &metadata) {
  // generate the vector and call Add
  for (const auto &label : labelArray) {
    Add(label, metadata);
  }
}

///
/// The internal routine for allocating a particle swarm.  This subroutine
/// is topology aware and will allocate accordingly.
///
/// @param label the name of the variable
/// @param metadata the metadata associated with the particle
void SwarmContainer::Add(const std::string &label, const Metadata &metadata) {
  if (swarmMap_.find(label) != swarmMap_.end()) {
    throw std::invalid_argument("swarm " + label + " already enrolled during Add()!");
  }

  auto swarm = std::make_shared<Swarm>(label, metadata);
  swarm->SetBlockPointer(GetBlockPointer());
  PARTHENON_WARN("We commented out AllocateComms! This code is going away!");
  //swarm->AllocateComms(GetBlockPointer());
  swarmVector_.push_back(swarm);
  swarmMap_[label] = swarm;
}

void SwarmContainer::Remove(const std::string &label) {
  // Find index of swarm
  int isize = swarmVector_.size();
  int idx = 0;
  for (const auto &s : swarmVector_) {
    if (!label.compare(s->label())) {
      break;
    }
    idx++;
  }
  if (idx >= isize) {
    throw std::invalid_argument("swarm not found in Remove()");
  }

  // first delete the variable
  swarmVector_[idx].reset();

  // Next move the last element into idx and pop last entry
  isize--;
  if (isize >= 0) swarmVector_[idx] = std::move(swarmVector_.back());
  swarmVector_.pop_back();

  // Also remove swarm from map
  swarmMap_.erase(label);
}

TaskStatus SwarmContainer::Defrag(double min_occupancy) {
  Kokkos::Profiling::pushRegion("Task_SwarmContainer_Defrag");
  PARTHENON_REQUIRE_THROWS(min_occupancy >= 0. && min_occupancy <= 1.,
                           "Max fractional occupancy of swarm must be >= 0 and <= 1");

  for (auto &s : swarmVector_) {
    if (s->GetNumActive() > 0 &&
        s->GetNumActive() / (s->GetMaxActiveIndex() + 1.0) < min_occupancy) {
      s->Defrag();
    }
  }

  Kokkos::Profiling::popRegion();

  return TaskStatus::complete;
}

TaskStatus SwarmContainer::SortParticlesByCell() {
  Kokkos::Profiling::pushRegion("Task_SwarmContainer_SortParticlesByCell");

  for (auto &s : swarmVector_) {
    s->SortParticlesByCell();
  }

  Kokkos::Profiling::popRegion();

  return TaskStatus::complete;
}

void SwarmContainer::SendBoundaryBuffers() {}

void SwarmContainer::SetupPersistentMPI() {
  for (auto &s : swarmVector_) {
    s->SetupPersistentMPI();
  }
}

bool SwarmContainer::ReceiveBoundaryBuffers() { return true; }

void SwarmContainer::ReceiveAndSetBoundariesWithWait() {}

void SwarmContainer::SetBoundaries() {}

void SwarmContainer::AllocateBoundaries() {
  for (auto &s : swarmVector_) {
    s->AllocateBoundaries();
  }
}

TaskStatus SwarmContainer::Send(BoundaryCommSubset phase) {
  Kokkos::Profiling::pushRegion("Task_SwarmContainer_Send");

  for (auto &s : swarmVector_) {
    s->Send(phase);
  }

  Kokkos::Profiling::popRegion(); // Task_SwarmContainer_Send
  return TaskStatus::complete;
}

TaskStatus SwarmContainer::Receive(BoundaryCommSubset phase) {
  Kokkos::Profiling::pushRegion("Task_SwarmContainer_Receive");

  int success = 0, total = 0;
  for (auto &s : swarmVector_) {
    if (s->Receive(phase)) {
      success++;
    }
    total++;
  }

  Kokkos::Profiling::popRegion(); // Task_SwarmContainer_Receive
  if (success == total) return TaskStatus::complete;
  return TaskStatus::incomplete;
}

TaskStatus SwarmContainer::ResetCommunication() {
  Kokkos::Profiling::pushRegion("Task_SwarmContainer_ResetCommunication");

  for (auto &s : swarmVector_) {
    s->ResetCommunication();
  }

  Kokkos::Profiling::popRegion(); // Task_SwarmContainer_ResetCommunication
  return TaskStatus::complete;
}

TaskStatus SwarmContainer::FinalizeCommunicationIterative() {
  Kokkos::Profiling::pushRegion("Task_SwarmContainer_FinalizeCommunicationIterative");

  PARTHENON_THROW("FinalizeCommunicationIterative not yet fully implemented!")

  int success = 0, total = 0;
  for (auto &s : swarmVector_) {
    if (s->FinalizeCommunicationIterative()) {
      success++;
    }
    total++;
  }

  Kokkos::Profiling::popRegion(); // Task_SwarmContainer_FinalizeCommunicationIterative
  if (success == total) return TaskStatus::complete;
  return TaskStatus::incomplete;
}

void SwarmContainer::ClearBoundary(BoundaryCommSubset phase) {}

void SwarmContainer::Print() const {
  std::cout << "Swarms are:\n";
  for (const auto &s : swarmMap_) {
    std::cout << "  " << s.second->info() << std::endl;
  }
}

} // namespace parthenon
