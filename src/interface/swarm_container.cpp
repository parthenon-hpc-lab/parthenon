//========================================================================================
// (C) (or copyright) 2020-2026. Triad National Security, LLC. All rights reserved.
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
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "globals.hpp" // my_rank
#include "mesh/mesh.hpp"
#include "swarm_container.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

void SwarmContainer::Initialize(const std::shared_ptr<StateDescriptor> resolved_packages,
                                const std::shared_ptr<MeshBlock> pmb) {
  SetBlockPointer(pmb);

  for (auto const &q : resolved_packages->AllSwarms()) {
    Add(q.first, q.second);
    // Populate swarm values
    auto &swarm = Get(q.first);
    for (auto const &m : resolved_packages->AllSwarmValues(q.first)) {
      swarm->Add(m.first, m.second);
    }
  }
}

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
  swarm->AllocateComms(GetBlockPointer());
  Add(swarm);
}

// TODO(JMM): Should we support this operation
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
    PARTHENON_FAIL("swarm not found in Remove()");
  }

  // Pull out metadata
  const SP_Swarm pswarm = swarmVector_[idx];
  const Metadata &m = pswarm->metadata();

  // Delete the variable
  swarmVector_[idx].reset();

  // Next move the last element into idx and pop last entry
  isize--;
  if (isize >= 0) swarmVector_[idx] = std::move(swarmVector_.back());
  swarmVector_.pop_back();

  // Also remove swarm from map
  swarmMap_.erase(label);
  for (const auto &flag : m.Flags()) {
    swarmMetadataMap_[flag].erase(pswarm);
  }
}

// Return swarms meeting some conditions
SwarmSet SwarmContainer::GetSwarmsByFlag(const Metadata::FlagCollection &flags) {
  PARTHENON_INSTRUMENT

  auto swarms = MetadataUtils::GetByFlag<SwarmSet>(flags, swarmMap_, swarmMetadataMap_);

  return swarms;
}

TaskStatus SwarmContainer::Defrag(double min_occupancy) {
  PARTHENON_INSTRUMENT
  PARTHENON_REQUIRE_THROWS(min_occupancy >= 0. && min_occupancy <= 1.,
                           "Max fractional occupancy of swarm must be >= 0 and <= 1");

  for (auto &s : swarmVector_) {
    if (s->GetNumActive() > 0 &&
        s->GetNumActive() / (s->GetMaxActiveIndex() + 1.0) < min_occupancy) {
      s->Defrag();
    }
  }

  return TaskStatus::complete;
}

TaskStatus SwarmContainer::DefragAll() {
  PARTHENON_INSTRUMENT
  for (auto &s : swarmVector_) {
    s->Defrag();
  }
  return TaskStatus::complete;
}

TaskStatus SwarmContainer::SortParticlesByCell() {
  PARTHENON_INSTRUMENT

  for (auto &s : swarmVector_) {
    s->SortParticlesByCell();
  }

  return TaskStatus::complete;
}

void SwarmContainer::SetupPersistentMPI() {
  for (auto &s : swarmVector_) {
    s->SetupPersistentMPI();
  }
}

TaskStatus SwarmContainer::Send(BoundaryCommSubset phase) {
  PARTHENON_INSTRUMENT
  for (auto &s : swarmVector_) {
    s->Send(phase);
  }

  return TaskStatus::complete;
}

TaskStatus SwarmContainer::Receive(BoundaryCommSubset phase) {
  PARTHENON_INSTRUMENT
  int success = 0, total = 0;
  for (auto &s : swarmVector_) {
    if (s->Receive(phase)) {
      success++;
      ApplySwarmBoundaryConditions(s);
      s->RemoveMarkedParticles();
    }
    total++;
  }

  if (success == total) return TaskStatus::complete;
  return TaskStatus::incomplete;
}

TaskStatus SwarmContainer::ResetCommunication() {
  PARTHENON_INSTRUMENT
  for (auto &s : swarmVector_) {
    s->ResetCommunication();
  }

  return TaskStatus::complete;
}

void SwarmContainer::Print() const {
  std::cout << "Swarms are:\n";
  for (const auto &s : swarmMap_) {
    std::cout << "  " << s.second->info() << std::endl;
  }
}

bool SwarmContainer::operator==(const SwarmContainer &cmp) {
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

TaskStatus SendSwarmsMesh(std::shared_ptr<MeshData<Real>> &md) {
  PARTHENON_INSTRUMENT
  for (int b = 0; b < md->NumBlocks(); b++) {
    md->GetBlockData(b)->GetSwarmData()->Send(BoundaryCommSubset::all);
  }

  return TaskStatus::complete;
}

TaskStatus ReceiveSwarmsMesh(std::shared_ptr<MeshData<Real>> &md) {
  PARTHENON_INSTRUMENT
  TaskStatus status = TaskStatus::complete;
  for (int b = 0; b < md->NumBlocks(); b++) {
    if (md->GetBlockData(b)->GetSwarmData()->Receive(BoundaryCommSubset::all) ==
        TaskStatus::incomplete) {
      status = TaskStatus::incomplete;
    }
  }

  return status;
}

TaskStatus ResetSwarmsCommunicationMesh(std::shared_ptr<MeshData<Real>> &md) {
  PARTHENON_INSTRUMENT
  for (int b = 0; b < md->NumBlocks(); b++) {
    md->GetBlockData(b)->GetSwarmData()->ResetCommunication();
  }

  return TaskStatus::complete;
}

TaskStatus RemoveMarkedParticlesMesh(std::shared_ptr<MeshData<Real>> &md,
                                     const std::string &swarm_name) {
  PARTHENON_INSTRUMENT
  for (int b = 0; b < md->NumBlocks(); b++) {
    md->GetBlockData(b)->GetSwarmData()->Get(swarm_name)->RemoveMarkedParticles();
  }

  return TaskStatus::complete;
}

TaskStatus DefragSwarmsMesh(std::shared_ptr<MeshData<Real>> &md,
                            const Real &min_occupancy) {
  PARTHENON_INSTRUMENT
  for (int b = 0; b < md->NumBlocks(); b++) {
    md->GetBlockData(b)->GetSwarmData()->Defrag(min_occupancy);
  }

  return TaskStatus::complete;
}

TaskStatus DefragAllSwarmsMesh(std::shared_ptr<MeshData<Real>> &md) {
  PARTHENON_INSTRUMENT
  for (int b = 0; b < md->NumBlocks(); b++) {
    md->GetBlockData(b)->GetSwarmData()->DefragAll();
  }

  return TaskStatus::complete;
}

} // namespace parthenon
