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

  std::stringstream msg;
  auto &bcs = pmb->pmy_mesh->mesh_bcs;
  // Check that, if we are using user BCs, they are actually enrolled, and unsupported BCs
  // are not being used
  for (int iFace = 0; iFace < 6; iFace++) {
    if (bcs[iFace] == BoundaryFlag::user) {
      if (pmb->pmy_mesh->MeshSwarmBndryFnctn[iFace] == nullptr) {
        msg << (iFace % 2 == 0 ? "i" : "o") << "x" << iFace / 2 + 1
            << " user boundary requested but provided function is null!";
        PARTHENON_THROW(msg);
      }
    } else if (bcs[iFace] != BoundaryFlag::outflow &&
               bcs[iFace] != BoundaryFlag::periodic) {
      msg << (iFace % 2 == 0 ? "i" : "o") << "x" << iFace / 2 + 1 << " boundary flag "
          << static_cast<int>(bcs[iFace]) << " not supported!";
      PARTHENON_THROW(msg);
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

void SwarmContainer::SendBoundaryBuffers() {}

void SwarmContainer::SetupPersistentMPI() {
  for (auto &s : swarmVector_) {
    s->SetupPersistentMPI();
  }
}

bool SwarmContainer::ReceiveBoundaryBuffers() { return true; }

void SwarmContainer::ReceiveAndSetBoundariesWithWait() {}

void SwarmContainer::SetBoundaries() {}

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
  printf("here?\n");

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

TaskStatus SwarmContainer::FinalizeCommunicationIterative() {
  PARTHENON_INSTRUMENT

  PARTHENON_THROW("FinalizeCommunicationIterative not yet fully implemented!")

  int success = 0, total = 0;
  for (auto &s : swarmVector_) {
    if (s->FinalizeCommunicationIterative()) {
      success++;
    }
    total++;
  }

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
