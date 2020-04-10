//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
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
#include "bvals/cc/bvals_cc.hpp"
#include "SwarmContainer.hpp"
#include "globals.hpp" // my_rank
#include "mesh/mesh.hpp"

namespace parthenon {

void SwarmContainer::Add(const std::vector<std::string> labelArray,
                       const SwarmMetadata &smetadata) {
  // generate the vector and call Add
  for (auto label : labelArray) {
    Add(label, smetadata);
  }
}

///
/// The internal routine for allocating a particle swarm.  This subroutine
/// is topology aware and will allocate accordingly.
///
/// @param label the name of the variable
/// @param pmetadata the metadata associated with the particle
void SwarmContainer::Add(const std::string label,
                       const SwarmMetadata &smetadata) {
  auto swarm = std::make_shared<Swarm>(label, smetadata);
  // TODO(BRR) check that swarm isn't already enrolled?
  swarms.push_back(swarm);
}

void SwarmContainer::Remove(const std::string label) {
  int idx, isize;

  // Find index of swarm
  isize = swarms.size();
  idx = 0;
  for (auto s : swarms) {
    if ( ! label.compare(s->label())) {
      break;
    }
    idx++;
  }
  if ( idx >= isize) {
    throw std::invalid_argument ("swarm not found in Remove()");
  }

  // first delete the variable
  swarms[idx].reset();

  // Next move the last element into idx and pop last entry
  isize--;
  if ( isize >= 0) swarms[idx] = std::move(swarms.back());
  swarms.pop_back();
  return;
}

void SwarmContainer::SendBoundaryBuffers() {}

void SwarmContainer::SetupPersistentMPI() {}

bool SwarmContainer::ReceiveBoundaryBuffers() {}

void SwarmContainer::ReceiveAndSetBoundariesWithWait() {}

void SwarmContainer::SetBoundaries() {}

void SwarmContainer::StartReceiving(BoundaryCommSubset phase) {}

void SwarmContainer::ClearBoundary(BoundaryCommSubset phase) {}

void SwarmContainer::print() {
  std::cout << "Swarms are:\n";
  for (auto s : swarms) { std::cout << "  " << s->info() << std::endl; }
}

static void AddSwarm(Swarm&V, std::vector<Swarm>& sRet) {
  // adds aliases to sRet
  sRet.push_back(Swarm(V));
}


/// Gets an array of real variables from container.
/// @param index_ret is returned with starting index for each name
/// @param count_ret is returned with number of arrays for each name
/// @param sparse_ids if specified, only those sparse IDs are returned
int SwarmContainer::GetVariables(const std::vector<std::string>& names,
                               std::vector<Swarm>& sRet,
                               std::map<std::string,std::pair<int,int>>& indexCount,
                               const std::vector<int>& sparse_ids) {
  // First count how many entries we need and fill in index and count
  indexCount.clear();

  int index = 0;
  for (auto label : names) {
    int count = 0;
    try { // normal variable
      Swarm& S = Get(label);
      AddSwarm(S, sRet);
      count++;
    }
    catch (const std::invalid_argument& x) {
      throw std::invalid_argument (" Unable to find swarm " +
                                     label + " in container");
    }
    indexCount[label] = std::make_pair(index,count);
    index += count;
  } // (auto label : names)

  return index;
}

} // namespace parthenon
