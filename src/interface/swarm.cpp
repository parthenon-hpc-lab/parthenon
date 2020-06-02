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
#include "swarm.hpp"
#include "globals.hpp" // my_rank
#include "mesh/mesh.hpp"

namespace parthenon {

void Swarm::Add(const std::vector<std::string> labelArray,
                       const Metadata &metadata) {
  // generate the vector and call Add
  for (auto label : labelArray) {
    Add(label, metadata);
  }
}

std::shared_ptr<Swarm> Swarm::AllocateCopy(const bool allocComms,
                                    MeshBlock *pmb) {
  Metadata m = m_;

  auto swarm = std::make_shared<Swarm>(label(), m, nmax_pool_);

  return swarm;
}

///
/// The internal routine for allocating a particle swarm.  This subroutine
/// is topology aware and will allocate accordingly.
///
/// @param label the name of the variable
/// @param metadata the metadata associated with the particle
void Swarm::Add(const std::string label, const Metadata &metadata) {
  // labels must be unique, even between different types of data
  if (intMap_.count(label) > 0 ||
      realMap_.count(label) > 0 ||
      stringMap_.count(label) > 0) {
    throw std::invalid_argument ("swarm variable " + label + " already enrolled during Add()!");
  }

  std::array<int, 6> arrDims = {nmax_pool_, 1, 1, 1, 1, 1};

  printf("Adding: %s\n", label.c_str());

  if (metadata.Type() == Metadata::Integer) {
    printf("int!\n");
    auto var = std::make_shared<ParticleVariable<int>>(label, nmax_pool_, metadata);
    intVector_.push_back(var);
    intMap_[label] = var;
  } else if (metadata.Type() == Metadata::Real) {
    printf("real!\n");
    auto var = std::make_shared<ParticleVariable<Real>>(label, nmax_pool_, metadata);
    realVector_.push_back(var);
    realMap_[label] = var;
  } else if (metadata.Type() == Metadata::String) {
    auto var = std::make_shared<ParticleVariable<std::string>>(label, nmax_pool_, metadata);
    stringVector_.push_back(var);
    stringMap_[label] = var;
  } else {
    throw std::invalid_argument ("swarm variable " + label + " does not have a valid type during Add()");
  }
}

void Swarm::Remove(const std::string label) {
  printf("SWARM::REMOVE!!\n");
  int idx, isize;
  bool found = false;

  // Find index of variable
  idx = 0;
  for (auto v : intVector_) {
    if ( label == v->label() ) {
      found = true;
      break;
    }
    idx++;
  }
  if (found == true) {
    // first delete the variable
    intVector_[idx].reset();

    // Next move the last element into idx and pop last entry
    if ( intVector_.size() > 1) intVector_[idx] = std::move(intVector_.back());
    intVector_.pop_back();

    // Also remove variable from map
    intMap_.erase(label);
  }

  if (found == false) {
    idx = 0;
    for (auto v : realVector_) {
      if ( label == v->label() ) {
        found = true;
        break;
      }
      idx++;
    }
  }
  if (found == true) {
    realVector_[idx].reset();
    if ( realVector_.size() > 1) realVector_[idx] = std::move(realVector_.back());
    realVector_.pop_back();
    realMap_.erase(label);
  }

  if (found == false) {
    idx = 0;
    for (auto v : stringVector_) {
      if ( label == v->label() ) {
        found = true;
        break;
      }
      idx++;
    }
  }
  if (found == true) {
    stringVector_[idx].reset();
    if ( stringVector_.size() > 1) stringVector_[idx] = std::move(stringVector_.back());
    stringVector_.pop_back();
    stringMap_.erase(label);
  }

  if (found == false) {
    throw std::invalid_argument ("swarm not found in Remove()");
  }
}

} // namespace parthenon
