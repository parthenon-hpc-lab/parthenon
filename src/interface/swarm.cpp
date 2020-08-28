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
#include <utility>

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

  if (metadata.Type() == Metadata::Integer) {
    auto var = std::make_shared<ParticleVariable<int>>(label, nmax_pool_, metadata);
    intVector_.push_back(var);
    intMap_[label] = var;
  } else if (metadata.Type() == Metadata::Real) {
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
    throw std::invalid_argument ("swarm variable not found in Remove()");
  }
}

void Swarm::setPoolMax(const int nmax_pool) {
  if (nmax_pool < nmax_pool_) {
    printf("Must increase pool size!\n");
    exit(-1);
  }
  int n_new_begin = nmax_pool_;
  int n_new = nmax_pool - nmax_pool_;

  for (int n = 0; n < n_new; n++) {
    free_indices_.push_back(n + n_new_begin);
  }

  // Resize and copy data

  auto oldvar = mask_;
  auto newvar = ParticleVariable<int>(oldvar.label(), nmax_pool, oldvar.metadata());
  for (int m = 0; m < nmax_pool_; m++) {
    newvar(m) = oldvar(m);
  }
  mask_ = newvar;

  for (int n = 0; n < intVector_.size(); n++) {
    auto oldvar = intVector_[n];
    auto newvar = std::make_shared<ParticleVariable<int>>(oldvar->label(),
                                                          nmax_pool,
                                                          oldvar->metadata());
    for (int m = 0; m < nmax_pool_; m++) {
      (*newvar)(m) = (*oldvar)(m);
    }
    intVector_[n] = newvar;
    intMap_[oldvar->label()] = newvar;
  }

  for (int n = 0; n < realVector_.size(); n++) {
    auto oldvar = realVector_[n];
    auto newvar = std::make_shared<ParticleVariable<Real>>(oldvar->label(),
                                                           nmax_pool,
                                                           oldvar->metadata());
    for (int m = 0; m < nmax_pool_; m++) {
      (*newvar)(m) = (*oldvar)(m);
    }
    realVector_[n] = newvar;
    realMap_[oldvar->label()] = newvar;
  }

  for (int n = 0; n < stringVector_.size(); n++) {
    auto oldvar = stringVector_[n];
    auto newvar = std::make_shared<ParticleVariable<std::string>>(oldvar->label(),
                                                           nmax_pool,
                                                           oldvar->metadata());
    for (int m = 0; m < nmax_pool_; m++) {
      (*newvar)(m) = (*oldvar)(m);
    }
    stringVector_[n] = newvar;
    stringMap_[oldvar->label()] = newvar;
  }

  nmax_pool_ = nmax_pool;
}

void Swarm::Defrag() {
  // TODO(BRR) Could this algorithm be more efficient?
  // Add 1 to convert max index to max number
  int num_free = (nmax_active_ + 1) - num_active_;
  printf("num free: %i nmax_active_: %i num_active: %i\n", num_free,
  nmax_active_, num_active_);
  for (int n = 0; n <= nmax_active_; n++) {
    printf("mask[%i] = %i\n", n, mask_(n));
  }

  free_indices_.sort();
  for (auto index : free_indices_) {
    printf("free: %i\n", index);
  }

  std::list<std::pair<int, int>> from_to_indices;

  int index = nmax_active_;
  for (int n = 0; n < num_free; n++) {
    while (mask_(index) == 0) {
      index--;
    }
    int index_to_move_from = index;
    index--;

    int index_to_move_to = free_indices_.front();
    free_indices_.pop_front();

    // index_to_move_to isn't always correct... some of the "moved" particles
    // should actually stay in place
    if (index_to_move_from < num_active_) {
      break;
    }
    //  printf("not moving %i to %i!\n", index_to_move_from, index_to_move_to);
    //} else {
    from_to_indices.push_back(std::pair<int, int>(index_to_move_from, index_to_move_to));
    //}

  }

  // Swap straggler particles into empty slots at lower indices
  for (auto pair : from_to_indices) {
    int from = pair.first;
    int to = pair.second;
    printf("from: %i to: %i\n", from, to);

    // Update swarm variables
    for (int n = 0; n < intVector_.size(); n++) {
      auto var = *(intVector_[n]);
      var(to) = var(from);
    }
    for (int n = 0; n < realVector_.size(); n++) {
      auto var = *(realVector_[n]);
      var(to) = var(from);
    }
    for (int n = 0; n < stringVector_.size(); n++) {
      auto var = *(stringVector_[n]);
      var(to) = var(from);
    }

    // Update mask
    mask_(from) = 0;
    mask_(to) = 1;

    // Update free indices
    free_indices_.push_back(from);
  }

  // Update nmax_active_
  nmax_active_ = num_active_ - 1;
}

} // namespace parthenon
