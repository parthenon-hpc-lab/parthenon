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

Swarm::Swarm(const std::string label, const Metadata &metadata, const int nmax_pool_in)
    : label_(label), m_(metadata), nmax_pool_(nmax_pool_in),
      mask_("mask", nmax_pool_, Metadata({Metadata::Integer})), mpiStatus(true) {
  Add("x", Metadata({Metadata::Real}));
  Add("y", Metadata({Metadata::Real}));
  Add("z", Metadata({Metadata::Real}));
  // TODO BRR should this actually be a private variable so users can't
  // mess with it? Have to update other variables when this changes
  // Add("mask", Metadata({Metadata::Integer}));
  // auto &mask = GetInteger("mask");
  for (int n = 0; n < nmax_pool_; n++) {
    mask_(n) = 0;
    free_indices_.push_back(n);
    //printf("free_index: %i (%i)\n", free_indices_.last(), n);
  }
  for (auto index : free_indices_) {
    printf("begin index: %i\n", index);
  }
}

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

  printf("Increasing pool max! %i\n", nmax_pool);
  for (auto index : free_indices_) {
    printf("index: %i\n", index);
  }

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

int Swarm::AddEmptyParticle()
{
  printf("B\n");
  if (free_indices_.size() == 0) {
    increasePoolMax();
  }
  printf("B\n");

  auto free_index_iter = free_indices_.begin();
  int free_index = *free_index_iter;
  free_indices_.erase(free_index_iter);
  printf("B\n");

  // ParticleVariable<int> &mask = GetInteger("mask");
  ParticleVariable<Real> &x = GetReal("x");
  ParticleVariable<Real> &y = GetReal("y");
  ParticleVariable<Real> &z = GetReal("z");
  printf("B\n");

  mask_(free_index) = 1;
  nmax_active_ = std::max<int>(nmax_active_, free_index);
  num_active_ += 1;
  printf("B\n");

  x(free_index) = 0.;
  y(free_index) = 0.;
  z(free_index) = 0.;
  printf("B\n");

  return free_index;
}

std::vector<int> Swarm::AddEmptyParticles(int num_to_add) {
  printf("C\n");
  while (free_indices_.size() < num_to_add) {
    increasePoolMax();
  }

  printf("C\n");
  std::vector<int> indices(num_to_add);

  printf("free_indices_.size: %i\n", free_indices_.size());
  for (auto &index: free_indices_) {
    printf("  index: %i\n", index);
  }

  auto free_index = free_indices_.begin();

  printf("C\n");
  // ParticleVariable<int> &mask = GetInteger("mask");
  ParticleVariable<Real> &x = GetReal("x");
  ParticleVariable<Real> &y = GetReal("y");
  ParticleVariable<Real> &z = GetReal("z");

  printf("C\n");
  for (int n = 0; n < num_to_add; n++) {
    printf("n: %i\n", n);
    indices[n] = *free_index;
    printf("free_index: %i\n", *free_index);
    printf("D\n");
    mask_(*free_index) = 1;
    printf("D\n");
    nmax_active_ = std::max<int>(nmax_active_, *free_index);
    printf("D\n");

    x(*free_index) = 0.;
    printf("D\n");
    y(*free_index) = 0.;
    printf("D\n");
    z(*free_index) = 0.;
    printf("D\n");

    free_index = free_indices_.erase(free_index);
    printf("D\n");
  }
  printf("C\n");

  num_active_ += num_to_add;

  return indices;
}

void Swarm::RemoveParticle(int index) {
  // ParticleVariable<int> &mask = GetInteger("mask");
  mask_(index) = 0;
  free_indices_.push_back(index);
  num_active_ -= 1;
  if (index == nmax_active_) {
    // TODO BRR this isn't actually right
    nmax_active_ -= 1;
  }
}

void Swarm::Defrag() {
  // TODO(BRR) Could this algorithm be more efficient?
  // Add 1 to convert max index to max number
  int num_free = (nmax_active_ + 1) - num_active_;

  free_indices_.sort();

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
    from_to_indices.push_back(std::pair<int, int>(index_to_move_from, index_to_move_to));
  }

  // Swap straggler particles into empty slots at lower indices
  for (auto pair : from_to_indices) {
    int from = pair.first;
    int to = pair.second;

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

std::vector<int> Swarm::AddUniformParticles(int num_to_add) {
  while (free_indices_.size() < num_to_add) {
    increasePoolMax();
  }

  num_active_ += num_to_add;

  return std::vector<int>();
}

} // namespace parthenon
