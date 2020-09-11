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
      mask_("mask", nmax_pool_, Metadata({Metadata::Integer})),
      marked_for_removal_("mfr", nmax_pool_, Metadata({Metadata::Integer})),
      mpiStatus(true) {
        printf("A!\n");
  Add("x", Metadata({Metadata::Real}));
  printf("B");
  Add("y", Metadata({Metadata::Real}));
  printf("C");
  Add("z", Metadata({Metadata::Real}));
  num_active_ = 0;
  max_active_index_ = 0;
  printf("here!");

  auto mask_h = mask_.data.GetHostMirror();
  auto marked_for_removal_h = marked_for_removal_.data.GetHostMirror();

  for (int n = 0; n < nmax_pool_; n++) {
    mask_h(n) = 0;
    marked_for_removal_h(n) = 0;
    //mask_(n) = 0;
    //marked_for_removal_(n) = false;
    free_indices_.push_back(n);
  }

  mask_.data.DeepCopy(mask_h);
  marked_for_removal_.data.DeepCopy(marked_for_removal_h);
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
  printf("AISUDHASIUDH\n");
  if (intMap_.count(label) > 0 ||
      realMap_.count(label) > 0) {
      //stringMap_.count(label) > 0) {
    throw std::invalid_argument ("swarm variable " + label + " already enrolled during Add()!");
  }
  printf("CC!");

  if (metadata.Type() == Metadata::Integer) {
    auto var = std::make_shared<ParticleVariable<int>>(label, nmax_pool_, metadata);
    intVector_.push_back(var);
    intMap_[label] = var;
  } else if (metadata.Type() == Metadata::Real) {
    printf("DD!");
    auto var = std::make_shared<ParticleVariable<Real>>(label, nmax_pool_, metadata);
    printf("EE!");
    realVector_.push_back(var);
    realMap_[label] = var;
  //} else if (metadata.Type() == Metadata::String) {
    //auto var = std::make_shared<ParticleVariable<std::string>>(label, nmax_pool_, metadata);
    //stringVector_.push_back(var);
    //stringMap_[label] = var;
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

  /*if (found == false) {
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
  }*/

  if (found == false) {
    throw std::invalid_argument ("swarm variable not found in Remove()");
  }
}

void Swarm::setPoolMax(const int nmax_pool) {
  printf("Setting swarm pool max!");
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
  auto oldvar_data = oldvar.data;
  auto newvar_data = newvar.data;

  pmy_block->par_for("setPoolMax_mask", 0, nmax_pool - 1,
    KOKKOS_LAMBDA(const int n) {
      newvar_data(n) = 0;
      if (n < nmax_pool_) {
        newvar_data(n) = oldvar_data(n);
      } else {
        newvar_data(n) = 0;
      }
    });

  /*for (int m = 0; m < nmax_pool_; m++) {
    newvar(m) = oldvar(m);
  }
  // Fill new data with false
  for (int m = nmax_pool_; m < nmax_pool; m++) {
    newvar(m) = 0;
  }*/

  mask_ = newvar;

  auto oldvar_bool = marked_for_removal_;
  auto newvar_bool = ParticleVariable<int>(oldvar_bool.label(), nmax_pool, oldvar_bool.metadata());
  auto oldvar_bool_data = oldvar_bool.data;
  auto newvar_bool_data = newvar_bool.data;
  pmy_block->par_for("setPoolMax_marked_for_removal", 0, nmax_pool - 1,
    KOKKOS_LAMBDA(const int n) {
      if (n < nmax_pool_) {
        newvar_data(n) = oldvar_data(n);
      } else {
        newvar_data(n) = false;
      }
    });
  /*for (int m = 0; m < nmax_pool_; m++) {
    newvar_bool(m) = oldvar_bool(m);
  }
  // Fill new data with false
  for (int m = nmax_pool_; m < nmax_pool; m++) {
    newvar_bool(m) = false;
  }*/
  marked_for_removal_ = newvar_bool;

  // TODO this is not an efficient loop ordering, probably
  for (int n = 0; n < intVector_.size(); n++) {
    auto oldvar = intVector_[n];
    auto newvar = std::make_shared<ParticleVariable<int>>(oldvar->label(),
                                                          nmax_pool,
                                                          oldvar->metadata());
    auto oldvar_data = oldvar->data;
    auto newvar_data = newvar->data;
    pmy_block->par_for("setPoolMax_int", 0, nmax_pool_ - 1,
      KOKKOS_LAMBDA(const int m) {
        newvar_data(m) = oldvar_data(m);
      });

    //for (int m = 0; m < nmax_pool_; m++) {
    //  (*newvar)(m) = (*oldvar)(m);
    //}
    intVector_[n] = newvar;
    intMap_[oldvar->label()] = newvar;
  }

  for (int n = 0; n < realVector_.size(); n++) {
    auto oldvar = realVector_[n];
    auto newvar = std::make_shared<ParticleVariable<Real>>(oldvar->label(),
                                                           nmax_pool,
                                                           oldvar->metadata());
    auto oldvar_data = oldvar->data;
    auto newvar_data = newvar->data;
    pmy_block->par_for("setPoolMax_real", 0, nmax_pool_ - 1,
      KOKKOS_LAMBDA(const int m) {
        newvar_data(m) = oldvar_data(m);
      });
    //for (int m = 0; m < nmax_pool_; m++) {
    //  (*newvar)(m) = (*oldvar)(m);
    //}
    realVector_[n] = newvar;
    realMap_[oldvar->label()] = newvar;
  }

  /*for (int n = 0; n < stringVector_.size(); n++) {
    auto oldvar = stringVector_[n];
    auto newvar = std::make_shared<ParticleVariable<std::string>>(oldvar->label(),
                                                           nmax_pool,
                                                           oldvar->metadata());
    for (int m = 0; m < nmax_pool_; m++) {
      (*newvar)(m) = (*oldvar)(m);
    }
    stringVector_[n] = newvar;
    stringMap_[oldvar->label()] = newvar;
  }*/

  nmax_pool_ = nmax_pool;
  printf("Done setting swarm pool max!");
}

int Swarm::AddEmptyParticle()
{
  if (free_indices_.size() == 0) {
    increasePoolMax();
  }

  auto free_index_iter = free_indices_.begin();
  int free_index = *free_index_iter;
  free_indices_.erase(free_index_iter);

  // ParticleVariable<int> &mask = GetInteger("mask");
  ParticleVariable<Real> &x = GetReal("x");
  ParticleVariable<Real> &y = GetReal("y");
  ParticleVariable<Real> &z = GetReal("z");

  mask_(free_index) = 1;
  max_active_index_ = std::max<int>(max_active_index_, free_index);
  num_active_ += 1;

  x(free_index) = 0.;
  y(free_index) = 0.;
  z(free_index) = 0.;

  return free_index;
}

std::vector<int> Swarm::AddEmptyParticles(int num_to_add) {
  printf("Adding empty particles!");
  while (free_indices_.size() < num_to_add) {
    increasePoolMax();
  }

  std::vector<int> indices(num_to_add);
  std::vector<int> mask(nmax_pool_, 0);

  auto free_index = free_indices_.begin();

  // ParticleVariable<int> &mask = GetInteger("mask");
  ParticleVariable<Real> &x = GetReal("x");
  ParticleVariable<Real> &y = GetReal("y");
  ParticleVariable<Real> &z = GetReal("z");

  for (int n = 0; n < num_to_add; n++) {
    indices[n] = *free_index;
    mask_(*free_index) = 1;
    max_active_index_ = std::max<int>(max_active_index_, *free_index);

    x(*free_index) = 0.;
    y(*free_index) = 0.;
    z(*free_index) = 0.;
    mask[*free_index] = 1;

    free_index = free_indices_.erase(free_index);
  }

  num_active_ += num_to_add;
  printf("Done adding empty particles!");

  return mask;
}

// TODO BRR this should be Kokkosified
void Swarm::RemoveMarkedParticles() {
  printf("Removing marked particles!");
  int new_max_active_index = -1; // TODO BRR this is a magic number, needed for Defrag()
  for (int n = 0; n <= max_active_index_; n++) {
    if (mask_(n)) {
      if (marked_for_removal_(n)) {
        mask_(n) = 0;
        free_indices_.push_back(n);
        num_active_ -= 1;
        if (n == max_active_index_) {
          max_active_index_ -= 1;
        }
        marked_for_removal_(n) = false;
      } else {
        new_max_active_index = n;
      }
    }
  }
  max_active_index_ = new_max_active_index;
  printf("Done removing marked particles!");
}

void Swarm::RemoveParticle(int index) {
  // ParticleVariable<int> &mask = GetInteger("mask");
  mask_(index) = 0;
  free_indices_.push_back(index);
  num_active_ -= 1;
  if (index == max_active_index_) {
    // TODO BRR this isn't actually right
    max_active_index_ -= 1;
  }
}

void Swarm::Defrag() {
  printf("Defragging!");
  // TODO(BRR) Could this algorithm be more efficient?
  // Add 1 to convert max index to max number
  int num_free = (max_active_index_ + 1) - num_active_;

  free_indices_.sort();

  std::list<std::pair<int, int>> from_to_indices;

  int index = max_active_index_;
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
    //for (int n = 0; n < stringVector_.size(); n++) {
    //  auto var = *(stringVector_[n]);
    //  var(to) = var(from);
    //}

    // Update mask
    mask_(from) = 0;
    mask_(to) = 1;

    // Update free indices
    free_indices_.push_back(from);
  }

  // Update max_active_index_
  max_active_index_ = num_active_ - 1;
  printf("Done defragging!");
}

std::vector<int> Swarm::AddUniformParticles(int num_to_add) {
  while (free_indices_.size() < num_to_add) {
    increasePoolMax();
  }

  num_active_ += num_to_add;

  return std::vector<int>();
}

} // namespace parthenon
