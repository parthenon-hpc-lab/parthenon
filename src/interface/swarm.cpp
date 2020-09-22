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
      mask_("mask", nmax_pool_, Metadata({Metadata::Boolean})),
      marked_for_removal_("mfr", nmax_pool_, Metadata({Metadata::Boolean})),
      mpiStatus(true) {
  Add("x", Metadata({Metadata::Real}));
  Add("y", Metadata({Metadata::Real}));
  Add("z", Metadata({Metadata::Real}));
  num_active_ = 0;
  max_active_index_ = 0;

  auto mask_h = mask_.data.GetHostMirror();
  auto marked_for_removal_h = marked_for_removal_.data.GetHostMirror();

  for (int n = 0; n < nmax_pool_; n++) {
    mask_h(n) = false;
    marked_for_removal_h(n) = false;
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
  if (intMap_.count(label) > 0 ||
      realMap_.count(label) > 0) {
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
    throw std::invalid_argument ("swarm variable not found in Remove()");
  }
}

void Swarm::setPoolMax(const int nmax_pool) {
  PARTHENON_REQUIRE(nmax_pool > nmax_pool_, "Must request larger pool size!");
  int n_new_begin = nmax_pool_;
  int n_new = nmax_pool - nmax_pool_;

  for (int n = 0; n < n_new; n++) {
    free_indices_.push_back(n + n_new_begin);
  }

  // Resize and copy data

  auto oldvar = mask_;
  auto newvar = ParticleVariable<bool>(oldvar.label(), nmax_pool, oldvar.metadata());
  auto &oldvar_data = oldvar.Get();
  auto &newvar_data = newvar.Get();

  pmy_block->par_for("setPoolMax_mask_1", 0, nmax_pool_ - 1,
    KOKKOS_LAMBDA(const int n) {
        newvar_data(n) = oldvar_data(n);
    });
  pmy_block->par_for("setPoolMax_mask_2", nmax_pool_, nmax_pool - 1,
    KOKKOS_LAMBDA(const int n) {
        newvar_data(n) = 0;
    });

  mask_ = newvar;

  auto oldvar_bool = marked_for_removal_;
  auto newvar_bool = ParticleVariable<bool>(oldvar_bool.label(), nmax_pool, oldvar_bool.metadata());
  auto oldvar_bool_data = oldvar_bool.data;
  auto newvar_bool_data = newvar_bool.data;
  pmy_block->par_for("setPoolMax_mark_1", 0, nmax_pool_ - 1,
    KOKKOS_LAMBDA(const int n) {
        newvar_bool_data(n) = oldvar_bool_data(n);
    });
  pmy_block->par_for("setPoolMax_mark_2", nmax_pool_, nmax_pool - 1,
    KOKKOS_LAMBDA(const int n) {
        newvar_bool_data(n) = 0;
    });
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
    realVector_[n] = newvar;
    realMap_[oldvar->label()] = newvar;
  }

  nmax_pool_ = nmax_pool;
}

ParArrayND<bool> Swarm::AddEmptyParticles(int num_to_add) {
  while (free_indices_.size() < num_to_add) {
    increasePoolMax();
  }

  ParArrayND<bool> new_mask("Newly created particles", nmax_pool_);
  auto new_mask_h = new_mask.GetHostMirror();
  for (int n = 0; n < nmax_pool_; n++) {
    new_mask_h(n) = false;
  }

  auto mask_h = mask_.data.GetHostMirror();
  mask_h.DeepCopy(mask_.data);

  auto free_index = free_indices_.begin();

  // Don't bother sanitizing the memory
  for (int n = 0; n < num_to_add; n++) {
    mask_h(*free_index) = true;
    new_mask_h(*free_index) = true;
    max_active_index_ = std::max<int>(max_active_index_, *free_index);

    free_index = free_indices_.erase(free_index);
  }

  num_active_ += num_to_add;

  new_mask.DeepCopy(new_mask_h);
  mask_.data.DeepCopy(mask_h);

  return new_mask;
}

// No active particles: nmax_active_index = -1
// No particles removed: nmax_active_index unchanged
// Particles removed: nmax_active_index is new max active index
void Swarm::RemoveMarkedParticles() {
  int new_max_active_index = -1; // TODO BRR this is a magic number, needed for Defrag()

  auto mask_h = mask_.data.GetHostMirrorAndCopy();
  auto marked_for_removal_h = marked_for_removal_.data.GetHostMirror();
  marked_for_removal_h.DeepCopy(marked_for_removal_.data);

  // loop backwards to keep free_indices_ updated correctly
  for (int n = max_active_index_; n >= 0; n--) {
    if (mask_h(n)) {
      if (marked_for_removal_h(n)) {
        mask_h(n) = false;
        free_indices_.push_front(n);
        num_active_ -= 1;
        if (n == max_active_index_) {
          max_active_index_ -= 1;
        }
        marked_for_removal_h(n) = false;
      } else {
        new_max_active_index = n;
      }
    }
  }

  mask_.data.DeepCopy(mask_h);
  marked_for_removal_.data.DeepCopy(marked_for_removal_h);
}

void Swarm::Defrag() {
  // TODO(BRR) Could this algorithm be more efficient? Does it matter?
  // Add 1 to convert max index to max number
  int num_free = (max_active_index_ + 1) - num_active_;

  ParArrayND<int> from_to_indices("from_to_indices", max_active_index_ + 1);
  auto from_to_indices_h = from_to_indices.GetHostMirror();

  auto mask_h = mask_.data.GetHostMirrorAndCopy();

  for (int n = 0; n < max_active_index_; n++) {
    from_to_indices_h(n) = -1;
  }

  std::list<int> new_free_indices;

  int index = max_active_index_;
  for (int n = 0; n < num_free; n++) {
    while (mask_h(index) == false) {
      index--;
    }
    int index_to_move_from = index;
    index--;

    // Below this number "moved" particles should actually stay in place
    if (index_to_move_from < num_active_) {
      break;
    }
    int index_to_move_to = free_indices_.front();
    free_indices_.pop_front();
    new_free_indices.push_back(index_to_move_from);
    from_to_indices_h(index_to_move_from) = index_to_move_to;
  }

  // Not all these sorts may be necessary
  free_indices_.sort();
  new_free_indices.sort();
  free_indices_.merge(new_free_indices);

  from_to_indices.DeepCopy(from_to_indices_h);

  pmy_block->par_for("Swarm::DefragMask", 0, max_active_index_,
    KOKKOS_LAMBDA(const int n) {
      if (from_to_indices(n) >= 0) {
        mask_(from_to_indices(n)) = mask_(n);
        mask_(n) = false;
      }
    });

  for (int m = 0; m < intVector_.size(); m++) {
    auto &vec = intVector_[m]->Get();
    pmy_block->par_for("Swarm::DefragInt", 0, max_active_index_,
      KOKKOS_LAMBDA(const int n) {
        if (from_to_indices(n) >= 0) {
          vec(from_to_indices(n)) = vec(n);
        }
      });
  }

  for (int m = 0; m < realVector_.size(); m++) {
    auto &vec = realVector_[m]->Get();
    pmy_block->par_for("Swarm::DefragReal", 0, max_active_index_,
      KOKKOS_LAMBDA(const int n) {
        if (from_to_indices(n) >= 0) {
          vec(from_to_indices(n)) = vec(n);
        }
      });
  }

  mask_h.DeepCopy(mask_.data);

  // Update max_active_index_
  max_active_index_ = num_active_ - 1;
}

} // namespace parthenon
