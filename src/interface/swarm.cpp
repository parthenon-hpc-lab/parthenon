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
#include <algorithm>
#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>

#include "bvals/cc/bvals_cc.hpp"
#include "mesh/mesh.hpp"
#include "swarm.hpp"

namespace parthenon {

SwarmDeviceContext Swarm::GetDeviceContext() const {
  SwarmDeviceContext context;
  context.marked_for_removal_ = marked_for_removal_.data;
  context.mask_ = mask_.data;
  context.blockIndex_ = blockIndex_;
  context.neighborIndices_ = neighborIndices_;
  // context.neighbor_send_index_ = neighbor_send_index.data;

  auto pmb = GetBlockPointer();

  const IndexRange &ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange &jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange &kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  context.x_min_ = pmb->coords.x1f(ib.s);
  context.y_min_ = pmb->coords.x2f(jb.s);
  context.z_min_ = pmb->coords.x3f(kb.s);
  context.x_max_ = pmb->coords.x1f(ib.e + 1);
  context.y_max_ = pmb->coords.x2f(jb.e + 1);
  context.z_max_ = pmb->coords.x3f(kb.e + 1);
  context.ndim_ = pmb->pmy_mesh->ndim;
  return context;
}

Swarm::Swarm(const std::string &label, const Metadata &metadata, const int nmax_pool_in)
    : label_(label), m_(metadata), nmax_pool_(nmax_pool_in),
      mask_("mask", nmax_pool_, Metadata({Metadata::Boolean})),
      marked_for_removal_("mfr", nmax_pool_, Metadata({Metadata::Boolean})),
      neighbor_send_index_("nsi", nmax_pool_, Metadata({Metadata::Integer})),
      blockIndex_("blockIndex_", nmax_pool_),
      neighborIndices_("neighborIndices_", 4, 4, 4), mpiStatus(true) {
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

bool Swarm::FinishCommunication(BoundaryCommSubset phase) {
  printf("[%i] FinishCommunication\n", Globals::my_rank);
  GetBlockPointer()->exec_space.fence();

  if (allreduce_request_ == MPI_REQUEST_NULL) { // No outstanding Iallreduce request
    printf("NULL!\n");
    MPI_Iallreduce(&local_num_completed_, &global_num_completed_, 1, MPI_INT, MPI_SUM,
                   MPI_COMM_WORLD, &allreduce_request_);
  } else { // Outstanding Iallreduce request
    int flag;
    MPI_Test(&allreduce_request_, &flag, MPI_STATUS_IGNORE);
    printf("Not NULL! flag: %i\n", flag);
    if (flag) { // Iallreduce completed

      // TODO(BRR) temporary!
      mpiStatus = true;
      return true;

      printf("[%i] incomplete: %i completed: %i\n", Globals::my_rank,
             global_num_incomplete_, global_num_completed_);

      /*if (global_num_incomplete_ > global_num_completed) { // Transport not done
        return false;
      } else { // Transport completed
        mpiStatus = true;
        return true;
      }*/

      // TODO(BRR) change the name of these vars
      printf("incomp: %i comp: %i\n", global_num_incomplete_, global_num_completed_);
      if (global_num_incomplete_ == global_num_completed_) {
        // Transport completed
        mpiStatus = true;
        return true;
      }

      allreduce_request_ = MPI_REQUEST_NULL;
    }
  }

  return false;
}
/*
    // Check that global_num_incomplete = 0
    // TODO(BRR) if splitting particles during a push, just add 1 to global_num_incomplete
  update

    //int num_completed = 0;
    //int global_num_completed = num_completed;
    int global_num_completed;
    MPI_Iallreduce(&local_num_completed_, &global_num_completed, 1, MPI_INT,
      MPI_SUM, MPI_COMM_WORLD, &allreduce_request_);
    //global_num_incomplete_ -= global_num_completed;

    printf("[%i] incomplete: %i completed: %i\n", Globals::my_rank,
  global_num_incomplete_, global_num_completed);

    if (global_num_incomplete_ > global_num_completed) {
      return false;
    }

    mpiStatus = true;

    printf("[%i] Finishing comm!\n", Globals::my_rank);
    return true;
  }*/

void Swarm::Add(const std::vector<std::string> &labelArray, const Metadata &metadata) {
  // generate the vector and call Add
  for (auto label : labelArray) {
    Add(label, metadata);
  }
}

std::shared_ptr<Swarm> Swarm::AllocateCopy(const bool allocComms, MeshBlock *pmb) {
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
void Swarm::Add(const std::string &label, const Metadata &metadata) {
  // labels must be unique, even between different types of data
  if (intMap_.count(label) > 0 || realMap_.count(label) > 0) {
    throw std::invalid_argument("swarm variable " + label +
                                " already enrolled during Add()!");
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
    throw std::invalid_argument("swarm variable " + label +
                                " does not have a valid type during Add()");
  }
}

void Swarm::Remove(const std::string &label) {
  bool found = false;

  // Find index of variable
  int idx = 0;
  for (auto v : intVector_) {
    if (label == v->label()) {
      found = true;
      break;
    }
    idx++;
  }
  if (found == true) {
    // first delete the variable
    intVector_[idx].reset();

    // Next move the last element into idx and pop last entry
    if (intVector_.size() > 1) intVector_[idx] = std::move(intVector_.back());
    intVector_.pop_back();

    // Also remove variable from map
    intMap_.erase(label);
  }

  if (found == false) {
    idx = 0;
    for (const auto &v : realVector_) {
      if (label == v->label()) {
        found = true;
        break;
      }
      idx++;
    }
  }
  if (found == true) {
    realVector_[idx].reset();
    if (realVector_.size() > 1) realVector_[idx] = std::move(realVector_.back());
    realVector_.pop_back();
    realMap_.erase(label);
  }

  if (found == false) {
    throw std::invalid_argument("swarm variable not found in Remove()");
  }
}

template <typename T>
void Swarm::ResizeParArray(ParArrayND<T> &var, int n_old, int n_new) {
  auto oldvar = var;
  auto newvar = ParArrayND<T>(oldvar.label(), n_new);
  GetBlockPointer()->par_for("ResizeParArray", 0, n_old - 1,
      KOKKOS_LAMBDA(const int n) { newvar(n) = oldvar(n); });
  var = newvar;
}

void Swarm::setPoolMax(const int nmax_pool) {
  GetBlockPointer()->exec_space.fence();
  PARTHENON_REQUIRE(nmax_pool > nmax_pool_, "Must request larger pool size!");
  int n_new_begin = nmax_pool_;
  int n_new = nmax_pool - nmax_pool_;

  auto pmb = GetBlockPointer();

  for (int n = 0; n < n_new; n++) {
    free_indices_.push_back(n + n_new_begin);
  }

  // Resize and copy data

  auto oldvar = mask_;
  auto newvar = ParticleVariable<bool>(oldvar.label(), nmax_pool, oldvar.metadata());
  auto &oldvar_data = oldvar.Get();
  auto &newvar_data = newvar.Get();

  pmb->par_for(
      "setPoolMax_mask_1", 0, nmax_pool_ - 1,
      KOKKOS_LAMBDA(const int n) { newvar_data(n) = oldvar_data(n); });
  pmb->par_for(
      "setPoolMax_mask_2", nmax_pool_, nmax_pool - 1,
      KOKKOS_LAMBDA(const int n) { newvar_data(n) = 0; });

  mask_ = newvar;

  auto oldvar_bool = marked_for_removal_;
  auto newvar_bool =
      ParticleVariable<bool>(oldvar_bool.label(), nmax_pool, oldvar_bool.metadata());
  auto oldvar_bool_data = oldvar_bool.data;
  auto newvar_bool_data = newvar_bool.data;
  pmb->par_for(
      "setPoolMax_mark_1", 0, nmax_pool_ - 1,
      KOKKOS_LAMBDA(const int n) { newvar_bool_data(n) = oldvar_bool_data(n); });
  pmb->par_for(
      "setPoolMax_mark_2", nmax_pool_, nmax_pool - 1,
      KOKKOS_LAMBDA(const int n) { newvar_bool_data(n) = 0; });
  marked_for_removal_ = newvar_bool;

  // TODO(BRR) make this operation a private member function for reuse
  auto oldvar_int = neighbor_send_index_;
  auto newvar_int =
      ParticleVariable<int>(oldvar_int.label(), nmax_pool, oldvar_int.metadata());
  auto oldvar_int_data = oldvar_int.data;
  auto newvar_int_data = newvar_int.data;
  pmb->par_for(
      "setPoolMax_neighbor_send_index", 0, nmax_pool_ - 1,
      KOKKOS_LAMBDA(const int n) { newvar_int_data(n) = oldvar_int_data(n); });
  neighbor_send_index_ = newvar_int;

  // Do something better about this...
  ResizeParArray(blockIndex_, nmax_pool_, nmax_pool);

  // TODO(BRR) this is not an efficient loop ordering, probably
  for (int n = 0; n < intVector_.size(); n++) {
    auto oldvar = intVector_[n];
    auto newvar = std::make_shared<ParticleVariable<int>>(oldvar->label(), nmax_pool,
                                                          oldvar->metadata());
    auto oldvar_data = oldvar->data;
    auto newvar_data = newvar->data;
    pmb->par_for(
        "setPoolMax_int", 0, nmax_pool_ - 1,
        KOKKOS_LAMBDA(const int m) { newvar_data(m) = oldvar_data(m); });

    intVector_[n] = newvar;
    intMap_[oldvar->label()] = newvar;
  }

  for (int n = 0; n < realVector_.size(); n++) {
    auto oldvar = realVector_[n];
    printf("RESIZING %s\n", oldvar->label().c_str());
    auto newvar = std::make_shared<ParticleVariable<Real>>(oldvar->label(), nmax_pool,
                                                           oldvar->metadata());
    auto oldvar_data = oldvar->data;
    auto newvar_data = newvar->data;
    pmb->par_for(
        "setPoolMax_real", 0, nmax_pool_ - 1,
        KOKKOS_LAMBDA(const int m) { newvar_data(m) = oldvar_data(m); });
    /*pmb->par_for(
        "setPoolMax_real", nmax_pool_, nmax_pool,
        KOKKOS_LAMBDA(const int m) {
          printf("  %e\n", newvar_data(m));
        });*/
    realVector_[n] = newvar;
    realMap_[oldvar->label()] = newvar;
  }

  nmax_pool_ = nmax_pool;

  GetBlockPointer()->exec_space.fence();
}

ParArrayND<bool> Swarm::AddEmptyParticles(const int num_to_add) {
  GetBlockPointer()->exec_space.fence();
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
  auto blockIndex_h = blockIndex_.GetHostMirrorAndCopy();

  auto free_index = free_indices_.begin();

  // Don't bother sanitizing the memory
  for (int n = 0; n < num_to_add; n++) {
    mask_h(*free_index) = true;
    new_mask_h(*free_index) = true;
    blockIndex_h(*free_index) = -1;
    max_active_index_ = std::max<int>(max_active_index_, *free_index);

    free_index = free_indices_.erase(free_index);
  }

  num_active_ += num_to_add;

  new_mask.DeepCopy(new_mask_h);
  mask_.data.DeepCopy(mask_h);
  blockIndex_.DeepCopy(blockIndex_h);

  return new_mask;
}

// No active particles: nmax_active_index = -1
// No particles removed: nmax_active_index unchanged
// Particles removed: nmax_active_index is new max active index
void Swarm::RemoveMarkedParticles() {
  GetBlockPointer()->exec_space.fence();
  int new_max_active_index = -1; // TODO(BRR) this is a magic number, needed for Defrag()

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
  GetBlockPointer()->exec_space.fence();
  // TODO(BRR) Could this algorithm be more efficient? Does it matter?
  // Add 1 to convert max index to max number
  int num_free = (max_active_index_ + 1) - num_active_;
  auto pmb = GetBlockPointer();

  ParArrayND<int> from_to_indices("from_to_indices", max_active_index_ + 1);
  auto from_to_indices_h = from_to_indices.GetHostMirror();

  auto mask_h = mask_.data.GetHostMirrorAndCopy();

  for (int n = 0; n <= max_active_index_; n++) {
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

  auto mask = mask_.Get();
  pmb->par_for(
      "Swarm::DefragMask", 0, max_active_index_, KOKKOS_LAMBDA(const int n) {
        if (from_to_indices(n) >= 0) {
          mask(from_to_indices(n)) = mask(n);
          mask(n) = false;
        }
      });

  for (int m = 0; m < intVector_.size(); m++) {
    auto &vec = intVector_[m]->Get();
    pmb->par_for(
        "Swarm::DefragInt", 0, max_active_index_, KOKKOS_LAMBDA(const int n) {
          if (from_to_indices(n) >= 0) {
            vec(from_to_indices(n)) = vec(n);
          }
        });
  }

  for (int m = 0; m < realVector_.size(); m++) {
    auto &vec = realVector_[m]->Get();
    pmb->par_for(
        "Swarm::DefragReal", 0, max_active_index_, KOKKOS_LAMBDA(const int n) {
          if (from_to_indices(n) >= 0) {
            vec(from_to_indices(n)) = vec(n);
          }
        });
  }

  // Update max_active_index_
  max_active_index_ = num_active_ - 1;

  GetBlockPointer()->exec_space.fence();
}

void Swarm::SetupPersistentMPI() {
  printf("[%i] SetupPersistentMPI\n", Globals::my_rank);
  vbvar->SetupPersistentMPI();

  allreduce_request_ = MPI_REQUEST_NULL;

  // Index into neighbor blocks
  auto pmb = GetBlockPointer();
  auto neighborIndices_h = neighborIndices_.GetHostMirror();
  printf("my level: %i\n", pmb->loc.level);
  printf("Dimensions: %i\n", pmb->pmy_mesh->ndim);

  // Region belonging to this meshblock
  // for (int k = 1; k < 3; k++) {
  for (int k = 0; k < 4; k++) { // 2D ONLY!
    for (int j = 1; j < 3; j++) {
      for (int i = 1; i < 3; i++) {
        neighborIndices_h(k, j, i) = -1;
      }
    }
  }

  int ndim = pmb->pmy_mesh->ndim;

  PARTHENON_REQUIRE(ndim == 2, "Only 2D tested right now!");

  for (int n = 0; n < pmb->pbval->nneighbor; n++) {
    NeighborBlock &nb = pmb->pbval->neighbor[n];
    printf("[%i] indices: %i %i %i\n", n, nb.ni.ox1, nb.ni.ox2, nb.ni.ox3);
    printf("    fi1: %i fi2: %i\n", nb.ni.fi1, nb.ni.fi2);
    printf("    rank: %i level: %i local ID: %i global ID: %i\n", nb.snb.rank,
           nb.snb.level, nb.snb.lid, nb.snb.gid);

    PARTHENON_REQUIRE(pmb->loc.level == nb.snb.level,
                      "Particles do not currently support AMR!");

    int i = nb.ni.ox1;
    int j = nb.ni.ox2;
    int k = nb.ni.ox3;

    if (ndim == 1) {
      if (i == -1) {
        neighborIndices_h(0, 0, 0) = n;
      } else if (i == 0) {
        neighborIndices_h(0, 0, 1) = n;
        neighborIndices_h(0, 0, 2) = n;
      } else {
        neighborIndices_h(0, 0, 3) = n;
      }
    } else if (ndim == 2) {
      if (i == -1) {
        if (j == -1) {
          neighborIndices_h(0, 0, 0) = n;
        } else if (j == 0) {
          neighborIndices_h(0, 1, 0) = n;
          neighborIndices_h(0, 2, 0) = n;
        } else if (j == 1) {
          neighborIndices_h(0, 3, 0) = n;
        }
      } else if (i == 0) {
        if (j == -1) {
          neighborIndices_h(0, 0, 1) = n;
          neighborIndices_h(0, 0, 2) = n;
        } else if (j == 1) {
          neighborIndices_h(0, 3, 1) = n;
          neighborIndices_h(0, 3, 2) = n;
        }
      } else if (i == 1) {
        if (j == -1) {
          neighborIndices_h(0, 0, 3) = n;
        } else if (j == 0) {
          neighborIndices_h(0, 1, 3) = n;
          neighborIndices_h(0, 2, 3) = n;
        } else if (j == 1) {
          neighborIndices_h(0, 3, 3) = n;
        }
      }
    } else {
      PARTHENON_FAIL("3D particles not currently supported!");
    }
  }

/*    if (i == -1) {
      if (j == -1) {
        neighborIndices_h(0, 0, 0) = n;
      } else if (j == 0) {
        neighborIndices_h(0, 1, 0) = n;
        neighborIndices_h(0, 2, 0) = n;
      } else if (j == 1) {
        neighborIndices_h(0, 3, 0) = n;
      }
    } else if (i == 0) {
      if (j == -1) {
        neighborIndices_h(0, 0, 1) = n;
        neighborIndices_h(0, 0, 2) = n;
      } else if (j == 1) {
        neighborIndices_h(0, 3, 1) = n;
        neighborIndices_h(0, 3, 2) = n;
      }
    } else if (i == 1) {
      if (j == -1) {
        neighborIndices_h(0, 0, 3) = n;
      } else if (j == 0) {
        neighborIndices_h(0, 1, 3) = n;
        neighborIndices_h(0, 2, 3) = n;
      } else if (j == 1) {
        neighborIndices_h(0, 3, 3) = n;
      }
    }
  }*/

  // Copy indices along z in 2D
  /*if (pmb->pmy_mesh->ndim == 2) {
    for (int k = 1; k < 4; k++) {
      for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 4; i++) {
          neighborIndices_h(k, j, i) = neighborIndices_h(0, j, i);
        }
      }
    }
  }*/

    /*for (int k = 0; k < 4; k++) {
      for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 4; i++) {
          printf("neighborIndices(%i, %i, %i) = %i\n", k,j,i,neighborIndices_h(k,j,i));
          //neighborIndices_h(k, j, i) = neighborIndices_h(0, j, i);
        }
      }
    }
    exit(-1);*/

  neighborIndices_.DeepCopy(neighborIndices_h);

  // exit(-1);
}

bool Swarm::Send(BoundaryCommSubset phase) {
  GetBlockPointer()->exec_space.fence();
  printf("[%i] Send\n", Globals::my_rank);
  if (mpiStatus == true) {
    printf("mpiStatus is true!\n");
    return true;
  }

  auto blockIndex_h = blockIndex_.GetHostMirrorAndCopy();
  auto mask_h = mask_.data.GetHostMirrorAndCopy();

  auto pmb = GetBlockPointer();

  // Fence to make sure particles aren't currently being transported locally
  pmb->exec_space.fence();

  /*NeighborBlock &nb = pmb->pbval->neighbor[n];
  printf("[%i] indices: %i %i %i\n", n, nb.ni.ox1, nb.ni.ox2, nb.ni.ox3);
  printf("    fi1: %i fi2: %i\n", nb.ni.fi1, nb.ni.fi2);
  printf("    rank: %i level: %i local ID: %i global ID: %i\n", nb.snb.rank,
         nb.snb.level, nb.snb.lid, nb.snb.gid);*/
  printf("max active index: %i\n", max_active_index_);
  printf("blockIndex_h.GetSize(): %i\n", blockIndex_h.GetSize());
  printf("mask_h.GetSize(): %i\n", mask_h.GetSize());

  printf("num neighboring meshblocks: %i\n", vbvar->bd_var_.nbmax);

  int nbmax = vbvar->bd_var_.nbmax;
  //std::vector<int> num_particles_to_send(nbmax, 0);
  //std::vector<ParArrayND<int>> particle_indices_to_send(nbmax);
  ParArrayND<int> num_particles_to_send("npts", nbmax);
  auto num_particles_to_send_h = num_particles_to_send.GetHostMirror();
  for (int n = 0; n < nbmax; n++) {
    num_particles_to_send_h(n) = 0;
  }
  int particle_size = GetParticleDataSize();

  int max_indices_size = 0;
  for (int n = 0; n <= max_active_index_; n++) {
    printf("%i\n", n);
    printf("mask: %i\n", mask_h(n));
    if (mask_h(n)) {
      printf("blockIndex_h(n): %i\n", blockIndex_h(n));
      printf("[%i] particle %i: -> %i (rank %i)\n", Globals::my_rank, n, blockIndex_h(n),
             pmb->pbval->neighbor[blockIndex_h(n)]);
      // This particle should be sent
      if (blockIndex_h(n) >= 0) {
        num_particles_to_send_h(blockIndex_h(n))++;
    //    particle_indices_to_send[blockIndex_h(n)].push_back(n);
        if (max_indices_size < num_particles_to_send_h(blockIndex_h(n))) {
          max_indices_size = num_particles_to_send_h(blockIndex_h(n));
        }
      }
    }
  }
  // Not a ragged-right array, just for convenience
  ParArrayND<int> particle_indices_to_send("Particle indices to send", nbmax, max_indices_size);
  auto particle_indices_to_send_h = particle_indices_to_send.GetHostMirror();
  std::vector<int> counter(nbmax, 0);
  for (int n = 0; n <= max_active_index_; n++) {
    if (mask_h(n)) {
      if (blockIndex_h(n) >= 0) {
        particle_indices_to_send_h(blockIndex_h(n), counter[blockIndex_h(n)]) = n;
        counter[blockIndex_h(n)]++;
      }
    }
  }
  num_particles_to_send.DeepCopy(num_particles_to_send_h);
  particle_indices_to_send.DeepCopy(particle_indices_to_send_h);

  for (int n = 0; n < nbmax; n++) {
    printf("SEND %i PARTICLES TO %i!\n", num_particles_to_send_h(n), n);
    // Resize buffer if too small
    //int olddim = vbvar->bd_var_.send[n].extent(0);
    auto sendbuf = vbvar->bd_var_.send[n];
    printf("olddim: %i\n", sendbuf.extent(0));
    if (sendbuf.extent(0) < num_particles_to_send_h(n)*particle_size) {
      sendbuf = ParArray1D<Real>("Buffer", num_particles_to_send_h(n)*particle_size);
      vbvar->bd_var_.send[n] = sendbuf;
    }
    printf("newdim: %i\n", sendbuf.extent(0));
    //for (auto index : particle_indices_to_send[n]) {
    //  printf("  %i\n", index);
    //}
    //if (vbvar->bd_var_.send[n].GetSize() < num_particles_to_send[n]*particle_size) {
    //  vbvar->bd_var_.send[n] = ParArray1D<Real>("Buffer", 2.*vbvar->bd_var_.send[n].GetSize());
    //}
  }

  pmb->par_for("Pack Buffers", 0, max_indices_size,
    KOKKOS_LAMBDA(const int n) { // Max index
      for (int m = 0; m < nbmax; m++) { // Number of neighbors
        if (n < num_particles_to_send(m)) {
          printf("copy this particle! %i %i\n", n, m);
        }
      }
    });

  exit(-1);

  // Count all the particles that are Active and Not on this block, if nonzero,
  // copy into buffers (if no send already for that buffer) and send

  vbvar->Send(phase);
  exit(-1);
  return false;
}

/*template <typename T>
VariablePack<T> Swarm::PackVariables(const std::vector<std::string> &names,
                                     const vpack_types::VarList<T> &vars,
                                     PackIndexMap &vmap) {
  auto kvpair = varPackMap_.find(names);
  if (kvpair == varPackMap_.end()) {
    auto pack = MakeSwarmPack<T>(vars, &vmap);
    PackIndxPair<T> value;
    value.pack = pack;
    value.map = vmap;
    varPackMap_[names] = value;
    return pack;
  }
  vmap = (kvpair->second).map;
  return (kvpair->second).pack;
}*/
VariablePack<Real> Swarm::PackVariablesReal(const std::vector<std::string> &names,
                                     const vpack_types::VarList<Real> &vars,
                                     PackIndexMap &vmap) {
  auto kvpair = varPackMapReal_.find(names);
  if (kvpair == varPackMapReal_.end()) {
    auto pack = MakeSwarmPack<Real>(vars, &vmap);
    PackIndxPair<Real> value;
    value.pack = pack;
    value.map = vmap;
    varPackMapReal_[names] = value;
    return pack;
  }
  vmap = (kvpair->second).map;
  return (kvpair->second).pack;
}
VariablePack<int> Swarm::PackVariablesInt(const std::vector<std::string> &names,
                                     const vpack_types::VarList<int> &vars,
                                     PackIndexMap &vmap) {
  auto kvpair = varPackMapInt_.find(names);
  if (kvpair == varPackMapInt_.end()) {
    auto pack = MakeSwarmPack<int>(vars, &vmap);
    PackIndxPair<int> value;
    value.pack = pack;
    value.map = vmap;
    varPackMapInt_[names] = value;
    return pack;
  }
  vmap = (kvpair->second).map;
  return (kvpair->second).pack;
}

/*VariablePack<Real> Swarm::PackRealVariables() {
  std::vector<std::string> names{"f"};
  return MapToVariablePack<Real>().find(names);
}
VariablePack<int> Swarm::PackIntVariables() {
  std::vector<std::string> names{"f"};
  return MapToVariablePack<int>().find(names);
}*/

bool Swarm::Receive(BoundaryCommSubset phase) {
  GetBlockPointer()->exec_space.fence();
  printf("[%i] Receive\n", Globals::my_rank);
  if (mpiStatus == true) {
    return true;
  }
  vbvar->Receive(phase);
  return false;
}

// TODO(BRR) move to BoundarySwarm
/*void Swarm::SetupPersistentMPI() {
#ifdef MPI_PARALLEL
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  int &mylevel = pmb->loc.level;

  // Initialize neighbor communications to other ranks
  for (int n = 0; n < pmb->pbval->nneighbor; n++) {
    NeighborBlock &nb = pmb->pbval->neighbor[n];

    printf("rank: %i Neighbor: %i Neighbor rank: %i\n", Globals::my_rank, n, nb.snb.rank);

    // Neighbor on different MPI process
    if (nb.snb.rank != Globals::my_rank) {
    }
  }
#endif
}*/
void Swarm::allocateComms(std::weak_ptr<MeshBlock> wpmb) {
  printf("[%i] allocateComms\n", Globals::my_rank);
  if (wpmb.expired()) return;

  std::shared_ptr<MeshBlock> pmb = wpmb.lock();

  // Create the boundary object
  vbvar = std::make_shared<BoundarySwarm>(pmb);

  // Enroll SwarmVariables
}

} // namespace parthenon
