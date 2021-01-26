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

#include "mesh/mesh.hpp"
#include "swarm.hpp"

namespace parthenon {

SwarmDeviceContext Swarm::GetDeviceContext() const {
  SwarmDeviceContext context;
  context.marked_for_removal_ = marked_for_removal_.data;
  context.mask_ = mask_.data;
  context.blockIndex_ = blockIndex_;
  context.neighborIndices_ = neighborIndices_;

  auto pmb = GetBlockPointer();
  auto pmesh = pmb->pmy_mesh;
  auto mesh_size = pmesh->mesh_size;

  const IndexRange &ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange &jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange &kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  context.x_min_ = pmb->coords.x1f(ib.s);
  context.y_min_ = pmb->coords.x2f(jb.s);
  context.z_min_ = pmb->coords.x3f(kb.s);
  context.x_max_ = pmb->coords.x1f(ib.e + 1);
  context.y_max_ = pmb->coords.x2f(jb.e + 1);
  context.z_max_ = pmb->coords.x3f(kb.e + 1);
  context.x_min_global_ = mesh_size.x1min;
  context.x_max_global_ = mesh_size.x1max;
  context.y_min_global_ = mesh_size.x2min;
  context.y_max_global_ = mesh_size.x2max;
  context.z_min_global_ = mesh_size.x3min;
  context.z_max_global_ = mesh_size.x3max;
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
/// The routine for allocating a particle variable in the current swarm.
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

///
/// The routine for removing a variable from a particle swarm.
///
/// @param label the name of the variable
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

///
/// The routine for resizing a 1D ParArrayND while retaining existing data
///
/// @param var The ParArrayND to be resized
/// @param n_old The length of existing data to be copied
/// @param n_new The requested length of the new data
template <typename T>
void Swarm::ResizeParArray(ParArrayND<T> &var, const int n_old, const int n_new) {
  auto oldvar = var;
  auto newvar = ParArrayND<T>(oldvar.label(), n_new);
  PARTHENON_DEBUG_REQUIRE(n_new > n_old, "Resized ParArrayND must be larger!");
  GetBlockPointer()->par_for(
      "ResizeParArray", 0, n_old - 1,
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
  ResizeParArray(mask_.Get(), nmax_pool_, nmax_pool);
  pmb->par_for(
      "setPoolMax_mask", nmax_pool_, nmax_pool - 1,
      KOKKOS_LAMBDA(const int n) { mask_(n) = 0; });

  ResizeParArray(marked_for_removal_.Get(), nmax_pool_, nmax_pool);
  pmb->par_for(
      "setPoolMax_marked_for_removal", nmax_pool_, nmax_pool - 1,
      KOKKOS_LAMBDA(const int n) { marked_for_removal_(n) = false; });

  ResizeParArray(neighbor_send_index_.Get(), nmax_pool_, nmax_pool);

  ResizeParArray(blockIndex_, nmax_pool_, nmax_pool);

  // TODO(BRR) Use ParticleVariable packs to reduce kernel launches
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
    auto newvar = std::make_shared<ParticleVariable<Real>>(oldvar->label(), nmax_pool,
                                                           oldvar->metadata());
    auto oldvar_data = oldvar->data;
    auto newvar_data = newvar->data;
    pmb->par_for(
        "setPoolMax_real", 0, nmax_pool_ - 1,
        KOKKOS_LAMBDA(const int m) { newvar_data(m) = oldvar_data(m); });
    realVector_[n] = newvar;
    realMap_[oldvar->label()] = newvar;
  }

  nmax_pool_ = nmax_pool;

  GetBlockPointer()->exec_space.fence();
}

ParArrayND<bool> Swarm::AddEmptyParticles(const int num_to_add,
                                          ParArrayND<int> &new_indices) {
  PARTHENON_REQUIRE(num_to_add > 0, "Attempting to add fewer than 1 new particles!");
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

  new_indices = ParArrayND<int>("New indices", num_to_add);
  auto new_indices_h = new_indices.GetHostMirror();

  // Don't bother sanitizing the memory
  for (int n = 0; n < num_to_add; n++) {
    mask_h(*free_index) = true;
    new_mask_h(*free_index) = true;
    blockIndex_h(*free_index) = this_block_;
    max_active_index_ = std::max<int>(max_active_index_, *free_index);
    new_indices_h(n) = *free_index;

    free_index = free_indices_.erase(free_index);
  }

  new_indices.DeepCopy(new_indices_h);

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
      }
    }
  }

  mask_.data.DeepCopy(mask_h);
  marked_for_removal_.data.DeepCopy(marked_for_removal_h);
}

void Swarm::Defrag() {
  printf("%s:%i\n", __FILE__, __LINE__);
  if (Globals::my_rank != 0) return;
  if (get_num_active() == 0) {
    return;
  }
  // TODO(BRR) Could this algorithm be more efficient? Does it matter?
  // Add 1 to convert max index to max number
  int num_free = (max_active_index_ + 1) - num_active_;
  auto pmb = GetBlockPointer();
  printf("%s:%i\n", __FILE__, __LINE__);

  ParArrayND<int> from_to_indices("from_to_indices", max_active_index_ + 1);
  auto from_to_indices_h = from_to_indices.GetHostMirror();
  printf("%s:%i\n", __FILE__, __LINE__);

  auto mask_h = mask_.data.GetHostMirrorAndCopy();
  printf("%s:%i\n", __FILE__, __LINE__);

  for (int n = 0; n <= max_active_index_; n++) {
    from_to_indices_h(n) = unset_index_;
  }
  printf("%s:%i\n", __FILE__, __LINE__);

  std::list<int> new_free_indices;
  printf("%s:%i\n", __FILE__, __LINE__);
  printf("num_free: %i num_active: %i max_active_index: %i\n", num_free, num_active_, max_active_index_);
  for (int n = 0; n <= max_active_index_; n++) {
    printf("mask(%i) = %i\n", n, mask_(n));
  }

  int index = max_active_index_;
  printf("max_active_index_: %i\n", max_active_index_);
  int num_to_move = std::min<int>(num_free, num_active_);
  for (int n = 0; n < num_to_move; n++) {
  printf("%s:%i\n", __FILE__, __LINE__);
    while (mask_h(index) == false) {
  printf("%s:%i\n", __FILE__, __LINE__);
  printf("index: %i\n", index);
      index--;
    }
  printf("%s:%i\n", __FILE__, __LINE__);
    int index_to_move_from = index;
  printf("%s:%i\n", __FILE__, __LINE__);
    index--;
  printf("%s:%i\n", __FILE__, __LINE__);

    // Below this number "moved" particles should actually stay in place
    if (index_to_move_from < num_active_) {
      break;
    }
  printf("%s:%i\n", __FILE__, __LINE__);
    int index_to_move_to = free_indices_.front();
  printf("%s:%i\n", __FILE__, __LINE__);
    free_indices_.pop_front();
  printf("%s:%i\n", __FILE__, __LINE__);
    new_free_indices.push_back(index_to_move_from);
  printf("%s:%i\n", __FILE__, __LINE__);
    from_to_indices_h(index_to_move_from) = index_to_move_to;
    printf("MOVE %i TO %i\n", index_to_move_from, index_to_move_to);
  printf("%s:%i\n", __FILE__, __LINE__);
  }
  printf("%s:%i\n", __FILE__, __LINE__);

  // TODO(BRR) Not all these sorts may be necessary
  free_indices_.sort();
  new_free_indices.sort();
  free_indices_.merge(new_free_indices);

  from_to_indices.DeepCopy(from_to_indices_h);
  printf("%s:%i\n", __FILE__, __LINE__);

  auto mask = mask_.Get();
  pmb->par_for(
      "Swarm::DefragMask", 0, max_active_index_, KOKKOS_LAMBDA(const int n) {
        if (from_to_indices(n) >= 0) {
          mask(from_to_indices(n)) = mask(n);
          mask(n) = false;
        }
      });
  printf("%s:%i\n", __FILE__, __LINE__);

  SwarmVariablePack<Real> vreal;
  SwarmVariablePack<int> vint;
  PackIndexMap rmap;
  PackIndexMap imap;
  PackAllVariables(vreal, vint, rmap, imap);
  int real_vars_size = realVector_.size();
  int int_vars_size = intVector_.size();
  printf("%s:%i\n", __FILE__, __LINE__);

  pmb->par_for(
      "Swarm::DefragVariables", 0, max_active_index_, KOKKOS_LAMBDA(const int n) {
        if (from_to_indices(n) >= 0) {
          for (int i = 0; i < real_vars_size; i++) {
            vreal(i, from_to_indices(n)) = vreal(i, n);
          }
          for (int i = 0; i < int_vars_size; i++) {
            vint(i, from_to_indices(n)) = vint(i, n);
          }
        }
      });
  printf("%s:%i\n", __FILE__, __LINE__);

  // Update max_active_index_
  max_active_index_ = num_active_ - 1;
}

void Swarm::SetupPersistentMPI() {
  vbswarm->SetupPersistentMPI();

  allreduce_request_ = MPI_REQUEST_NULL;

  // Index into neighbor blocks
  auto pmb = GetBlockPointer();
  auto neighborIndices_h = neighborIndices_.GetHostMirror();

  // Region belonging to this meshblock
  // for (int k = 1; k < 3; k++) {
  for (int k = 0; k < 4; k++) { // 2D ONLY!
    for (int j = 1; j < 3; j++) {
      for (int i = 1; i < 3; i++) {
        neighborIndices_h(k, j, i) = this_block_;
      }
    }
  }

  // TODO(BRR) Checks against some current limitations
  int ndim = pmb->pmy_mesh->ndim;
  PARTHENON_REQUIRE(ndim == 2, "Only 2D tested right now!");
  auto mesh_bcs = pmb->pmy_mesh->mesh_bcs;
  for (int n = 0; n < 2 * ndim; n++) {
    PARTHENON_REQUIRE(mesh_bcs[n] == BoundaryFlag::periodic,
                      "Only periodic boundaries supported right now!");
  }

  for (int n = 0; n < pmb->pbval->nneighbor; n++) {
    NeighborBlock &nb = pmb->pbval->neighbor[n];

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

  neighborIndices_.DeepCopy(neighborIndices_h);
}

bool Swarm::Send(BoundaryCommSubset phase) {
  printf("[%i] Send\n", Globals::my_rank);
  auto blockIndex_h = blockIndex_.GetHostMirrorAndCopy();
  auto mask_h = mask_.data.GetHostMirrorAndCopy();
  auto swarm_d = GetDeviceContext();

  auto pmb = GetBlockPointer();
  /*{int gid = 0;
  {
  MeshBlock &tb = *pmb->pmy_mesh->FindMeshBlock(gid);
      std::shared_ptr<BoundarySwarm> pbs =
          tb.pbswarm->bswarms[0];
  printf("%s:%i\n", __FILE__, __LINE__);
  printf("[%i] (%p): ", 0, pbs.get());
  for (int n = 0; n < pbs->bd_var_.nbmax; n++) {
    NeighborBlock &nb = tb.pbval->neighbor[n];
    printf("%i ", pbs->bd_var_.flag[nb.bufid]);
  } printf("\n");}
  gid = 1;
  {
  MeshBlock &tb = *pmb->pmy_mesh->FindMeshBlock(gid);
  std::shared_ptr<BoundarySwarm> pbs =
          tb.pbswarm->bswarms[0];
  printf("[%i] (%p): ", 0, pbs.get());
  for (int n = 0; n < pbs->bd_var_.nbmax; n++) {
    NeighborBlock &nb = tb.pbval->neighbor[n];
    printf("%i ", pbs->bd_var_.flag[nb.bufid]);
  } printf("\n\n\n");
  }}*/

  // Fence to make sure particles aren't currently being transported locally
  pmb->exec_space.fence();

  int nbmax = vbswarm->bd_var_.nbmax;
  ParArrayND<int> num_particles_to_send("npts", nbmax);
  auto num_particles_to_send_h = num_particles_to_send.GetHostMirror();
  for (int n = 0; n < nbmax; n++) {
    num_particles_to_send_h(n) = 0;
    auto &nb = pmb->pbval->neighbor[n];
    //vbswarm->bd_var_.flag[nb.bufid] = BoundaryStatus::waiting;
  }
  int particle_size = GetParticleDataSize();
  vbswarm->particle_size = particle_size;

  int max_indices_size = 0;
  for (int n = 0; n <= max_active_index_; n++) {
    if (mask_h(n)) {
      // This particle should be sent
      if (blockIndex_h(n) >= 0) {
        num_particles_to_send_h(blockIndex_h(n))++;
        if (max_indices_size < num_particles_to_send_h(blockIndex_h(n))) {
          max_indices_size = num_particles_to_send_h(blockIndex_h(n));
        }
      }
    }
  }
  // Size-0 arrays not permitted
  max_indices_size = std::max<int>(1, max_indices_size);
  // Not a ragged-right array, just for convenience
  ParArrayND<int> particle_indices_to_send("Particle indices to send", nbmax,
                                           max_indices_size);
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

  num_particles_sent_ = 0;
  for (int n = 0; n < nbmax; n++) {
    // Resize buffer if too small
    auto sendbuf = vbswarm->bd_var_.send[n];
    if (sendbuf.extent(0) < num_particles_to_send_h(n) * particle_size) {
      sendbuf = ParArray1D<Real>("Buffer", num_particles_to_send_h(n) * particle_size);
      vbswarm->bd_var_.send[n] = sendbuf;
    }
    vbswarm->send_size[n] = num_particles_to_send_h(n) * particle_size;
    num_particles_sent_ += num_particles_to_send_h(n);
  }

  SwarmVariablePack<Real> vreal;
  SwarmVariablePack<int> vint;
  PackIndexMap rmap;
  PackIndexMap imap;
  PackAllVariables(vreal, vint, rmap, imap);
  int real_vars_size = realVector_.size();
  int int_vars_size = intVector_.size();
  const int ix = rmap["x"].first;
  const int iy = rmap["y"].first;
  const int iz = rmap["z"].first;

  ParArrayND<int> nrank("Neighbor rank", nbmax);
  auto nrank_h = nrank.GetHostMirrorAndCopy();
  for (int n = 0; n < nbmax; n++) {
    NeighborBlock &nb = pmb->pbval->neighbor[n];
    nrank_h(n) = nb.snb.rank;
  }
  nrank.DeepCopy(nrank_h);

  auto &bdvar = vbswarm->bd_var_;
  pmb->par_for(
      "Pack Buffers", 0, max_indices_size,
      KOKKOS_LAMBDA(const int n) {        // Max index
        for (int m = 0; m < nbmax; m++) { // Number of neighbors
          if (n < num_particles_to_send(m)) {
            int sidx = particle_indices_to_send(m, n);
            int buffer_index = n * particle_size;
            swarm_d.MarkParticleForRemoval(sidx);
            for (int i = 0; i < real_vars_size; i++) {
              bdvar.send[m](buffer_index) = vreal(i, sidx);
              buffer_index++;
            }
            for (int i = 0; i < int_vars_size; i++) {
              bdvar.send[m](buffer_index) = static_cast<Real>(vint(i, sidx));
              buffer_index++;
            }
            // If rank is shared, apply boundary conditions here
            // TODO(BRR) Don't hardcode periodic boundary conditions
            if (nrank(m) == Globals::my_rank) {
              double &x = vreal(ix, sidx);
              double &y = vreal(iy, sidx);
              double &z = vreal(iz, sidx);
              if (x < swarm_d.x_min_global_) {
                x = swarm_d.x_max_global_ - (swarm_d.x_min_global_ - x);
              }
              if (x > swarm_d.x_max_global_) {
                x = swarm_d.x_min_global_ + (x - swarm_d.x_max_global_);
              }
              if (y < swarm_d.y_min_global_) {
                y = swarm_d.y_max_global_ - (swarm_d.y_min_global_ - y);
              }
              if (y > swarm_d.y_max_global_) {
                y = swarm_d.y_min_global_ + (y - swarm_d.y_max_global_);
              }
              if (z < swarm_d.z_min_global_) {
                z = swarm_d.z_max_global_ - (swarm_d.z_min_global_ - z);
              }
              if (z > swarm_d.z_max_global_) {
                z = swarm_d.z_min_global_ + (z - swarm_d.z_max_global_);
              }
            }
          }
        }
      });

  // Count all the particles that are Active and Not on this block, if nonzero,
  // copy into buffers (if no send already for that buffer) and send

  RemoveMarkedParticles();
  /*{int gid = 0;
  {
  MeshBlock &tb = *pmb->pmy_mesh->FindMeshBlock(gid);
      std::shared_ptr<BoundarySwarm> pbs =
          tb.pbswarm->bswarms[0];
  printf("%s:%i\n", __FILE__, __LINE__);
  printf("[%i] (%p): ", 0, pbs.get());
  for (int n = 0; n < pbs->bd_var_.nbmax; n++) {
    NeighborBlock &nb = tb.pbval->neighbor[n];
    printf("%i ", pbs->bd_var_.flag[nb.bufid]);
  } printf("\n");}
  gid = 1;
  {
  MeshBlock &tb = *pmb->pmy_mesh->FindMeshBlock(gid);
  std::shared_ptr<BoundarySwarm> pbs =
          tb.pbswarm->bswarms[0];
  printf("[%i] (%p): ", 0, pbs.get());
  for (int n = 0; n < pbs->bd_var_.nbmax; n++) {
    NeighborBlock &nb = tb.pbval->neighbor[n];
    printf("%i ", pbs->bd_var_.flag[nb.bufid]);
  } printf("\n\n\n");
  }}*/

  vbswarm->Send(phase);
  /*{int gid = 0;
  {
  MeshBlock &tb = *pmb->pmy_mesh->FindMeshBlock(gid);
      std::shared_ptr<BoundarySwarm> pbs =
          tb.pbswarm->bswarms[0];
  printf("%s:%i\n", __FILE__, __LINE__);
  printf("[%i] (%p): ", 0, pbs.get());
  for (int n = 0; n < pbs->bd_var_.nbmax; n++) {
    NeighborBlock &nb = tb.pbval->neighbor[n];
    printf("%i ", pbs->bd_var_.flag[nb.bufid]);
  } printf("\n");}
  gid = 1;
  {
  MeshBlock &tb = *pmb->pmy_mesh->FindMeshBlock(gid);
  std::shared_ptr<BoundarySwarm> pbs =
          tb.pbswarm->bswarms[0];
  printf("[%i] (%p): ", 0, pbs.get());
  for (int n = 0; n < pbs->bd_var_.nbmax; n++) {
    NeighborBlock &nb = tb.pbval->neighbor[n];
    printf("%i ", pbs->bd_var_.flag[nb.bufid]);
  } printf("\n\n\n");
  }}*/
  return true;
}

vpack_types::SwarmVarList<Real> Swarm::MakeRealList_(std::vector<std::string> &names) {
  int size = 0;
  vpack_types::SwarmVarList<Real> vars;

  for (auto it = realVector_.rbegin(); it != realVector_.rend(); ++it) {
    auto v = *it;
    vars.push_front(v);
    size++;
  }
  // Get names in same order as list
  names.resize(size);
  int it = 0;
  for (auto &v : vars) {
    names[it++] = v->label();
  }
  return vars;
}

vpack_types::SwarmVarList<int> Swarm::MakeIntList_(std::vector<std::string> &names) {
  int size = 0;
  vpack_types::SwarmVarList<int> vars;

  for (auto it = intVector_.rbegin(); it != intVector_.rend(); ++it) {
    auto v = *it;
    vars.push_front(v);
    size++;
  }
  // Get names in same order as list
  names.resize(size);
  int it = 0;
  for (auto &v : vars) {
    names[it++] = v->label();
  }
  return vars;
}

SwarmVariablePack<Real> Swarm::PackAllVariablesReal(PackIndexMap &vmap) {
  std::vector<std::string> names;
  for (auto &v : realVector_) {
    names.push_back(v->label());
  }
  return PackVariablesReal(names, vmap);
}

SwarmVariablePack<Real> Swarm::PackVariablesReal(const std::vector<std::string> &names,
                                                 PackIndexMap &vmap) {
  std::vector<std::string> expanded_names = names;
  vpack_types::SwarmVarList<Real> vars = MakeRealList_(expanded_names);

  auto pack = MakeSwarmPack<Real>(vars, &vmap);
  SwarmPackIndxPair<Real> value;
  value.pack = pack;
  value.map = vmap;
  return pack;
}
SwarmVariablePack<int> Swarm::PackVariablesInt(const std::vector<std::string> &names,
                                               PackIndexMap &vmap) {
  std::vector<std::string> expanded_names = names;
  vpack_types::SwarmVarList<int> vars = MakeIntList_(expanded_names);

  auto pack = MakeSwarmPack<int>(vars, &vmap);
  SwarmPackIndxPair<int> value;
  value.pack = pack;
  value.map = vmap;
  return pack;
}

static int count = 0;
bool Swarm::Receive(BoundaryCommSubset phase) {
  //if (count > 5) exit(-1);
  //count++;
  printf("[%i] Receive\n", Globals::my_rank);
  // Ensure all local deep copies marked BoundaryStatus::completed are actually received
  GetBlockPointer()->exec_space.fence();
  auto pmb = GetBlockPointer();

  // Populate buffers
  vbswarm->Receive(phase);

  /*{int gid = 0;
  {
  MeshBlock &tb = *pmb->pmy_mesh->FindMeshBlock(gid);
      std::shared_ptr<BoundarySwarm> pbs =
          tb.pbswarm->bswarms[0];
  printf("%s:%i\n", __FILE__, __LINE__);
  printf("[%i] (%p): ", 0, pbs.get());
  for (int n = 0; n < pbs->bd_var_.nbmax; n++) {
    NeighborBlock &nb = tb.pbval->neighbor[n];
    printf("%i ", pbs->bd_var_.flag[nb.bufid]);
  } printf("\n");}
  gid = 1;
  {
  MeshBlock &tb = *pmb->pmy_mesh->FindMeshBlock(gid);
  std::shared_ptr<BoundarySwarm> pbs =
          tb.pbswarm->bswarms[0];
  printf("[%i] (%p): ", 0, pbs.get());
  for (int n = 0; n < pbs->bd_var_.nbmax; n++) {
    NeighborBlock &nb = tb.pbval->neighbor[n];
    printf("%i ", pbs->bd_var_.flag[nb.bufid]);
  } printf("\n\n\n");
  }}*/

  /*printf("[%i] After vbswarm (%p): ", pmb->gid, vbswarm.get());
  for (int n = 0; n < vbswarm->bd_var_.nbmax; n++) {
    NeighborBlock &nb = pmb->pbval->neighbor[n];
    printf("%i ", vbswarm->bd_var_.flag[nb.bufid]);
    //printf("[%i] After vbswarm (%p) Neighbor %i bufid %i: BoundaryStatus: %i\n", pmb->gid, vbswarm.get(), nb.snb.gid, nb.bufid, vbswarm->bd_var_.flag[nb.bufid]);
  }
  printf("\n");*/

  // Copy buffers into swarm data on this proc
  int maxneighbor = vbswarm->bd_var_.nbmax;
  int total_received_particles = 0;
  std::vector<int> neighbor_received_particles(maxneighbor);
  for (int n = 0; n < maxneighbor; n++) {
    if (vbswarm->bd_var_.flag[pmb->pbval->neighbor[n].bufid] == BoundaryStatus::arrived) {
      total_received_particles += vbswarm->recv_size[n] / vbswarm->particle_size;
      neighbor_received_particles[n] = vbswarm->recv_size[n] / vbswarm->particle_size;
    } else {
      neighbor_received_particles[n] = 0;
    }
  }

  auto &bdvar = vbswarm->bd_var_;

  if (total_received_particles > 0) {
    ParArrayND<int> new_indices;
    auto new_mask = AddEmptyParticles(total_received_particles, new_indices);
    SwarmVariablePack<Real> vreal;
    SwarmVariablePack<int> vint;
    PackIndexMap rmap;
    PackIndexMap imap;
    PackAllVariables(vreal, vint, rmap, imap);
    int real_vars_size = realVector_.size();
    int int_vars_size = intVector_.size();
    const int ix = rmap["x"].first;
    const int iy = rmap["y"].first;
    const int iz = rmap["z"].first;

    ParArrayND<int> neighbor_index("Neighbor index", total_received_particles);
    ParArrayND<int> buffer_index("Buffer index", total_received_particles);
    auto neighbor_index_h = neighbor_index.GetHostMirror();
    auto buffer_index_h = buffer_index.GetHostMirror();
    int nid = 0;
    int per_neighbor_count = 0;

    int id = 0;
    for (int n = 0; n < maxneighbor; n++) {
      for (int m = 0; m < neighbor_received_particles[n]; m++) {
        neighbor_index_h(id) = n;
        buffer_index_h(id) = m;
        id++;
      }
    }
    neighbor_index.DeepCopy(neighbor_index_h);
    buffer_index.DeepCopy(buffer_index_h);

    // construct map from buffer index to swarm index (or just return vector of indices!)
    int particle_size = GetParticleDataSize();
    auto swarm_d = GetDeviceContext();

    pmb->par_for(
        "Unpack buffers", 0, total_received_particles - 1, KOKKOS_LAMBDA(const int n) {
          int sid = new_indices(n);
          int nid = neighbor_index(n);
          int bid = buffer_index(n);
          for (int i = 0; i < real_vars_size; i++) {
            vreal(i, sid) = bdvar.recv[nid](bid * particle_size + i);
          }
          for (int i = 0; i < int_vars_size; i++) {
            vint(i, sid) = static_cast<int>(
                bdvar.recv[nid]((real_vars_size + bid) * particle_size + i));
          }

          double &x = vreal(ix, sid);
          double &y = vreal(iy, sid);
          double &z = vreal(iz, sid);
          // TODO(BRR) Don't hardcode periodic boundary conditions
          if (x < swarm_d.x_min_global_) {
            x = swarm_d.x_max_global_ - (swarm_d.x_min_global_ - x);
          }
          if (x > swarm_d.x_max_global_) {
            x = swarm_d.x_min_global_ + (x - swarm_d.x_max_global_);
          }
          if (y < swarm_d.y_min_global_) {
            y = swarm_d.y_max_global_ - (swarm_d.y_min_global_ - y);
          }
          if (y > swarm_d.y_max_global_) {
            y = swarm_d.y_min_global_ + (y - swarm_d.y_max_global_);
          }
          if (z < swarm_d.z_min_global_) {
            z = swarm_d.z_max_global_ - (swarm_d.z_min_global_ - z);
          }
          if (z > swarm_d.z_max_global_) {
            z = swarm_d.z_min_global_ + (z - swarm_d.z_max_global_);
          }
        });
  }
  /*{int gid = 0;
  {
  MeshBlock &tb = *pmb->pmy_mesh->FindMeshBlock(gid);
      std::shared_ptr<BoundarySwarm> pbs =
          tb.pbswarm->bswarms[0];
  printf("%s:%i\n", __FILE__, __LINE__);
  printf("[%i] (%p): ", 0, pbs.get());
  for (int n = 0; n < pbs->bd_var_.nbmax; n++) {
    NeighborBlock &nb = tb.pbval->neighbor[n];
    printf("%i ", pbs->bd_var_.flag[nb.bufid]);
  } printf("\n");}
  gid = 1;
  {
  MeshBlock &tb = *pmb->pmy_mesh->FindMeshBlock(gid);
  std::shared_ptr<BoundarySwarm> pbs =
          tb.pbswarm->bswarms[0];
  printf("[%i] (%p): ", 0, pbs.get());
  for (int n = 0; n < pbs->bd_var_.nbmax; n++) {
    NeighborBlock &nb = tb.pbval->neighbor[n];
    printf("%i ", pbs->bd_var_.flag[nb.bufid]);
  } printf("\n\n\n");
  }}*/

  bool all_boundaries_received = true;
  for (int n = 0; n < pmb->pbval->nneighbor; n++) {
    NeighborBlock &nb = pmb->pbval->neighbor[n];
    //printf("[%i] Neighbor %i: BoundaryStatus: %i\n", pmb->gid, n, bdvar.flag[nb.bufid]);
    if (bdvar.flag[nb.bufid] == BoundaryStatus::arrived) {
      bdvar.flag[nb.bufid] = BoundaryStatus::completed;
    } else if (bdvar.flag[nb.bufid] == BoundaryStatus::waiting) {
      all_boundaries_received = false;
    }
  }
  /*{int gid = 0;
  {
  MeshBlock &tb = *pmb->pmy_mesh->FindMeshBlock(gid);
      std::shared_ptr<BoundarySwarm> pbs =
          tb.pbswarm->bswarms[0];
  printf("%s:%i\n", __FILE__, __LINE__);
  printf("[%i] (%p): ", 0, pbs.get());
  for (int n = 0; n < pbs->bd_var_.nbmax; n++) {
    NeighborBlock &nb = tb.pbval->neighbor[n];
    printf("%i ", pbs->bd_var_.flag[nb.bufid]);
  } printf("\n");}
  gid = 1;
  {
  MeshBlock &tb = *pmb->pmy_mesh->FindMeshBlock(gid);
  std::shared_ptr<BoundarySwarm> pbs =
          tb.pbswarm->bswarms[0];
  printf("[%i] (%p): ", 0, pbs.get());
  for (int n = 0; n < pbs->bd_var_.nbmax; n++) {
    NeighborBlock &nb = tb.pbval->neighbor[n];
    printf("%i ", pbs->bd_var_.flag[nb.bufid]);
  } printf("\n\n\n");
  }}*/

  if (all_boundaries_received) {
    printf("[%i] ALL BOUNDARIES RECEIVED\n", pmb->gid);
    return true;
  } else {
    printf("[%i] ALL BOUNDARIES NOT RECEIVED\n", pmb->gid);
    return false;
  }
}
void Swarm::PackAllVariables(SwarmVariablePack<Real> &vreal,
                             SwarmVariablePack<int> &vint) {
  PackIndexMap rmap, imap;
  std::vector<std::string> real_vars;
  std::vector<std::string> int_vars;
  for (auto &realVar : realVector_) {
    real_vars.push_back(realVar->label());
  }
  int real_vars_size = realVector_.size();
  int int_vars_size = intVector_.size();
  for (auto &intVar : intVector_) {
    int_vars.push_back(intVar->label());
  }
  vreal = PackVariablesReal(real_vars, rmap);
  vint = PackVariablesInt(int_vars, imap);
}

void Swarm::PackAllVariables(SwarmVariablePack<Real> &vreal, SwarmVariablePack<int> &vint,
                             PackIndexMap &rmap, PackIndexMap &imap) {
  std::vector<std::string> real_vars;
  std::vector<std::string> int_vars;
  for (auto &realVar : realVector_) {
    real_vars.push_back(realVar->label());
  }
  int real_vars_size = realVector_.size();
  int int_vars_size = intVector_.size();
  for (auto &intVar : intVector_) {
    int_vars.push_back(intVar->label());
  }
  vreal = PackVariablesReal(real_vars, rmap);
  vint = PackVariablesInt(int_vars, imap);
}

void Swarm::allocateComms(std::weak_ptr<MeshBlock> wpmb) {
  if (wpmb.expired()) return;

  std::shared_ptr<MeshBlock> pmb = wpmb.lock();

  // Create the boundary object
  vbswarm = std::make_shared<BoundarySwarm>(pmb);

  // Enroll SwarmVariable object
  vbswarm->bswarm_index = pmb->pbswarm->bswarms.size();
  pmb->pbswarm->bswarms.push_back(vbswarm);
}

} // namespace parthenon
