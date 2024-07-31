//========================================================================================
// (C) (or copyright) 2020-2024. Triad National Security, LLC. All rights reserved.
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
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "mesh/mesh.hpp"
#include "swarm.hpp"
#include "swarm_default_names.hpp"
#include "utils/error_checking.hpp"
#include "utils/sort.hpp"

namespace parthenon {

SwarmDeviceContext Swarm::GetDeviceContext() const {
  SwarmDeviceContext context;
  context.marked_for_removal_ = marked_for_removal_;
  context.mask_ = mask_;
  context.block_index_ = block_index_;
  context.neighbor_indices_ = neighbor_indices_;
  context.cell_sorted_ = cell_sorted_;
  context.cell_sorted_begin_ = cell_sorted_begin_;
  context.cell_sorted_number_ = cell_sorted_number_;

  auto pmb = GetBlockPointer();
  auto pmesh = pmb->pmy_mesh;
  auto mesh_size = pmesh->mesh_size;

  const IndexRange &ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange &jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange &kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  context.ib_s_ = ib.s;
  context.jb_s_ = jb.s;
  context.kb_s_ = kb.s;
  context.x_min_ = pmb->coords.Xf<1>(ib.s);
  context.y_min_ = pmb->coords.Xf<2>(jb.s);
  context.z_min_ = pmb->coords.Xf<3>(kb.s);
  context.x_max_ = pmb->coords.Xf<1>(ib.e + 1);
  context.y_max_ = pmb->coords.Xf<2>(jb.e + 1);
  context.z_max_ = pmb->coords.Xf<3>(kb.e + 1);
  context.x_min_global_ = mesh_size.xmin(X1DIR);
  context.x_max_global_ = mesh_size.xmax(X1DIR);
  context.y_min_global_ = mesh_size.xmin(X2DIR);
  context.y_max_global_ = mesh_size.xmax(X2DIR);
  context.z_min_global_ = mesh_size.xmin(X3DIR);
  context.z_max_global_ = mesh_size.xmax(X3DIR);
  context.ndim_ = pmb->pmy_mesh->ndim;
  context.my_rank_ = Globals::my_rank;
  context.coords_ = pmb->coords;
  return context;
}

Swarm::Swarm(const std::string &label, const Metadata &metadata, const int nmax_pool_in)
    : label_(label), m_(metadata), nmax_pool_(nmax_pool_in), mask_("mask", nmax_pool_),
      marked_for_removal_("mfr", nmax_pool_),
      empty_indices_("empty_indices_", nmax_pool_),
      block_index_("block_index_", nmax_pool_),
      neighbor_indices_("neighbor_indices_", 4, 4, 4),
      new_indices_("new_indices_", nmax_pool_),
      from_to_indices_("from_to_indices_", nmax_pool_ + 1),
      recv_neighbor_index_("recv_neighbor_index_", nmax_pool_),
      recv_buffer_index_("recv_buffer_index_", nmax_pool_),
      num_particles_to_send_("num_particles_to_send_", NMAX_NEIGHBORS),
      cell_sorted_("cell_sorted_", nmax_pool_), mpiStatus(true) {
  PARTHENON_REQUIRE_THROWS(typeid(Coordinates_t) == typeid(UniformCartesian),
                           "SwarmDeviceContext only supports a uniform Cartesian mesh!");

  uid_ = get_uid_(label_);

  Add(swarm_position::x::name(), Metadata({Metadata::Real}));
  Add(swarm_position::y::name(), Metadata({Metadata::Real}));
  Add(swarm_position::z::name(), Metadata({Metadata::Real}));
  num_active_ = 0;
  max_active_index_ = inactive_max_active_index;

  // Initialize pool metadata arrays
  parthenon::par_for(
      PARTHENON_AUTO_LABEL, 0, nmax_pool_ - 1, KOKKOS_LAMBDA(const int n) {
        mask_(n) = false;
        marked_for_removal_(n) = false;
        empty_indices_(n) = n;
      });
}

void Swarm::Add(const std::vector<std::string> &label_array, const Metadata &metadata) {
  // generate the vector and call Add
  for (auto label : label_array) {
    Add(label, metadata);
  }
}

std::shared_ptr<Swarm> Swarm::AllocateCopy(MeshBlock * /*pmb*/) {
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
  //  if (intMap_.count(label) > 0 || realMap_.count(label) > 0) {
  if (std::get<getType<int>()>(maps_).count(label) > 0 ||
      std::get<getType<Real>()>(maps_).count(label) > 0) {
    throw std::invalid_argument("swarm variable " + label +
                                " already enrolled during Add()!");
  }

  Metadata newm(metadata);
  newm.Set(Metadata::Particle);

  if (newm.Type() == Metadata::Integer) {
    Add_<int>(label, newm);
  } else if (newm.Type() == Metadata::Real) {
    Add_<Real>(label, newm);
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

  auto &int_map = std::get<getType<int>()>(maps_);
  auto &int_vector = std::get<getType<int>()>(vectors_);
  auto &real_map = std::get<getType<Real>()>(maps_);
  auto &real_vector = std::get<getType<Real>()>(vectors_);

  // Find index of variable
  int idx = 0;
  for (auto v : int_vector) {
    if (label == v->label()) {
      found = true;
      break;
    }
    idx++;
  }
  if (found == true) {
    // first delete the variable
    int_vector[idx].reset();

    // Next move the last element into idx and pop last entry
    if (int_vector.size() > 1) int_vector[idx] = std::move(int_vector.back());
    int_vector.pop_back();

    // Also remove variable from map
    int_map.erase(label);
  }

  if (found == false) {
    idx = 0;
    for (const auto &v : real_vector) {
      if (label == v->label()) {
        found = true;
        break;
      }
      idx++;
    }
  }
  if (found == true) {
    real_vector[idx].reset();
    if (real_vector.size() > 1) real_vector[idx] = std::move(real_vector.back());
    real_vector.pop_back();
    real_map.erase(label);
  }

  if (found == false) {
    throw std::invalid_argument("swarm variable not found in Remove()");
  }
}

void Swarm::setPoolMax(const std::int64_t nmax_pool) {
  printf(" === setPoolMax\n");
  PARTHENON_REQUIRE(nmax_pool > nmax_pool_, "Must request larger pool size!");
  std::int64_t n_new_begin = nmax_pool_;
  std::int64_t n_new = nmax_pool - nmax_pool_;

  auto pmb = GetBlockPointer();
  auto pm = pmb->pmy_mesh;

  // Rely on Kokkos setting the newly added values to false for these arrays
  Kokkos::resize(mask_, nmax_pool);
  Kokkos::resize(marked_for_removal_, nmax_pool);
  Kokkos::resize(empty_indices_, nmax_pool);
  Kokkos::resize(new_indices_, nmax_pool);
  Kokkos::resize(from_to_indices_, nmax_pool + 1);
  Kokkos::resize(recv_neighbor_index_, nmax_pool);
  Kokkos::resize(recv_buffer_index_, nmax_pool);

  pmb->LogMemUsage(2 * n_new * sizeof(bool));

  Kokkos::resize(cell_sorted_, nmax_pool);
  pmb->LogMemUsage(n_new * sizeof(SwarmKey));

  block_index_.Resize(nmax_pool);
  pmb->LogMemUsage(n_new * sizeof(int));

  auto &int_vector = std::get<getType<int>()>(vectors_);
  auto &real_vector = std::get<getType<Real>()>(vectors_);

  for (auto &d : int_vector) {
    d->data.Resize(d->data.GetDim(6), d->data.GetDim(5), d->data.GetDim(4),
                   d->data.GetDim(3), d->data.GetDim(2), nmax_pool);
    pmb->LogMemUsage(n_new * sizeof(int));
  }

  for (auto &d : real_vector) {
    d->data.Resize(d->data.GetDim(6), d->data.GetDim(5), d->data.GetDim(4),
                   d->data.GetDim(3), d->data.GetDim(2), nmax_pool);
    pmb->LogMemUsage(n_new * sizeof(Real));
  }

  nmax_pool_ = nmax_pool;

  // Populate new empty indices
  UpdateEmptyIndices();

  // Eliminate any cached SwarmPacks, as they will need to be rebuilt following setPoolMax
  pmb->meshblock_data.Get()->ClearSwarmCaches();
  pm->mesh_data.Get("base")->ClearSwarmCaches();
  for (auto &partition : pm->GetDefaultBlockPartitions()) {
    pm->mesh_data.Add("base", partition)->ClearSwarmCaches();
  }
  printf(" ===\n");
}

NewParticlesContext Swarm::AddEmptyParticles(const int num_to_add) {
  PARTHENON_DEBUG_REQUIRE(num_to_add >= 0, "Cannot add negative numbers of particles!");

  auto pmb = GetBlockPointer();

  printf(" === AddEmptyParticles\n");
  printf("num_to_add = %i\n", num_to_add);

  if (num_to_add > 0) {
    while (nmax_pool_ - num_active_ < num_to_add) {
      increasePoolMax();
    }

    auto &new_indices = new_indices_;
    auto &empty_indices = empty_indices_;
    auto &mask = mask_;

    // Update mask and new particle indices
    // pmb->par_for(
    //    PARTHENON_AUTO_LABEL, 0, num_to_add - 1, KOKKOS_LAMBDA(const int n) {
    //      new_indices(n) = empty_indices(n);
    //      mask(n) = true;
    //    });

    int max_new_active_index = 0;
    parthenon::par_reduce(
        PARTHENON_AUTO_LABEL, 0, num_to_add - 1,
        KOKKOS_LAMBDA(const int n, int &max_ind) {
          new_indices(n) = empty_indices(n);
          mask(new_indices(n)) = true;

          // Record vote for max active index
          max_ind = new_indices(n);
          printf("  [%i] new_indices(n) = %i max_ind = %i\n", n, new_indices(n), max_ind);
        },
        Kokkos::Max<int>(max_new_active_index));

    printf("max_new_active_index = %i\n", max_new_active_index);

    // Update max active index if necessary
    max_active_index_ = std::max(max_active_index_, max_new_active_index);

    new_indices_max_idx_ = num_to_add - 1;
    num_active_ += num_to_add;

    UpdateEmptyIndices();
  } else {
    new_indices_max_idx_ = -1;
  }

  // Create and return NewParticlesContext
  return NewParticlesContext(new_indices_max_idx_, new_indices_);

  // if (num_to_add > 0) {
  //  while (free_indices_.size() < num_to_add) {
  //    increasePoolMax();
  //  }

  //  // TODO(BRR) Use par_scan on device rather than do this on host
  //  auto mask_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), mask_);

  //  auto block_index_h = block_index_.GetHostMirrorAndCopy();

  //  auto free_index = free_indices_.begin();

  //  auto new_indices_h = new_indices_.GetHostMirror();

  //  // Don't bother sanitizing the memory
  //  for (int n = 0; n < num_to_add; n++) {
  //    mask_h(*free_index) = true;
  //    block_index_h(*free_index) = this_block_;
  //    max_active_index_ = std::max<int>(max_active_index_, *free_index);
  //    new_indices_h(n) = *free_index;

  //    free_index = free_indices_.erase(free_index);
  //  }

  //  new_indices_.DeepCopy(new_indices_h);

  //  num_active_ += num_to_add;

  //  Kokkos::deep_copy(mask_, mask_h);
  //  block_index_.DeepCopy(block_index_h);
  //  new_indices_max_idx_ = num_to_add - 1;
  //} else {
  //  new_indices_max_idx_ = -1;
  //}

  // return NewParticlesContext(new_indices_max_idx_, new_indices_);
}

void Swarm::UpdateEmptyIndices() {
  auto pmb = GetBlockPointer();

  int &max_active_index = max_active_index_;

  auto &mask = mask_;
  auto &empty_indices = empty_indices_;

  printf(" === UpdateEmptyIndices\n");
  printf("max_active_index = %i\n", max_active_index);

  ParArray1D<int> scan_scratch("Scan scratch", nmax_pool_);

  // Update list of empty indices
  Kokkos::parallel_scan(
      "Set empty indices prefix sum", nmax_pool_,
      KOKKOS_LAMBDA(const int n, int &update, const bool &final) {
        printf("[%i] update: %i final: %i mask: %i\n", n, update, final, mask(n));

        const int val = !mask(n);
        if (val) {
          update += 1;
        }
        // if (final) empty_indices(n) = update;
        if (final) scan_scratch(n) = update;
        // update += val;

        // if (final && !mask(n)) {
        //  empty_indices(n) = update;
        //}
        // if (!mask(n)) {
        //  update += 1;
        //}

        // const bool is_free = !mask(n);
        // if (final) {
        //  if (n > 0) {
        //    empty_indices(n) = update;
        //    printf("final, empty_indices(%i) = %i\n", n, empty_indices(n));
        //  }
        //  // if (n == max_active_index) {
        //  if (n == nmax_pool_ - 1) {
        //    // printf("LAST final, n: %i update: %i\n", n, update);
        //    empty_indices(n) = update + is_free;
        //    printf("LAST final, empty_indices(%i) = %i\n", n, empty_indices(n));
        //  }
        //  update += is_free;
        //}
      });

  const int n_empty = nmax_pool_ - num_active_;

  printf("n empty: %i\n", n_empty);

  for (int n = 0; n < nmax_pool_; n++) {
    printf("  mask[%i] = %i\n", n, mask(n));
  }

  for (int n = 0; n < nmax_pool_ - num_active_; n++) {
    printf("  scan[%i] = %i\n", n, scan_scratch(n));
  }

  parthenon::par_for(
      PARTHENON_AUTO_LABEL, 0, nmax_pool_ - 1, KOKKOS_LAMBDA(const int n) {
        if (!mask(n)) {
          empty_indices(scan_scratch(n) - 1) = n;
        }
      });

  for (int n = 0; n < nmax_pool_ - num_active_; n++) {
    printf("  empty_indices[%i] = %i\n", n, empty_indices(n));
  }

  printf("empty indices: %i (nmax pool: %i num active: %i)\n", nmax_pool_ - num_active_,
         nmax_pool_, num_active_);
  printf(" === (UpdateEmptyIndices)\n");
}

// No active particles: nmax_active_index = inactive_max_active_index (= -1)
// No particles removed: nmax_active_index unchanged
// Particles removed: nmax_active_index is new max active index
void Swarm::RemoveMarkedParticles() {
  printf(" === RemoveMarkedParticles\n");

  int &max_active_index = max_active_index_;

  auto &mask = mask_;
  auto &marked_for_removal = marked_for_removal_;

  // Update mask, count number of removed particles
  int num_removed = 0;
  parthenon::par_reduce(
      PARTHENON_AUTO_LABEL, 0, max_active_index,
      KOKKOS_LAMBDA(const int n, int &removed) {
        if (mask(n)) {
          if (marked_for_removal(n)) {
            mask(n) = false;
            marked_for_removal(n) = false;
            removed += 1;
          }
        }
      },
      Kokkos::Sum<int>(num_removed));

  num_active_ -= num_removed;
  printf("num_removed: %i num_active: %i\n", num_removed, num_active_);

  UpdateEmptyIndices();
  printf(" === (RemoveMarkedParticles)\n");

  //  // Update list of empty indices
  //  Kokkos::parallel_scan(
  //      "Set empty indices", max_active_index + 1,
  //      KOKKOS_LAMBDA(const int n, int &update, const bool &final) {
  //        const bool is_free = !mask_(n);
  //        if (final) {
  //          if (n > 0) {
  //            empty_indices_(n) = update;
  //          }
  //          if (n == max_active_index) {
  //            empty_indices_(n) = update + is_free;
  //          }
  //          update += is_free;
  //        }
  //      });

  // number of empty indices = nmax_pool_ - num_active_

  // TODO(BRR) Use par_scan to do this on device rather than host
  // auto mask_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), mask_);
  // auto marked_for_removal_h =
  //    Kokkos::create_mirror_view_and_copy(HostMemSpace(), marked_for_removal_);

  //// loop backwards to keep free_indices_ updated correctly
  // for (int n = max_active_index_; n >= 0; n--) {
  //  if (mask_h(n)) {
  //    if (marked_for_removal_h(n)) {
  //      mask_h(n) = false;
  //      free_indices_.push_front(n);
  //      num_active_ -= 1;
  //      if (n == max_active_index_) {
  //        max_active_index_ -= 1;
  //      }
  //      marked_for_removal_h(n) = false;
  //    }
  //  }
  //}

  // Kokkos::deep_copy(mask_, mask_h);
  // Kokkos::deep_copy(marked_for_removal_, marked_for_removal_h);
}

void Swarm::Defrag() {
  printf(" === Defrag\n");

  if (GetNumActive() == 0) {
    return;
  }

  // TODO(BRR) store scan_scratch with swarm
  ParArray1D<int> scan_scratch_towrite("Scan scratch write", nmax_pool_);
  ParArray1D<int> scan_scratch_toread("Scan scratch read", nmax_pool_);
  ParArray1D<int> map("Scan scratch map", nmax_pool_);

  auto &mask = mask_;

  for (int n = 0; n < nmax_pool_; n++) {
    printf("  mask(%i) = %i\n", n, static_cast<int>(mask(n)));
  }

  // Update list of empty indices
  Kokkos::parallel_scan(
      "Set empty indices prefix sum", nmax_pool_,
      KOKKOS_LAMBDA(const int n, int &update, const bool &final) {
        const int val = !mask(n);
        if (val) {
          update += 1;
        }
        if (final) scan_scratch_towrite(n) = update;
      });

  const int &nmax_pool = nmax_pool_;
  const int &num_active = num_active_;
  Kokkos::parallel_scan(
      //"Set empty indices prefix sum", num_active_, nmax_pool_ - 1,
      "Set empty indices prefix sum", nmax_pool_ - num_active_,
      KOKKOS_LAMBDA(const int nn, int &update, const bool &final) {
        const int n = nn + num_active;
        const int val = mask(n);
        if (val) {
          update += 1;
        }
        if (final) scan_scratch_toread(n) = update;
      });

  parthenon::par_for(
      // PARTHENON_AUTO_LABEL, num_active_, nmax_pool_ - 1, KOKKOS_LAMBDA(const int n) {
      PARTHENON_AUTO_LABEL, 0, nmax_pool_ - 1, KOKKOS_LAMBDA(const int n) {
        if (n >= num_active_) {
          if (mask(n)) {
            map(scan_scratch_toread(n) - 1) = n;
          }
          mask(n) = false;
          //} else {
          //  mask(n) = true;
        }
      });

  for (int n = 0; n < nmax_pool_; n++) {
    printf("[%i] towrite: %i toread: %i map: %i\n", n, scan_scratch_towrite(n),
           scan_scratch_toread(n), map(n));
  }

  auto &int_vector = std::get<getType<int>()>(vectors_);
  auto &real_vector = std::get<getType<Real>()>(vectors_);
  PackIndexMap real_imap;
  PackIndexMap int_imap;
  auto vreal = PackAllVariables_<Real>(real_imap);
  auto vint = PackAllVariables_<int>(int_imap);
  int real_vars_size = real_vector.size();
  int int_vars_size = int_vector.size();
  auto real_map = real_imap.Map();
  auto int_map = int_imap.Map();
  const int realPackDim = vreal.GetDim(2);
  const int intPackDim = vint.GetDim(2);

  // Loop over only the active number of particles, and if mask is empty, copy in particle
  // using address from prefix sum
  parthenon::par_for(
      PARTHENON_AUTO_LABEL, 0, num_active_ - 1, KOKKOS_LAMBDA(const int n) {
        if (!mask(n)) {
          const int nread = map(scan_scratch_towrite(n) - 1);
          // const int nn =
          //    scan_scratch_toread(scan_scratch_towrite(n) + num_active) + num_active;
          printf("    n: %i nread: %i = map(%i)\n", n, nread, scan_scratch_towrite(n));
          for (int vidx = 0; vidx < realPackDim; vidx++) {
            vreal(vidx, n) = vreal(vidx, nread);
          }
          for (int vidx = 0; vidx < intPackDim; vidx++) {
            vint(vidx, n) = vint(vidx, nread);
          }
          mask(n) = true;
        }
      });

  printf(" === (Defrag)\n");

  // Loop over only the active number of particles, and if mask is empty, copy in particle
  // using address from prefix sum
  //  pmb->par_for(
  //      PARTHENON_AUTO_LABEL, 0, num_active_ - 1, KOKKOS_LAMBDA(const int n) {
  //        if (!mask(n)) {
  //          for (int vidx = 0; vidx < realPackDim; vidx++) {
  //            vreal(vidx, from_to_indices(n)) = vreal(vidx, n);
  //          }
  //          for (int vidx = 0; vidx < intPackDim; vidx++) {
  //            vint(vidx, from_to_indices(n)) = vint(vidx, n);
  //          }
  //        }
  //
  //        if (from_to_indices(n) >= 0) {
  //          for (int vidx = 0; vidx < realPackDim; vidx++) {
  //            vreal(vidx, from_to_indices(n)) = vreal(vidx, n);
  //          }
  //          for (int vidx = 0; vidx < intPackDim; vidx++) {
  //            vint(vidx, from_to_indices(n)) = vint(vidx, n);
  //          }
  //        }
  //      });

  // Update max_active_index_
  max_active_index_ = num_active_ - 1;

  //// TODO(BRR) Could this algorithm be more efficient? Does it matter?
  //// Add 1 to convert max index to max number
  // std::int64_t num_free = (max_active_index_ + 1) - num_active_;
  // auto pmb = GetBlockPointer();

  // auto from_to_indices_h = from_to_indices_.GetHostMirror();

  // auto mask_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), mask_);

  // for (int n = 0; n <= max_active_index_; n++) {
  //  from_to_indices_h(n) = unset_index_;
  //}

  //// std::list<int> new_free_indices;

  //// free_indices_.sort();

  // int index = max_active_index_;
  // int num_to_move = std::min<int>(num_free, num_active_);
  // for (int n = 0; n < num_to_move; n++) {
  //  while (mask_h(index) == false) {
  //    index--;
  //  }
  //  int index_to_move_from = index;
  //  index--;

  //  // Below this number "moved" particles should actually stay in place
  //  if (index_to_move_from < num_active_) {
  //    break;
  //  }
  //  int index_to_move_to = free_indices_.front();
  //  free_indices_.pop_front();
  //  new_free_indices.push_back(index_to_move_from);
  //  from_to_indices_h(index_to_move_from) = index_to_move_to;
  //}

  //// TODO(BRR) Not all these sorts may be necessary
  // new_free_indices.sort();
  // free_indices_.merge(new_free_indices);

  // from_to_indices_.DeepCopy(from_to_indices_h);

  // auto from_to_indices = from_to_indices_;

  // auto &mask = mask_;
  // pmb->par_for(
  //    PARTHENON_AUTO_LABEL, 0, max_active_index_, KOKKOS_LAMBDA(const int n) {
  //      if (from_to_indices(n) >= 0) {
  //        mask(from_to_indices(n)) = mask(n);
  //        mask(n) = false;
  //      }
  //    });

  // auto &int_vector = std::get<getType<int>()>(vectors_);
  // auto &real_vector = std::get<getType<Real>()>(vectors_);
  // PackIndexMap real_imap;
  // PackIndexMap int_imap;
  // auto vreal = PackAllVariables_<Real>(real_imap);
  // auto vint = PackAllVariables_<int>(int_imap);
  // int real_vars_size = real_vector.size();
  // int int_vars_size = int_vector.size();
  // auto real_map = real_imap.Map();
  // auto int_map = int_imap.Map();
  // const int realPackDim = vreal.GetDim(2);
  // const int intPackDim = vint.GetDim(2);

  // pmb->par_for(
  //    PARTHENON_AUTO_LABEL, 0, max_active_index_, KOKKOS_LAMBDA(const int n) {
  //      if (from_to_indices(n) >= 0) {
  //        for (int vidx = 0; vidx < realPackDim; vidx++) {
  //          vreal(vidx, from_to_indices(n)) = vreal(vidx, n);
  //        }
  //        for (int vidx = 0; vidx < intPackDim; vidx++) {
  //          vint(vidx, from_to_indices(n)) = vint(vidx, n);
  //        }
  //      }
  //    });

  //// Update max_active_index_
  // max_active_index_ = num_active_ - 1;
}

///
/// Routine to sort particles by cell. Updates internal swarm variables:
///  cell_sorted_: 1D Per-cell sorted array of swarm memory indices
///  (SwarmKey::swarm_index_) cell_sorted_begin_: Per-cell array of starting indices in
///  cell_sorted_ cell_sorted_number_: Per-cell array of number of particles in each cell
///
void Swarm::SortParticlesByCell() {
  auto pmb = GetBlockPointer();

  auto &x = Get<Real>(swarm_position::x::name()).Get();
  auto &y = Get<Real>(swarm_position::y::name()).Get();
  auto &z = Get<Real>(swarm_position::z::name()).Get();

  const int nx1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
  const int nx2 = pmb->cellbounds.ncellsj(IndexDomain::entire);
  const int nx3 = pmb->cellbounds.ncellsk(IndexDomain::entire);
  PARTHENON_REQUIRE(nx1 * nx2 * nx3 < std::numeric_limits<int>::max(),
                    "Too many cells for an int32 to store cell_idx_1d below!");

  auto cell_sorted = cell_sorted_;
  int ncells = pmb->cellbounds.GetTotal(IndexDomain::entire);
  int num_active = num_active_;
  int max_active_index = max_active_index_;

  // Allocate data if necessary
  if (cell_sorted_begin_.GetDim(1) == 0) {
    cell_sorted_begin_ = ParArrayND<int>("cell_sorted_begin_", nx3, nx2, nx1);
    cell_sorted_number_ = ParArrayND<int>("cell_sorted_number_", nx3, nx2, nx1);
  }
  auto cell_sorted_begin = cell_sorted_begin_;
  auto cell_sorted_number = cell_sorted_number_;
  auto swarm_d = GetDeviceContext();

  // Write an unsorted list
  pmb->par_for(
      PARTHENON_AUTO_LABEL, 0, max_active_index_, KOKKOS_LAMBDA(const int n) {
        int i, j, k;
        swarm_d.Xtoijk(x(n), y(n), z(n), i, j, k);
        const int64_t cell_idx_1d = i + nx1 * (j + nx2 * k);
        cell_sorted(n) = SwarmKey(static_cast<int>(cell_idx_1d), n);
      });

  sort(cell_sorted, SwarmKeyComparator(), 0, max_active_index);

  // Update per-cell arrays for easier accessing later
  const IndexRange &ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  const IndexRange &jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  const IndexRange &kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  pmb->par_for(
      PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        int cell_idx_1d = i + nx1 * (j + nx2 * k);
        // Find starting index, first by guessing
        int start_index =
            static_cast<int>((cell_idx_1d * static_cast<Real>(num_active) / ncells));
        int n = 0;
        while (true) {
          n++;
          // Check if we left the list
          if (start_index < 0 || start_index > max_active_index) {
            start_index = -1;
            break;
          }

          if (cell_sorted(start_index).cell_idx_1d_ == cell_idx_1d) {
            if (start_index == 0) {
              break;
            } else if (cell_sorted(start_index - 1).cell_idx_1d_ != cell_idx_1d) {
              break;
            } else {
              start_index--;
              continue;
            }
          }
          if (cell_sorted(start_index).cell_idx_1d_ >= cell_idx_1d) {
            start_index--;
            if (start_index < 0) {
              start_index = -1;
              break;
            }
            if (cell_sorted(start_index).cell_idx_1d_ < cell_idx_1d) {
              start_index = -1;
              break;
            }
            continue;
          }
          if (cell_sorted(start_index).cell_idx_1d_ < cell_idx_1d) {
            start_index++;
            if (start_index > max_active_index) {
              start_index = -1;
              break;
            }
            if (cell_sorted(start_index).cell_idx_1d_ > cell_idx_1d) {
              start_index = -1;
              break;
            }
            continue;
          }
        }
        cell_sorted_begin(k, j, i) = start_index;
        if (start_index == -1) {
          cell_sorted_number(k, j, i) = 0;
        } else {
          int number = 0;
          int current_index = start_index;
          while (current_index <= max_active_index &&
                 cell_sorted(current_index).cell_idx_1d_ == cell_idx_1d) {
            current_index++;
            number++;
            cell_sorted_number(k, j, i) = number;
          }
        }
      });
}

} // namespace parthenon
