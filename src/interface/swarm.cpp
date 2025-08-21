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
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "interface/metadata.hpp"
#include "mesh/mesh.hpp"
#include "pack/swarm_default_names.hpp"
#include "swarm.hpp"
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
  context.buffer_sorted_ = buffer_sorted_;
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
      new_indices_("new_indices_", nmax_pool_), scratch_a_("scratch_a_", nmax_pool_),
      scratch_b_("scratch_b_", nmax_pool_),
      num_particles_to_send_("num_particles_to_send_", NMAX_NEIGHBORS),
      buffer_start_("buffer_start_", NMAX_NEIGHBORS),
      neighbor_received_particles_("neighbor_received_particles_", NMAX_NEIGHBORS),
      cell_sorted_("cell_sorted_", nmax_pool_),
      buffer_sorted_("buffer_sorted_", nmax_pool_), mpiStatus(true) {
  PARTHENON_REQUIRE_THROWS(typeid(Coordinates_t) == typeid(UniformCartesian),
                           "SwarmDeviceContext only supports a uniform Cartesian mesh!");

  uid_ = get_uid_(label_);

  // Add default swarm fields
  if (!metadata.IsSet(Metadata::NoPersistentParticleIds)) {
    Add(swarm_position::id::name(), Metadata({Metadata::UInt64}));
  }
  Add(swarm_position::x::name(), Metadata({Metadata::Real}));
  Add(swarm_position::y::name(), Metadata({Metadata::Real}));
  Add(swarm_position::z::name(), Metadata({Metadata::Real}));

  // Initialize index metadata
  num_active_ = 0;
  max_active_index_ = inactive_max_active_index;
  UpdateEmptyIndices();
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
      std::get<getType<Real>()>(maps_).count(label) > 0 ||
      std::get<getType<std::uint64_t>()>(maps_).count(label) > 0) {
    throw std::invalid_argument("swarm variable " + label +
                                " already enrolled during Add()!");
  }

  Metadata newm(metadata);
  newm.Set(Metadata::Particle);

  if (newm.Type() == Metadata::Integer) {
    Add_<int>(label, newm);
  } else if (newm.Type() == Metadata::UInt64) {
    Add_<std::uint64_t>(label, newm);
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
  auto &uint64_map = std::get<getType<std::uint64_t>()>(maps_);
  auto &uint64_vector = std::get<getType<std::uint64_t>()>(vectors_);
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
  if (found) {
    // first delete the variable
    int_vector[idx].reset();

    // Next move the last element into idx and pop last entry
    if (int_vector.size() > 1) int_vector[idx] = std::move(int_vector.back());
    int_vector.pop_back();

    // Also remove variable from map
    int_map.erase(label);
    return;
  }

  // search next variable type (real)
  idx = 0;
  for (const auto &v : real_vector) {
    if (label == v->label()) {
      found = true;
      break;
    }
    idx++;
  }
  if (found) {
    real_vector[idx].reset();
    if (real_vector.size() > 1) real_vector[idx] = std::move(real_vector.back());
    real_vector.pop_back();
    real_map.erase(label);
    return;
  }

  // search next variable type (uint64_t)
  idx = 0;
  for (const auto &v : uint64_vector) {
    if (label == v->label()) {
      found = true;
      break;
    }
    idx++;
  }
  if (found) {
    uint64_vector[idx].reset();
    if (uint64_vector.size() > 1) uint64_vector[idx] = std::move(uint64_vector.back());
    uint64_vector.pop_back();
    uint64_map.erase(label);
    return;
  }

  throw std::invalid_argument("swarm variable not found in Remove()");
}

void Swarm::SetPoolMax(const std::int64_t nmax_pool) {
  PARTHENON_REQUIRE(nmax_pool > nmax_pool_, "Must request larger pool size!");
  std::int64_t n_new = nmax_pool - nmax_pool_;

  auto pmb = GetBlockPointer();
  auto pm = pmb->pmy_mesh;

  // Rely on Kokkos setting the newly added values to false for these arrays
  Kokkos::resize(mask_, nmax_pool);
  Kokkos::resize(marked_for_removal_, nmax_pool);
  Kokkos::resize(empty_indices_, nmax_pool);
  Kokkos::resize(new_indices_, nmax_pool);
  Kokkos::resize(scratch_a_, nmax_pool);
  Kokkos::resize(scratch_b_, nmax_pool);

  pmb->LogMemUsage(2 * n_new * sizeof(bool));

  Kokkos::resize(cell_sorted_, nmax_pool);
  pmb->LogMemUsage(n_new * sizeof(SwarmKey));

  Kokkos::resize(buffer_sorted_, nmax_pool);
  pmb->LogMemUsage(n_new * sizeof(SwarmKey));

  block_index_.Resize(nmax_pool);
  pmb->LogMemUsage(n_new * sizeof(int));

  auto &int_vector = std::get<getType<int>()>(vectors_);
  auto &uint64_vector = std::get<getType<std::uint64_t>()>(vectors_);
  auto &real_vector = std::get<getType<Real>()>(vectors_);

  for (auto &d : int_vector) {
    d->data.Resize(d->data.GetDim(6), d->data.GetDim(5), d->data.GetDim(4),
                   d->data.GetDim(3), d->data.GetDim(2), nmax_pool);
    pmb->LogMemUsage(n_new * sizeof(int));
  }

  for (auto &d : uint64_vector) {
    d->data.Resize(d->data.GetDim(6), d->data.GetDim(5), d->data.GetDim(4),
                   d->data.GetDim(3), d->data.GetDim(2), nmax_pool);
    pmb->LogMemUsage(n_new * sizeof(std::uint64_t));
  }

  for (auto &d : real_vector) {
    d->data.Resize(d->data.GetDim(6), d->data.GetDim(5), d->data.GetDim(4),
                   d->data.GetDim(3), d->data.GetDim(2), nmax_pool);
    pmb->LogMemUsage(n_new * sizeof(Real));
  }

  nmax_pool_ = nmax_pool;

  // Populate new empty indices
  UpdateEmptyIndices();

  // Eliminate any cached SwarmPacks, as they will need to be rebuilt following SetPoolMax
  pmb->meshblock_data.Get()->ClearSwarmCaches();
  pm->mesh_data.Get("base")->ClearSwarmCaches();
  for (auto &partition : pm->GetDefaultBlockPartitions()) {
    pm->mesh_data.Add("base", partition)->ClearSwarmCaches();
  }
}

NewParticlesContext Swarm::AddEmptyParticles(const int num_to_add) {
  PARTHENON_DEBUG_REQUIRE(num_to_add >= 0, "Cannot add negative numbers of particles!");

  auto pmb = GetBlockPointer();

  if (num_to_add > 0) {
    while (nmax_pool_ - num_active_ < num_to_add) {
      IncreasePoolMax();
    }

    auto &new_indices = new_indices_;
    auto &empty_indices = empty_indices_;
    auto &mask = mask_;

    int max_new_active_index = 0;
    parthenon::par_reduce(
        PARTHENON_AUTO_LABEL, 0, num_to_add - 1,
        KOKKOS_LAMBDA(const int n, int &max_ind) {
          new_indices(n) = empty_indices(n);
          mask(new_indices(n)) = true;

          // Record vote for max active index
          max_ind = new_indices(n);
        },
        Kokkos::Max<int>(max_new_active_index));

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
}

// Updates the empty_indices_ array so the first N elements contain an ascending list of
// indices into empty elements of the swarm pool, where N is the number of empty indices
void Swarm::UpdateEmptyIndices() {
  auto &mask = mask_;
  auto &empty_indices = empty_indices_;

  // Associate scratch memory
  auto &empty_indices_scan = scratch_a_;

  // Calculate prefix sum of empty indices
  parthenon::par_scan(
      "Set empty indices prefix sum", 0, nmax_pool_ - 1,
      KOKKOS_LAMBDA(const int n, int &update, const bool &final) {
        const int val = !mask(n);
        if (val) {
          update += 1;
        }

        if (final) {
          empty_indices_scan(n) = update;
        }
      });

  // Update list of empty indices such that it is contiguous and in ascending order
  parthenon::par_for(
      PARTHENON_AUTO_LABEL, 0, nmax_pool_ - 1, KOKKOS_LAMBDA(const int n) {
        if (!mask(n)) {
          empty_indices(empty_indices_scan(n) - 1) = n;
        }
      });
}

// No active particles: nmax_active_index = inactive_max_active_index (= -1)
// No particles removed: nmax_active_index unchanged
// Particles removed: nmax_active_index is new max active index
void Swarm::RemoveMarkedParticles() {
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

  UpdateEmptyIndices();
}

void Swarm::Defrag() {
  if (GetNumActive() == 0) {
    return;
  }

  // Associate scratch memory
  auto &scan_scratch_toread = scratch_a_;
  auto &map = scratch_b_;

  auto &mask = mask_;

  const int &num_active = num_active_;
  auto empty_indices = empty_indices_;
  parthenon::par_scan(
      "Set empty indices prefix sum", num_active, nmax_pool_ - 1,
      KOKKOS_LAMBDA(const int n, int &update, const bool &final) {
        const int val = mask(n);
        if (val) {
          update += 1;
        }
        if (final) {
          scan_scratch_toread(n) = update;
          empty_indices(n - num_active) = n;
        }
      });

  parthenon::par_for(
      PARTHENON_AUTO_LABEL, 0, nmax_pool_ - 1, KOKKOS_LAMBDA(const int n) {
        if (n >= num_active) {
          if (mask(n)) {
            map(scan_scratch_toread(n) - 1) = n;
          }
          mask(n) = false;
        }
      });

  // Reuse scratch memory
  auto &scan_scratch_towrite = scan_scratch_toread;

  // Update list of empty indices
  parthenon::par_scan(
      "Set empty indices prefix sum", 0, num_active_ - 1,
      KOKKOS_LAMBDA(const int n, int &update, const bool &final) {
        const int val = !mask(n);
        if (val) {
          update += 1;
        }
        if (final) scan_scratch_towrite(n) = update;
      });

  // Get all dynamical variables in swarm
  auto &int_vector = std::get<getType<int>()>(vectors_);
  auto &uint64_vector = std::get<getType<std::uint64_t>()>(vectors_);
  auto &real_vector = std::get<getType<Real>()>(vectors_);
  PackIndexMap real_imap;
  PackIndexMap int_imap;
  PackIndexMap uint64_imap;
  auto vreal = PackAllVariables_<Real>(real_imap);
  auto vint = PackAllVariables_<int>(int_imap);
  auto vuint64 = PackAllVariables_<std::uint64_t>(uint64_imap);
  int real_vars_size = real_vector.size();
  int int_vars_size = int_vector.size();
  int uint64_vars_size = uint64_vector.size();
  auto real_map = real_imap.Map();
  auto int_map = int_imap.Map();
  auto uint64_map = uint64_imap.Map();
  const int realPackDim = vreal.GetDim(2);
  const int intPackDim = vint.GetDim(2);
  const int uint64PackDim = vuint64.GetDim(2);

  // Loop over only the active number of particles, and if mask is empty, copy in particle
  // using address from prefix sum
  parthenon::par_for(
      PARTHENON_AUTO_LABEL, 0, num_active_ - 1, KOKKOS_LAMBDA(const int n) {
        if (!mask(n)) {
          const int nread = map(scan_scratch_towrite(n) - 1);
          for (int vidx = 0; vidx < realPackDim; vidx++) {
            vreal(vidx, n) = vreal(vidx, nread);
          }
          for (int vidx = 0; vidx < intPackDim; vidx++) {
            vint(vidx, n) = vint(vidx, nread);
          }
          for (int vidx = 0; vidx < uint64PackDim; vidx++) {
            vuint64(vidx, n) = vuint64(vidx, nread);
          }
          mask(n) = true;
        }
      });

  // Update max_active_index_
  max_active_index_ = num_active_ - 1;
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

          if (cell_sorted(start_index).sort_idx_ == cell_idx_1d) {
            if (start_index == 0) {
              break;
            } else if (cell_sorted(start_index - 1).sort_idx_ != cell_idx_1d) {
              break;
            } else {
              start_index--;
              continue;
            }
          }
          if (cell_sorted(start_index).sort_idx_ >= cell_idx_1d) {
            start_index--;
            if (start_index < 0) {
              start_index = -1;
              break;
            }
            if (cell_sorted(start_index).sort_idx_ < cell_idx_1d) {
              start_index = -1;
              break;
            }
            continue;
          }
          if (cell_sorted(start_index).sort_idx_ < cell_idx_1d) {
            start_index++;
            if (start_index > max_active_index) {
              start_index = -1;
              break;
            }
            if (cell_sorted(start_index).sort_idx_ > cell_idx_1d) {
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
                 cell_sorted(current_index).sort_idx_ == cell_idx_1d) {
            current_index++;
            number++;
            cell_sorted_number(k, j, i) = number;
          }
        }
      });
}

void Swarm::Validate(bool test_comms) const {
  auto mask = mask_;
  auto neighbor_indices = neighbor_indices_;
  auto empty_indices = empty_indices_;

  // Check that number of unmasked particles is number of active particles
  int nactive = 0;
  parthenon::par_reduce(
      PARTHENON_AUTO_LABEL, 0, nmax_pool_ - 1,
      KOKKOS_LAMBDA(const int n, int &nact) {
        if (mask(n)) {
          nact += 1;
        }
      },
      Kokkos::Sum<int>(nactive));
  PARTHENON_REQUIRE(nactive == num_active_, "Mask and num_active counter do not agree!");

  // Check that region of neighbor indices corresponding to this block is correct
  // This is optional because the relevant infrastructure for comms isn't always allocated
  // in testing.
  int num_err = 0;
  if (test_comms) {
    parthenon::par_reduce(
        parthenon::loop_pattern_mdrange_tag, PARTHENON_AUTO_LABEL, DevExecSpace(), 1, 2,
        1, 2, 1, 2,
        KOKKOS_LAMBDA(const int k, const int j, const int i, int &nerr) {
          if (neighbor_indices(k, j, i) != this_block_) {
            nerr += 1;
          }
        },
        Kokkos::Sum<int>(num_err));
    PARTHENON_REQUIRE(num_err == 0,
                      "This block region of neighbor indices is incorrect!");
  }

  num_err = 0;
  parthenon::par_reduce(
      PARTHENON_AUTO_LABEL, 0, nmax_pool_ - num_active_ - 1,
      KOKKOS_LAMBDA(const int n, int &nerr) {
        if (mask(empty_indices(n)) == true) {
          nerr += 1;
        }
      },
      Kokkos::Sum<int>(num_err));
  PARTHENON_REQUIRE(num_err == 0,
                    "empty_indices_ array pointing to unmasked particle indices!");
}

} // namespace parthenon
