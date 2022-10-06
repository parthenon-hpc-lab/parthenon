//========================================================================================
// (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
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
#include "utils/error_checking.hpp"
#include "utils/sort.hpp"

namespace parthenon {

SwarmDeviceContext Swarm::GetDeviceContext() const {
  SwarmDeviceContext context;
  context.marked_for_removal_ = marked_for_removal_;
  context.mask_ = mask_;
  context.blockIndex_ = blockIndex_;
  context.neighborIndices_ = neighborIndices_;
  context.cellSorted_ = cellSorted_;
  context.cellSortedBegin_ = cellSortedBegin_;
  context.cellSortedNumber_ = cellSortedNumber_;

  auto pmb = GetBlockPointer();
  auto pmesh = pmb->pmy_mesh;
  auto mesh_size = pmesh->mesh_size;

  const IndexRange &ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange &jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange &kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  context.ib_s_ = ib.s;
  context.jb_s_ = jb.s;
  context.kb_s_ = kb.s;
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
  context.my_rank_ = Globals::my_rank;
  context.coords_ = pmb->coords;
  return context;
}

Swarm::Swarm(const std::string &label, const Metadata &metadata, const int nmax_pool_in)
    : label_(label), m_(metadata), nmax_pool_(nmax_pool_in), mask_("mask", nmax_pool_),
      marked_for_removal_("mfr", nmax_pool_), blockIndex_("blockIndex_", nmax_pool_),
      neighborIndices_("neighborIndices_", 4, 4, 4),
      cellSorted_("cellSorted_", nmax_pool_), mpiStatus(true) {
  PARTHENON_REQUIRE_THROWS(typeid(Coordinates_t) == typeid(UniformCartesian),
                           "SwarmDeviceContext only supports a uniform Cartesian mesh!");

  Add("x", Metadata({Metadata::Real}));
  Add("y", Metadata({Metadata::Real}));
  Add("z", Metadata({Metadata::Real}));
  num_active_ = 0;
  max_active_index_ = 0;

  auto mask_h = Kokkos::create_mirror_view(HostMemSpace(), mask_);
  auto marked_for_removal_h =
      Kokkos::create_mirror_view(HostMemSpace(), marked_for_removal_);

  for (int n = 0; n < nmax_pool_; n++) {
    mask_h(n) = false;
    marked_for_removal_h(n) = false;
    free_indices_.push_back(n);
  }

  Kokkos::deep_copy(mask_, mask_h);
  Kokkos::deep_copy(marked_for_removal_, marked_for_removal_h);
}

template <class BOutflow, class BPeriodic, int iFace>
void Swarm::AllocateBoundariesImpl_(MeshBlock *pmb) {
  std::stringstream msg;
  auto &bcs = pmb->pmy_mesh->mesh_bcs;
  if (bcs[iFace] == BoundaryFlag::outflow) {
    bounds_uptrs[iFace] = DeviceAllocate<BOutflow>();
  } else if (bcs[iFace] == BoundaryFlag::periodic) {
    bounds_uptrs[iFace] = DeviceAllocate<BPeriodic>();
  } else if (bcs[iFace] == BoundaryFlag::user) {
    if (pmb->pmy_mesh->SwarmBndryFnctn[iFace] != nullptr) {
      bounds_uptrs[iFace] = pmb->pmy_mesh->SwarmBndryFnctn[iFace]();
    } else {
      msg << "ix" << iFace + 1
          << " user boundary requested but provided function is null!";
      PARTHENON_THROW(msg);
    }
  } else {
    msg << "ix" << iFace + 1 << " boundary flag " << static_cast<int>(bcs[iFace])
        << " not supported!";
    PARTHENON_THROW(msg);
  }
}

void Swarm::AllocateBoundaries() {
  auto pmb = GetBlockPointer();
  std::stringstream msg;

  auto &bcs = pmb->pmy_mesh->mesh_bcs;

  AllocateBoundariesImpl_<ParticleBoundIX1Outflow, ParticleBoundIX1Periodic, 0>(
      pmb.get());
  AllocateBoundariesImpl_<ParticleBoundOX1Outflow, ParticleBoundOX1Periodic, 1>(
      pmb.get());
  AllocateBoundariesImpl_<ParticleBoundIX2Outflow, ParticleBoundIX2Periodic, 2>(
      pmb.get());
  AllocateBoundariesImpl_<ParticleBoundOX2Outflow, ParticleBoundOX2Periodic, 3>(
      pmb.get());
  AllocateBoundariesImpl_<ParticleBoundIX3Outflow, ParticleBoundIX3Periodic, 4>(
      pmb.get());
  AllocateBoundariesImpl_<ParticleBoundOX3Outflow, ParticleBoundOX3Periodic, 5>(
      pmb.get());

  for (int n = 0; n < 6; n++) {
    bounds_d.bounds[n] = bounds_uptrs[n].get();
    std::stringstream msg;
    msg << "Boundary condition on face " << n << " missing.\n"
        << "Please set it to `outflow`, `periodic`, or `user` in the input deck.\n"
        << "If you set it to user, you must also manually set "
        << "the swarm boundary pointer in your application." << std::endl;
    PARTHENON_REQUIRE(bounds_d.bounds[n] != nullptr, msg);
  }
}

void Swarm::Add(const std::vector<std::string> &labelArray, const Metadata &metadata) {
  // generate the vector and call Add
  for (auto label : labelArray) {
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
  if (std::get<getType<int>()>(Maps_).count(label) > 0 ||
      std::get<getType<Real>()>(Maps_).count(label) > 0) {
    throw std::invalid_argument("swarm variable " + label +
                                " already enrolled during Add()!");
  }

  if (metadata.Type() == Metadata::Integer) {
    Add_<int>(label, metadata);
  } else if (metadata.Type() == Metadata::Real) {
    Add_<Real>(label, metadata);
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

  auto &intMap_ = std::get<getType<int>()>(Maps_);
  auto &intVector_ = std::get<getType<int>()>(Vectors_);
  auto &realMap_ = std::get<getType<Real>()>(Maps_);
  auto &realVector_ = std::get<getType<Real>()>(Vectors_);

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

void Swarm::setPoolMax(const int nmax_pool) {
  PARTHENON_REQUIRE(nmax_pool > nmax_pool_, "Must request larger pool size!");
  int n_new_begin = nmax_pool_;
  int n_new = nmax_pool - nmax_pool_;

  auto pmb = GetBlockPointer();

  for (int n = 0; n < n_new; n++) {
    free_indices_.push_back(n + n_new_begin);
  }

  // Rely on Kokkos setting the newly added values to false for these arrays
  Kokkos::resize(mask_, nmax_pool);
  Kokkos::resize(marked_for_removal_, nmax_pool);

  Kokkos::resize(cellSorted_, nmax_pool);

  blockIndex_.Resize(nmax_pool);

  auto &intMap_ = std::get<getType<int>()>(Maps_);
  auto &intVector_ = std::get<getType<int>()>(Vectors_);
  auto &realMap_ = std::get<getType<Real>()>(Maps_);
  auto &realVector_ = std::get<getType<Real>()>(Vectors_);

  for (auto &d : intVector_) {
    d->data.Resize(d->data.GetDim(6), d->data.GetDim(5), d->data.GetDim(4),
                   d->data.GetDim(3), d->data.GetDim(2), nmax_pool);
  }

  for (auto &d : realVector_) {
    d->data.Resize(d->data.GetDim(6), d->data.GetDim(5), d->data.GetDim(4),
                   d->data.GetDim(3), d->data.GetDim(2), nmax_pool);
  }

  nmax_pool_ = nmax_pool;
}

ParArray1D<bool> Swarm::AddEmptyParticles(const int num_to_add,
                                          ParArrayND<int> &new_indices) {
  if (num_to_add <= 0) {
    new_indices = ParArrayND<int>();
    return ParArray1D<bool>();
  }

  while (free_indices_.size() < num_to_add) {
    increasePoolMax();
  }

  ParArray1D<bool> new_mask("Newly created particles", nmax_pool_);
  auto new_mask_h = Kokkos::create_mirror_view(HostMemSpace(), new_mask);
  for (int n = 0; n < nmax_pool_; n++) {
    new_mask_h(n) = false;
  }

  auto mask_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), mask_);

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

  Kokkos::deep_copy(new_mask, new_mask_h);
  Kokkos::deep_copy(mask_, mask_h);
  blockIndex_.DeepCopy(blockIndex_h);

  return new_mask;
}

// No active particles: nmax_active_index = -1
// No particles removed: nmax_active_index unchanged
// Particles removed: nmax_active_index is new max active index
void Swarm::RemoveMarkedParticles() {
  auto mask_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), mask_);
  auto marked_for_removal_h =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), marked_for_removal_);

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

  Kokkos::deep_copy(mask_, mask_h);
  Kokkos::deep_copy(marked_for_removal_, marked_for_removal_h);
}

void Swarm::Defrag() {
  if (GetNumActive() == 0) {
    return;
  }
  // TODO(BRR) Could this algorithm be more efficient? Does it matter?
  // Add 1 to convert max index to max number
  int num_free = (max_active_index_ + 1) - num_active_;
  auto pmb = GetBlockPointer();

  ParArrayND<int> from_to_indices("from_to_indices", max_active_index_ + 1);
  auto from_to_indices_h = from_to_indices.GetHostMirror();

  auto mask_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), mask_);

  for (int n = 0; n <= max_active_index_; n++) {
    from_to_indices_h(n) = unset_index_;
  }

  std::list<int> new_free_indices;

  int index = max_active_index_;
  int num_to_move = std::min<int>(num_free, num_active_);
  for (int n = 0; n < num_to_move; n++) {
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

  // TODO(BRR) Not all these sorts may be necessary
  free_indices_.sort();
  new_free_indices.sort();
  free_indices_.merge(new_free_indices);

  from_to_indices.DeepCopy(from_to_indices_h);

  auto &mask = mask_;
  pmb->par_for(
      "Swarm::DefragMask", 0, max_active_index_, KOKKOS_LAMBDA(const int n) {
        if (from_to_indices(n) >= 0) {
          mask(from_to_indices(n)) = mask(n);
          mask(n) = false;
        }
      });

  auto &intVector_ = std::get<getType<int>()>(Vectors_);
  auto &realVector_ = std::get<getType<Real>()>(Vectors_);
  PackIndexMap real_imap;
  PackIndexMap int_imap;
  auto vreal = PackAllVariables_<Real>(real_imap);
  auto vint = PackAllVariables_<int>(int_imap);
  int real_vars_size = realVector_.size();
  int int_vars_size = intVector_.size();
  auto real_map = real_imap.Map();
  auto int_map = int_imap.Map();
  const int realPackDim = vreal.GetDim(2);
  const int intPackDim = vint.GetDim(2);

  pmb->par_for(
      "Swarm::DefragVariables", 0, max_active_index_, KOKKOS_LAMBDA(const int n) {
        if (from_to_indices(n) >= 0) {
          for (int vidx = 0; vidx < realPackDim; vidx++) {
            vreal(vidx, from_to_indices(n)) = vreal(vidx, n);
          }
          for (int vidx = 0; vidx < intPackDim; vidx++) {
            vint(vidx, from_to_indices(n)) = vint(vidx, n);
          }
        }
      });

  // Update max_active_index_
  max_active_index_ = num_active_ - 1;
}

///
/// Routine to sort particles by cell. Updates internal swarm variables:
///  cellSorted_: 1D Per-cell sorted array of swarm memory indices
///  (SwarmKey::swarm_index_) cellSortedBegin_: Per-cell array of starting indices in
///  cellSorted_ cellSortedNumber_: Per-cell array of number of particles in each cell
///
void Swarm::SortParticlesByCell() {
  auto pmb = GetBlockPointer();

  auto &x = Get<Real>("x").Get();
  auto &y = Get<Real>("y").Get();
  auto &z = Get<Real>("z").Get();

  const int nx1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
  const int nx2 = pmb->cellbounds.ncellsj(IndexDomain::entire);
  const int nx3 = pmb->cellbounds.ncellsk(IndexDomain::entire);
  PARTHENON_REQUIRE(nx1 * nx2 * nx3 < std::numeric_limits<int>::max(),
                    "Too many cells for an int32 to store cell_idx_1d below!");

  auto cellSorted = cellSorted_;
  int ncells = pmb->cellbounds.GetTotal(IndexDomain::entire);
  int num_active = num_active_;
  int max_active_index = max_active_index_;

  // Allocate data if necessary
  if (cellSortedBegin_.GetDim(1) == 0) {
    cellSortedBegin_ = ParArrayND<int>("cellSortedBegin_", nx3, nx2, nx1);
    cellSortedNumber_ = ParArrayND<int>("cellSortedNumber_", nx3, nx2, nx1);
  }
  auto cellSortedBegin = cellSortedBegin_;
  auto cellSortedNumber = cellSortedNumber_;
  auto swarm_d = GetDeviceContext();

  // Write an unsorted list
  pmb->par_for(
      "Write unsorted list", 0, max_active_index_, KOKKOS_LAMBDA(const int n) {
        int i, j, k;
        swarm_d.Xtoijk(x(n), y(n), z(n), i, j, k);
        const int64_t cell_idx_1d = i + nx1 * (j + nx2 * k);
        cellSorted(n) = SwarmKey(static_cast<int>(cell_idx_1d), n);
      });

  sort(cellSorted, SwarmKeyComparator(), 0, max_active_index);

  // Update per-cell arrays for easier accessing later
  const IndexRange &ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  const IndexRange &jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  const IndexRange &kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  pmb->par_for(
      "Update per-cell arrays", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
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

          if (cellSorted(start_index).cell_idx_1d_ == cell_idx_1d) {
            if (start_index == 0) {
              break;
            } else if (cellSorted(start_index - 1).cell_idx_1d_ != cell_idx_1d) {
              break;
            } else {
              start_index--;
              continue;
            }
          }
          if (cellSorted(start_index).cell_idx_1d_ >= cell_idx_1d) {
            start_index--;
            if (start_index < 0) {
              start_index = -1;
              break;
            }
            if (cellSorted(start_index).cell_idx_1d_ < cell_idx_1d) {
              start_index = -1;
              break;
            }
            continue;
          }
          if (cellSorted(start_index).cell_idx_1d_ < cell_idx_1d) {
            start_index++;
            if (start_index > max_active_index) {
              start_index = -1;
              break;
            }
            if (cellSorted(start_index).cell_idx_1d_ > cell_idx_1d) {
              start_index = -1;
              break;
            }
            continue;
          }
        }
        cellSortedBegin(k, j, i) = start_index;
        if (start_index == -1) {
          cellSortedNumber(k, j, i) = 0;
        } else {
          int number = 0;
          int current_index = start_index;
          while (current_index <= max_active_index &&
                 cellSorted(current_index).cell_idx_1d_ == cell_idx_1d) {
            current_index++;
            number++;
            cellSortedNumber(k, j, i) = number;
          }
        }
      });
}

///
/// Routine for precomputing neighbor indices to efficiently compute particle
/// position in terms of neighbor blocks based on spatial position. See
/// GetNeighborBlockIndex()
///
void Swarm::SetNeighborIndices1D_() {
  auto pmb = GetBlockPointer();
  const int ndim = pmb->pmy_mesh->ndim;
  auto neighborIndices_h = neighborIndices_.GetHostMirror();

  // Initialize array in event of zero neighbors
  for (int k = 0; k < 4; k++) {
    for (int j = 0; j < 4; j++) {
      for (int i = 0; i < 4; i++) {
        neighborIndices_h(k, j, i) = no_block_;
      }
    }
  }

  // Indicate which neighbor regions correspond to this meshblock
  const int kmin = 0;
  const int kmax = 4;
  const int jmin = 0;
  const int jmax = 4;
  const int imin = 1;
  const int imax = 3;
  for (int k = kmin; k < kmax; k++) {
    for (int j = jmin; j < jmax; j++) {
      for (int i = imin; i < imax; i++) {
        neighborIndices_h(k, j, i) = this_block_;
      }
    }
  }

  auto mesh_bcs = pmb->pmy_mesh->mesh_bcs;
  // Indicate which neighbor regions correspond to each neighbor meshblock
  for (int n = 0; n < pmb->pbval->nneighbor; n++) {
    NeighborBlock &nb = pmb->pbval->neighbor[n];

    const int i = nb.ni.ox1;

    if (i == -1) {
      neighborIndices_h(0, 0, 0) = n;
    } else if (i == 0) {
      neighborIndices_h(0, 0, 1) = n;
      neighborIndices_h(0, 0, 2) = n;
    } else {
      neighborIndices_h(0, 0, 3) = n;
    }
  }

  neighborIndices_.DeepCopy(neighborIndices_h);
}

void Swarm::SetNeighborIndices2D_() {
  auto pmb = GetBlockPointer();
  const int ndim = pmb->pmy_mesh->ndim;
  auto neighborIndices_h = neighborIndices_.GetHostMirror();

  // Initialize array in event of zero neighbors
  for (int k = 0; k < 4; k++) {
    for (int j = 0; j < 4; j++) {
      for (int i = 0; i < 4; i++) {
        neighborIndices_h(k, j, i) = no_block_;
      }
    }
  }

  // Indicate which neighbor regions correspond to this meshblock
  const int kmin = 0;
  const int kmax = 4;
  const int jmin = 1;
  const int jmax = 3;
  const int imin = 1;
  const int imax = 3;
  for (int k = kmin; k < kmax; k++) {
    for (int j = jmin; j < jmax; j++) {
      for (int i = imin; i < imax; i++) {
        neighborIndices_h(k, j, i) = this_block_;
      }
    }
  }

  // Indicate which neighbor regions correspond to each neighbor meshblock
  for (int n = 0; n < pmb->pbval->nneighbor; n++) {
    NeighborBlock &nb = pmb->pbval->neighbor[n];

    const int i = nb.ni.ox1;
    const int j = nb.ni.ox2;

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
  }

  neighborIndices_.DeepCopy(neighborIndices_h);
}

void Swarm::SetNeighborIndices3D_() {
  auto pmb = GetBlockPointer();
  const int ndim = pmb->pmy_mesh->ndim;
  auto neighborIndices_h = neighborIndices_.GetHostMirror();

  // Initialize array in event of zero neighbors
  for (int k = 0; k < 4; k++) {
    for (int j = 0; j < 4; j++) {
      for (int i = 0; i < 4; i++) {
        neighborIndices_h(k, j, i) = no_block_;
      }
    }
  }

  // Indicate which neighbor regions correspond to this meshblock
  const int kmin = 1;
  const int kmax = 3;
  const int jmin = 1;
  const int jmax = 3;
  const int imin = 1;
  const int imax = 3;
  for (int k = kmin; k < kmax; k++) {
    for (int j = jmin; j < jmax; j++) {
      for (int i = imin; i < imax; i++) {
        neighborIndices_h(k, j, i) = this_block_;
      }
    }
  }

  // Indicate which neighbor regions correspond to each neighbor meshblock
  for (int n = 0; n < pmb->pbval->nneighbor; n++) {
    NeighborBlock &nb = pmb->pbval->neighbor[n];

    const int i = nb.ni.ox1;
    const int j = nb.ni.ox2;
    const int k = nb.ni.ox3;

    if (i == -1) {
      if (j == -1) {
        if (k == -1) {
          neighborIndices_h(0, 0, 0) = n;
        } else if (k == 0) {
          neighborIndices_h(1, 0, 0) = n;
          neighborIndices_h(2, 0, 0) = n;
        } else if (k == 1) {
          neighborIndices_h(3, 0, 0) = n;
        }
      } else if (j == 0) {
        if (k == -1) {
          neighborIndices_h(0, 1, 0) = n;
          neighborIndices_h(0, 2, 0) = n;
        } else if (k == 0) {
          neighborIndices_h(1, 1, 0) = n;
          neighborIndices_h(1, 2, 0) = n;
          neighborIndices_h(2, 1, 0) = n;
          neighborIndices_h(2, 2, 0) = n;
        } else if (k == 1) {
          neighborIndices_h(3, 1, 0) = n;
          neighborIndices_h(3, 2, 0) = n;
        }
      } else if (j == 1) {
        if (k == -1) {
          neighborIndices_h(0, 3, 0) = n;
        } else if (k == 0) {
          neighborIndices_h(1, 3, 0) = n;
          neighborIndices_h(2, 3, 0) = n;
        } else if (k == 1) {
          neighborIndices_h(3, 3, 0) = n;
        }
      }
    } else if (i == 0) {
      if (j == -1) {
        if (k == -1) {
          neighborIndices_h(0, 0, 1) = n;
          neighborIndices_h(0, 0, 2) = n;
        } else if (k == 0) {
          neighborIndices_h(1, 0, 1) = n;
          neighborIndices_h(1, 0, 2) = n;
          neighborIndices_h(2, 0, 1) = n;
          neighborIndices_h(2, 0, 2) = n;
        } else if (k == 1) {
          neighborIndices_h(3, 0, 1) = n;
          neighborIndices_h(3, 0, 2) = n;
        }
      } else if (j == 0) {
        if (k == -1) {
          neighborIndices_h(0, 1, 1) = n;
          neighborIndices_h(0, 1, 2) = n;
          neighborIndices_h(0, 2, 1) = n;
          neighborIndices_h(0, 2, 2) = n;
        } else if (k == 1) {
          neighborIndices_h(3, 1, 1) = n;
          neighborIndices_h(3, 1, 2) = n;
          neighborIndices_h(3, 2, 1) = n;
          neighborIndices_h(3, 2, 2) = n;
        }
      } else if (j == 1) {
        if (k == -1) {
          neighborIndices_h(0, 3, 1) = n;
          neighborIndices_h(0, 3, 2) = n;
        } else if (k == 0) {
          neighborIndices_h(1, 3, 1) = n;
          neighborIndices_h(1, 3, 2) = n;
          neighborIndices_h(2, 3, 1) = n;
          neighborIndices_h(2, 3, 2) = n;
        } else if (k == 1) {
          neighborIndices_h(3, 3, 1) = n;
          neighborIndices_h(3, 3, 2) = n;
        }
      }
    } else if (i == 1) {
      if (j == -1) {
        if (k == -1) {
          neighborIndices_h(0, 0, 3) = n;
        } else if (k == 0) {
          neighborIndices_h(1, 0, 3) = n;
          neighborIndices_h(2, 0, 3) = n;
        } else if (k == 1) {
          neighborIndices_h(3, 0, 3) = n;
        }
      } else if (j == 0) {
        if (k == -1) {
          neighborIndices_h(0, 1, 3) = n;
          neighborIndices_h(0, 2, 3) = n;
        } else if (k == 0) {
          neighborIndices_h(1, 1, 3) = n;
          neighborIndices_h(1, 2, 3) = n;
          neighborIndices_h(2, 1, 3) = n;
          neighborIndices_h(2, 2, 3) = n;
        } else if (k == 1) {
          neighborIndices_h(3, 1, 3) = n;
          neighborIndices_h(3, 2, 3) = n;
        }
      } else if (j == 1) {
        if (k == -1) {
          neighborIndices_h(0, 3, 3) = n;
        } else if (k == 0) {
          neighborIndices_h(1, 3, 3) = n;
          neighborIndices_h(2, 3, 3) = n;
        } else if (k == 1) {
          neighborIndices_h(3, 3, 3) = n;
        }
      }
    }
  }

  neighborIndices_.DeepCopy(neighborIndices_h);
}

void Swarm::SetupPersistentMPI() {
  auto pmb = GetBlockPointer();
  vbswarm->SetupPersistentMPI();

  const int ndim = pmb->pmy_mesh->ndim;

  const int nbmax = vbswarm->bd_var_.nbmax;
  num_particles_to_send_ = ParArrayND<int>("npts", nbmax);

  // Build up convenience array of neighbor indices
  if (ndim == 1) {
    SetNeighborIndices1D_();
  } else if (ndim == 2) {
    SetNeighborIndices2D_();
  } else if (ndim == 3) {
    SetNeighborIndices3D_();
  } else {
    PARTHENON_FAIL("ndim must be 1, 2, or 3 for particles!");
  }

  neighbor_received_particles_.resize(nbmax);

  // Build device array mapping neighbor index to neighbor bufid
  if (pmb->pbval->nneighbor > 0) {
    ParArrayND<int> neighbor_buffer_index("Neighbor buffer index", pmb->pbval->nneighbor);
    auto neighbor_buffer_index_h = neighbor_buffer_index.GetHostMirror();
    for (int n = 0; n < pmb->pbval->nneighbor; n++) {
      neighbor_buffer_index_h(n) = pmb->pbval->neighbor[n].bufid;
    }
    neighbor_buffer_index.DeepCopy(neighbor_buffer_index_h);
    neighbor_buffer_index_ = neighbor_buffer_index;
  }
}

int Swarm::CountParticlesToSend_() {
  auto blockIndex_h = blockIndex_.GetHostMirrorAndCopy();
  auto mask_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), mask_);
  auto swarm_d = GetDeviceContext();
  auto pmb = GetBlockPointer();
  const int nbmax = vbswarm->bd_var_.nbmax;

  // Fence to make sure particles aren't currently being transported locally
  pmb->exec_space.fence();
  auto num_particles_to_send_h = num_particles_to_send_.GetHostMirror();
  for (int n = 0; n < pmb->pbval->nneighbor; n++) {
    num_particles_to_send_h(n) = 0;
  }
  const int particle_size = GetParticleDataSize();
  vbswarm->particle_size = particle_size;

  int max_indices_size = 0;
  int total_noblock_particles = 0;
  for (int n = 0; n <= max_active_index_; n++) {
    if (mask_h(n)) {
      // This particle should be sent
      if (blockIndex_h(n) >= 0) {
        num_particles_to_send_h(blockIndex_h(n))++;
        if (max_indices_size < num_particles_to_send_h(blockIndex_h(n))) {
          max_indices_size = num_particles_to_send_h(blockIndex_h(n));
        }
      }
      if (blockIndex_h(n) == no_block_) {
        total_noblock_particles++;
      }
    }
  }
  // Size-0 arrays not permitted but we don't want to short-circuit subsequent logic that
  // indicates completed communications
  max_indices_size = std::max<int>(1, max_indices_size);

  // Not a ragged-right array, just for convenience
  if (total_noblock_particles > 0) {
    auto noblock_indices =
        ParArrayND<int>("Particles with no block", total_noblock_particles);
    auto noblock_indices_h = noblock_indices.GetHostMirror();
    int counter = 0;
    for (int n = 0; n <= max_active_index_; n++) {
      if (mask_h(n)) {
        if (blockIndex_h(n) == no_block_) {
          noblock_indices_h(counter) = n;
          counter++;
        }
      }
    }
    noblock_indices.DeepCopy(noblock_indices_h);
    ApplyBoundaries_(total_noblock_particles, noblock_indices);
  }

  particle_indices_to_send_ =
      ParArrayND<int>("Particle indices to send", nbmax, max_indices_size);
  auto particle_indices_to_send_h = particle_indices_to_send_.GetHostMirror();
  std::vector<int> counter(nbmax, 0);
  for (int n = 0; n <= max_active_index_; n++) {
    if (mask_h(n)) {
      if (blockIndex_h(n) >= 0) {
        particle_indices_to_send_h(blockIndex_h(n), counter[blockIndex_h(n)]) = n;
        counter[blockIndex_h(n)]++;
      }
    }
  }
  num_particles_to_send_.DeepCopy(num_particles_to_send_h);
  particle_indices_to_send_.DeepCopy(particle_indices_to_send_h);

  num_particles_sent_ = 0;
  for (int n = 0; n < pmb->pbval->nneighbor; n++) {
    // Resize buffer if too small
    const int bufid = pmb->pbval->neighbor[n].bufid;
    auto sendbuf = vbswarm->bd_var_.send[bufid];
    if (sendbuf.extent(0) < num_particles_to_send_h(n) * particle_size) {
      sendbuf = BufArray1D<Real>("Buffer", num_particles_to_send_h(n) * particle_size);
      vbswarm->bd_var_.send[bufid] = sendbuf;
    }
    vbswarm->send_size[bufid] = num_particles_to_send_h(n) * particle_size;
    num_particles_sent_ += num_particles_to_send_h(n);
  }

  return max_indices_size;
}

void Swarm::LoadBuffers_(const int max_indices_size) {
  auto swarm_d = GetDeviceContext();
  auto pmb = GetBlockPointer();
  const int particle_size = GetParticleDataSize();
  const int nneighbor = pmb->pbval->nneighbor;

  auto &intVector_ = std::get<getType<int>()>(Vectors_);
  auto &realVector_ = std::get<getType<Real>()>(Vectors_);
  PackIndexMap real_imap;
  PackIndexMap int_imap;
  auto vreal = PackAllVariables_<Real>(real_imap);
  auto vint = PackAllVariables_<int>(int_imap);
  const int realPackDim = vreal.GetDim(2);
  const int intPackDim = vint.GetDim(2);

  // Pack index:
  // [variable start] [swarm idx]

  auto &bdvar = vbswarm->bd_var_;
  auto num_particles_to_send = num_particles_to_send_;
  auto particle_indices_to_send = particle_indices_to_send_;
  auto neighbor_buffer_index = neighbor_buffer_index_;
  pmb->par_for(
      "Pack Buffers", 0, max_indices_size - 1,
      KOKKOS_LAMBDA(const int n) {            // Max index
        for (int m = 0; m < nneighbor; m++) { // Number of neighbors
          const int bufid = neighbor_buffer_index(m);
          if (n < num_particles_to_send(m)) {
            const int sidx = particle_indices_to_send(m, n);
            int buffer_index = n * particle_size;
            swarm_d.MarkParticleForRemoval(sidx);
            for (int i = 0; i < realPackDim; i++) {
              bdvar.send[bufid](buffer_index) = vreal(i, sidx);
              buffer_index++;
            }
            for (int i = 0; i < intPackDim; i++) {
              bdvar.send[bufid](buffer_index) = static_cast<Real>(vint(i, sidx));
              buffer_index++;
            }
          }
        }
      });

  RemoveMarkedParticles();
}

void Swarm::Send(BoundaryCommSubset phase) {
  auto pmb = GetBlockPointer();
  const int nneighbor = pmb->pbval->nneighbor;
  auto swarm_d = GetDeviceContext();

  if (nneighbor == 0) {
    // Process physical boundary conditions on "sent" particles
    auto blockIndex_h = blockIndex_.GetHostMirrorAndCopy();
    auto mask_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), mask_);

    int total_sent_particles = 0;
    pmb->par_reduce(
        "total sent particles", 0, max_active_index_,
        KOKKOS_LAMBDA(int n, int &total_sent_particles) {
          if (swarm_d.IsActive(n)) {
            if (!swarm_d.IsOnCurrentMeshBlock(n)) {
              total_sent_particles++;
            }
          }
        },
        Kokkos::Sum<int>(total_sent_particles));

    if (total_sent_particles > 0) {
      ParArrayND<int> new_indices("new indices", total_sent_particles);
      auto new_indices_h = new_indices.GetHostMirrorAndCopy();
      int sent_particle_index = 0;
      for (int n = 0; n <= max_active_index_; n++) {
        if (mask_h(n)) {
          if (blockIndex_h(n) >= 0 || blockIndex_h(n) == no_block_) {
            new_indices_h(sent_particle_index) = n;
            sent_particle_index++;
          }
        }
      }
      new_indices.DeepCopy(new_indices_h);

      ApplyBoundaries_(total_sent_particles, new_indices);
    }
  } else {
    // Query particles for those to be sent
    int max_indices_size = CountParticlesToSend_();

    // Prepare buffers for send operations
    LoadBuffers_(max_indices_size);

    // Send buffer data
    vbswarm->Send(phase);
  }
}

void Swarm::CountReceivedParticles_() {
  auto pmb = GetBlockPointer();
  total_received_particles_ = 0;
  for (int n = 0; n < pmb->pbval->nneighbor; n++) {
    const int bufid = pmb->pbval->neighbor[n].bufid;
    if (vbswarm->bd_var_.flag[bufid] == BoundaryStatus::arrived) {
      PARTHENON_DEBUG_REQUIRE(vbswarm->recv_size[bufid] % vbswarm->particle_size == 0,
                              "Receive buffer is not divisible by particle size!");
      neighbor_received_particles_[n] =
          vbswarm->recv_size[bufid] / vbswarm->particle_size;
      total_received_particles_ += neighbor_received_particles_[n];
    } else {
      neighbor_received_particles_[n] = 0;
    }
  }
}

void Swarm::UpdateNeighborBufferReceiveIndices_(ParArrayND<int> &neighbor_index,
                                                ParArrayND<int> &buffer_index) {
  auto pmb = GetBlockPointer();
  auto neighbor_index_h = neighbor_index.GetHostMirror();
  auto buffer_index_h =
      buffer_index.GetHostMirror(); // Index of each particle in its received buffer

  int id = 0;
  for (int n = 0; n < pmb->pbval->nneighbor; n++) {
    for (int m = 0; m < neighbor_received_particles_[n]; m++) {
      neighbor_index_h(id) = n;
      buffer_index_h(id) = m;
      id++;
    }
  }
  neighbor_index.DeepCopy(neighbor_index_h);
  buffer_index.DeepCopy(buffer_index_h);
}

void Swarm::UnloadBuffers_() {
  auto pmb = GetBlockPointer();

  CountReceivedParticles_();

  auto &bdvar = vbswarm->bd_var_;

  if (total_received_particles_ > 0) {
    ParArrayND<int> new_indices;
    auto new_mask = AddEmptyParticles(total_received_particles_, new_indices);

    ParArrayND<int> neighbor_index("Neighbor index", total_received_particles_);
    ParArrayND<int> buffer_index("Buffer index", total_received_particles_);
    UpdateNeighborBufferReceiveIndices_(neighbor_index, buffer_index);
    auto neighbor_buffer_index = neighbor_buffer_index_;

    auto &intVector_ = std::get<getType<int>()>(Vectors_);
    auto &realVector_ = std::get<getType<Real>()>(Vectors_);
    PackIndexMap real_imap;
    PackIndexMap int_imap;
    auto vreal = PackAllVariables_<Real>(real_imap);
    auto vint = PackAllVariables_<int>(int_imap);
    int realPackDim = vreal.GetDim(2);
    int intPackDim = vint.GetDim(2);

    // construct map from buffer index to swarm index (or just return vector of indices!)
    const int particle_size = GetParticleDataSize();
    auto swarm_d = GetDeviceContext();

    pmb->par_for(
        "Unload buffers", 0, total_received_particles_ - 1, KOKKOS_LAMBDA(const int n) {
          const int sid = new_indices(n);
          const int nid = neighbor_index(n);
          int bid = buffer_index(n) * particle_size;
          const int nbid = neighbor_buffer_index(nid);
          for (int i = 0; i < realPackDim; i++) {
            vreal(i, sid) = bdvar.recv[nbid](bid);
            bid++;
          }
          for (int i = 0; i < intPackDim; i++) {
            vint(i, sid) = static_cast<int>(bdvar.recv[nbid](bid));
            bid++;
          }
        });

    ApplyBoundaries_(total_received_particles_, new_indices);
  }
}

void Swarm::ApplyBoundaries_(const int nparticles, ParArrayND<int> indices) {
  auto pmb = GetBlockPointer();
  auto &x = Get<Real>("x").Get();
  auto &y = Get<Real>("y").Get();
  auto &z = Get<Real>("z").Get();
  auto swarm_d = GetDeviceContext();
  auto bcs = this->bounds_d;

  pmb->par_for(
      "Swarm::ApplyBoundaries", 0, nparticles - 1, KOKKOS_LAMBDA(const int n) {
        const int sid = indices(n);
        for (int l = 0; l < 6; l++) {
          bcs.bounds[l]->Apply(sid, x(sid), y(sid), z(sid), swarm_d);
        }
      });

  RemoveMarkedParticles();
}

bool Swarm::Receive(BoundaryCommSubset phase) {
  auto pmb = GetBlockPointer();
  const int nneighbor = pmb->pbval->nneighbor;

  if (nneighbor == 0) {
    // Do nothing; no boundaries to receive
    return true;
  } else {
    // Ensure all local deep copies marked BoundaryStatus::completed are actually received
    pmb->exec_space.fence();

    // Populate buffers
    vbswarm->Receive(phase);

    // Transfer data from buffers to swarm memory pool
    UnloadBuffers_();

    auto &bdvar = vbswarm->bd_var_;
    bool all_boundaries_received = true;
    for (int n = 0; n < nneighbor; n++) {
      NeighborBlock &nb = pmb->pbval->neighbor[n];
      if (bdvar.flag[nb.bufid] == BoundaryStatus::arrived) {
        bdvar.flag[nb.bufid] = BoundaryStatus::completed;
      } else if (bdvar.flag[nb.bufid] == BoundaryStatus::waiting) {
        all_boundaries_received = false;
      }
    }

    return all_boundaries_received;
  }
}

void Swarm::ResetCommunication() {
  auto pmb = GetBlockPointer();
#ifdef MPI_PARALLEL
  for (int n = 0; n < pmb->pbval->nneighbor; n++) {
    NeighborBlock &nb = pmb->pbval->neighbor[n];
    vbswarm->bd_var_.req_send[nb.bufid] = MPI_REQUEST_NULL;
  }
#endif

  // Reset boundary statuses
  for (int n = 0; n < pmb->pbval->nneighbor; n++) {
    auto &nb = pmb->pbval->neighbor[n];
    vbswarm->bd_var_.flag[nb.bufid] = BoundaryStatus::waiting;
  }
}

bool Swarm::FinalizeCommunicationIterative() {
  PARTHENON_THROW("FinalizeCommunicationIterative not yet implemented!");
  return true;
}

void Swarm::AllocateComms(std::weak_ptr<MeshBlock> wpmb) {
  if (wpmb.expired()) return;
  printf("%s:%i:%s\n", __FILE__, __LINE__, __FILE__);

  std::shared_ptr<MeshBlock> pmb = wpmb.lock();
  printf("%s:%i:%s\n", __FILE__, __LINE__, __FILE__);
  printf("pmb: %i\n", pmb.get() == nullptr);

  // Create the boundary object
  vbswarm = std::make_shared<BoundarySwarm>(pmb, label_);
  printf("%s:%i:%s\n", __FILE__, __LINE__, __FILE__);

  // Enroll SwarmVariable object
  vbswarm->bswarm_index = pmb->pbswarm->bswarms.size();
  printf("%s:%i:%s\n", __FILE__, __LINE__, __FILE__);
  pmb->pbswarm->bswarms.push_back(vbswarm);
  printf("%s:%i:%s\n", __FILE__, __LINE__, __FILE__);
}

} // namespace parthenon
