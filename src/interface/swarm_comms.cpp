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
#include <unordered_map>
#include <utility>
#include <vector>

#include "mesh/mesh.hpp"
#include "pack/swarm_default_names.hpp"
#include "swarm.hpp"
#include "utils/error_checking.hpp"
#include "utils/sort.hpp"

namespace parthenon {

///
/// Routine for precomputing neighbor indices to efficiently compute particle position
/// in terms of neighbor blocks based on spatial position for communication. See
/// GetNeighborBlockIndex()
///
void Swarm::SetNeighborIndices_() {
  auto pmb = GetBlockPointer();
  const int ndim = pmb->pmy_mesh->ndim;
  auto neighbor_indices_h = neighbor_indices_.GetHostMirror();

  // Initialize array in event of zero neighbors
  for (int k = 0; k <= 3; k++) {
    for (int j = 0; j <= 3; j++) {
      for (int i = 0; i <= 3; i++) {
        neighbor_indices_h(k, j, i) = no_block_;
      }
    }
  }

  // Create a point in the center of each ghost halo region at maximum refinement level
  // and then test whether each neighbor block includes that point.
  const auto &bsize = pmb->block_size;
  const std::array<Real, 3> dx_test = {(bsize.xmax_[0] - bsize.xmin_[0]) / 2.,
                                       (bsize.xmax_[1] - bsize.xmin_[1]) / 2.,
                                       (bsize.xmax_[2] - bsize.xmin_[2]) / 2.};
  for (int k = 0; k < 4; k++) {
    for (int j = 0; j < 4; j++) {
      for (int i = 0; i < 4; i++) {
        // Midpoint of meshblock at highest possible refinement level in this i,j,k
        // ghost halo.
        std::array<Real, 3> x_test = {bsize.xmin_[0] + (i - 0.5) * dx_test[0],
                                      ndim < 2 ? (bsize.xmin_[1] + bsize.xmax_[1]) / 2.
                                               : bsize.xmin_[1] + (j - 0.5) * dx_test[1],
                                      ndim < 3 ? (bsize.xmin_[2] + bsize.xmax_[2]) / 2.
                                               : bsize.xmin_[2] + (k - 0.5) * dx_test[2]};

        // Account for periodic boundary conditions by applying BCs to x_test
        // Assumes mesh is hyper-rectangular and Mesh::mesh_size represents the entire
        // domain.
        auto &msize = pmb->pmy_mesh->mesh_size;
        if (pmb->boundary_flag[0] == BoundaryFlag::periodic) {
          if (x_test[0] < msize.xmin(X1DIR)) {
            x_test[0] = msize.xmax(X1DIR) - (msize.xmin(X1DIR) - x_test[0]);
          }
        }
        if (pmb->boundary_flag[1] == BoundaryFlag::periodic) {
          if (x_test[0] > msize.xmax(X1DIR)) {
            x_test[0] = msize.xmin(X1DIR) + (x_test[0] - msize.xmax(X1DIR));
          }
        }
        if (ndim > 1) {
          if (pmb->boundary_flag[2] == BoundaryFlag::periodic) {
            if (x_test[1] < msize.xmin(X2DIR)) {
              x_test[1] = msize.xmax(X2DIR) - (msize.xmin(X2DIR) - x_test[1]);
            }
          }
          if (pmb->boundary_flag[3] == BoundaryFlag::periodic) {
            if (x_test[1] > msize.xmax(X2DIR)) {
              x_test[1] = msize.xmin(X2DIR) + (x_test[1] - msize.xmax(X2DIR));
            }
          }
        }
        if (ndim > 2) {
          if (pmb->boundary_flag[4] == BoundaryFlag::periodic) {
            if (x_test[2] < msize.xmin(X3DIR)) {
              x_test[2] = msize.xmax(X3DIR) - (msize.xmin(X3DIR) - x_test[2]);
            }
          }
          if (pmb->boundary_flag[5] == BoundaryFlag::periodic) {
            if (x_test[2] > msize.xmax(X3DIR)) {
              x_test[2] = msize.xmin(X3DIR) + (x_test[2] - msize.xmax(X3DIR));
            }
          }
        }

        // Loop over neighbor blocks and see if any contains this test point.
        for (int n = 0; n < pmb->neighbors.size(); n++) {
          NeighborBlock &nb = pmb->neighbors[n];
          const auto &nbsize = nb.block_size;
          if ((x_test[0] > nbsize.xmin_[0] && x_test[0] < nbsize.xmax_[0]) &&
              (x_test[1] > nbsize.xmin_[1] && x_test[1] < nbsize.xmax_[1]) &&
              (x_test[2] > nbsize.xmin_[2] && x_test[2] < nbsize.xmax_[2])) {
            neighbor_indices_h(k, j, i) = n;
            break;
          }
        }
      }
    }
  }

  // Indicate which neighbor regions correspond to this meshblock
  const int kmin = ndim < 3 ? 0 : 1;
  const int kmax = ndim < 3 ? 3 : 2;
  const int jmin = ndim < 2 ? 0 : 1;
  const int jmax = ndim < 2 ? 3 : 2;
  const int imin = 1;
  const int imax = 2;
  for (int k = kmin; k <= kmax; k++) {
    for (int j = jmin; j <= jmax; j++) {
      for (int i = imin; i <= imax; i++) {
        neighbor_indices_h(k, j, i) = this_block_;
      }
    }
  }

  neighbor_indices_.DeepCopy(neighbor_indices_h);
}

void Swarm::SetupPersistentMPI() {
  auto pmb = GetBlockPointer();
  vbswarm->SetupPersistentMPI();

  const int ndim = pmb->pmy_mesh->ndim;

  const int nbmax = vbswarm->bd_var_.nbmax;

  // Build up convenience array of neighbor indices
  SetNeighborIndices_();

  // Build device array mapping neighbor index to neighbor bufid
  if (pmb->neighbors.size() > 0) {
    ParArrayND<int> neighbor_buffer_index("Neighbor buffer index", pmb->neighbors.size());
    auto neighbor_buffer_index_h = neighbor_buffer_index.GetHostMirror();
    for (int n = 0; n < pmb->neighbors.size(); n++) {
      neighbor_buffer_index_h(n) = pmb->neighbors[n].bufid;
    }
    neighbor_buffer_index.DeepCopy(neighbor_buffer_index_h);
    neighbor_buffer_index_ = neighbor_buffer_index;
  }
}

void Swarm::LoadBuffers_() {
  auto swarm_d = GetDeviceContext();
  auto pmb = GetBlockPointer();
  const int particle_size = GetParticleDataSize();
  vbswarm->particle_size = particle_size;
  const int nneighbor = pmb->neighbors.size();
  // Fence to make sure particles aren't currently being transported locally
  pmb->exec_space.fence();

  auto &int_vector = std::get<getType<int>()>(vectors_);
  auto &uint64_vector = std::get<getType<std::uint64_t>()>(vectors_);
  auto &real_vector = std::get<getType<Real>()>(vectors_);
  PackIndexMap real_imap;
  PackIndexMap int_imap;
  PackIndexMap uint64_imap;
  auto vreal = PackAllVariables_<Real>(real_imap);
  auto vint = PackAllVariables_<int>(int_imap);
  auto vuint64 = PackAllVariables_<std::uint64_t>(uint64_imap);
  const int realPackDim = vreal.GetDim(2);
  const int intPackDim = vint.GetDim(2);
  const int uint64PackDim = vuint64.GetDim(2);

  auto &x = Get<Real>(swarm_position::x::name()).Get();
  auto &y = Get<Real>(swarm_position::y::name()).Get();
  auto &z = Get<Real>(swarm_position::z::name()).Get();

  if (max_active_index_ >= 0) {
    auto &buffer_sorted = buffer_sorted_;
    auto &buffer_start = buffer_start_;

    pmb->par_for(
        PARTHENON_AUTO_LABEL, 0, max_active_index_, KOKKOS_LAMBDA(const int n) {
          if (swarm_d.IsActive(n)) {
            bool on_current_mesh_block = true;
            const int m =
                swarm_d.GetNeighborBlockIndex(n, x(n), y(n), z(n), on_current_mesh_block);
            buffer_sorted(n) = SwarmKey(m, n);
          } else {
            buffer_sorted(n) = SwarmKey(this_block_, n);
          }
        });

    // sort by buffer index
    sort(buffer_sorted, SwarmKeyComparator(), 0, max_active_index_);

    // use discontinuity check to determine start of a buffer in buffer_sorted array
    auto &num_particles_to_send = num_particles_to_send_;
    auto max_active_index = max_active_index_;

    // Zero out number of particles to send before accumulating
    pmb->par_for(
        PARTHENON_AUTO_LABEL, 0, NMAX_NEIGHBORS - 1, KOKKOS_LAMBDA(const int n) {
          num_particles_to_send[n] = 0;
          buffer_start[n] = 0;
        });

    pmb->par_for(
        PARTHENON_AUTO_LABEL, 0, max_active_index_, KOKKOS_LAMBDA(const int n) {
          auto m = buffer_sorted(n).sort_idx_;
          // start checks (used for index of particle in buffer)
          if (m >= 0 && n == 0) {
            buffer_start(m) = 0;
          } else if (m >= 0 && m != buffer_sorted(n - 1).sort_idx_) {
            buffer_start(m) = n;
          }
          // end checks (used to to size particle buffers)
          if (m >= 0 && n == max_active_index) {
            num_particles_to_send(m) = n + 1;
          } else if (m >= 0 && m != buffer_sorted(n + 1).sort_idx_) {
            num_particles_to_send(m) = n + 1;
          }
        });

    // copy values back to host for buffer sizing
    auto num_particles_to_send_h = num_particles_to_send_.GetHostMirrorAndCopy();
    auto buffer_start_h = buffer_start.GetHostMirrorAndCopy();

    // Resize send buffers if too small
    for (int n = 0; n < pmb->neighbors.size(); n++) {
      num_particles_to_send_h(n) -= buffer_start_h(n);
      const int bufid = pmb->neighbors[n].bufid;
      auto sendbuf = vbswarm->bd_var_.send[bufid];
      if (sendbuf.extent(0) < num_particles_to_send_h(n) * particle_size) {
        sendbuf = BufArray1D<Real>("Buffer", num_particles_to_send_h(n) * particle_size);
        vbswarm->bd_var_.send[bufid] = sendbuf;
      }
      vbswarm->send_size[bufid] = num_particles_to_send_h(n) * particle_size;
    }

    auto &bdvar = vbswarm->bd_var_;
    auto neighbor_buffer_index = neighbor_buffer_index_;
    // Loop over active particles buffer_sorted, use index n and buffer_start to
    /// get index in buffer m, pack that particle in buffer
    pmb->par_for(
        PARTHENON_AUTO_LABEL, 0, max_active_index_, KOKKOS_LAMBDA(const int n) {
          auto p_index = buffer_sorted(n).swarm_idx_;
          if (swarm_d.IsActive(p_index)) {
            const int m = buffer_sorted(n).sort_idx_;
            if (m >= 0) {
              const int bufid = neighbor_buffer_index(m);
              const int bid = n - buffer_start[m];
              int buffer_index = bid * particle_size;
              swarm_d.MarkParticleForRemoval(p_index);
              for (int i = 0; i < realPackDim; i++) {
                bdvar.send[bufid](buffer_index) = vreal(i, p_index);
                buffer_index++;
              }
              // Making sure we catch/update this, when allowing Real = float again
              static_assert(sizeof(Real) == 2 * sizeof(int));
              for (int i = 0; i < intPackDim; i++) {
                bdvar.send[bufid](buffer_index) = static_cast<Real>(vint(i, p_index));
                buffer_index++;
              }
              // Should eventually be a bit_cast once we're on C++20
              static_assert(sizeof(Real) == sizeof(std::uint64_t));
              for (int i = 0; i < uint64PackDim; i++) {
                std::memcpy(&bdvar.send[bufid](buffer_index), &vuint64(i, p_index),
                            sizeof(std::uint64_t));
                buffer_index++;
              }
            }
          }
        });

    // Remove particles that were loaded to send to another block from this block
    RemoveMarkedParticles();
  } else {
    for (int n = 0; n < pmb->neighbors.size(); n++) {
      const int bufid = pmb->neighbors[n].bufid;
      vbswarm->send_size[bufid] = 0;
    }
  }
}

void Swarm::Send(BoundaryCommSubset phase) {
  auto pmb = GetBlockPointer();
  const int nneighbor = pmb->neighbors.size();
  auto swarm_d = GetDeviceContext();

  // Potentially resize buffer, get consistent index from particle array, get ready to
  // send
  LoadBuffers_();

  // Send buffer data
  vbswarm->Send(phase);
}

void Swarm::UnloadBuffers_() {
  auto pmb = GetBlockPointer();

  // Count received particles
  total_received_particles_ = 0;
  auto &neighbor_received_particles = neighbor_received_particles_;
  auto neighbor_received_particles_h = neighbor_received_particles.GetHostMirror();
  for (int n = 0; n < pmb->neighbors.size(); n++) {
    const int bufid = pmb->neighbors[n].bufid;
    if (vbswarm->bd_var_.flag[bufid] == BoundaryStatus::arrived) {
      PARTHENON_DEBUG_REQUIRE(vbswarm->recv_size[bufid] % vbswarm->particle_size == 0,
                              "Receive buffer is not divisible by particle size!");
      neighbor_received_particles_h(n) =
          vbswarm->recv_size[bufid] / vbswarm->particle_size;
      total_received_particles_ += neighbor_received_particles_h(n);
    } else {
      neighbor_received_particles_h(n) = 0;
    }
  }

  auto &bdvar = vbswarm->bd_var_;
  const int nbmax = vbswarm->bd_var_.nbmax;

  if (total_received_particles_ > 0) {
    auto newParticlesContext = AddEmptyParticles(total_received_particles_);

    auto neighbor_buffer_index = neighbor_buffer_index_;

    auto &int_vector = std::get<getType<int>()>(vectors_);
    auto &uint64_vector = std::get<getType<std::uint64_t>()>(vectors_);
    auto &real_vector = std::get<getType<Real>()>(vectors_);
    PackIndexMap real_imap;
    PackIndexMap int_imap;
    PackIndexMap uint64_imap;
    auto vreal = PackAllVariables_<Real>(real_imap);
    auto vint = PackAllVariables_<int>(int_imap);
    auto vuint64 = PackAllVariables_<std::uint64_t>(uint64_imap);
    const int realPackDim = vreal.GetDim(2);
    const int intPackDim = vint.GetDim(2);
    const int uint64PackDim = vuint64.GetDim(2);

    const int particle_size = GetParticleDataSize();
    auto swarm_d = GetDeviceContext();

    // Change meaning of neighbor_received_particles from particles per neighbor to
    // cumulative particles per neighbor
    int val_prev = 0;
    for (int n = 0; n < nbmax; n++) {
      int val_curr = neighbor_received_particles_h(n);
      neighbor_received_particles_h(n) += val_prev;
      val_prev += val_curr;
    }
    neighbor_received_particles.DeepCopy(neighbor_received_particles_h);

    pmb->par_for(
        PARTHENON_AUTO_LABEL, 0, newParticlesContext.GetNewParticlesMaxIndex(),
        // n is both new particle index and index over buffer values
        KOKKOS_LAMBDA(const int n) {
          const int sid = newParticlesContext.GetNewParticleIndex(n);
          // Search for neighbor id over cumulative indices
          int nid = 0;
          while (n >= neighbor_received_particles(nid) && nid < nbmax - 1) {
            nid++;
          }

          // Convert neighbor id to buffer id
          int bid = nid == 0 ? n * particle_size
                             : (n - neighbor_received_particles(nid - 1)) * particle_size;
          const int nbid = neighbor_buffer_index(nid);
          for (int i = 0; i < realPackDim; i++) {
            vreal(i, sid) = bdvar.recv[nbid](bid);
            bid++;
          }
          // Making sure we catch/update this, when allowing Real = float again
          static_assert(sizeof(Real) == 2 * sizeof(int));
          for (int i = 0; i < intPackDim; i++) {
            vint(i, sid) = static_cast<int>(bdvar.recv[nbid](bid));
            bid++;
          }
          // Should eventually be a bit_cast once we're on C++20
          static_assert(sizeof(Real) == sizeof(std::uint64_t));
          for (int i = 0; i < uint64PackDim; i++) {
            std::memcpy(&vuint64(i, sid), &bdvar.recv[nbid](bid), sizeof(std::uint64_t));
            bid++;
          }
        });
  }
}

bool Swarm::Receive(BoundaryCommSubset phase) {
  auto pmb = GetBlockPointer();
  const int nneighbor = pmb->neighbors.size();

  if (nneighbor == 0) {
    // Do nothing; no boundaries to receive
    return true;
  }

  // Ensure all local deep copies marked BoundaryStatus::completed are actually
  // received
  pmb->exec_space.fence();

  // Populate buffers
  vbswarm->Receive(phase);

  // Transfer data from buffers to swarm memory pool
  UnloadBuffers_();

  auto &bdvar = vbswarm->bd_var_;
  bool all_boundaries_received = true;
  for (int n = 0; n < nneighbor; n++) {
    NeighborBlock &nb = pmb->neighbors[n];
    if (bdvar.flag[nb.bufid] == BoundaryStatus::arrived) {
      bdvar.flag[nb.bufid] = BoundaryStatus::completed;
    } else if (bdvar.flag[nb.bufid] == BoundaryStatus::waiting) {
      all_boundaries_received = false;
    }
  }

  return all_boundaries_received;
}

void Swarm::ResetCommunication() {
  auto pmb = GetBlockPointer();
#ifdef MPI_PARALLEL
  for (int n = 0; n < pmb->neighbors.size(); n++) {
    NeighborBlock &nb = pmb->neighbors[n];
    if (vbswarm->bd_var_.req_send[nb.bufid] != MPI_REQUEST_NULL) {
      MPI_Request_free(&(vbswarm->bd_var_.req_send[nb.bufid]));
    }
  }
#endif

  // Reset boundary statuses
  for (int n = 0; n < pmb->neighbors.size(); n++) {
    auto &nb = pmb->neighbors[n];
    vbswarm->bd_var_.flag[nb.bufid] = BoundaryStatus::waiting;
  }
}

bool Swarm::FinalizeCommunicationIterative() {
  PARTHENON_THROW("FinalizeCommunicationIterative not yet implemented!");
  return true;
}

void Swarm::AllocateComms(std::weak_ptr<MeshBlock> wpmb) {
  if (wpmb.expired()) return;

  std::shared_ptr<MeshBlock> pmb = wpmb.lock();

  // Create the boundary object
  vbswarm = std::make_shared<BoundarySwarm>(pmb, label_);

  // Enroll SwarmVariable object
  vbswarm->bswarm_index = pmb->pbswarm->bswarms.size();
  pmb->pbswarm->bswarms.push_back(vbswarm);

  // Evaluate neigbor indices
  SetNeighborIndices_();
}

} // namespace parthenon
