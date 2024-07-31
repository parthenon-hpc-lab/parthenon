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

  // Draft alternative approach due to LFR utilizing mesh offset comparisons at the
  // highest refinement level
  // auto ll_block = pmb->loc.GetDaughter(0, 0, 0);
  // int finest_level = pmb->loc.level() + 1;
  // for (auto &n : pmb->neighbors) {
  //  std::vector<LogicalLocation> dlocs;
  //  auto &nloc =
  //      n.loc; // Would need to use the location in the coordinates of the origin tree
  //  if (nloc.level() == finest_level) {
  //    dlocs.emplace_back(nloc);
  //  } else if (nloc.level() == finest_level) {
  //    dlocs = nloc.GetDaughters(ndim);
  //  } else if (nloc.level() == finest_level - 2) {
  //    auto tlocs = nloc.GetDaughters(ndim);
  //    for (auto &t : tlocs) {
  //      auto gdlocs = t.GetDaughters(ndim);
  //      dlocs.insert(dlocs.end(), gdlocs.begin(), gdlocs.end());
  //    }
  //  } else {
  //    PARTHENON_FAIL("Proper nesting is not being respected.");
  //  }
  //  for (auto &d : dlocs) {
  //    const int k = d.lx3() - ll_block.lx3() + 1;
  //    const int j = d.lx2() - ll_block.lx2() + 1;
  //    const int i = d.lx1() - ll_block.lx1() + 1;
  //    if (i >= 0 && i <= 3 && j >= 0 && j <= 3 && k >= 0 && k <= 3)
  //      neighbor_indices_h(k, j, i) = n.gid;
  //  }
  //}

  neighbor_indices_.DeepCopy(neighbor_indices_h);
}

void Swarm::SetupPersistentMPI() {
  auto pmb = GetBlockPointer();
  vbswarm->SetupPersistentMPI();

  const int ndim = pmb->pmy_mesh->ndim;

  const int nbmax = vbswarm->bd_var_.nbmax;

  // Build up convenience array of neighbor indices
  SetNeighborIndices_();

  neighbor_received_particles_.resize(nbmax);

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

int Swarm::CountParticlesToSend_() {
  auto mask_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), mask_);
  auto swarm_d = GetDeviceContext();
  auto pmb = GetBlockPointer();
  const int nbmax = vbswarm->bd_var_.nbmax;

  // Fence to make sure particles aren't currently being transported locally
  // TODO(BRR) do this operation on device.
  pmb->exec_space.fence();
  auto num_particles_to_send_h = num_particles_to_send_.GetHostMirror();
  for (int n = 0; n < pmb->neighbors.size(); n++) {
    num_particles_to_send_h(n) = 0;
  }
  const int particle_size = GetParticleDataSize();
  vbswarm->particle_size = particle_size;

  // TODO(BRR) This kernel launch should be folded into the subsequent logic once we
  // convert that to kernel-based reductions
  auto &x = Get<Real>(swarm_position::x::name()).Get();
  auto &y = Get<Real>(swarm_position::y::name()).Get();
  auto &z = Get<Real>(swarm_position::z::name()).Get();
  const int max_active_index = GetMaxActiveIndex();
  pmb->par_for(
      PARTHENON_AUTO_LABEL, 0, max_active_index, KOKKOS_LAMBDA(const int n) {
        if (swarm_d.IsActive(n)) {
          bool on_current_mesh_block = true;
          swarm_d.GetNeighborBlockIndex(n, x(n), y(n), z(n), on_current_mesh_block);
        }
      });

  // Facilitate lambda captures
  auto &block_index = block_index_;
  auto &num_particles_to_send = num_particles_to_send_;

  int max_indices_size = 0;
  parthenon::par_reduce(
      PARTHENON_AUTO_LABEL, 0, max_active_index,
      KOKKOS_LAMBDA(const int n, int &red) {
        if (swarm_d.IsActive(n)) {
          bool on_current_mesh_block = true;
          swarm_d.GetNeighborBlockIndex(n, x(n), y(n), z(n), on_current_mesh_block);

          if (block_index(n) >= 0) {
            red = Kokkos::atomic_add_fetch(&num_particles_to_send(block_index(n)), 1);
          }
        }
      },
      Kokkos::Max<int>(max_indices_size));

  auto num_particles_to_send_h = num_particles_to_send_.GetHostMirrorAndCopy();

  // Size-0 arrays not permitted but we don't want to short-circuit subsequent logic
  // that indicates completed communications
  max_indices_size = std::max<int>(1, max_indices_size);

  //// TODO(BRR) don't allocate dynamically
  // particle_indices_to_send_ =
  //    ParArrayND<int>("Particle indices to send", nbmax, max_indices_size);
  // auto particle_indices_to_send_h = particle_indices_to_send_.GetHostMirror();
  // std::vector<int> counter(nbmax, 0);
  // for (int n = 0; n <= max_active_index_; n++) {
  //  if (mask_h(n)) {
  //    if (block_index_h(n) >= 0) {
  //      particle_indices_to_send_h(block_index_h(n), counter[block_index_h(n)]) = n;
  //      counter[block_index_h(n)]++;
  //    }
  //  }
  //}
  // num_particles_to_send_.DeepCopy(num_particles_to_send_h);
  // particle_indices_to_send_.DeepCopy(particle_indices_to_send_h);

  // Count total particles sent and resize buffers if too small
  num_particles_sent_ = 0;
  for (int n = 0; n < pmb->neighbors.size(); n++) {
    // Resize buffer if too small
    const int bufid = pmb->neighbors[n].bufid;
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
  // TODO(BRR) rewrite without particle_indices_to_send
  auto swarm_d = GetDeviceContext();
  auto pmb = GetBlockPointer();
  const int particle_size = GetParticleDataSize();
  const int nneighbor = pmb->neighbors.size();

  auto &int_vector = std::get<getType<int>()>(vectors_);
  auto &real_vector = std::get<getType<Real>()>(vectors_);
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
      PARTHENON_AUTO_LABEL, 0, max_indices_size - 1,
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
  const int nneighbor = pmb->neighbors.size();
  auto swarm_d = GetDeviceContext();

  if (nneighbor == 0) {
    PARTHENON_FAIL("Shouldn't be here... just remove this logic?");
    // TODO(BRR) Do we ever reach this branch?
    // Process physical boundary conditions on "sent" particles
    auto block_index_h = block_index_.GetHostMirrorAndCopy();
    auto mask_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), mask_);

    int total_sent_particles = 0;
    pmb->par_reduce(
        PARTHENON_AUTO_LABEL, 0, max_active_index_,
        KOKKOS_LAMBDA(int n, int &total_sent_particles) {
          if (swarm_d.IsActive(n)) {
            if (!swarm_d.IsOnCurrentMeshBlock(n)) {
              total_sent_particles++;
            }
          }
        },
        Kokkos::Sum<int>(total_sent_particles));

    if (total_sent_particles > 0) {
      ParArray1D<int> new_indices("new indices", total_sent_particles);
      auto new_indices_h = new_indices.GetHostMirrorAndCopy();
      int sent_particle_index = 0;
      for (int n = 0; n <= max_active_index_; n++) {
        if (mask_h(n)) {
          if (block_index_h(n) >= 0 || block_index_h(n) == no_block_) {
            new_indices_h(sent_particle_index) = n;
            sent_particle_index++;
          }
        }
      }
      new_indices.DeepCopy(new_indices_h);
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
  for (int n = 0; n < pmb->neighbors.size(); n++) {
    const int bufid = pmb->neighbors[n].bufid;
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

void Swarm::UpdateNeighborBufferReceiveIndices_(ParArray1D<int> &neighbor_index,
                                                ParArray1D<int> &buffer_index) {
  auto pmb = GetBlockPointer();
  auto neighbor_index_h = neighbor_index.GetHostMirror();
  auto buffer_index_h =
      buffer_index.GetHostMirror(); // Index of each particle in its received buffer

  int id = 0;
  for (int n = 0; n < pmb->neighbors.size(); n++) {
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
    auto newParticlesContext = AddEmptyParticles(total_received_particles_);

    auto &recv_neighbor_index = recv_neighbor_index_;
    auto &recv_buffer_index = recv_buffer_index_;
    UpdateNeighborBufferReceiveIndices_(recv_neighbor_index, recv_buffer_index);
    auto neighbor_buffer_index = neighbor_buffer_index_;

    auto &int_vector = std::get<getType<int>()>(vectors_);
    auto &real_vector = std::get<getType<Real>()>(vectors_);
    PackIndexMap real_imap;
    PackIndexMap int_imap;
    auto vreal = PackAllVariables_<Real>(real_imap);
    auto vint = PackAllVariables_<int>(int_imap);
    int realPackDim = vreal.GetDim(2);
    int intPackDim = vint.GetDim(2);

    // construct map from buffer index to swarm index (or just return vector of
    // indices!)
    const int particle_size = GetParticleDataSize();
    auto swarm_d = GetDeviceContext();

    pmb->par_for(
        PARTHENON_AUTO_LABEL, 0, newParticlesContext.GetNewParticlesMaxIndex(),
        // n is both new particle index and index over buffer values
        KOKKOS_LAMBDA(const int n) {
          const int sid = newParticlesContext.GetNewParticleIndex(n);
          const int nid = recv_neighbor_index(n);
          int bid = recv_buffer_index(n) * particle_size;
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
}

} // namespace parthenon
