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
#include "utils/error_checking.hpp"
#include "utils/sort.hpp"

namespace parthenon {

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
      msg << (iFace % 2 == 0 ? "i" : "o") << "x" << iFace / 2 + 1
          << " user boundary requested but provided function is null!";
      PARTHENON_THROW(msg);
    }
  } else {
    msg << (iFace % 2 == 0 ? "i" : "o") << "x" << iFace / 2 + 1 << " boundary flag "
        << static_cast<int>(bcs[iFace]) << " not supported!";
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

///
/// Routine for precomputing neighbor indices to efficiently compute particle
/// position in terms of neighbor blocks based on spatial position. See
/// GetNeighborBlockIndex()
///
void Swarm::SetNeighborIndices1D_() {
  auto pmb = GetBlockPointer();
  const int ndim = pmb->pmy_mesh->ndim;
  auto neighbor_indices_h = neighbor_indices_.GetHostMirror();

  // Initialize array in event of zero neighbors
  for (int k = 0; k < 4; k++) {
    for (int j = 0; j < 4; j++) {
      for (int i = 0; i < 4; i++) {
        neighbor_indices_h(k, j, i) = no_block_;
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
        neighbor_indices_h(k, j, i) = this_block_;
      }
    }
  }

  auto mesh_bcs = pmb->pmy_mesh->mesh_bcs;
  // Indicate which neighbor regions correspond to each neighbor meshblock
  for (int n = 0; n < pmb->neighbors.size(); n++) {
    NeighborBlock &nb = pmb->neighbors[n];
    const int i = nb.ni.ox1;

    if (i == -1) {
      neighbor_indices_h(0, 0, 0) = n;
    } else if (i == 0) {
      neighbor_indices_h(0, 0, 1) = n;
      neighbor_indices_h(0, 0, 2) = n;
    } else {
      neighbor_indices_h(0, 0, 3) = n;
    }
  }

  neighbor_indices_.DeepCopy(neighbor_indices_h);
}

void Swarm::SetNeighborIndices2D_() {
  auto pmb = GetBlockPointer();
  const int ndim = pmb->pmy_mesh->ndim;
  auto neighbor_indices_h = neighbor_indices_.GetHostMirror();

  // Initialize array in event of zero neighbors
  for (int k = 0; k < 4; k++) {
    for (int j = 0; j < 4; j++) {
      for (int i = 0; i < 4; i++) {
        neighbor_indices_h(k, j, i) = no_block_;
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
        neighbor_indices_h(k, j, i) = this_block_;
      }
    }
  }

  // Indicate which neighbor regions correspond to each neighbor meshblock
  for (int n = 0; n < pmb->neighbors.size(); n++) {
    NeighborBlock &nb = pmb->neighbors[n];
    const int i = nb.ni.ox1;
    const int j = nb.ni.ox2;

    if (i == -1) {
      if (j == -1) {
        neighbor_indices_h(0, 0, 0) = n;
      } else if (j == 0) {
        neighbor_indices_h(0, 1, 0) = n;
        neighbor_indices_h(0, 2, 0) = n;
      } else if (j == 1) {
        neighbor_indices_h(0, 3, 0) = n;
      }
    } else if (i == 0) {
      if (j == -1) {
        neighbor_indices_h(0, 0, 1) = n;
        neighbor_indices_h(0, 0, 2) = n;
      } else if (j == 1) {
        neighbor_indices_h(0, 3, 1) = n;
        neighbor_indices_h(0, 3, 2) = n;
      }
    } else if (i == 1) {
      if (j == -1) {
        neighbor_indices_h(0, 0, 3) = n;
      } else if (j == 0) {
        neighbor_indices_h(0, 1, 3) = n;
        neighbor_indices_h(0, 2, 3) = n;
      } else if (j == 1) {
        neighbor_indices_h(0, 3, 3) = n;
      }
    }
  }

  neighbor_indices_.DeepCopy(neighbor_indices_h);
}

void Swarm::SetNeighborIndices3D_() {
  auto pmb = GetBlockPointer();
  const int ndim = pmb->pmy_mesh->ndim;
  auto neighbor_indices_h = neighbor_indices_.GetHostMirror();

  // Initialize array in event of zero neighbors
  for (int k = 0; k < 4; k++) {
    for (int j = 0; j < 4; j++) {
      for (int i = 0; i < 4; i++) {
        neighbor_indices_h(k, j, i) = no_block_;
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
        neighbor_indices_h(k, j, i) = this_block_;
      }
    }
  }

  // Indicate which neighbor regions correspond to each neighbor meshblock
  for (int n = 0; n < pmb->neighbors.size(); n++) {
    NeighborBlock &nb = pmb->neighbors[n];
    const int i = nb.ni.ox1;
    const int j = nb.ni.ox2;
    const int k = nb.ni.ox3;

    if (i == -1) {
      if (j == -1) {
        if (k == -1) {
          neighbor_indices_h(0, 0, 0) = n;
        } else if (k == 0) {
          neighbor_indices_h(1, 0, 0) = n;
          neighbor_indices_h(2, 0, 0) = n;
        } else if (k == 1) {
          neighbor_indices_h(3, 0, 0) = n;
        }
      } else if (j == 0) {
        if (k == -1) {
          neighbor_indices_h(0, 1, 0) = n;
          neighbor_indices_h(0, 2, 0) = n;
        } else if (k == 0) {
          neighbor_indices_h(1, 1, 0) = n;
          neighbor_indices_h(1, 2, 0) = n;
          neighbor_indices_h(2, 1, 0) = n;
          neighbor_indices_h(2, 2, 0) = n;
        } else if (k == 1) {
          neighbor_indices_h(3, 1, 0) = n;
          neighbor_indices_h(3, 2, 0) = n;
        }
      } else if (j == 1) {
        if (k == -1) {
          neighbor_indices_h(0, 3, 0) = n;
        } else if (k == 0) {
          neighbor_indices_h(1, 3, 0) = n;
          neighbor_indices_h(2, 3, 0) = n;
        } else if (k == 1) {
          neighbor_indices_h(3, 3, 0) = n;
        }
      }
    } else if (i == 0) {
      if (j == -1) {
        if (k == -1) {
          neighbor_indices_h(0, 0, 1) = n;
          neighbor_indices_h(0, 0, 2) = n;
        } else if (k == 0) {
          neighbor_indices_h(1, 0, 1) = n;
          neighbor_indices_h(1, 0, 2) = n;
          neighbor_indices_h(2, 0, 1) = n;
          neighbor_indices_h(2, 0, 2) = n;
        } else if (k == 1) {
          neighbor_indices_h(3, 0, 1) = n;
          neighbor_indices_h(3, 0, 2) = n;
        }
      } else if (j == 0) {
        if (k == -1) {
          neighbor_indices_h(0, 1, 1) = n;
          neighbor_indices_h(0, 1, 2) = n;
          neighbor_indices_h(0, 2, 1) = n;
          neighbor_indices_h(0, 2, 2) = n;
        } else if (k == 1) {
          neighbor_indices_h(3, 1, 1) = n;
          neighbor_indices_h(3, 1, 2) = n;
          neighbor_indices_h(3, 2, 1) = n;
          neighbor_indices_h(3, 2, 2) = n;
        }
      } else if (j == 1) {
        if (k == -1) {
          neighbor_indices_h(0, 3, 1) = n;
          neighbor_indices_h(0, 3, 2) = n;
        } else if (k == 0) {
          neighbor_indices_h(1, 3, 1) = n;
          neighbor_indices_h(1, 3, 2) = n;
          neighbor_indices_h(2, 3, 1) = n;
          neighbor_indices_h(2, 3, 2) = n;
        } else if (k == 1) {
          neighbor_indices_h(3, 3, 1) = n;
          neighbor_indices_h(3, 3, 2) = n;
        }
      }
    } else if (i == 1) {
      if (j == -1) {
        if (k == -1) {
          neighbor_indices_h(0, 0, 3) = n;
        } else if (k == 0) {
          neighbor_indices_h(1, 0, 3) = n;
          neighbor_indices_h(2, 0, 3) = n;
        } else if (k == 1) {
          neighbor_indices_h(3, 0, 3) = n;
        }
      } else if (j == 0) {
        if (k == -1) {
          neighbor_indices_h(0, 1, 3) = n;
          neighbor_indices_h(0, 2, 3) = n;
        } else if (k == 0) {
          neighbor_indices_h(1, 1, 3) = n;
          neighbor_indices_h(1, 2, 3) = n;
          neighbor_indices_h(2, 1, 3) = n;
          neighbor_indices_h(2, 2, 3) = n;
        } else if (k == 1) {
          neighbor_indices_h(3, 1, 3) = n;
          neighbor_indices_h(3, 2, 3) = n;
        }
      } else if (j == 1) {
        if (k == -1) {
          neighbor_indices_h(0, 3, 3) = n;
        } else if (k == 0) {
          neighbor_indices_h(1, 3, 3) = n;
          neighbor_indices_h(2, 3, 3) = n;
        } else if (k == 1) {
          neighbor_indices_h(3, 3, 3) = n;
        }
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
  auto block_index_h = block_index_.GetHostMirrorAndCopy();
  auto mask_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), mask_);
  auto swarm_d = GetDeviceContext();
  auto pmb = GetBlockPointer();
  const int nbmax = vbswarm->bd_var_.nbmax;

  // Fence to make sure particles aren't currently being transported locally
  // TODO(BRR) do this operation on device.
  pmb->exec_space.fence();
  auto num_particles_to_send_h = num_particles_to_send_.GetHostMirror();
  PARTHENON_REQUIRE(pmb->pbval->nneighbor == pmb->neighbors.size(), "neighbor sizes don't agree.");
  for (int n = 0; n < pmb->neighbors.size(); n++) {
    printf("%i %i (%i %i)\n",
      pmb->pbval->neighbor[n].bufid, pmb->neighbors[n].bufid,
      pmb->pbval->neighbor[n].targetid, pmb->neighbors[n].targetid);
    num_particles_to_send_h(n) = 0;
  }
  const int particle_size = GetParticleDataSize();
  vbswarm->particle_size = particle_size;

  int max_indices_size = 0;
  int total_noblock_particles = 0;
  for (int n = 0; n <= max_active_index_; n++) {
    if (mask_h(n)) {
      // This particle should be sent
      if (block_index_h(n) >= 0) {
        num_particles_to_send_h(block_index_h(n))++;
        if (max_indices_size < num_particles_to_send_h(block_index_h(n))) {
          max_indices_size = num_particles_to_send_h(block_index_h(n));
        }
      }
      if (block_index_h(n) == no_block_) {
        total_noblock_particles++;
      }
    }
  }
  // Size-0 arrays not permitted but we don't want to short-circuit subsequent logic
  // that indicates completed communications
  max_indices_size = std::max<int>(1, max_indices_size);

  // Not a ragged-right array, just for convenience
  if (total_noblock_particles > 0) {
    auto noblock_indices =
        ParArray1D<int>("Particles with no block", total_noblock_particles);
    auto noblock_indices_h = noblock_indices.GetHostMirror();
    int counter = 0;
    for (int n = 0; n <= max_active_index_; n++) {
      if (mask_h(n)) {
        if (block_index_h(n) == no_block_) {
          noblock_indices_h(counter) = n;
          counter++;
        }
      }
    }
    noblock_indices.DeepCopy(noblock_indices_h);
    ApplyBoundaries_(total_noblock_particles, noblock_indices);
  }

  // TODO(BRR) don't allocate dynamically
  particle_indices_to_send_ =
      ParArrayND<int>("Particle indices to send", nbmax, max_indices_size);
  auto particle_indices_to_send_h = particle_indices_to_send_.GetHostMirror();
  std::vector<int> counter(nbmax, 0);
  for (int n = 0; n <= max_active_index_; n++) {
    if (mask_h(n)) {
      if (block_index_h(n) >= 0) {
        particle_indices_to_send_h(block_index_h(n), counter[block_index_h(n)]) = n;
        counter[block_index_h(n)]++;
      }
    }
  }
  num_particles_to_send_.DeepCopy(num_particles_to_send_h);
  particle_indices_to_send_.DeepCopy(particle_indices_to_send_h);

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

    ApplyBoundaries_(total_received_particles_, new_indices_);
  }
}

void Swarm::ApplyBoundaries_(const int nparticles, ParArray1D<int> indices) {
  auto pmb = GetBlockPointer();
  auto &x = Get<Real>("x").Get();
  auto &y = Get<Real>("y").Get();
  auto &z = Get<Real>("z").Get();
  auto swarm_d = GetDeviceContext();
  auto bcs = this->bounds_d;

  pmb->par_for(
      PARTHENON_AUTO_LABEL, 0, nparticles - 1, KOKKOS_LAMBDA(const int n) {
        const int sid = indices(n);
        for (int l = 0; l < 6; l++) {
          bcs.bounds[l]->Apply(sid, x(sid), y(sid), z(sid), swarm_d);
        }
      });

  RemoveMarkedParticles();
}

bool Swarm::Receive(BoundaryCommSubset phase) {
  auto pmb = GetBlockPointer();
  const int nneighbor = pmb->neighbors.size();

  if (nneighbor == 0) {
    // Do nothing; no boundaries to receive
    return true;
  } else {
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
}

void Swarm::ResetCommunication() {
  auto pmb = GetBlockPointer();
#ifdef MPI_PARALLEL
  for (int n = 0; n < pmb->neighbors.size(); n++) {
    NeighborBlock &nb = pmb->neighbors[n];
    vbswarm->bd_var_.req_send[nb.bufid] = MPI_REQUEST_NULL;
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
