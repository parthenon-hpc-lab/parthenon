//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2026 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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

// This file was made in part with generative AI

#include "mesh/swarm_amr_remesh.hpp"

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "parthenon_mpi.hpp"

#include "defs.hpp"
#include "globals.hpp"
#include "interface/swarm.hpp"
#include "mesh/amr_particle_ownership.hpp"
#include "mesh/mesh.hpp"
#include "pack/swarm_pack/swarm_default_names.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

namespace {

using RealSwarmHostMirror =
    decltype(std::declval<ParticleVariable<Real>>().GetHostMirrorAndCopy());
using IntSwarmHostMirror =
    decltype(std::declval<ParticleVariable<int>>().GetHostMirrorAndCopy());
using UInt64SwarmHostMirror =
    decltype(std::declval<ParticleVariable<std::uint64_t>>().GetHostMirrorAndCopy());

// This remesh path intentionally works with host mirrors and byte buffers rather than the
// usual neighbor-send swarm buffers. Existing swarm communication packs directly from
// device into per-neighbor Real buffers for the current mesh topology. AMR remeshing is a
// different problem: the destination is the post-remesh leaf mesh, which may change both
// owning block and owning rank. The host-side serializer below therefore prioritizes a
// schema-generic "complete particle record" exchange over reusing the neighbor-only comms
// path verbatim.

template <typename T, typename F>
// Visit every logical component stored in a particle variable so the remesh serializer
// can treat scalars and small fixed-size arrays uniformly.
void ForEachParticleComponent(const ParticleVariable<T> &var, F &&f) {
  for (int n6 = 0; n6 < var.GetDim(6); ++n6) {
    for (int n5 = 0; n5 < var.GetDim(5); ++n5) {
      for (int n4 = 0; n4 < var.GetDim(4); ++n4) {
        for (int n3 = 0; n3 < var.GetDim(3); ++n3) {
          for (int n2 = 0; n2 < var.GetDim(2); ++n2) {
            f(n6, n5, n4, n3, n2);
          }
        }
      }
    }
  }
}

inline void AppendBytes(std::vector<char> &buffer, const void *data, std::size_t size) {
  const auto old_size = buffer.size();
  buffer.resize(old_size + size);
  std::memcpy(buffer.data() + old_size, data, size);
}

template <typename T>
// Append all components of one particle variable for one particle to the byte buffer.
void AppendParticleVariableData(const std::shared_ptr<ParticleVariable<T>> &var,
                                const decltype(var->GetHostMirrorAndCopy()) &host,
                                int particle_idx, std::vector<char> &buffer) {
  ForEachParticleComponent(
      *var, [&](const int n6, const int n5, const int n4, const int n3, const int n2) {
        const T value = host(n6, n5, n4, n3, n2, particle_idx);
        AppendBytes(buffer, &value, sizeof(T));
      });
}

template <typename T, typename HostMirror>
// Reconstruct one particle variable from a serialized byte stream into a host mirror.
void LoadParticleVariableData(const std::shared_ptr<ParticleVariable<T>> &var,
                              HostMirror &host, int particle_idx,
                              const char *&buffer_ptr) {
  ForEachParticleComponent(
      *var, [&](const int n6, const int n5, const int n4, const int n3, const int n2) {
        T value;
        std::memcpy(&value, buffer_ptr, sizeof(T));
        host(n6, n5, n4, n3, n2, particle_idx) = value;
        buffer_ptr += sizeof(T);
      });
}

std::size_t GetSwarmRemeshRecordSizeBytes(const std::shared_ptr<Swarm> &swarm) {
  std::size_t size = sizeof(int);
  for (const auto &var : swarm->GetVariableVector<Real>()) {
    size += var->NumComponents() * sizeof(Real);
  }
  for (const auto &var : swarm->GetVariableVector<int>()) {
    size += var->NumComponents() * sizeof(int);
  }
  for (const auto &var : swarm->GetVariableVector<std::uint64_t>()) {
    size += var->NumComponents() * sizeof(std::uint64_t);
  }
  return size;
}

} // namespace

// Remap every active particle in every resolved swarm from the old leaf mesh onto the new
// leaf mesh. The owning destination is chosen from the post-remesh geometry and rank
// layout, so the same routine covers pure AMR changes, pure load-balancing changes, and
// combined AMR+load-balance steps.
void RemeshSwarms(const std::shared_ptr<StateDescriptor> &resolved_packages,
                  const BlockList_t &old_block_list, Mesh *pmesh,
                  const SwarmRemeshContext &context) {
  if (resolved_packages->AllSwarms().empty()) return;

  for (const auto &swarm_pair : resolved_packages->AllSwarms()) {
    const auto &swarm_name = swarm_pair.first;
    std::vector<char> recv_buffer;
#ifdef MPI_PARALLEL
    std::vector<std::vector<char>> send_buffers(Globals::nranks);
#else
    std::vector<std::vector<char>> send_buffers(1);
#endif

    for (int on = context.old_start_gid; on <= context.old_end_gid; ++on) {
      const int nn = context.old_to_new_gid[on];
      const bool same_rank_same_level =
          (context.old_ranks[on] == context.new_ranks[nn]) &&
          (context.old_locs[on].level() == context.new_locs[nn].level());
      if (same_rank_same_level) continue;

      auto &old_pmb = old_block_list[on - context.old_start_gid];
      auto &old_swarm_container = old_pmb->meshblock_data.Get()->GetSwarmData();
      if (!old_swarm_container->Contains(swarm_name)) continue;

      auto swarm = old_swarm_container->Get(swarm_name);
      if (swarm->GetNumActive() == 0) continue;

      const auto mask_h = swarm->GetMask().GetHostMirrorAndCopy();
      const auto real_vars = swarm->GetVariableVector<Real>();
      const auto int_vars = swarm->GetVariableVector<int>();
      const auto uint64_vars = swarm->GetVariableVector<std::uint64_t>();

      // Pull the swarm record to host once so we can pack a complete, mixed-type particle
      // payload into byte buffers for all-to-all exchange. This looks heavier than field
      // remesh because swarm records are heterogeneous and are not represented as one
      // structured mesh array with an existing prolong/restrict operator.
      std::vector<RealSwarmHostMirror> real_h;
      std::vector<IntSwarmHostMirror> int_h;
      std::vector<UInt64SwarmHostMirror> uint64_h;
      real_h.reserve(real_vars.size());
      int_h.reserve(int_vars.size());
      uint64_h.reserve(uint64_vars.size());
      for (const auto &var : real_vars) {
        real_h.emplace_back(var->GetHostMirrorAndCopy());
      }
      for (const auto &var : int_vars) {
        int_h.emplace_back(var->GetHostMirrorAndCopy());
      }
      for (const auto &var : uint64_vars) {
        uint64_h.emplace_back(var->GetHostMirrorAndCopy());
      }

      auto &x = swarm->Get<Real>(swarm_position::x::name()).Get();
      auto &y = swarm->Get<Real>(swarm_position::y::name()).Get();
      auto &z = swarm->Get<Real>(swarm_position::z::name()).Get();
      auto x_h = x.GetHostMirrorAndCopy();
      auto y_h = y.GetHostMirrorAndCopy();
      auto z_h = z.GetHostMirrorAndCopy();

      for (int n = 0; n <= swarm->GetMaxActiveIndex(); ++n) {
        if (!mask_h(n)) continue;

        const int dest_gid =
            amr::FindOwningBlock(pmesh, context.new_locs, x_h(n), y_h(n), z_h(n));
        PARTHENON_REQUIRE(dest_gid >= 0,
                          "Failed to find destination block for remeshed particle.");

        auto &buffer = send_buffers[context.new_ranks[dest_gid]];
        AppendBytes(buffer, &dest_gid, sizeof(int));
        for (std::size_t i = 0; i < real_vars.size(); ++i) {
          AppendParticleVariableData(real_vars[i], real_h[i], n, buffer);
        }
        for (std::size_t i = 0; i < int_vars.size(); ++i) {
          AppendParticleVariableData(int_vars[i], int_h[i], n, buffer);
        }
        for (std::size_t i = 0; i < uint64_vars.size(); ++i) {
          AppendParticleVariableData(uint64_vars[i], uint64_h[i], n, buffer);
        }
      }
    }

#ifdef MPI_PARALLEL
    std::vector<int> send_counts(Globals::nranks, 0);
    std::vector<int> recv_counts(Globals::nranks, 0);
    for (int rank = 0; rank < Globals::nranks; ++rank) {
      PARTHENON_REQUIRE(send_buffers[rank].size() <= std::numeric_limits<int>::max(),
                        "Swarm remesh send buffer too large for MPI.");
      send_counts[rank] = static_cast<int>(send_buffers[rank].size());
    }
    PARTHENON_MPI_CHECK(MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(),
                                     1, MPI_INT, MPI_COMM_WORLD));

    std::vector<int> send_displs(Globals::nranks, 0);
    std::vector<int> recv_displs(Globals::nranks, 0);
    int send_total = 0;
    int recv_total = 0;
    for (int rank = 0; rank < Globals::nranks; ++rank) {
      send_displs[rank] = send_total;
      recv_displs[rank] = recv_total;
      send_total += send_counts[rank];
      recv_total += recv_counts[rank];
    }

    std::vector<char> send_buffer(send_total);
    for (int rank = 0; rank < Globals::nranks; ++rank) {
      std::memcpy(send_buffer.data() + send_displs[rank], send_buffers[rank].data(),
                  send_buffers[rank].size());
    }

    recv_buffer.resize(recv_total);
    PARTHENON_MPI_CHECK(MPI_Alltoallv(send_buffer.data(), send_counts.data(),
                                      send_displs.data(), MPI_BYTE, recv_buffer.data(),
                                      recv_counts.data(), recv_displs.data(), MPI_BYTE,
                                      MPI_COMM_WORLD));
#else
    recv_buffer = std::move(send_buffers[0]);
#endif

    if (recv_buffer.empty()) continue;

    auto prototype_swarm =
        pmesh->block_list.front()->meshblock_data.Get()->GetSwarmData()->Get(swarm_name);
    const auto record_size = GetSwarmRemeshRecordSizeBytes(prototype_swarm);
    PARTHENON_REQUIRE(record_size > 0, "Invalid swarm remesh record size.");
    PARTHENON_REQUIRE(recv_buffer.size() % record_size == 0,
                      "Swarm remesh receive buffer has invalid size.");

    std::unordered_map<int, int> received_per_gid;
    for (std::size_t offset = 0; offset < recv_buffer.size(); offset += record_size) {
      int gid;
      std::memcpy(&gid, recv_buffer.data() + offset, sizeof(int));
      received_per_gid[gid]++;
    }

    for (const auto &[gid, count] : received_per_gid) {
      auto pmb = pmesh->FindMeshBlock(gid);
      auto swarm = pmb->meshblock_data.Get()->GetSwarmData()->Get(swarm_name);
      auto new_particles = swarm->AddEmptyParticles(count);

      const auto real_vars = swarm->GetVariableVector<Real>();
      const auto int_vars = swarm->GetVariableVector<int>();
      const auto uint64_vars = swarm->GetVariableVector<std::uint64_t>();

      std::vector<RealSwarmHostMirror> real_h;
      std::vector<IntSwarmHostMirror> int_h;
      std::vector<UInt64SwarmHostMirror> uint64_h;
      real_h.reserve(real_vars.size());
      int_h.reserve(int_vars.size());
      uint64_h.reserve(uint64_vars.size());
      for (const auto &var : real_vars) {
        real_h.emplace_back(var->GetHostMirrorAndCopy());
      }
      for (const auto &var : int_vars) {
        int_h.emplace_back(var->GetHostMirrorAndCopy());
      }
      for (const auto &var : uint64_vars) {
        uint64_h.emplace_back(var->GetHostMirrorAndCopy());
      }

      int new_particle_idx = 0;
      for (std::size_t offset = 0; offset < recv_buffer.size(); offset += record_size) {
        const char *record_ptr = recv_buffer.data() + offset;
        int recv_gid;
        std::memcpy(&recv_gid, record_ptr, sizeof(int));
        record_ptr += sizeof(int);
        if (recv_gid != gid) continue;

        const int particle_idx = new_particles.GetNewParticleIndex(new_particle_idx++);
        for (std::size_t i = 0; i < real_vars.size(); ++i) {
          LoadParticleVariableData(real_vars[i], real_h[i], particle_idx, record_ptr);
        }
        for (std::size_t i = 0; i < int_vars.size(); ++i) {
          LoadParticleVariableData(int_vars[i], int_h[i], particle_idx, record_ptr);
        }
        for (std::size_t i = 0; i < uint64_vars.size(); ++i) {
          LoadParticleVariableData(uint64_vars[i], uint64_h[i], particle_idx, record_ptr);
        }
      }

      PARTHENON_REQUIRE(new_particle_idx == count,
                        "Mismatch while unpacking remeshed swarm particles.");

      for (std::size_t i = 0; i < real_vars.size(); ++i) {
        real_vars[i]->Get().DeepCopy(real_h[i]);
      }
      for (std::size_t i = 0; i < int_vars.size(); ++i) {
        int_vars[i]->Get().DeepCopy(int_h[i]);
      }
      for (std::size_t i = 0; i < uint64_vars.size(); ++i) {
        uint64_vars[i]->Get().DeepCopy(uint64_h[i]);
      }
    }
  }
}

// Swarm layout changes can invalidate cached SwarmPacks at both the MeshBlockData and
// MeshData levels. `mesh_data.clear()` destroys most MeshData containers in this remesh
// path already, but clearing any surviving MeshData caches here as well is cheap and
// makes the post-remesh cleanup rule explicit and conservative.
void ClearSwarmCachesAfterRemesh(Mesh *pmesh, const BlockList_t &block_list) {
  for (const auto &pmb : block_list) {
    for (const auto &[label, mbd] : pmb->meshblock_data.Stages()) {
      mbd->ClearSwarmCaches();
    }
  }
  for (const auto &[label, md] : pmesh->mesh_data.Stages()) {
    md->ClearSwarmCaches();
  }
}

// Find the post-remesh leaf block that owns the given particle position. When a point
// lies on a shared face, edge, or node, break ties with the same ownership ordering
// Parthenon already uses for shared field elements.
int amr::FindOwningBlock(const Mesh *pmesh, const std::vector<LogicalLocation> &locs,
                         const Real x, const Real y, const Real z) {
  int owner_gid = -1;
  for (int gid = 0; gid < static_cast<int>(locs.size()); ++gid) {
    if (PointInBlock(pmesh->GetBlockSize(locs[gid]), x, y, z, pmesh->ndim)) {
      if (owner_gid < 0 || OwnershipLessThan(locs[owner_gid], locs[gid])) {
        owner_gid = gid;
      }
    }
  }
  return owner_gid;
}

} // namespace parthenon
