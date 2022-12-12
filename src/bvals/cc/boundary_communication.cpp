//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2022 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2022. Triad National Security, LLC. All rights reserved.
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
#include <iostream> // debug
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "bnd_info.hpp"
#include "bvals_cc_in_one.hpp"
#include "bvals_utils.hpp"
#include "config.hpp"
#include "globals.hpp"
#include "interface/variable.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/refinement_in_one.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {
namespace cell_centered_bvars {

using namespace impl;

template <BoundaryType bound_type>
TaskStatus SendBoundBufs(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::Profiling::pushRegion("Task_LoadAndSendBoundBufs");

  Mesh *pmesh = md->GetMeshPointer();
  auto &cache = md->GetBvarsCache().GetSubCache(bound_type, true);

  if (cache.buf_vec.size() == 0)
    InitializeBufferCache<bound_type>(md, &(pmesh->boundary_comm_map), &cache, SendKey,
                                      true);

  auto [rebuild, nbound, other_communication_unfinished] =
      CheckSendBufferCacheForRebuild<bound_type, true>(md);

  if (nbound == 0) {
    Kokkos::Profiling::popRegion(); // Task_LoadAndSendBoundBufs
    return TaskStatus::complete;
  }
  if (other_communication_unfinished) {
    Kokkos::Profiling::popRegion(); // Task_LoadAndSendBoundBufs
    return TaskStatus::incomplete;
  }

  if (rebuild) RebuildBufferCache<bound_type, true>(md, nbound, BndInfo::GetSendBndInfo);

  // Restrict
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  StateDescriptor *resolved_packages = pmb->resolved_packages.get();
  refinement::Restrict(resolved_packages, cache, pmb->cellbounds, pmb->c_cellbounds);

  // Load buffer data
  auto &bnd_info = cache.bnd_info;
  PARTHENON_DEBUG_REQUIRE(bnd_info.size() == nbound, "Need same size for boundary info");
  auto &sending_nonzero_flags = cache.sending_non_zero_flags;
  auto &sending_nonzero_flags_h = cache.sending_non_zero_flags_h;
  Kokkos::parallel_for(
      "SendBoundBufs",
      Kokkos::TeamPolicy<>(parthenon::DevExecSpace(), nbound, Kokkos::AUTO),
      KOKKOS_LAMBDA(parthenon::team_mbr_t team_member) {
        const int b = team_member.league_rank();

        if (!bnd_info(b).allocated) {
          if (team_member.team_rank() == 0) {
            sending_nonzero_flags(b) = false;
          }
          return;
        }

        const int &si = bnd_info(b).si;
        const int &ei = bnd_info(b).ei;
        const int &sj = bnd_info(b).sj;
        const int &ej = bnd_info(b).ej;
        const int &sk = bnd_info(b).sk;
        const int &ek = bnd_info(b).ek;
        const int Ni = ei + 1 - si;
        const int Nj = ej + 1 - sj;
        const int Nk = ek + 1 - sk;

        const int &Nt = bnd_info(b).Nt;
        const int &Nu = bnd_info(b).Nu;
        const int &Nv = bnd_info(b).Nv;

        const int NjNi = Nj * Ni;
        const int NkNjNi = Nk * NjNi;
        const int NvNkNjNi = Nv * NkNjNi;
        const int NuNvNkNjNi = Nu * NvNkNjNi;
        const int NtNuNvNkNjNi = Nt * NuNvNkNjNi;

        Real threshold = bnd_info(b).var.allocation_threshold;
        bool non_zero = false;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange<>(team_member, NtNuNvNkNjNi),
            [&](const int idx, bool &lnon_zero) {
              const int t = idx / NuNvNkNjNi;
              const int u = (idx % NuNvNkNjNi) / NvNkNjNi;
              const int v = (idx % NvNkNjNi) / NkNjNi;
              const int k = (idx % NkNjNi) / NjNi + sk;
              const int j = (idx % NjNi) / Ni + sj;
              const int i = idx % Ni + si;

              const Real &val = bnd_info(b).var(t, u, v, k, j, i);
              bnd_info(b).buf(idx) = val;
              if (std::abs(val) >= threshold) {
                lnon_zero = true;
              }
            },
            Kokkos::BOr<bool, parthenon::DevMemSpace>(non_zero));

        if (team_member.team_rank() == 0) {
          sending_nonzero_flags(b) = non_zero;
        }
      });

  // Send buffers
  if (Globals::sparse_config.enabled)
    Kokkos::deep_copy(sending_nonzero_flags_h, sending_nonzero_flags);
#ifdef MPI_PARALLEL
  if (bound_type == BoundaryType::any || bound_type == BoundaryType::nonlocal)
    Kokkos::fence();
#endif

  for (int ibuf = 0; ibuf < cache.buf_vec.size(); ++ibuf) {
    auto &buf = *cache.buf_vec[ibuf];
    if (sending_nonzero_flags_h(ibuf) || !Globals::sparse_config.enabled)
      buf.Send();
    else
      buf.SendNull();
  }

  Kokkos::Profiling::popRegion(); // Task_LoadAndSendBoundBufs
  return TaskStatus::complete;
}

template TaskStatus SendBoundBufs<BoundaryType::any>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus SendBoundBufs<BoundaryType::local>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus
SendBoundBufs<BoundaryType::nonlocal>(std::shared_ptr<MeshData<Real>> &);

template <BoundaryType bound_type>
TaskStatus StartReceiveBoundBufs(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::Profiling::pushRegion("Task_StartReceiveBoundBufs");
  Mesh *pmesh = md->GetMeshPointer();
  auto &cache = md->GetBvarsCache().GetSubCache(BoundaryType::flxcor_send, false);
  if (cache.buf_vec.size() == 0)
    InitializeBufferCache<bound_type>(md, &(pmesh->boundary_comm_map), &cache, ReceiveKey,
                                      false);

  std::for_each(std::begin(cache.buf_vec), std::end(cache.buf_vec),
                [](auto pbuf) { pbuf->TryStartReceive(); });

  Kokkos::Profiling::popRegion(); // Task_StartReceiveBoundBufs
  return TaskStatus::complete;
}

template TaskStatus
StartReceiveBoundBufs<BoundaryType::any>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus
StartReceiveBoundBufs<BoundaryType::local>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus
StartReceiveBoundBufs<BoundaryType::nonlocal>(std::shared_ptr<MeshData<Real>> &);

template <BoundaryType bound_type>
TaskStatus ReceiveBoundBufs(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::Profiling::pushRegion("Task_ReceiveBoundBufs");

  Mesh *pmesh = md->GetMeshPointer();
  auto &cache = md->GetBvarsCache().GetSubCache(bound_type, false);
  if (cache.buf_vec.size() == 0)
    InitializeBufferCache<bound_type>(md, &(pmesh->boundary_comm_map), &cache, ReceiveKey,
                                      false);

  bool all_received = true;
  std::for_each(
      std::begin(cache.buf_vec), std::end(cache.buf_vec),
      [&all_received](auto pbuf) { all_received = pbuf->TryReceive() && all_received; });

  int ibound = 0;
  if (Globals::sparse_config.enabled) {
    ForEachBoundary<bound_type>(
        md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb, const sp_cv_t v, OffsetIndices &no) {
          const std::size_t ibuf = cache.idx_vec[ibound];
          auto &buf = *cache.buf_vec[ibuf];

          // Allocate variable if it is receiving actual data in any boundary
          // (the state could also be BufferState::received_null, which corresponds to no
          // data)
          if (buf.GetState() == BufferState::received && !v->IsAllocated()) {
            constexpr bool flag_uninitialized = true;
            constexpr bool only_control = true;
            pmb->AllocateSparse(v->label(), only_control, flag_uninitialized);
          }
          ++ibound;
        });
  }
  Kokkos::Profiling::popRegion(); // Task_ReceiveBoundBufs
  if (all_received) return TaskStatus::complete;
  return TaskStatus::incomplete;
}

template TaskStatus
ReceiveBoundBufs<BoundaryType::any>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus
ReceiveBoundBufs<BoundaryType::local>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus
ReceiveBoundBufs<BoundaryType::nonlocal>(std::shared_ptr<MeshData<Real>> &);

template <BoundaryType bound_type>
TaskStatus SetBounds(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::Profiling::pushRegion("Task_SetInternalBoundaries");

  Mesh *pmesh = md->GetMeshPointer();
  auto &cache = md->GetBvarsCache().GetSubCache(bound_type, false);

  auto [rebuild, nbound] = CheckReceiveBufferCacheForRebuild<bound_type, false>(md);
  if (rebuild) RebuildBufferCache<bound_type, false>(md, nbound, BndInfo::GetSetBndInfo);

  // const Real threshold = Globals::sparse_config.allocation_threshold;
  auto &bnd_info = cache.bnd_info;
  Kokkos::parallel_for(
      "SetBoundaryBuffers",
      Kokkos::TeamPolicy<>(parthenon::DevExecSpace(), nbound, Kokkos::AUTO),
      KOKKOS_LAMBDA(parthenon::team_mbr_t team_member) {
        const int b = team_member.league_rank();

        const int &si = bnd_info(b).si;
        const int &ei = bnd_info(b).ei;
        const int &sj = bnd_info(b).sj;
        const int &ej = bnd_info(b).ej;
        const int &sk = bnd_info(b).sk;
        const int &ek = bnd_info(b).ek;

        const int Ni = ei + 1 - si;
        const int Nj = ej + 1 - sj;
        const int Nk = ek + 1 - sk;
        const int &Nv = bnd_info(b).Nv;
        const int &Nu = bnd_info(b).Nu;
        const int &Nt = bnd_info(b).Nt;

        const int NjNi = Nj * Ni;
        const int NkNjNi = Nk * NjNi;
        const int NvNkNjNi = Nv * NkNjNi;
        const int NuNvNkNjNi = Nu * NvNkNjNi;
        const int NtNuNvNkNjNi = Nt * NuNvNkNjNi;

        if (bnd_info(b).allocated) {
          Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team_member, NtNuNvNkNjNi),
                               [&](const int idx) {
                                 const int t = idx / NuNvNkNjNi;
                                 const int u = (idx % NuNvNkNjNi) / NvNkNjNi;
                                 const int v = (idx % NvNkNjNi) / NkNjNi;
                                 const int k = (idx % NkNjNi) / NjNi + sk;
                                 const int j = (idx % NjNi) / Ni + sj;
                                 const int i = idx % Ni + si;

                                 bnd_info(b).var(t, u, v, k, j, i) = bnd_info(b).buf(idx);
                               });
        } else if (bnd_info(b).var.size() > 0) {
          const Real default_val = bnd_info(b).var.sparse_default_val;
          Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team_member, NtNuNvNkNjNi),
                               [&](const int idx) {
                                 const int t = idx / NuNvNkNjNi;
                                 const int u = (idx % NuNvNkNjNi) / NvNkNjNi;
                                 const int v = (idx % NvNkNjNi) / NkNjNi;
                                 const int k = (idx % NkNjNi) / NjNi + sk;
                                 const int j = (idx % NjNi) / Ni + sj;
                                 const int i = idx % Ni + si;
                                 bnd_info(b).var(t, u, v, k, j, i) = default_val;
                               });
        }
      });
#ifdef MPI_PARALLEL
  Kokkos::fence();
#endif
  std::for_each(std::begin(cache.buf_vec), std::end(cache.buf_vec),
                [](auto pbuf) { pbuf->Stale(); });

  Kokkos::Profiling::popRegion(); // Task_SetInternalBoundaries
  return TaskStatus::complete;
}

// Restricts all relevant meshblock boundaries, but doesn't
// communicate at all.
TaskStatus RestrictGhostHalos(std::shared_ptr<MeshData<Real>> &md, bool reset_cache) {
  constexpr BoundaryType bound_type = BoundaryType::restricted;
  Kokkos::Profiling::pushRegion("Task_RestrictGhostHalos");
  Mesh *pmesh = md->GetMeshPointer();
  BvarsSubCache_t &cache = md->GetBvarsCache().GetSubCache(bound_type, false);
  // JMM: No buffers to communicate, but we still want the buffer info
  // cache so we don't bother using the initialization routine, we
  // just set the index to linear and go.
  if (reset_cache || cache.idx_vec.size() == 0) {
    cache.clear();
    int buff_idx = 0;
    ForEachBoundary<bound_type>(md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb,
                                        const sp_cv_t v, const OffsetIndices &no) {
      cache.idx_vec.push_back(buff_idx++);
      // must fill buf_vec even if we don't allocate new buffers
      // because it's passed into the BoundaryCreator struct
      cache.buf_vec.push_back(nullptr);
    });
  }
  auto [rebuild, nbound] = CheckNoCommCacheForRebuild<bound_type, false>(md);
  if (rebuild || reset_cache) {
    RebuildBufferCache<bound_type, false>(md, nbound, BndInfo::GetCCRestrictInfo);
  }
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  StateDescriptor *resolved_packages = pmb->resolved_packages.get();
  refinement::Restrict(resolved_packages, cache, pmb->cellbounds, pmb->c_cellbounds);
  Kokkos::Profiling::popRegion(); // Task_RestrictGhostHalos
  return TaskStatus::complete;
}

template TaskStatus SetBounds<BoundaryType::any>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus SetBounds<BoundaryType::local>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus SetBounds<BoundaryType::nonlocal>(std::shared_ptr<MeshData<Real>> &);

} // namespace cell_centered_bvars
} // namespace parthenon
