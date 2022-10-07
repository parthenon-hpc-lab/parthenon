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

#include "bvals_cc_in_one.hpp"
#include "bvals_utils.hpp"
#include "config.hpp"
#include "globals.hpp"
#include "interface/variable.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/refinement_cc_in_one.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {
namespace cell_centered_bvars {

using namespace impl;

// pmesh->boundary_comm_map.clear() after every remesh
// in InitializeBlockTimeStepsAndBoundaries()
TaskStatus BuildSparseBoundaryBuffers(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::Profiling::pushRegion("Task_BuildSendBoundBufs");
  Mesh *pmesh = md->GetMeshPointer();
  auto &all_caches = md->GetBvarsCache();

  // Clear the fast access vectors for this block since they are no longer valid
  // after all MeshData call BuildSparseBoundaryBuffers
  all_caches.clear();

  // Build buffers for all boundaries, both local and nonlocal
  ForEachBoundary(md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb, const sp_cv_t v) {
    // Calculate the required size of the buffer for this boundary
    int buf_size = GetBufferSize(pmb, nb, v);

    // Add a buffer pool if one does not exist for this size
    if (pmesh->pool_map.count(buf_size) == 0) {
      pmesh->pool_map.emplace(std::make_pair(
          buf_size, buf_pool_t<Real>([buf_size](buf_pool_t<Real> *pool) {
            using buf_t = buf_pool_t<Real>::base_t;
            // TODO(LFR): Make nbuf a user settable parameter
            const int nbuf = 200;
            buf_t chunk("pool buffer", buf_size * nbuf);
            for (int i = 1; i < nbuf; ++i) {
              pool->AddFreeObjectToPool(
                  buf_t(chunk, std::make_pair(i * buf_size, (i + 1) * buf_size)));
            }
            return buf_t(chunk, std::make_pair(0, buf_size));
          })));
    }

    const int receiver_rank = nb.snb.rank;
    const int sender_rank = Globals::my_rank;

    int tag = 0;
#ifdef MPI_PARALLEL
    // Get a bi-directional mpi tag for this pair of blocks
    tag = pmesh->tag_map.GetTag(pmb, nb);

    mpi_comm_t comm = pmesh->GetMPIComm(v->label());
    mpi_comm_t comm_flxcor = comm;
    if (nb.snb.level != pmb->loc.level)
      comm_flxcor = pmesh->GetMPIComm(v->label() + "_flcor");
#else
    // Setting to zero is fine here since this doesn't actually get used when everything
    // is on the same rank
    mpi_comm_t comm = 0;
    mpi_comm_t comm_flxcor = 0;
#endif
    // Build send buffers
    auto s_key = SendKey(pmb, nb, v);
    PARTHENON_DEBUG_REQUIRE(pmesh->boundary_comm_map.count(s_key) == 0,
                            "Two communication buffers have the same key.");

    auto get_resource_method = [pmesh, buf_size]() {
      return buf_pool_t<Real>::owner_t(pmesh->pool_map.at(buf_size).Get());
    };

    bool use_sparse_buffers = v->IsSet(Metadata::Sparse);
    pmesh->boundary_comm_map[s_key] = CommBuffer<buf_pool_t<Real>::owner_t>(
        tag, sender_rank, receiver_rank, comm, get_resource_method, use_sparse_buffers);

    // Separate flxcor buffer if needed, first part of if statement checks that this
    // is fine to coarse and the second checks the two blocks share a face
    if ((nb.snb.level == pmb->loc.level - 1) &&
        (std::abs(nb.ni.ox1) + std::abs(nb.ni.ox2) + std::abs(nb.ni.ox3) == 1))
      pmesh->boundary_comm_flxcor_map[s_key] = CommBuffer<buf_pool_t<Real>::owner_t>(
          tag, sender_rank, receiver_rank, comm_flxcor, get_resource_method,
          use_sparse_buffers);

    // Also build the non-local receive buffers here
    if (sender_rank != receiver_rank) {
      auto r_key = ReceiveKey(pmb, nb, v);
      pmesh->boundary_comm_map[r_key] = CommBuffer<buf_pool_t<Real>::owner_t>(
          tag, receiver_rank, sender_rank, comm, get_resource_method, use_sparse_buffers);
      // Separate flxcor buffer if needed
      if ((nb.snb.level - 1 == pmb->loc.level) &&
          (std::abs(nb.ni.ox1) + std::abs(nb.ni.ox2) + std::abs(nb.ni.ox3) == 1))
        pmesh->boundary_comm_flxcor_map[r_key] = CommBuffer<buf_pool_t<Real>::owner_t>(
            tag, receiver_rank, sender_rank, comm_flxcor, get_resource_method,
            use_sparse_buffers);
    }
  });

  Kokkos::Profiling::popRegion(); // "Task_BuildSendBoundBufs"
  return TaskStatus::complete;
}



template <BoundaryType bound_type>
TaskStatus SendBoundBufs(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::Profiling::pushRegion("Task_LoadAndSendBoundBufs");

  Mesh *pmesh = md->GetMeshPointer();
  auto &cache = md->GetBvarsCache().GetSubCache(bound_type, true);

  if (cache.buf_vec.size() == 0) {
    BuildBufferCache<bound_type>(md, &(pmesh->boundary_comm_map), &(cache.buf_vec), 
                                 &(cache.idx_vec), SendKey);
    const int nbound = cache.buf_vec.size();
    if (nbound > 0) {
      cache.sending_non_zero_flags = ParArray1D<bool>("sending_nonzero_flags", nbound);
      cache.sending_non_zero_flags_h =
          Kokkos::create_mirror_view(cache.sending_non_zero_flags);
    }
  } else {
    PARTHENON_REQUIRE(cache.buf_vec.size() == cache.sending_non_zero_flags.size(),
                      "Flag arrays incorrectly allocated.");
    PARTHENON_REQUIRE(cache.buf_vec.size() == cache.sending_non_zero_flags_h.size(),
                      "Flag arrays incorrectly allocated.");
  }
  
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


  //if (rebuild) RebuildBufferCache<bound_type, true>(md, nbound, BndInfo::GetSendBndInfo);
  if (rebuild) {
    cache.bnd_info = BufferCache_t("boundary_info", nbound);
    cache.bnd_info_h = Kokkos::create_mirror_view(cache.bnd_info);

    int ibound = 0;
    ForEachBoundary<bound_type>(
        md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb, const sp_cv_t v) {
          const std::size_t ibuf = cache.idx_vec[ibound];
          cache.bnd_info_h(ibuf).allocated = v->IsAllocated();
          if (v->IsAllocated()) {
            cache.bnd_info_h(ibuf) = BndInfo::GetSendBndInfo(pmb, nb, v);
            auto &buf = *cache.buf_vec[ibuf];
            cache.bnd_info_h(ibuf).buf = buf.buffer();
          }
          ++ibound;
        });
    Kokkos::deep_copy(cache.bnd_info, cache.bnd_info_h);
  }

  // Restrict
  auto &rc = md->GetBlockData(0);
  auto pmb = rc->GetBlockPointer();
  IndexShape cellbounds = pmb->cellbounds;
  IndexShape c_cellbounds = pmb->c_cellbounds;
  cell_centered_refinement::Restrict(cache.bnd_info, cache.bnd_info_h,
                                     cellbounds, c_cellbounds);

  // Load buffer data
  const Real threshold = Globals::sparse_config.allocation_threshold;
  auto &bnd_info = cache.bnd_info;
  PARTHENON_DEBUG_REQUIRE(bnd_info.size() == nbound, "Need same size for boundary info");
  auto &sending_nonzero_flags = cache.sending_non_zero_flags;
  auto &sending_nonzero_flags_h = cache.sending_non_zero_flags_h;
  Kokkos::parallel_for(
      "SendBoundBufs",
      Kokkos::TeamPolicy<>(parthenon::DevExecSpace(), nbound, Kokkos::AUTO),
      KOKKOS_LAMBDA(parthenon::team_mbr_t team_member) {
        const int b = team_member.league_rank();

        sending_nonzero_flags(b) = false;
        if (!bnd_info(b).allocated) return;

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

        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team_member, NtNuNvNkNjNi),
                             [&](const int idx) {
                               const int t = idx / NuNvNkNjNi;
                               const int u = (idx % NuNvNkNjNi) / NvNkNjNi;
                               const int v = (idx % NvNkNjNi) / NkNjNi;
                               const int k = (idx % NkNjNi) / NjNi + sk;
                               const int j = (idx % NjNi) / Ni + sj;
                               const int i = idx % Ni + si;

                               const Real &val = bnd_info(b).var(t, u, v, k, j, i);
                               bnd_info(b).buf(idx) = val;
                               if (std::abs(val) > threshold)
                                 sending_nonzero_flags(b) = true;
                             });
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
    BuildBufferCache<bound_type>(md, &(pmesh->boundary_comm_map), &(cache.buf_vec), 
                                 &(cache.idx_vec), ReceiveKey);

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
    BuildBufferCache<bound_type>(md, &(pmesh->boundary_comm_map), &(cache.buf_vec), 
                                 &(cache.idx_vec), ReceiveKey);

  bool all_received = true;
  std::for_each(
      std::begin(cache.buf_vec), std::end(cache.buf_vec),
      [&all_received](auto pbuf) { all_received = pbuf->TryReceive() && all_received; });

  int ibound = 0;
  if (Globals::sparse_config.enabled) {
    ForEachBoundary<bound_type>(
        md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb, const sp_cv_t v) {
          const std::size_t ibuf = cache.idx_vec[ibound];
          auto &buf = *cache.buf_vec[ibuf];

          // Allocate variable if it is receiving actual data in any boundary
          // (the state could also be BufferState::received_null, which corresponds to no
          // data)
          if (buf.GetState() == BufferState::received && !v->IsAllocated()) {
            pmb->AllocateSparse(v->label());
            // TODO(lfroberts): Need to flag this so that the array gets filled with
            //                  something sensible, currently just defaulted to zero.
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
  //if (rebuild) RebuildBufferCache<bound_type, false>(md, nbound, BndInfo::GetSetBndInfo);

  if (rebuild) {
    cache.bnd_info = BufferCache_t("boundary_info", nbound);
    cache.bnd_info_h = Kokkos::create_mirror_view(cache.bnd_info);
    int iarr = 0;
    ForEachBoundary<bound_type>(
        md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb, const sp_cv_t v) {
          const std::size_t ibuf = cache.idx_vec[iarr];
          auto &buf = *cache.buf_vec[ibuf];
          if (v->IsAllocated())
            cache.bnd_info_h(ibuf) = BndInfo::GetSetBndInfo(pmb, nb, v);

          cache.bnd_info_h(ibuf).buf = buf.buffer();
          if (buf.GetState() == BufferState::received) {
            cache.bnd_info_h(ibuf).allocated = true;
            PARTHENON_DEBUG_REQUIRE(v->IsAllocated(),
                                    "Variable must be allocated to receive");
          } else if (buf.GetState() == BufferState::received_null) {
            cache.bnd_info_h(ibuf).allocated = false;
          } else {
            PARTHENON_FAIL("Buffer should be in a received state.");
          }

          ++iarr;
        });
    Kokkos::deep_copy(cache.bnd_info, cache.bnd_info_h);
  }

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
          Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team_member, NtNuNvNkNjNi),
                               [&](const int idx) {
                                 const int t = idx / NuNvNkNjNi;
                                 const int u = (idx % NuNvNkNjNi) / NvNkNjNi;
                                 const int v = (idx % NvNkNjNi) / NkNjNi;
                                 const int k = (idx % NkNjNi) / NjNi + sk;
                                 const int j = (idx % NjNi) / Ni + sj;
                                 const int i = idx % Ni + si;
                                 bnd_info(b).var(t, u, v, k, j, i) = 0.0;
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

template TaskStatus SetBounds<BoundaryType::any>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus SetBounds<BoundaryType::local>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus SetBounds<BoundaryType::nonlocal>(std::shared_ptr<MeshData<Real>> &);

} // namespace cell_centered_bvars
} // namespace parthenon
