//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020 The Parthenon collaboration
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
    if (receiver_rank != sender_rank) tag = SendMPITag(pmb, nb, v);

#ifdef MPI_PARALLEL
    const comm_t comm = pmesh->GetMPIComm(v->label() + "_sparse_comm");
#else
    // Setting to zero is fine here since this doesn't actually get used when everything
    // is on the same rank
    const comm_t comm = 0;
#endif
    // Build sending buffers
    auto s_tag = SendKey(pmb, nb, v);
    PARTHENON_DEBUG_REQUIRE(pmesh->boundary_comm_map.count(s_tag) == 0,
                            "Two communication buffers have the same key.");

    pmesh->boundary_comm_map[s_tag] = CommBuffer<buf_pool_t<Real>::owner_t>(
        tag, sender_rank, receiver_rank, comm,
        [pmesh, buf_size]() { return pmesh->pool_map.at(buf_size).Get(); });

    // Also build the non-local receive buffers here
    if (sender_rank != receiver_rank) {
      int tag_r = ReceiveMPITag(pmb, nb, v);
      pmesh->boundary_comm_map[ReceiveKey(pmb, nb, v)] =
          CommBuffer<buf_pool_t<Real>::owner_t>(
              tag_r, receiver_rank, sender_rank, comm,
              [pmesh, buf_size]() { return pmesh->pool_map.at(buf_size).Get(); });
    }
  });

  Kokkos::Profiling::popRegion();
  return TaskStatus::complete;
}

template <BoundaryType bound_type> 
TaskStatus SendBoundBufs(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::Profiling::pushRegion("Task_LoadAndSendBoundBufs");
  
  Mesh *pmesh = md->GetMeshPointer();
  auto &cache = md->GetBvarsCache()[bound_type];

  if (cache.send_buf_vec.size() == 0) {
    int nbound = 0;
    ForEachBoundary<bound_type>(md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb, const sp_cv_t v) {
      PARTHENON_DEBUG_REQUIRE(pmesh->boundary_comm_map.count(SendKey(pmb, nb, v)) > 0,
                              "Boundary communicator does not exist");
      cache.send_buf_vec.push_back(&(pmesh->boundary_comm_map[SendKey(pmb, nb, v)]));
      nbound++;
    });
    cache.sending_non_zero_flags = ParArray1D<bool>("sending_nonzero_flags", nbound);
    cache.sending_non_zero_flags_h =
        Kokkos::create_mirror_view(cache.sending_non_zero_flags);
  } else {
    assert(cache.send_buf_vec.size() == cache.sending_non_zero_flags.size());
    assert(cache.send_buf_vec.size() == cache.sending_non_zero_flags_h.size());
  }

  // Allocate channels sending from active data and then check to see if
  // if buffers have changed
  bool rebuild = false;
  bool other_communication_unfinished = false;
  int nbound = 0;
  ForEachBoundary<bound_type>(md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb, const sp_cv_t v) {
    auto &buf = *cache.send_buf_vec[nbound];

    if (!buf.IsAvailableForWrite()) other_communication_unfinished = true;

    if (v->IsAllocated()) {
      buf.Allocate();
    } else {
      buf.Free();
    }

    if (nbound < cache.send_bnd_info_h.size()) {
      rebuild =
          rebuild || !UsingSameResource(cache.send_bnd_info_h(nbound).buf, buf.buffer());
    } else {
      rebuild = true;
    }

    ++nbound;
  });

  if (other_communication_unfinished) return TaskStatus::incomplete;

  if (rebuild) {
    cache.send_bnd_info = BufferCache_t("send_boundary_info", nbound);
    cache.send_bnd_info_h = Kokkos::create_mirror_view(cache.send_bnd_info);

    int iarr = 0;
    ForEachBoundary<bound_type>(md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb, const sp_cv_t v) {
      cache.send_bnd_info_h(iarr).allocated = v->IsAllocated();
      if (v->IsAllocated()) {
        cache.send_bnd_info_h(iarr) = BndInfo::GetSendBndInfo(pmb, nb, v);
        auto &buf = *cache.send_buf_vec[iarr];
        cache.send_bnd_info_h(iarr).buf = buf.buffer();
      }
      ++iarr;
    });
    Kokkos::deep_copy(cache.send_bnd_info, cache.send_bnd_info_h);
  }

  // Restrict
  auto &rc = md->GetBlockData(0);
  auto pmb = rc->GetBlockPointer();
  IndexShape cellbounds = pmb->cellbounds;
  IndexShape c_cellbounds = pmb->c_cellbounds;
  cell_centered_refinement::Restrict(cache.send_bnd_info, cellbounds, c_cellbounds);

  // Load buffer data
  const Real threshold = Globals::sparse_config.allocation_threshold;
  auto &bnd_info = cache.send_bnd_info;
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

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange<>(team_member, NtNuNvNkNjNi), [&](const int idx) {
              const int t = idx / NuNvNkNjNi;
              const int u = (idx % NuNvNkNjNi) / NvNkNjNi;
              const int v = (idx % NvNkNjNi) / NkNjNi;
              const int k = (idx % NkNjNi) / NjNi + sk;
              const int j = (idx % NjNi) / Ni + sj;
              const int i = idx % Ni + si;

              const Real val = bnd_info(b).var(t, u, v, k, j, i);
              const size_t flat_idx = i - si + Ni * (j - sj) + NjNi * (k - sk) +
                                      NkNjNi * v + NvNkNjNi * u + NuNvNkNjNi * t;
              bnd_info(b).buf(flat_idx) = val;
              if (std::abs(val) > threshold) sending_nonzero_flags(b) = true;
            });
      });

  // Send buffers
  if (Globals::sparse_config.enabled)
    Kokkos::deep_copy(sending_nonzero_flags_h, sending_nonzero_flags);
#ifdef MPI_PARALLEL
  if (bound_type == BoundaryType::any || bound_type == BoundaryType::nonlocal) Kokkos::fence();
#endif

  int iarr = 0;
  ForEachBoundary<bound_type>(md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb, const sp_cv_t v) {
    auto &buf = *cache.send_buf_vec[iarr];
    if (sending_nonzero_flags_h(iarr) || !Globals::sparse_config.enabled)
      buf.Send();
    else
      buf.SendNull();
    ++iarr;
  });

  Kokkos::Profiling::popRegion();
  return TaskStatus::complete;
}

template TaskStatus SendBoundBufs<BoundaryType::any>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus SendBoundBufs<BoundaryType::local>(std::shared_ptr<MeshData<Real>> &); 
template TaskStatus SendBoundBufs<BoundaryType::nonlocal>(std::shared_ptr<MeshData<Real>> &); 

template <BoundaryType bound_type> 
TaskStatus ReceiveBoundBufs(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::Profiling::pushRegion("Task_ReceiveBoundBufs");
  
  
  Mesh *pmesh = md->GetMeshPointer();
  auto &cache = md->GetBvarsCache()[bound_type];
  
  if (cache.recv_buf_vec.size() == 0) {
    ForEachBoundary<bound_type>(md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb, const sp_cv_t v) {
      PARTHENON_DEBUG_REQUIRE(pmesh->boundary_comm_map.count(ReceiveKey(pmb, nb, v)) > 0,
                              "Boundary communicator does not exist");
      cache.recv_buf_vec.push_back(&(pmesh->boundary_comm_map[ReceiveKey(pmb, nb, v)]));
    });
  }

  int ibound = 0;
  bool all_received = true;
  ForEachBoundary<bound_type>(md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb, const sp_cv_t v) {
    auto &buf = *cache.recv_buf_vec[ibound];

    all_received = all_received && buf.TryReceive();
    
    // Allocate variable if it is receiving actual data in any boundary
    // (the state could also be BufferState::received_null, which corresponds to no data)
    if (Globals::sparse_config.enabled && buf.GetState() == BufferState::received &&
        !v->IsAllocated()) {
      pmb->AllocateSparse(v->label());
      // TODO(lfroberts): Need to flag this so that the array gets filled with
      //                  something sensible, currently just defaulted to zero.
    }
    ++ibound;
  });
  Kokkos::Profiling::popRegion();

  if (all_received) return TaskStatus::complete;

  return TaskStatus::incomplete;
}

template TaskStatus ReceiveBoundBufs<BoundaryType::any>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus ReceiveBoundBufs<BoundaryType::local>(std::shared_ptr<MeshData<Real>> &); 
template TaskStatus ReceiveBoundBufs<BoundaryType::nonlocal>(std::shared_ptr<MeshData<Real>> &); 

template <BoundaryType bound_type> 
TaskStatus SetBounds(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::Profiling::pushRegion("Task_SetInternalBoundaries");
  
  Mesh *pmesh = md->GetMeshPointer();
  auto &cache = md->GetBvarsCache()[bound_type];

  // Check for rebuild
  bool rebuild = false;
  int nbound = 0;

  ForEachBoundary<bound_type>(md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb, const sp_cv_t v) {
    auto &buf = *cache.recv_buf_vec[nbound];
    if (nbound < cache.recv_bnd_info_h.size()) {
      rebuild =
          rebuild || !UsingSameResource(cache.recv_bnd_info_h(nbound).buf, buf.buffer());
      if ((buf.GetState() == BufferState::received) &&
          !cache.recv_bnd_info_h(nbound).allocated) {
        rebuild = true;
      }
      if ((buf.GetState() == BufferState::received_null) &&
          cache.recv_bnd_info_h(nbound).allocated) {
        rebuild = true;
      }
    } else {
      rebuild = true;
    }
    ++nbound;
  });

  if (rebuild) {
    cache.recv_bnd_info = BufferCache_t("recv_boundary_info", nbound);
    cache.recv_bnd_info_h = Kokkos::create_mirror_view(cache.recv_bnd_info);
    int iarr = 0;
    ForEachBoundary<bound_type>(md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb, const sp_cv_t v) {
      auto &buf = *cache.recv_buf_vec[iarr];
      if (v->IsAllocated())
        cache.recv_bnd_info_h(iarr) = BndInfo::GetSetBndInfo(pmb, nb, v);

      cache.recv_bnd_info_h(iarr).buf = buf.buffer();
      if (buf.GetState() == BufferState::received) {
        cache.recv_bnd_info_h(iarr).allocated = true;
        PARTHENON_DEBUG_REQUIRE(v->IsAllocated(),
                                "Variable must be allocated to receive");
      } else if (buf.GetState() == BufferState::received_null) {
        cache.recv_bnd_info_h(iarr).allocated = false;
      } else {
        PARTHENON_FAIL("Buffer should be in a received state.");
      }

      ++iarr;
    });
    Kokkos::deep_copy(cache.recv_bnd_info, cache.recv_bnd_info_h);
  }

  // const Real threshold = Globals::sparse_config.allocation_threshold;
  auto &bnd_info = cache.recv_bnd_info;
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
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange<>(team_member, NtNuNvNkNjNi), [&](const int idx) {
                // TODO(LFR): Make sure the tuv indexing is correct, what is currently
                // here doesn't seem right.
                const int t = idx / NuNvNkNjNi;
                const int u = (idx % NuNvNkNjNi) / NvNkNjNi;
                const int v = (idx % NvNkNjNi) / NkNjNi;
                const int k = (idx % NkNjNi) / NjNi + sk;
                const int j = (idx % NjNi) / Ni + sj;
                const int i = idx % Ni + si;

                const size_t flat_idx = i - si + Ni * (j - sj) + NjNi * (k - sk) +
                                        NkNjNi * v + NvNkNjNi * u + NuNvNkNjNi * t;

                const Real val = bnd_info(b).buf(flat_idx);
                bnd_info(b).var(t, u, v, k, j, i) = val;
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

  int iarr = 0;
  ForEachBoundary<bound_type>(md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb, const sp_cv_t v) {
    cache.recv_buf_vec[iarr]->Stale();
    ++iarr;
  });

  Kokkos::Profiling::popRegion();
  return TaskStatus::complete;
}

template TaskStatus SetBounds<BoundaryType::any>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus SetBounds<BoundaryType::local>(std::shared_ptr<MeshData<Real>> &); 
template TaskStatus SetBounds<BoundaryType::nonlocal>(std::shared_ptr<MeshData<Real>> &); 

} // namespace cell_centered_bvars
} // namespace parthenon
