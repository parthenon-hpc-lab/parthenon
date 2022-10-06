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
#include <string>
#include <type_traits>
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

TaskStatus LoadAndSendFluxCorrections(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::Profiling::pushRegion("Task_LoadAndSendFluxCorrections");

  Mesh *pmesh = md->GetMeshPointer();
  auto &cache = md->GetBvarsCache()[BoundaryType::flxcor_send];
  const int ndim = pmesh->ndim;

  if (cache.send_buf_vec.size() == 0) {
    BuildBufferCache<BoundaryType::flxcor_send>(md, &(pmesh->boundary_comm_flxcor_map), 
                                 &(cache.send_buf_vec), &(cache.send_idx_vec), SendKey);
    const int nbound = cache.send_buf_vec.size();
    if (nbound > 0) {
      cache.sending_non_zero_flags = ParArray1D<bool>("sending_nonzero_flags", nbound);
      cache.sending_non_zero_flags_h =
          Kokkos::create_mirror_view(cache.sending_non_zero_flags);
    }
  } else {
    PARTHENON_REQUIRE(cache.send_buf_vec.size() == cache.sending_non_zero_flags.size(),
                      "Flag arrays incorrectly allocated.");
    PARTHENON_REQUIRE(cache.send_buf_vec.size() == cache.sending_non_zero_flags_h.size(),
                      "Flag arrays incorrectly allocated.");
  }

  // Allocate channels sending from active data and then check to see if
  // if buffers have changed
  bool rebuild = false;
  bool other_communication_unfinished = false;
  int nbound = 0;
  ForEachBoundary<BoundaryType::flxcor_send>(
      md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb, const sp_cv_t v) {
        const std::size_t ibuf = cache.send_idx_vec[nbound];
        auto &buf = *(cache.send_buf_vec[ibuf]);

        if (!buf.IsAvailableForWrite()) other_communication_unfinished = true;

        if (v->IsAllocated()) {
          buf.Allocate();
        } else {
          buf.Free();
        }

        if (ibuf < cache.send_bnd_info_h.size()) {
          rebuild = rebuild ||
                    !UsingSameResource(cache.send_bnd_info_h(ibuf).buf, buf.buffer());
        } else {
          rebuild = true;
        }

        ++nbound;
      });
  
  if (nbound == 0) {
    Kokkos::Profiling::popRegion(); // Task_LoadAndSendBoundBufs
    return TaskStatus::complete;
  }

  if (other_communication_unfinished) {
    Kokkos::Profiling::popRegion(); // Task_LoadAndSendBoundBufs
    return TaskStatus::incomplete;
  }

  if (rebuild) {
    cache.send_bnd_info = BufferCache_t("send_fluxcor_info", nbound);
    cache.send_bnd_info_h = Kokkos::create_mirror_view(cache.send_bnd_info);

    int ibound = 0;
    ForEachBoundary<BoundaryType::flxcor_send>(
        md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb, const sp_cv_t v) {
          const std::size_t ibuf = cache.send_idx_vec[ibound];
          cache.send_bnd_info_h(ibuf).allocated = v->IsAllocated();
          if (v->IsAllocated()) {
            cache.send_bnd_info_h(ibuf) = BndInfo::GetSendCCFluxCor(pmb, nb, v);
            auto &buf = *cache.send_buf_vec[ibuf];
            cache.send_bnd_info_h(ibuf).buf = buf.buffer();
          }
          ++ibound;
        });
    Kokkos::deep_copy(cache.send_bnd_info, cache.send_bnd_info_h);
  }
  
  auto &bnd_info = cache.send_bnd_info;
  PARTHENON_REQUIRE(bnd_info.size() == nbound, "Need same size for boundary info");
  printf("nbound: %i\n", nbound);
  Kokkos::parallel_for(
      "SendFluxCorrectionBufs",
      Kokkos::TeamPolicy<>(parthenon::DevExecSpace(), nbound, Kokkos::AUTO),
      KOKKOS_LAMBDA(parthenon::team_mbr_t team_member) {
        auto& binfo = bnd_info(team_member.league_rank());
        if (!binfo.allocated) return;
        auto& coords = binfo.coords;

        const int Ni = binfo.ei + 1 - binfo.si;
        const int Nj = binfo.ej + 1 - binfo.sj;
        const int Nk = binfo.ek + 1 - binfo.sk;

        const int &Nt = binfo.Nt;
        const int &Nu = binfo.Nu;
        const int &Nv = binfo.Nv;

        const int NjNi = Nj * Ni;
        const int NkNjNi = Nk * NjNi;
        const int NvNkNjNi = Nv * NkNjNi;
        const int NuNvNkNjNi = Nu * NvNkNjNi;
        const int NtNuNvNkNjNi = Nt * NuNvNkNjNi;

        int ioff = binfo.dir == X1DIR ? 0 : 1;
        int joff = (binfo.dir == X2DIR) || (ndim < 2) ? 0 : 1;
        int koff = (binfo.dir == X3DIR) || (ndim < 3) ? 0 : 1;

        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team_member, NtNuNvNkNjNi),
                             [&](const int idx) {
                               const int t = idx / NuNvNkNjNi;
                               const int u = (idx % NuNvNkNjNi) / NvNkNjNi;
                               const int v = (idx % NvNkNjNi) / NkNjNi;
                               const int ck = (idx % NkNjNi) / NjNi;
                               const int cj = (idx % NjNi) / Ni;
                               const int ci = idx % Ni;
                      
                               const int k = binfo.sk + 2 * ck;
                               const int j = binfo.sj + 2 * cj;
                               const int i = binfo.si + 2 * ci;

                               // For the given set of offsets, etc. this should work for any
                               // dimensionality since the same flux will be included multiple times
                               // in the average
                               const Real area00 = coords.Area(binfo.dir, k, j, i);
                               const Real area01 = coords.Area(binfo.dir, k, j + joff, i + ioff);
                               const Real area10 = coords.Area(binfo.dir, k + koff, j + joff, i);
                               const Real area11 = coords.Area(binfo.dir, k + koff, j, i + ioff);

                               Real avg_flx = area00 * binfo.var(t, u, v, k, j, i);
                               avg_flx += area01 * binfo.var(t, u, v, k + koff, j + joff, i);
                               avg_flx += area10 * binfo.var(t, u, v, k, j + joff, i + ioff);
                               avg_flx += area11 * binfo.var(t, u, v, k + koff, j, i + ioff);

                               avg_flx /= area00 + area01 + area10 + area11;
                               binfo.buf(idx) = avg_flx; 
                             });
      });
#ifdef MPI_PARALLEL
  Kokkos::fence();
#endif
  // Calling Send will send null if the underlying buffer is unallocated
  for (auto& buf : cache.send_buf_vec) buf->Send();
  Kokkos::Profiling::popRegion(); // Task_SetFluxCorrections
  return TaskStatus::complete;
}

TaskStatus StartReceiveFluxCorrections(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::Profiling::pushRegion("Task_ReceiveFluxCorrections");
  Mesh *pmesh = md->GetMeshPointer();
  auto &cache = md->GetBvarsCache()[BoundaryType::flxcor_recv];
  if (cache.recv_buf_vec.size() == 0)
    BuildBufferCache<BoundaryType::flxcor_recv>(md, &(pmesh->boundary_comm_flxcor_map), 
                                 &(cache.recv_buf_vec), &(cache.recv_idx_vec), ReceiveKey);

  std::for_each(std::begin(cache.recv_buf_vec), std::end(cache.recv_buf_vec),
                [](auto pbuf) { pbuf->TryStartReceive(); });

  Kokkos::Profiling::popRegion(); // Task_ReceiveFluxCorrections
  return TaskStatus::complete;
}

TaskStatus ReceiveFluxCorrections(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::Profiling::pushRegion("Task_ReceiveFluxCorrections");
  bool all_received = true;
  Mesh *pmesh = md->GetMeshPointer();
  ForEachBoundary<BoundaryType::flxcor_recv>(
      md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb, const sp_cv_t v) {
        PARTHENON_DEBUG_REQUIRE(
            pmesh->boundary_comm_flxcor_map.count(ReceiveKey(pmb, nb, v)) > 0,
            "Boundary communicator does not exist");
        auto &buf = pmesh->boundary_comm_flxcor_map[ReceiveKey(pmb, nb, v)];
        all_received = buf.TryReceive() && all_received;
      });

  Kokkos::Profiling::popRegion(); // Task_ReceiveFluxCorrections

  if (all_received) return TaskStatus::complete;
  return TaskStatus::incomplete;
}

TaskStatus SetFluxCorrections(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::Profiling::pushRegion("Task_SetFluxCorrections");

  Mesh *pmesh = md->GetMeshPointer();

  ForEachBoundary<BoundaryType::flxcor_recv>(
      md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb, const sp_cv_t v) {
        PARTHENON_DEBUG_REQUIRE(
            pmesh->boundary_comm_flxcor_map.count(ReceiveKey(pmb, nb, v)) > 0,
            "Boundary communicator does not exist");
        auto &buf = pmesh->boundary_comm_flxcor_map[ReceiveKey(pmb, nb, v)];

        // Check if this boundary requires flux correction
        if ((!v->IsAllocated()) || buf.GetState() == BufferState::received_null) {
          buf.Stale();
          return LoopControl::cont;
        }
        
        auto binfo = BndInfo::GetSetCCFluxCor(pmb, nb, v);

        binfo.buf = buf.buffer();
        const int nl = binfo.Nt; 
        const int nm = binfo.Nu; 
        const int nn = binfo.Nv; 
        const int nk = binfo.ek - binfo.sk + 1;
        const int nj = binfo.ej - binfo.sj + 1;
        const int ni = binfo.ei - binfo.si + 1;
   
        const int NjNi = nj * ni;
        const int NkNjNi = nk * NjNi;
        const int NnNkNjNi = nn * NkNjNi;
        const int NmNnNkNjNi = nm * NnNkNjNi;

        Kokkos::parallel_for(
            "SetFluxCorrections",
            Kokkos::RangePolicy<>(parthenon::DevExecSpace(), 0, nl * NmNnNkNjNi),
            KOKKOS_LAMBDA(const int loop_idx) {
              const int l = loop_idx / NmNnNkNjNi;
              const int m = (loop_idx % NmNnNkNjNi) / NnNkNjNi;
              const int n = (loop_idx % NnNkNjNi) / NkNjNi;
              const int k = (loop_idx % NkNjNi) / NjNi + binfo.sk;
              const int j = (loop_idx % NjNi) / ni + binfo.sj;
              const int i = loop_idx % ni + binfo.si;

              const int idx =
                  i - binfo.si + ni * (j - binfo.sj + nj * (k - binfo.sk + nk * (n + nn * (m + nm * l))));
              binfo.var(l, m, n, k, j, i) = binfo.buf(idx);
            });
        buf.Stale();
        return LoopControl::cont;
      });

  Kokkos::Profiling::popRegion(); // Task_SetFluxCorrections
  return TaskStatus::complete;
}

} // namespace cell_centered_bvars
} // namespace parthenon
