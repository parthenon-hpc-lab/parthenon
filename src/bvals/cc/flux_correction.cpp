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
#include "mesh/refinement_cc_in_one.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {
namespace cell_centered_bvars {

using namespace impl;

TaskStatus LoadAndSendFluxCorrections(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::Profiling::pushRegion("Task_LoadAndSendFluxCorrections");

  Mesh *pmesh = md->GetMeshPointer();
  auto &cache = md->GetBvarsCache().GetSubCache(BoundaryType::flxcor_send, true);
  const int ndim = pmesh->ndim;

  if (cache.buf_vec.size() == 0) 
    InitializeBufferCache<BoundaryType::flxcor_send>(
        md, &(pmesh->boundary_comm_flxcor_map), &cache, SendKey, false);

  auto [rebuild, nbound, other_communication_unfinished] =
      CheckSendBufferCacheForRebuild<BoundaryType::flxcor_send, true>(md);

  if (nbound == 0) {
    Kokkos::Profiling::popRegion(); // Task_LoadAndSendBoundBufs
    return TaskStatus::complete;
  }

  if (other_communication_unfinished) {
    Kokkos::Profiling::popRegion(); // Task_LoadAndSendBoundBufs
    return TaskStatus::incomplete;
  }

  if (rebuild)
    RebuildBufferCache<BoundaryType::flxcor_send, true>(md, nbound,
                                                        BndInfo::GetSendCCFluxCor);

  auto &bnd_info = cache.bnd_info;
  PARTHENON_REQUIRE(bnd_info.size() == nbound, "Need same size for boundary info");
  printf("nbound: %i\n", nbound);
  Kokkos::parallel_for(
      "SendFluxCorrectionBufs",
      Kokkos::TeamPolicy<>(parthenon::DevExecSpace(), nbound, Kokkos::AUTO),
      KOKKOS_LAMBDA(parthenon::team_mbr_t team_member) {
        auto &binfo = bnd_info(team_member.league_rank());
        if (!binfo.allocated) return;
        auto &coords = binfo.coords;

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

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange<>(team_member, NtNuNvNkNjNi), [&](const int idx) {
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
  for (auto &buf : cache.buf_vec)
    buf->Send();
  Kokkos::Profiling::popRegion(); // Task_SetFluxCorrections
  return TaskStatus::complete;
}

TaskStatus StartReceiveFluxCorrections(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::Profiling::pushRegion("Task_ReceiveFluxCorrections");
  Mesh *pmesh = md->GetMeshPointer();
  auto &cache = md->GetBvarsCache().GetSubCache(BoundaryType::flxcor_recv, false);
  if (cache.buf_vec.size() == 0)
    InitializeBufferCache<BoundaryType::flxcor_recv>(
        md, &(pmesh->boundary_comm_flxcor_map), &cache, ReceiveKey, false);

  std::for_each(std::begin(cache.buf_vec), std::end(cache.buf_vec),
                [](auto pbuf) { pbuf->TryStartReceive(); });

  Kokkos::Profiling::popRegion(); // Task_ReceiveFluxCorrections
  return TaskStatus::complete;
}

TaskStatus ReceiveFluxCorrections(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::Profiling::pushRegion("Task_ReceiveFluxCorrections");

  Mesh *pmesh = md->GetMeshPointer();
  auto &cache = md->GetBvarsCache().GetSubCache(BoundaryType::flxcor_recv, false);
  if (cache.buf_vec.size() == 0)
    InitializeBufferCache<BoundaryType::flxcor_recv>(
        md, &(pmesh->boundary_comm_flxcor_map), &cache, ReceiveKey, false);

  bool all_received = true;
  std::for_each(
      std::begin(cache.buf_vec), std::end(cache.buf_vec),
      [&all_received](auto pbuf) { all_received = pbuf->TryReceive() && all_received; });

  Kokkos::Profiling::popRegion(); // Task_ReceiveFluxCorrections

  if (all_received) return TaskStatus::complete;
  return TaskStatus::incomplete;
}

TaskStatus SetFluxCorrections(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::Profiling::pushRegion("Task_SetFluxCorrections");

  Mesh *pmesh = md->GetMeshPointer();
  auto &cache = md->GetBvarsCache().GetSubCache(BoundaryType::flxcor_recv, false);

  auto [rebuild, nbound] =
      CheckReceiveBufferCacheForRebuild<BoundaryType::flxcor_recv, false>(md);
  if (rebuild)
    RebuildBufferCache<BoundaryType::flxcor_recv, false>(md, nbound,
                                                         BndInfo::GetSetCCFluxCor);

  auto &bnd_info = cache.bnd_info;
  Kokkos::parallel_for(
      "SetFluxCorBuffers",
      Kokkos::TeamPolicy<>(parthenon::DevExecSpace(), nbound, Kokkos::AUTO),
      KOKKOS_LAMBDA(parthenon::team_mbr_t team_member) {
        const int b = team_member.league_rank();
        if (!bnd_info(b).allocated) return;

        const int Ni = bnd_info(b).ei + 1 - bnd_info(b).si;
        const int Nj = bnd_info(b).ej + 1 - bnd_info(b).sj;
        const int Nk = bnd_info(b).ek + 1 - bnd_info(b).sk;
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
                               const int k = (idx % NkNjNi) / NjNi + bnd_info(b).sk;
                               const int j = (idx % NjNi) / Ni + bnd_info(b).sj;
                               const int i = idx % Ni + bnd_info(b).si;

                               bnd_info(b).var(t, u, v, k, j, i) = bnd_info(b).buf(idx);
                             });
      });
#ifdef MPI_PARALLEL
  Kokkos::fence();
#endif
  std::for_each(std::begin(cache.buf_vec), std::end(cache.buf_vec),
                [](auto pbuf) { pbuf->Stale(); });

  Kokkos::Profiling::popRegion(); // Task_SetInternalBoundaries
  return TaskStatus::complete;
}

} // namespace cell_centered_bvars
} // namespace parthenon
