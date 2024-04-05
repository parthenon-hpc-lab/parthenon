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
#include "bvals_in_one.hpp"
#include "bvals_utils.hpp"
#include "config.hpp"
#include "globals.hpp"
#include "interface/variable.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock.hpp"
#include "prolong_restrict/prolong_restrict.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {
using namespace impl;

static std::array<std::mutex, 2 * NUM_BNDRY_TYPES> mutex;

TaskStatus LoadAndSendFluxCorrections(std::shared_ptr<MeshData<Real>> &md) {
  PARTHENON_INSTRUMENT

  int mutex_id = 2 * static_cast<int>(BoundaryType::flxcor_send) + 1;  
  mutex[mutex_id].lock();

  Mesh *pmesh = md->GetMeshPointer();
  auto &cache = md->GetBvarsCache().GetSubCache(BoundaryType::flxcor_send, true);
  const int ndim = pmesh->ndim;

  if (cache.buf_vec.size() == 0)
    InitializeBufferCache<BoundaryType::flxcor_send>(
        md, &(pmesh->boundary_comm_flxcor_map), &cache, SendKey, false);

  auto [rebuild, nbound, other_communication_unfinished] =
      CheckSendBufferCacheForRebuild<BoundaryType::flxcor_send, true>(md);

  if (nbound == 0) {
    mutex[mutex_id].unlock();
    return TaskStatus::complete;
  }

  if (other_communication_unfinished) {
    mutex[mutex_id].unlock();
    return TaskStatus::incomplete;
  }

  if (rebuild)
    RebuildBufferCache<BoundaryType::flxcor_send, true>(
        md, nbound, BndInfo::GetSendCCFluxCor, ProResInfo::GetSend);
  mutex[mutex_id].unlock();

  auto &bnd_info = cache.bnd_info;
  PARTHENON_REQUIRE(bnd_info.size() == nbound, "Need same size for boundary info");
  Kokkos::parallel_for(
      PARTHENON_AUTO_LABEL,
      Kokkos::TeamPolicy<>(parthenon::DevExecSpace(), nbound, Kokkos::AUTO),
      KOKKOS_LAMBDA(parthenon::team_mbr_t team_member) {
        auto &binfo = bnd_info(team_member.league_rank());
        if (!binfo.allocated) return;
        auto &coords = binfo.coords;

        int ioff = binfo.dir == X1DIR ? 0 : 1;
        int joff = (binfo.dir == X2DIR) || (ndim < 2) ? 0 : 1;
        int koff = (binfo.dir == X3DIR) || (ndim < 3) ? 0 : 1;
        auto &idxer = binfo.idxer[0];
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange<>(team_member, idxer.size()), [&](const int idx) {
              const auto [t, u, v, ck, cj, ci] = idxer(idx);
              // Move from a coarse grid index to the corresponding lower left fine grid
              // index, StartIdx<I>() should return the index of the first non-ghost zone
              // in direction I
              const int k =
                  idxer.template StartIdx<3>() + 2 * (ck - idxer.template StartIdx<3>());
              const int j =
                  idxer.template StartIdx<4>() + 2 * (cj - idxer.template StartIdx<4>());
              const int i =
                  idxer.template StartIdx<5>() + 2 * (ci - idxer.template StartIdx<5>());

              // For the given set of offsets, etc. this should work for any
              // dimensionality since the same flux will be included multiple times
              // in the average
              const Real area00 = coords.FaceAreaFA(binfo.dir, k, j, i);
              const Real area01 = coords.FaceAreaFA(binfo.dir, k, j + joff, i + ioff);
              const Real area10 = coords.FaceAreaFA(binfo.dir, k + koff, j + joff, i);
              const Real area11 = coords.FaceAreaFA(binfo.dir, k + koff, j, i + ioff);

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
  return TaskStatus::complete;
}

TaskStatus StartReceiveFluxCorrections(std::shared_ptr<MeshData<Real>> &md) {
  PARTHENON_INSTRUMENT

  int mutex_id = 2 * static_cast<int>(BoundaryType::flxcor_recv);
  mutex[mutex_id].lock();

  Mesh *pmesh = md->GetMeshPointer();
  auto &cache = md->GetBvarsCache().GetSubCache(BoundaryType::flxcor_recv, false);
  if (cache.buf_vec.size() == 0)
    InitializeBufferCache<BoundaryType::flxcor_recv>(
        md, &(pmesh->boundary_comm_flxcor_map), &cache, ReceiveKey, false);
  mutex[mutex_id].unlock();

  std::for_each(std::begin(cache.buf_vec), std::end(cache.buf_vec),
                [](auto pbuf) { pbuf->TryStartReceive(); });

  return TaskStatus::complete;
}

TaskStatus ReceiveFluxCorrections(std::shared_ptr<MeshData<Real>> &md) {
  PARTHENON_INSTRUMENT

  int mutex_id = 2 * static_cast<int>(BoundaryType::flxcor_recv);
  mutex[mutex_id].lock();

  Mesh *pmesh = md->GetMeshPointer();
  auto &cache = md->GetBvarsCache().GetSubCache(BoundaryType::flxcor_recv, false);
  if (cache.buf_vec.size() == 0)
    InitializeBufferCache<BoundaryType::flxcor_recv>(
        md, &(pmesh->boundary_comm_flxcor_map), &cache, ReceiveKey, false);
  mutex[mutex_id].unlock();

  bool all_received = true;
  std::for_each(
      std::begin(cache.buf_vec), std::end(cache.buf_vec),
      [&all_received](auto pbuf) { all_received = pbuf->TryReceive() && all_received; });

  if (all_received) return TaskStatus::complete;
  return TaskStatus::incomplete;
}

TaskStatus SetFluxCorrections(std::shared_ptr<MeshData<Real>> &md) {
  PARTHENON_INSTRUMENT

  int mutex_id = 2 * static_cast<int>(BoundaryType::flxcor_recv);
  mutex[mutex_id].lock();

  Mesh *pmesh = md->GetMeshPointer();
  auto &cache = md->GetBvarsCache().GetSubCache(BoundaryType::flxcor_recv, false);

  auto [rebuild, nbound] =
      CheckReceiveBufferCacheForRebuild<BoundaryType::flxcor_recv, false>(md);
  if (rebuild)
    RebuildBufferCache<BoundaryType::flxcor_recv, false>(
        md, nbound, BndInfo::GetSetCCFluxCor, ProResInfo::GetSend);
  mutex[mutex_id].unlock();

  auto &bnd_info = cache.bnd_info;
  Kokkos::parallel_for(
      PARTHENON_AUTO_LABEL,
      Kokkos::TeamPolicy<>(parthenon::DevExecSpace(), nbound, Kokkos::AUTO),
      KOKKOS_LAMBDA(parthenon::team_mbr_t team_member) {
        const int b = team_member.league_rank();
        if (!bnd_info(b).allocated) return;

        auto &idxer = bnd_info(b).idxer[0];
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team_member, idxer.size()),
                             [&](const int idx) {
                               const auto [t, u, v, k, j, i] = idxer(idx);
                               bnd_info(b).var(t, u, v, k, j, i) = bnd_info(b).buf(idx);
                             });
      });
#ifdef MPI_PARALLEL
  Kokkos::fence();
#endif
  std::for_each(std::begin(cache.buf_vec), std::end(cache.buf_vec),
                [](auto pbuf) { pbuf->Stale(); });

  return TaskStatus::complete;
}

} // namespace parthenon
