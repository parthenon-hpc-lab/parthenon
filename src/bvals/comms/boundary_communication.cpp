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
#include "bvals/boundary_conditions.hpp"
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
#include "tasks/task_id.hpp"
#include "tasks/task_list.hpp"
#include "utils/error_checking.hpp"
#include "utils/loop_utils.hpp"

namespace parthenon {

using namespace loops;
using namespace loops::shorthands;

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
          Kokkos::single(Kokkos::PerTeam(team_member),
                         [&]() { sending_nonzero_flags(b) = false; });
          return;
        }
        Real threshold = bnd_info(b).var.allocation_threshold;
        bool non_zero[3]{false, false, false};
        for (int iel = 0; iel < bnd_info(b).ntopological_elements; ++iel) {
          auto &idxer = bnd_info(b).idxer[iel];
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange<>(team_member, idxer.size()),
              [&](const int idx, bool &lnon_zero) {
                const auto [t, u, v, k, j, i] = idxer(idx);
                const Real &val = bnd_info(b).var(t, u, v, k, j, i);
                bnd_info(b).buf(idx) = val;
                lnon_zero = lnon_zero || (std::abs(val) >= threshold);
              },
              Kokkos::LOr<bool, parthenon::DevMemSpace>(non_zero[iel]));
        }
        Kokkos::single(Kokkos::PerTeam(team_member), [&]() {
          sending_nonzero_flags(b) = non_zero[0] || non_zero[1] || non_zero[2];
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
        md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb, const sp_cv_t v) {
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
        for (int iel = 0; iel < bnd_info(b).ntopological_elements; ++iel) {
          auto &idxer = bnd_info(b).idxer[iel];
          if (bnd_info(b).allocated) {
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange<>(team_member, idxer.size()), [&](const int idx) {
                  const auto [t, u, v, k, j, i] = idxer(idx);
                  bnd_info(b).var(t, u, v, k, j, i) = bnd_info(b).buf(idx);
                });
          } else if (bnd_info(b).var.size() > 0) {
            const Real default_val = bnd_info(b).var.sparse_default_val;
            Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team_member, idxer.size()),
                                 [&](const int idx) {
                                   const auto [t, u, v, k, j, i] = idxer(idx);
                                   bnd_info(b).var(t, u, v, k, j, i) = default_val;
                                 });
          }
        }
      });
#ifdef MPI_PARALLEL
  Kokkos::fence();
#endif
  std::for_each(std::begin(cache.buf_vec), std::end(cache.buf_vec),
                [](auto pbuf) { pbuf->Stale(); });
  if (nbound > 0 && pmesh->multilevel) {
    // Restrict
    auto pmb = md->GetBlockData(0)->GetBlockPointer();
    StateDescriptor *resolved_packages = pmb->resolved_packages.get();
    refinement::Restrict(resolved_packages, cache, pmb->cellbounds, pmb->c_cellbounds);

    // Apply coarse boundary conditions
    for (int block = 0; block < md->NumBlocks(); ++block) {
      ApplyBoundaryConditionsOnCoarseOrFine(md->GetBlockData(block), true);
    }

    // Prolongate from coarse buffer
    refinement::Prolongate(resolved_packages, cache, pmb->cellbounds, pmb->c_cellbounds);
  }
  Kokkos::Profiling::popRegion(); // Task_SetInternalBoundaries
  return TaskStatus::complete;
}

template TaskStatus SetBounds<BoundaryType::any>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus SetBounds<BoundaryType::local>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus SetBounds<BoundaryType::nonlocal>(std::shared_ptr<MeshData<Real>> &);

template <BoundaryType bound_type>
TaskStatus ProlongateBounds(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::Profiling::pushRegion("Task_ProlongateBoundaries");

  Mesh *pmesh = md->GetMeshPointer();
  auto &cache = md->GetBvarsCache().GetSubCache(bound_type, false);

  auto [rebuild, nbound] = CheckReceiveBufferCacheForRebuild<bound_type, false>(md);
  if (rebuild) RebuildBufferCache<bound_type, false>(md, nbound, BndInfo::GetSetBndInfo);
  if (nbound > 0 && pmesh->multilevel) {
    auto pmb = md->GetBlockData(0)->GetBlockPointer();
    StateDescriptor *resolved_packages = pmb->resolved_packages.get();

    // Prolongate from coarse buffer
    refinement::Prolongate(resolved_packages, cache, pmb->cellbounds, pmb->c_cellbounds);
  }
  Kokkos::Profiling::popRegion(); // Task_ProlongateBoundaries
  return TaskStatus::complete;
}

template TaskStatus
ProlongateBounds<BoundaryType::any>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus
ProlongateBounds<BoundaryType::local>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus
ProlongateBounds<BoundaryType::nonlocal>(std::shared_ptr<MeshData<Real>> &);

TaskStatus ApplyCoarseBoundaryConditions(std::shared_ptr<MeshData<Real>> &md) {
  if (!md->GetMeshPointer()->multilevel) return TaskStatus::complete;
  TaskStatus stat = TaskStatus::complete;
  for (int block = 0; block < md->NumBlocks(); ++block) {
    auto bstat = ApplyBoundaryConditionsOnCoarseOrFine(md->GetBlockData(block), true);
    // if (bstat != TaskStatus::complete) stat = bstat;
  }
  return stat;
}

// Adds all relevant boundary communication to a single task list
TaskID AddBoundaryExchangeTasks(TaskID dependency, TaskList &tl,
                                std::shared_ptr<MeshData<Real>> &md, bool multilevel) {
  const auto any = BoundaryType::any;
  const auto local = BoundaryType::local;
  const auto nonlocal = BoundaryType::nonlocal;

  // auto send = tl.AddTask(dependency, SendBoundBufs<nonlocal>, md);
  // auto send_local = tl.AddTask(dependency, SendBoundBufs<local>, md);

  // auto recv_local = tl.AddTask(dependency, ReceiveBoundBufs<local>, md);
  // auto set_local = tl.AddTask(recv_local, SetBounds<local>, md);

  // auto recv = tl.AddTask(dependency, ReceiveBoundBufs<nonlocal>, md);
  // auto set = tl.AddTask(recv, SetBounds<nonlocal>, md);

  // auto cbound = tl.AddTask(set, ApplyCoarseBoundaryConditions, md);

  // auto pro_local = tl.AddTask(cbound | set_local | set, ProlongateBounds<local>, md);
  // auto pro = tl.AddTask(cbound | set_local | set, ProlongateBounds<nonlocal>, md);

  // auto out = (pro_local | pro);

  // TODO(LFR): Splitting up the boundary tasks while doing prolongation in one
  //            breaks things, which I haven't completely figured out yet. To
  //            move to the commented out code above, this needs to be fixed and
  //            the prolongation and physical boundary conditions need to be removed
  //            from the end of SetBounds
  auto send = tl.AddTask(dependency, SendBoundBufs<any>, md);
  auto recv = tl.AddTask(dependency, ReceiveBoundBufs<any>, md);
  auto set = tl.AddTask(recv, SetBounds<any>, md);

  return set;
}
} // namespace parthenon
