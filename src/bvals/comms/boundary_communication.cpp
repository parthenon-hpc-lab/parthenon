//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2022 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2022-2024. Triad National Security, LLC. All rights reserved.
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
#include <cstdio>
#include <iostream> // debug
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "bnd_info.hpp"
#include "bvals/boundary_conditions.hpp"
#include "bvals_in_one.hpp"
#include "bvals_utils.hpp"
#include "combined_buffers.hpp"
#include "config.hpp"
#include "globals.hpp"
#include "interface/variable.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/forest/logical_coordinate_transformation.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock.hpp"
#include "prolong_restrict/prolong_restrict.hpp"

#include "tasks/tasks.hpp"
#include "utils/error_checking.hpp"
#include "utils/loop_utils.hpp"

namespace parthenon {

using namespace loops;
using namespace loops::shorthands;

template <BoundaryType bound_type>
TaskStatus SendBoundBufs(std::shared_ptr<MeshData<Real>> &md) {
  PARTHENON_INSTRUMENT

  Mesh *pmesh = md->GetMeshPointer();
  auto &cache = md->GetBvarsCache().GetSubCache(bound_type, true);

  if (cache.buf_vec.size() == 0)
    InitializeBufferCache<bound_type>(md, &(pmesh->boundary_comm_map), &cache, SendKey,
                                      true);

  auto [rebuild, nbound, other_communication_unfinished] =
      CheckSendBufferCacheForRebuild<bound_type, true>(md);

  if (nbound == 0) {
    return TaskStatus::complete;
  }

  bool can_write_combined =
      pmesh->pcombined_buffers->IsAvailableForWrite(md->partition, bound_type);
  if (other_communication_unfinished || !can_write_combined) {
    return TaskStatus::incomplete;
  }

  if (rebuild) {
    if constexpr (bound_type == BoundaryType::gmg_restrict_send) {
      RebuildBufferCache<bound_type, true>(md, nbound, BndInfo::GetSendBndInfo,
                                           ProResInfo::GetInteriorRestrict);
    } else if constexpr (bound_type == BoundaryType::gmg_prolongate_send) {
      RebuildBufferCache<bound_type, true>(md, nbound, BndInfo::GetSendBndInfo,
                                           ProResInfo::GetNull);
    } else {
      RebuildBufferCache<bound_type, true>(md, nbound, BndInfo::GetSendBndInfo,
                                           ProResInfo::GetSend);
    }
    pmesh->pcombined_buffers->RepointSendBuffers(pmesh, md->partition, bound_type);
  }
  // Restrict
  if (md->NumBlocks() > 0) {
    auto pmb = md->GetBlockData(0)->GetBlockPointer();
    StateDescriptor *resolved_packages = pmb->resolved_packages.get();
    refinement::Restrict(resolved_packages, cache.prores_cache, pmb->cellbounds,
                         pmb->c_cellbounds);
  }

  // Load buffer data
  auto &bnd_info = cache.bnd_info;
  PARTHENON_DEBUG_REQUIRE(bnd_info.size() == nbound, "Need same size for boundary info");
  auto &sending_nonzero_flags = cache.sending_non_zero_flags;
  auto &sending_nonzero_flags_h = cache.sending_non_zero_flags_h;

  Kokkos::parallel_for(
      PARTHENON_AUTO_LABEL,
      Kokkos::TeamPolicy<>(parthenon::DevExecSpace(), nbound, Kokkos::AUTO),
      KOKKOS_LAMBDA(parthenon::team_mbr_t team_member) {
        const int b = team_member.league_rank();

        if (!bnd_info(b).allocated || bnd_info(b).same_to_same) {
          Kokkos::single(Kokkos::PerTeam(team_member),
                         [&]() { sending_nonzero_flags(b) = false; });
          return;
        }
        Real threshold = bnd_info(b).var.allocation_threshold;
        bool non_zero[3]{false, false, false};
        int idx_offset = 0;
        for (int it = 0; it < bnd_info(b).ntopological_elements; ++it) {
          auto &idxer = bnd_info(b).idxer[it];
          const int iel = static_cast<int>(bnd_info(b).topo_idx[it]) % 3;
          const int Ni = idxer.template EndIdx<5>() - idxer.template StartIdx<5>() + 1;
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange<>(team_member, idxer.size() / Ni),
              [&](const int idx, bool &lnon_zero) {
                const auto [t, u, v, k, j, i] = idxer(idx * Ni);
                Real *var = &bnd_info(b).var(iel, t, u, v, k, j, i);
                Real *buf = &bnd_info(b).buf(idx * Ni + idx_offset);

                Kokkos::parallel_for(Kokkos::ThreadVectorRange<>(team_member, Ni),
                                     [&](int m) { buf[m] = var[m]; });

                bool mnon_zero = false;
                Kokkos::parallel_reduce(
                    Kokkos::ThreadVectorRange<>(team_member, Ni),
                    [&](int m, bool &llnon_zero) {
                      llnon_zero = llnon_zero || (std::abs(buf[m]) >= threshold);
                    },
                    Kokkos::LOr<bool, parthenon::DevMemSpace>(mnon_zero));

                lnon_zero = lnon_zero || mnon_zero;
                if (bound_type == BoundaryType::flxcor_send) lnon_zero = true;
              },
              Kokkos::LOr<bool, parthenon::DevMemSpace>(non_zero[iel]));
          idx_offset += idxer.size();
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

  // Send the combined buffers
  pmesh->pcombined_buffers->PackAndSend(md->partition, bound_type);

  for (int ibuf = 0; ibuf < cache.buf_vec.size(); ++ibuf) {
    auto &buf = *cache.buf_vec[ibuf];
    if (sending_nonzero_flags_h(ibuf) || !Globals::sparse_config.enabled)
      buf.SendLocal();
    else
      buf.SendNull();
  }

  return TaskStatus::complete;
}

template TaskStatus SendBoundBufs<BoundaryType::any>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus SendBoundBufs<BoundaryType::local>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus
SendBoundBufs<BoundaryType::nonlocal>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus
SendBoundBufs<BoundaryType::gmg_restrict_send>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus
SendBoundBufs<BoundaryType::gmg_prolongate_send>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus
SendBoundBufs<BoundaryType::flxcor_send>(std::shared_ptr<MeshData<Real>> &);

template <BoundaryType bound_type>
TaskStatus StartReceiveBoundBufs(std::shared_ptr<MeshData<Real>> &md) {
  PARTHENON_INSTRUMENT
  Mesh *pmesh = md->GetMeshPointer();
  auto &cache = md->GetBvarsCache().GetSubCache(bound_type, false);
  if (cache.buf_vec.size() == 0)
    InitializeBufferCache<bound_type>(md, &(pmesh->boundary_comm_map), &cache, ReceiveKey,
                                      false);

  // std::for_each(std::begin(cache.buf_vec), std::end(cache.buf_vec),
  //               [](auto pbuf) { pbuf->TryStartReceive(); });

  return TaskStatus::complete;
}

template TaskStatus
StartReceiveBoundBufs<BoundaryType::any>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus
StartReceiveBoundBufs<BoundaryType::local>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus
StartReceiveBoundBufs<BoundaryType::nonlocal>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus
StartReceiveBoundBufs<BoundaryType::gmg_restrict_recv>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus StartReceiveBoundBufs<BoundaryType::gmg_prolongate_recv>(
    std::shared_ptr<MeshData<Real>> &);
template TaskStatus
StartReceiveBoundBufs<BoundaryType::flxcor_recv>(std::shared_ptr<MeshData<Real>> &);

template <BoundaryType bound_type>
TaskStatus ReceiveBoundBufs(std::shared_ptr<MeshData<Real>> &md) {
  PARTHENON_INSTRUMENT

  static int ntotal_prints{0};

  Mesh *pmesh = md->GetMeshPointer();
  auto &cache = md->GetBvarsCache().GetSubCache(bound_type, false);
  if (cache.buf_vec.size() == 0)
    InitializeBufferCache<bound_type>(md, &(pmesh->boundary_comm_map), &cache, ReceiveKey,
                                      false);

  // Receive any messages that are around
  pmesh->pcombined_buffers->TryReceiveAny(pmesh, bound_type);

  bool all_received = true;
  int nreceived{0};
  std::for_each(std::begin(cache.buf_vec), std::end(cache.buf_vec),
                [&all_received, &nreceived](auto pbuf) {
                  all_received = pbuf->TryReceiveLocal() && all_received;
                  nreceived += pbuf->TryReceiveLocal();
                });
  // if (ntotal_prints++ < 1000)
  //   printf("rank = %i partition = %i nreceived = %i (%i)\n", Globals::my_rank,
  //          md->partition, nreceived, cache.buf_vec.size());

  int ibound = 0;
  if (Globals::sparse_config.enabled && all_received) {
    ForEachBoundary<bound_type>(
        md, [&](auto pmb, sp_mbd_t rc, nb_t &nb, const sp_cv_t v) {
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
  if (all_received) {
    MPI_Barrier(MPI_COMM_WORLD);
    return TaskStatus::complete;
  }
  return TaskStatus::incomplete;
}

template TaskStatus
ReceiveBoundBufs<BoundaryType::any>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus
ReceiveBoundBufs<BoundaryType::local>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus
ReceiveBoundBufs<BoundaryType::nonlocal>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus
ReceiveBoundBufs<BoundaryType::gmg_restrict_recv>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus
ReceiveBoundBufs<BoundaryType::gmg_prolongate_recv>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus
ReceiveBoundBufs<BoundaryType::flxcor_recv>(std::shared_ptr<MeshData<Real>> &);

template <BoundaryType bound_type>
TaskStatus SetBounds(std::shared_ptr<MeshData<Real>> &md) {
  PARTHENON_INSTRUMENT

  Mesh *pmesh = md->GetMeshPointer();
  auto &cache = md->GetBvarsCache().GetSubCache(bound_type, false);

  auto [rebuild, nbound] = CheckReceiveBufferCacheForRebuild<bound_type, false>(md);

  if (rebuild) {
    if constexpr (bound_type == BoundaryType::gmg_prolongate_recv) {
      RebuildBufferCache<bound_type, false>(md, nbound, BndInfo::GetSetBndInfo,
                                            ProResInfo::GetInteriorProlongate);
    } else if constexpr (bound_type == BoundaryType::gmg_restrict_recv) {
      RebuildBufferCache<bound_type, false>(md, nbound, BndInfo::GetSetBndInfo,
                                            ProResInfo::GetNull);
    } else {
      RebuildBufferCache<bound_type, false>(md, nbound, BndInfo::GetSetBndInfo,
                                            ProResInfo::GetSet);
    }
  }
  // const Real threshold = Globals::sparse_config.allocation_threshold;
  auto &bnd_info = cache.bnd_info;
  Kokkos::parallel_for(
      PARTHENON_AUTO_LABEL,
      Kokkos::TeamPolicy<>(parthenon::DevExecSpace(), nbound, Kokkos::AUTO),
      KOKKOS_LAMBDA(parthenon::team_mbr_t team_member) {
        const int b = team_member.league_rank();
        if (bnd_info(b).same_to_same) return;
        int idx_offset = 0;
        for (int it = 0; it < bnd_info(b).ntopological_elements; ++it) {
          auto &idxer = bnd_info(b).idxer[it];
          auto &lcoord_trans = bnd_info(b).lcoord_trans;
          auto &var = bnd_info(b).var;
          const auto [tel, ftemp] =
              lcoord_trans.InverseTransform(bnd_info(b).topo_idx[it]);
          Real fac = ftemp; // Can't capture structured bindings
          const int iel = static_cast<int>(tel) % 3;
          const int Ni = idxer.template EndIdx<5>() - idxer.template StartIdx<5>() + 1;
          if (bnd_info(b).buf_allocated && bnd_info(b).allocated) {
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange<>(team_member, idxer.size() / Ni),
                [&](const int idx) {
                  Real *buf = &bnd_info(b).buf(idx * Ni + idx_offset);
                  const auto [t, u, v, k, j, i] = idxer(idx * Ni);
                  // Have to do this because of some weird issue about structure bindings
                  // being captured
                  const int tt = t;
                  const int uu = u;
                  const int vv = v;
                  const int kk = k;
                  const int jj = j;
                  const int ii = i;
                  Kokkos::parallel_for(
                      Kokkos::ThreadVectorRange<>(team_member, Ni), [&](int m) {
                        const auto [il, jl, kl] =
                            lcoord_trans.InverseTransform({ii + m, jj, kk});
                        if (idxer.IsActive(kl, jl, il))
                          var(iel, tt, uu, vv, kl, jl, il) = fac * buf[m];
                      });
                });
          } else if (bnd_info(b).allocated && bound_type != BoundaryType::flxcor_recv) {
            const Real default_val = bnd_info(b).var.sparse_default_val;
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange<>(team_member, idxer.size() / Ni),
                [&](const int idx) {
                  const auto [t, u, v, k, j, i] = idxer(idx * Ni);
                  const int tt = t;
                  const int uu = u;
                  const int vv = v;
                  const int kk = k;
                  const int jj = j;
                  const int ii = i;
                  Kokkos::parallel_for(
                      Kokkos::ThreadVectorRange<>(team_member, Ni), [&](int m) {
                        const auto [il, jl, kl] =
                            lcoord_trans.InverseTransform({ii + m, jj, kk});
                        if (idxer.IsActive(kl, jl, il))
                          var(iel, tt, uu, vv, kl, jl, il) = default_val;
                      });
                });
          }
          idx_offset += idxer.size();
        }
      });
#ifdef MPI_PARALLEL
  Kokkos::fence();
#endif
  std::for_each(std::begin(cache.buf_vec), std::end(cache.buf_vec),
                [](auto pbuf) { pbuf->Stale(); });
  if (nbound > 0 && pmesh->multilevel && md->NumBlocks() > 0) {
    // Restrict
    auto pmb = md->GetBlockData(0)->GetBlockPointer();
    StateDescriptor *resolved_packages = pmb->resolved_packages.get();
    refinement::Restrict(resolved_packages, cache.prores_cache, pmb->cellbounds,
                         pmb->c_cellbounds);
  }
  return TaskStatus::complete;
}

template TaskStatus SetBounds<BoundaryType::any>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus SetBounds<BoundaryType::local>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus SetBounds<BoundaryType::nonlocal>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus
SetBounds<BoundaryType::gmg_restrict_recv>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus
SetBounds<BoundaryType::gmg_prolongate_recv>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus
SetBounds<BoundaryType::flxcor_recv>(std::shared_ptr<MeshData<Real>> &);

template <BoundaryType bound_type>
TaskStatus ProlongateBounds(std::shared_ptr<MeshData<Real>> &md) {
  PARTHENON_INSTRUMENT

  Mesh *pmesh = md->GetMeshPointer();
  auto &cache = md->GetBvarsCache().GetSubCache(bound_type, false);

  auto [rebuild, nbound] = CheckReceiveBufferCacheForRebuild<bound_type, false>(md);

  if (rebuild) {
    if constexpr (bound_type == BoundaryType::gmg_prolongate_recv) {
      RebuildBufferCache<bound_type, false>(md, nbound, BndInfo::GetSetBndInfo,
                                            ProResInfo::GetInteriorProlongate);
    } else if constexpr (bound_type == BoundaryType::gmg_restrict_recv) {
      RebuildBufferCache<bound_type, false>(md, nbound, BndInfo::GetSetBndInfo,
                                            ProResInfo::GetNull);
    } else {
      RebuildBufferCache<bound_type, false>(md, nbound, BndInfo::GetSetBndInfo,
                                            ProResInfo::GetSet);
    }
  }

  if (nbound > 0 && pmesh->multilevel && md->NumBlocks() > 0) {
    auto pmb = md->GetBlockData(0)->GetBlockPointer();
    StateDescriptor *resolved_packages = pmb->resolved_packages.get();

    // Prolongate from coarse buffer
    refinement::ProlongateShared(resolved_packages, cache.prores_cache, pmb->cellbounds,
                                 pmb->c_cellbounds);
    refinement::ProlongateInternal(resolved_packages, cache.prores_cache, pmb->cellbounds,
                                   pmb->c_cellbounds);
  }
  return TaskStatus::complete;
}

template TaskStatus
ProlongateBounds<BoundaryType::any>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus
ProlongateBounds<BoundaryType::local>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus
ProlongateBounds<BoundaryType::nonlocal>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus
ProlongateBounds<BoundaryType::gmg_prolongate_recv>(std::shared_ptr<MeshData<Real>> &);

// Adds all relevant boundary communication to a single task list
template <BoundaryType bounds>
TaskID AddBoundaryExchangeTasks(TaskID dependency, TaskList &tl,
                                std::shared_ptr<MeshData<Real>> &md, bool multilevel) {
  // TODO(LFR): Splitting up the boundary tasks while doing prolongation can cause some
  //            possible issues for sparse fields. In particular, the order in which
  //            fields are allocated and then set could potentially result in different
  //            results if the default sparse value is non-zero.
  // const auto any = BoundaryType::any;
  static_assert(bounds == BoundaryType::any || bounds == BoundaryType::gmg_same);
  // const auto local = BoundaryType::local;
  // const auto nonlocal = BoundaryType::nonlocal;

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

  auto send = tl.AddTask(dependency, TF(SendBoundBufs<bounds>), md);
  auto recv = tl.AddTask(dependency, TF(ReceiveBoundBufs<bounds>), md);
  auto set = tl.AddTask(recv, TF(SetBounds<bounds>), md);

  auto pro = set;
  if (md->GetMeshPointer()->multilevel) {
    auto cbound = tl.AddTask(set, TF(ApplyBoundaryConditionsOnCoarseOrFineMD), md, true);
    pro = tl.AddTask(cbound, TF(ProlongateBounds<bounds>), md);
  }
  auto fbound = tl.AddTask(pro, TF(ApplyBoundaryConditionsOnCoarseOrFineMD), md, false);

  return fbound;
}
template TaskID
AddBoundaryExchangeTasks<BoundaryType::any>(TaskID, TaskList &,
                                            std::shared_ptr<MeshData<Real>> &, bool);

template TaskID
AddBoundaryExchangeTasks<BoundaryType::gmg_same>(TaskID, TaskList &,
                                                 std::shared_ptr<MeshData<Real>> &, bool);

TaskID AddFluxCorrectionTasks(TaskID dependency, TaskList &tl,
                              std::shared_ptr<MeshData<Real>> &md, bool multilevel) {
  if (!multilevel) return dependency;
  tl.AddTask(dependency, TF(SendBoundBufs<BoundaryType::flxcor_send>), md);
  auto receive =
      tl.AddTask(dependency, TF(ReceiveBoundBufs<BoundaryType::flxcor_recv>), md);
  return tl.AddTask(receive, TF(SetBounds<BoundaryType::flxcor_recv>), md);
}
} // namespace parthenon
