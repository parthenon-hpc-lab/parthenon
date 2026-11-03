//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2022-2026 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2022-2026. Triad National Security, LLC. All rights reserved.
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// AI was used to assist with boundary communication kernel optimization.
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
#include <cstdlib>
#include <cstring>
#include <iostream> // debug
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "bnd_info.hpp"
#include "bvals/boundary_conditions.hpp"
#include "bvals_in_one.hpp"
#include "bvals_utils.hpp"
#include "coalesced_buffers.hpp"
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

namespace {
int BoundaryVectorLength() {
  static const int length = []() {
    const char *value = std::getenv("RIOT_BOUNDARY_VECTOR_LENGTH");
    if (value == nullptr) return 1;
    char *end = nullptr;
    const long n = std::strtol(value, &end, 10);
    PARTHENON_REQUIRE(end != value && *end == '\0' &&
                          (n == 1 || n == 2 || n == 4 || n == 8 || n == 16 || n == 32),
                      "RIOT_BOUNDARY_VECTOR_LENGTH must be a power of two from 1 to 32");
    return static_cast<int>(n);
  }();
  return length;
}

bool SkipDenseScan() {
  const char *value = std::getenv("RIOT_SKIP_DENSE_SCAN");
  return !Globals::sparse_config.enabled && value != nullptr &&
         std::strcmp(value, "1") == 0;
}

int GetNteamsPerBoundaryBuffer(const Mesh *pmesh, const int nbound) {
  PARTHENON_REQUIRE_THROWS(
      nbound > 0,
      "Cannot calculate the number of teams per boundary buffer without boundaries.");
  return std::max(1, (pmesh->minimum_number_of_teams_for_boundary_kernel + nbound - 1) /
                         nbound);
}
} // namespace

template <BoundaryType bound_type>
TaskStatus SendBoundBufsWithRestrictOption(std::shared_ptr<MeshData<Real>> &md,
                                           bool do_restriction, bool perform_transport) {
  PARTHENON_INSTRUMENT

  Mesh *pmesh = md->GetMeshPointer();
  auto &cache = md->GetBvarsCache().GetSubCache(bound_type, true);

  if (cache.RequiresReinitialize(pmesh))
    InitializeBufferCache<bound_type>(md, &(pmesh->boundary_comm_map), &cache, SendKey);

  auto [rebuild, nbound, other_communication_unfinished] =
      CheckSendBufferCacheForRebuild<bound_type, true>(md);

  if (nbound == 0) {
    return TaskStatus::complete;
  }

  bool can_write_combined{true};
  if (pmesh->do_coalesced_comms)
    can_write_combined =
        pmesh->pcoalesced_comms->IsAvailableForWrite(md.get(), bound_type);

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
    if (pmesh->pcoalesced_comms->UsesNCCL(bound_type))
      pmesh->pcoalesced_comms->SetNCCLDirectBuffers(&cache, bound_type, true);
  }
  // Restrict
  if (md->NumBlocks() > 0 && do_restriction) {
    auto pmb = md->GetBlockData(0)->GetBlockPointer();
    StateDescriptor *resolved_packages = pmb->resolved_packages.get();
    refinement::Restrict(resolved_packages, cache.prores_cache, pmb->cellbounds,
                         pmb->c_cellbounds);
  }

  // Load buffer data
  auto &bnd_info = cache.bnd_info;
  PARTHENON_DEBUG_REQUIRE(bnd_info.size() == nbound, "Need same size for boundary info");
  const int nteams_per_buffer = GetNteamsPerBoundaryBuffer(pmesh, nbound);
  const int work_chunk_size = pmesh->boundary_buffer_work_chunk_size;
  const bool skip_dense_scan = SkipDenseScan();
  auto &sending_nonzero_flags = cache.sending_non_zero_flags;
  auto &sending_nonzero_flags_h = cache.sending_non_zero_flags_h;
  if (sending_nonzero_flags.size() != (nbound * nteams_per_buffer)) {
    sending_nonzero_flags =
        ParArray1D<bool>("sending_nonzero_flags", nbound * nteams_per_buffer);
    sending_nonzero_flags_h = Kokkos::create_mirror_view(sending_nonzero_flags);
  }

  Kokkos::parallel_for(
      PARTHENON_AUTO_LABEL,
      Kokkos::TeamPolicy<>(parthenon::DevExecSpace(), nbound * nteams_per_buffer,
                           Kokkos::AUTO, BoundaryVectorLength()),
      KOKKOS_LAMBDA(parthenon::team_mbr_t team_member) {
        const int b = team_member.league_rank() / nteams_per_buffer;
        const int bteam = team_member.league_rank() % nteams_per_buffer;
        const int iflag = team_member.league_rank();

        if (!bnd_info(b).allocated || bnd_info(b).same_to_same) {
          Kokkos::single(Kokkos::PerTeam(team_member),
                         [&]() { sending_nonzero_flags(iflag) = false; });
          return;
        }
        Real threshold = bnd_info(b).var.allocation_threshold;
        bool non_zero[3]{false, false, false};
        int idx_offset = 0;
        for (int it = 0; it < bnd_info(b).ntopological_elements; ++it) {
          auto &idxer = bnd_info(b).idxer[it];
          const int iel = static_cast<int>(bnd_info(b).topo_idx[it]) % 3;
          const int Ni = idxer.template EndIdx<5>() - idxer.template StartIdx<5>() + 1;
          const auto idxer_size = idxer.size();
          if (Ni <= 0 || idxer_size == 0) continue;
          const int n_units = static_cast<int>(idxer_size / Ni);
          const SplitFlatIndexRangeAmongTeams split(nteams_per_buffer, work_chunk_size,
                                                    n_units);
          // TODO(LFR): Finish threading index splitting through reductions
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange<>(team_member, split.GetStart(bteam),
                                        split.GetEnd(bteam)),
              [&](const int idx, bool &lnon_zero) {
                const auto [t, u, v, k, j, i] = idxer(idx * Ni);
                Real *var = &bnd_info(b).var(iel, t, u, v, k, j, i);
                Real *buf = bnd_info(b).direct_buf != nullptr
                                ? bnd_info(b).direct_buf + idx * Ni + idx_offset
                                : &bnd_info(b).buf(idx * Ni + idx_offset);

                Kokkos::parallel_for(Kokkos::ThreadVectorRange<>(team_member, Ni),
                                     [&](int m) { buf[m] = var[m]; });

                bool mnon_zero = skip_dense_scan;
                if (!skip_dense_scan)
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
          sending_nonzero_flags(iflag) = non_zero[0] || non_zero[1] || non_zero[2];
        });
      });

  // Send buffers
  if (Globals::sparse_config.enabled)
    Kokkos::deep_copy(sending_nonzero_flags_h, sending_nonzero_flags);
  const bool coal_comm = pmesh->do_coalesced_comms;
#ifdef MPI_PARALLEL
  if (!coal_comm) Kokkos::fence();
#endif
  for (std::size_t ibuf = 0; ibuf < cache.buf_vec.size(); ++ibuf) {
    auto &buf = *cache.buf_vec[ibuf];
    if (!Globals::sparse_config.enabled) {
      buf.Send(coal_comm);
    } else {
      // Reduce flags over all of the teams that contributed to filling a given buffer
      bool sending_nonz{false};
      for (int i = ibuf * nteams_per_buffer; i < (ibuf + 1) * nteams_per_buffer; ++i)
        sending_nonz = sending_nonz || sending_nonzero_flags_h(i);

      if (sending_nonz) {
        buf.Send(coal_comm);
      } else {
        buf.SendNull(coal_comm);
      }
    }
  }
  if (pmesh->do_coalesced_comms) {
    if (pmesh->pcoalesced_comms->UsesNCCL(bound_type)) {
      PARTHENON_REQUIRE(!perform_transport,
                        "NCCL coalesced communication must use the coordinated task "
                        "exchange.");
      pmesh->pcoalesced_comms->PrepareNCCLSend(md.get(), bound_type);
    } else {
      pmesh->pcoalesced_comms->PackAndSend(md.get(), bound_type);
    }
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
template TaskStatus
SendBoundBufs<BoundaryType::gmg_same>(std::shared_ptr<MeshData<Real>> &);

template <BoundaryType bound_type>
TaskStatus StartReceiveBoundBufs(std::shared_ptr<MeshData<Real>> &md) {
  PARTHENON_INSTRUMENT
  Mesh *pmesh = md->GetMeshPointer();
  auto &cache = md->GetBvarsCache().GetSubCache(bound_type, false);
  if (cache.RequiresReinitialize(pmesh))
    InitializeBufferCache<bound_type>(md, &(pmesh->boundary_comm_map), &cache,
                                      ReceiveKey);
  if (!pmesh->do_coalesced_comms) {
    std::for_each(std::begin(cache.buf_vec), std::end(cache.buf_vec),
                  [](auto pbuf) { pbuf->TryStartReceive(); });
  }

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
template TaskStatus
StartReceiveBoundBufs<BoundaryType::gmg_same>(std::shared_ptr<MeshData<Real>> &);

template <BoundaryType bound_type>
TaskStatus ReceiveBoundBufs(std::shared_ptr<MeshData<Real>> &md) {
  PARTHENON_INSTRUMENT

  Mesh *pmesh = md->GetMeshPointer();
  auto &cache = md->GetBvarsCache().GetSubCache(bound_type, false);
  if (cache.RequiresReinitialize(pmesh))
    InitializeBufferCache<bound_type>(md, &(pmesh->boundary_comm_map), &cache,
                                      ReceiveKey);

  const bool coal_comm = pmesh->do_coalesced_comms;
  if (coal_comm && !pmesh->pcoalesced_comms->UsesNCCL(bound_type))
    pmesh->pcoalesced_comms->TryReceiveAny(md.get(), bound_type);
  bool all_received = true;
  std::for_each(std::begin(cache.buf_vec), std::end(cache.buf_vec),
                [&all_received, coal_comm](auto pbuf) {
                  all_received = pbuf->TryReceive(coal_comm) && all_received;
                });

  int ibound = 0;
  if (Globals::sparse_config.enabled && all_received) {
    ForEachBoundary<bound_type>(
        md, [&](auto pmb, sp_mbd_t rc, const nb_t &nb, const sp_cv_t v) {
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
  if (all_received) return TaskStatus::complete;
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
template TaskStatus
ReceiveBoundBufs<BoundaryType::gmg_same>(std::shared_ptr<MeshData<Real>> &);

template <BoundaryType bound_type>
TaskStatus SetBounds(std::shared_ptr<MeshData<Real>> &md) {
  PARTHENON_INSTRUMENT

  Mesh *pmesh = md->GetMeshPointer();
  auto &cache = md->GetBvarsCache().GetSubCache(bound_type, false);

  auto [rebuild, nbound] = CheckReceiveBufferCacheForRebuild<bound_type, false>(md);

  if (nbound == 0) {
    return TaskStatus::complete;
  }

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
    if (pmesh->pcoalesced_comms->UsesNCCL(bound_type))
      pmesh->pcoalesced_comms->SetNCCLDirectBuffers(&cache, bound_type, false);
  }
  // const Real threshold = Globals::sparse_config.allocation_threshold;
  auto &bnd_info = cache.bnd_info;
  const int nteams_per_buffer = GetNteamsPerBoundaryBuffer(pmesh, nbound);
  const int work_chunk_size = pmesh->boundary_buffer_work_chunk_size;

  if (pmesh->pcoalesced_comms->UsesNCCL(bound_type))
    pmesh->pcoalesced_comms->WaitForNCCLUnpack(bound_type, DevExecSpace());

  Kokkos::parallel_for(
      PARTHENON_AUTO_LABEL,
      Kokkos::TeamPolicy<>(parthenon::DevExecSpace(), nbound * nteams_per_buffer,
                           Kokkos::AUTO, BoundaryVectorLength()),
      KOKKOS_LAMBDA(parthenon::team_mbr_t team_member) {
        const int b = team_member.league_rank() / nteams_per_buffer;
        const int bteam = team_member.league_rank() % nteams_per_buffer;
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
          if (bnd_info(b).allocated) {
            const auto idxer_size = idxer.size();
            if (Ni <= 0 || idxer_size == 0) continue;
            const int n_units = static_cast<int>(idxer_size / Ni);
            const SplitFlatIndexRangeAmongTeams split(nteams_per_buffer, work_chunk_size,
                                                      n_units);
            if (bnd_info(b).buf_allocated) {
              Kokkos::parallel_for(
                  Kokkos::TeamThreadRange<>(team_member, split.GetStart(bteam),
                                            split.GetEnd(bteam)),
                  [&](const int idx) {
                    Real *buf = bnd_info(b).direct_buf != nullptr
                                    ? bnd_info(b).direct_buf + idx * Ni + idx_offset
                                    : &bnd_info(b).buf(idx * Ni + idx_offset);
                    const auto [t, u, v, k, j, i] = idxer(idx * Ni);
                    // Have to do this because of some weird issue about structure
                    // bindings being captured
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
            } else if (bound_type != BoundaryType::flxcor_recv) {
              const Real default_val = bnd_info(b).var.sparse_default_val;
              Kokkos::parallel_for(
                  Kokkos::TeamThreadRange<>(team_member, split.GetStart(bteam),
                                            split.GetEnd(bteam)),
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
template TaskStatus SetBounds<BoundaryType::gmg_same>(std::shared_ptr<MeshData<Real>> &);

template <BoundaryType bound_type>
TaskStatus ProlongateBounds(std::shared_ptr<MeshData<Real>> &md) {
  PARTHENON_INSTRUMENT

  Mesh *pmesh = md->GetMeshPointer();
  auto &cache = md->GetBvarsCache().GetSubCache(bound_type, false);

  auto [rebuild, nbound] = CheckReceiveBufferCacheForRebuild<bound_type, false>(md);

  // We should not rebuild the cache here even if it is flagged as such. When using
  // coalesced communication, the state of the buffers associated with this MeshData
  // partition can be updated by TryReceives on other MeshData partitions. As a result,
  // the buffer states and allocation statuses can change between calls to SetBounds and
  // ProlongateBounds. This would trigger a rebuild of the buffer cache when the receive
  // states are unknown (which is a requirement for rebuilding the receive buffer cache).

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
template TaskStatus
ProlongateBounds<BoundaryType::gmg_same>(std::shared_ptr<MeshData<Real>> &);

template <BoundaryType bound_type>
TaskStatus ProlongateInternalBounds(std::shared_ptr<MeshData<Real>> &md) {
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
    refinement::ProlongateInternal(resolved_packages, cache.prores_cache, pmb->cellbounds,
                                   pmb->c_cellbounds);
  }
  return TaskStatus::complete;
}
template TaskStatus
ProlongateInternalBounds<BoundaryType::any>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus
ProlongateInternalBounds<BoundaryType::local>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus
ProlongateInternalBounds<BoundaryType::nonlocal>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus ProlongateInternalBounds<BoundaryType::gmg_prolongate_recv>(
    std::shared_ptr<MeshData<Real>> &);
template TaskStatus
ProlongateInternalBounds<BoundaryType::gmg_same>(std::shared_ptr<MeshData<Real>> &);

bool IsMeshMultilevel(std::shared_ptr<MeshData<Real>> &md) {
  return md->GetMeshPointer()->multilevel;
}

bool UsesNCCLCoalescedComms(std::shared_ptr<MeshData<Real>> &md, BoundaryType btype) {
  return md->GetMeshPointer()->pcoalesced_comms->UsesNCCL(btype);
}

TaskStatus NCCLTaskRendezvous() { return TaskStatus::complete; }

template <BoundaryType send_type, BoundaryType recv_type>
TaskStatus LaunchNCCLExchange(std::shared_ptr<MeshData<Real>> &md) {
  return md->GetMeshPointer()->pcoalesced_comms->LaunchNCCLExchange(send_type, recv_type);
}

template <BoundaryType send_type>
TaskStatus PollNCCLExchange(std::shared_ptr<MeshData<Real>> &md) {
  return md->GetMeshPointer()->pcoalesced_comms->PollNCCLExchange(send_type);
}

template <BoundaryType send_type, BoundaryType recv_type>
TaskStatus FinalizeNCCLExchange(std::shared_ptr<MeshData<Real>> &md) {
  return md->GetMeshPointer()->pcoalesced_comms->FinalizeNCCLExchange(send_type,
                                                                      recv_type);
}

template TaskStatus LaunchNCCLExchange<BoundaryType::any, BoundaryType::any>(
    std::shared_ptr<MeshData<Real>> &);
template TaskStatus LaunchNCCLExchange<BoundaryType::gmg_same, BoundaryType::gmg_same>(
    std::shared_ptr<MeshData<Real>> &);
template TaskStatus
PollNCCLExchange<BoundaryType::any>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus
PollNCCLExchange<BoundaryType::gmg_same>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus
LaunchNCCLExchange<BoundaryType::flxcor_send, BoundaryType::flxcor_recv>(
    std::shared_ptr<MeshData<Real>> &);
template TaskStatus
PollNCCLExchange<BoundaryType::flxcor_send>(std::shared_ptr<MeshData<Real>> &);
template TaskStatus FinalizeNCCLExchange<BoundaryType::any, BoundaryType::any>(
    std::shared_ptr<MeshData<Real>> &);
template TaskStatus FinalizeNCCLExchange<BoundaryType::gmg_same, BoundaryType::gmg_same>(
    std::shared_ptr<MeshData<Real>> &);
template TaskStatus
FinalizeNCCLExchange<BoundaryType::flxcor_send, BoundaryType::flxcor_recv>(
    std::shared_ptr<MeshData<Real>> &);

TaskID AddFluxCorrectionTasks(TaskID dependency, TaskList &tl,
                              std::shared_ptr<MeshData<Real>> &md, bool multilevel) {
  if (!multilevel) return dependency;
  if (UsesNCCLCoalescedComms(md, BoundaryType::flxcor_send)) {
    auto pack =
        tl.AddTask(dependency, TF(PrepareNCCLBoundBufs<BoundaryType::flxcor_send>), md);
    auto rendezvous = tl.AddTask(TaskQualifier::local_sync, pack, TF(NCCLTaskRendezvous));
    auto launch = tl.AddTask(
        TaskQualifier::once_per_region, rendezvous,
        TF((LaunchNCCLExchange<BoundaryType::flxcor_send, BoundaryType::flxcor_recv>)),
        md);
    auto poll = tl.AddTask(TaskQualifier::once_per_region, launch,
                           TF((PollNCCLExchange<BoundaryType::flxcor_send>)), md);
    auto finalize = tl.AddTask(
        TaskQualifier::once_per_region, poll,
        TF((FinalizeNCCLExchange<BoundaryType::flxcor_send, BoundaryType::flxcor_recv>)),
        md);
    auto receive =
        tl.AddTask(finalize, TF(ReceiveBoundBufs<BoundaryType::flxcor_recv>), md);
    return tl.AddTask(receive, TF(SetBounds<BoundaryType::flxcor_recv>), md);
  }
  tl.AddTask(dependency, TF(SendBoundBufs<BoundaryType::flxcor_send>), md);
  auto receive =
      tl.AddTask(dependency, TF(ReceiveBoundBufs<BoundaryType::flxcor_recv>), md);
  return tl.AddTask(receive, TF(SetBounds<BoundaryType::flxcor_recv>), md);
}
} // namespace parthenon
