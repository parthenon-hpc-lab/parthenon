//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001
// for Los Alamos National Laboratory (LANL), which is operated by Triad
// National Security, LLC for the U.S. Department of Energy/National Nuclear
// Security Administration. All rights in the program are reserved by Triad
// National Security, LLC, and the U.S. Department of Energy/National Nuclear
// Security Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
// in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do
// so.
//========================================================================================

#include "mesh/refinement_cc_in_one.hpp"

#include "kokkos_abstraction.hpp"

namespace parthenon {
namespace cell_centered_refinement {

void Restrict(cell_centered_bvars::BufferCache_t &info, IndexShape &cellbounds,
              IndexShape &c_cellbounds) {
  const IndexDomain interior = IndexDomain::interior;
  const IndexDomain entire = IndexDomain::entire;
  auto ckb = c_cellbounds.GetBoundsK(interior);
  auto cjb = c_cellbounds.GetBoundsJ(interior);
  auto cib = c_cellbounds.GetBoundsI(interior);
  auto kb = cellbounds.GetBoundsK(interior);
  auto jb = cellbounds.GetBoundsJ(interior);
  auto ib = cellbounds.GetBoundsI(interior);

  // TODO(BRR) nbuffers is currently always 1. In the future nbuffers could be > 1 to
  // improve performance.
  const int nbuffers = info.extent_int(0);
  const int scratch_level = 1; // 0 is actual scratch (tiny); 1 is HBM
  size_t scratch_size_in_bytes = 1;

  if (cellbounds.ncellsk(entire) > 1) { // 3D
    par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, "RestrictCellCenteredValues3d", DevExecSpace(),
        scratch_size_in_bytes, scratch_level, 0, nbuffers - 1,
        KOKKOS_LAMBDA(team_mbr_t team_member, const int buf) {
          if (info(buf).allocated && info(buf).restriction) {
            par_for_inner(
                inner_loop_pattern_ttr_tag, team_member, 0, info(buf).Nt - 1, 0,
                info(buf).Nu - 1, 0, info(buf).Nv - 1, info(buf).sk, info(buf).ek,
                info(buf).sj, info(buf).ej, info(buf).si, info(buf).ei,
                [&](const int l, const int m, const int n, const int ck, const int cj,
                    const int ci) {
                  const int k = (ck - ckb.s) * 2 + kb.s;
                  const int j = (cj - cjb.s) * 2 + jb.s;
                  const int i = (ci - cib.s) * 2 + ib.s;
                  // KGF: add the off-centered quantities first to preserve FP
                  // symmetry
                  const Real vol000 = info(buf).coords.Volume(k, j, i);
                  const Real vol001 = info(buf).coords.Volume(k, j, i + 1);
                  const Real vol010 = info(buf).coords.Volume(k, j + 1, i);
                  const Real vol011 = info(buf).coords.Volume(k, j + 1, i + 1);
                  const Real vol100 = info(buf).coords.Volume(k + 1, j, i);
                  const Real vol101 = info(buf).coords.Volume(k + 1, j, i + 1);
                  const Real vol110 = info(buf).coords.Volume(k + 1, j + 1, i);
                  const Real vol111 = info(buf).coords.Volume(k + 1, j + 1, i + 1);
                  Real tvol = ((vol000 + vol010) + (vol001 + vol011)) +
                              ((vol100 + vol110) + (vol101 + vol111));
                  // KGF: add the off-centered quantities first to preserve FP
                  // symmetry
                  auto &coarse = info(buf).coarse;
                  auto &fine = info(buf).fine;
                  coarse(l, m, n, ck, cj, ci) =
                      (((fine(l, m, n, k, j, i) * vol000 +
                         fine(l, m, n, k, j + 1, i) * vol010) +
                        (fine(l, m, n, k, j, i + 1) * vol001 +
                         fine(l, m, n, k, j + 1, i + 1) * vol011)) +
                       ((fine(l, m, n, k + 1, j, i) * vol100 +
                         fine(l, m, n, k + 1, j + 1, i) * vol110) +
                        (fine(l, m, n, k + 1, j, i + 1) * vol101 +
                         fine(l, m, n, k + 1, j + 1, i + 1) * vol111))) /
                      tvol;
                });
          }
        });
  } else if (cellbounds.ncellsj(entire) > 1) { // 2D
    par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, "RestrictCellCenteredValues2d", DevExecSpace(),
        scratch_size_in_bytes, scratch_level, 0, nbuffers - 1,
        KOKKOS_LAMBDA(team_mbr_t team_member, const int buf) {
          if (info(buf).allocated && info(buf).restriction) {
            const int k = kb.s;
            par_for_inner(
                inner_loop_pattern_ttr_tag, team_member, 0, info(buf).Nt - 1, 0,
                info(buf).Nv - 1, 0, info(buf).Nv - 1, info(buf).sj, info(buf).ej,
                info(buf).si, info(buf).ei,
                [&](const int l, const int m, const int n, const int cj, const int ci) {
                  const int j = (cj - cjb.s) * 2 + jb.s;
                  const int i = (ci - cib.s) * 2 + ib.s;
                  // KGF: add the off-centered quantities first to preserve FP
                  // symmetry
                  const Real vol00 = info(buf).coords.Volume(k, j, i);
                  const Real vol10 = info(buf).coords.Volume(k, j + 1, i);
                  const Real vol01 = info(buf).coords.Volume(k, j, i + 1);
                  const Real vol11 = info(buf).coords.Volume(k, j + 1, i + 1);
                  Real tvol = (vol00 + vol10) + (vol01 + vol11);

                  // KGF: add the off-centered quantities first to preserve FP
                  // symmetry
                  auto &coarse = info(buf).coarse;
                  auto &fine = info(buf).fine;
                  coarse(l, m, n, 0, cj, ci) =
                      ((fine(l, m, n, 0, j, i) * vol00 +
                        fine(l, m, n, 0, j + 1, i) * vol10) +
                       (fine(l, m, n, 0, j, i + 1) * vol01 +
                        fine(l, m, n, 0, j + 1, i + 1) * vol11)) /
                      tvol;
                });
          }
        });
  } else if (cellbounds.ncellsi(entire) > 1) { // 1D
    par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, "RestrictCellCenteredValues1d", DevExecSpace(),
        scratch_size_in_bytes, scratch_level, 0, nbuffers - 1,
        KOKKOS_LAMBDA(team_mbr_t team_member, const int buf) {
          if (info(buf).allocated && info(buf).restriction) {
            const int ck = ckb.s;
            const int cj = cjb.s;
            const int k = kb.s;
            const int j = jb.s;
            par_for_inner(
                inner_loop_pattern_ttr_tag, team_member, 0, info(buf).Nt - 1, 0,
                info(buf).Nu - 1, 0, info(buf).Nv - 1, info(buf).si, info(buf).ei,
                [&](const int l, const int m, const int n, const int ci) {
                  const int i = (ci - cib.s) * 2 + ib.s;
                  const Real vol0 = info(buf).coords.Volume(k, j, i);
                  const Real vol1 = info(buf).coords.Volume(k, j, i + 1);
                  Real tvol = vol0 + vol1;
                  auto &coarse = info(buf).coarse;
                  auto &fine = info(buf).fine;
                  coarse(l, m, n, ck, cj, ci) = (fine(l, m, n, k, j, i) * vol0 +
                                                 fine(l, m, n, k, j, i + 1) * vol1) /
                                                tvol;
                });
          }
        });
  }
}

std::vector<bool> ComputePhysicalRestrictBoundsAllocStatus(MeshData<Real> *md) {
  Kokkos::Profiling::pushRegion("ComputePhysicalRestrictBoundsAllocStatus_MeshData");
  std::vector<bool> alloc_status;
  for (int block = 0; block < md->NumBlocks(); ++block) {
    auto &rc = md->GetBlockData(block);
    auto pmb = rc->GetBlockPointer();
    int nrestrictions = pmb->pbval->NumRestrictions();
    for (auto &v : rc->GetCellVariableVector()) {
      if (v->IsSet(parthenon::Metadata::FillGhost)) {
        int num_bufs = nrestrictions * (v->GetDim(6)) * (v->GetDim(5));
        for (int i = 0; i < num_bufs; ++i) {
          alloc_status.push_back(v->IsAllocated());
        }
      }
    }
  }

  Kokkos::Profiling::popRegion(); // ComputePhysicalRestrictBoundsAllocStatus_MeshData
  return alloc_status;
}

TaskStatus RestrictPhysicalBounds(MeshData<Real> *md) {
  Kokkos::Profiling::pushRegion("Task_RestrictPhysicalBounds_MeshData");

  // get alloc status
  auto alloc_status = ComputePhysicalRestrictBoundsAllocStatus(md);

  auto info = md->GetRestrictBuffers();
  if (!info.is_allocated() || (alloc_status != md->GetRestrictBufAllocStatus())) {
    ComputePhysicalRestrictBounds(md);
    info = md->GetRestrictBuffers();
  }

  auto &rc = md->GetBlockData(0);
  auto pmb = rc->GetBlockPointer();
  IndexShape cellbounds = pmb->cellbounds;
  IndexShape c_cellbounds = pmb->c_cellbounds;

  Restrict(info, cellbounds, c_cellbounds);

  Kokkos::Profiling::popRegion(); // Task_RestrictPhysicalBounds_MeshData
  return TaskStatus::complete;
}

void ComputePhysicalRestrictBounds(MeshData<Real> *md) {
  Kokkos::Profiling::pushRegion("ComputePhysicalRestrictBounds_MeshData");
  auto alloc_status = ComputePhysicalRestrictBoundsAllocStatus(md);

  cell_centered_bvars::BufferCache_t info("physical restriction bounds",
                                          alloc_status.size());
  auto info_h = Kokkos::create_mirror_view(info);
  int idx = 0;
  for (int block = 0; block < md->NumBlocks(); ++block) {
    auto &rc = md->GetBlockData(block);
    auto pmb = rc->GetBlockPointer();
    for (auto &v : rc->GetCellVariableVector()) {
      if (v->IsSet(parthenon::Metadata::FillGhost)) {
        pmb->pbval->FillRestrictionMetadata(info_h, idx, v);
      }
    }
  }
  PARTHENON_DEBUG_REQUIRE(idx == alloc_status.size(), "All buffers accounted for");
  Kokkos::deep_copy(info, info_h);

  md->SetRestrictBuffers(info, alloc_status);

  Kokkos::Profiling::popRegion(); // ComputePhysicalRestrictBounds_MeshData
}

} // namespace cell_centered_refinement
} // namespace parthenon
