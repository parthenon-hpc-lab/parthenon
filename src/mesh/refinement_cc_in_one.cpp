//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2022 The Parthenon collaboration
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

#include <algorithm>

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
          if (info(buf).allocated &&
              info(buf).refinement_op == RefinementOp_t::Restriction) {
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
          if (info(buf).allocated &&
              info(buf).refinement_op == RefinementOp_t::Restriction) {
            const int k = kb.s;
            par_for_inner(
                inner_loop_pattern_ttr_tag, team_member, 0, info(buf).Nt - 1, 0,
                info(buf).Nu - 1, 0, info(buf).Nv - 1, info(buf).sj, info(buf).ej,
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
          if (info(buf).allocated &&
              info(buf).refinement_op == RefinementOp_t::Restriction) {
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

void Prolongate(cell_centered_bvars::BufferCache_t &info, IndexShape &cellbounds,
                IndexShape &c_cellbounds) {
  const IndexDomain interior = IndexDomain::interior;
  const IndexDomain entire = IndexDomain::entire;
  auto ckb = c_cellbounds.GetBoundsK(interior);
  auto cjb = c_cellbounds.GetBoundsJ(interior);
  auto cib = c_cellbounds.GetBoundsI(interior);
  auto kb = cellbounds.GetBoundsK(interior);
  auto jb = cellbounds.GetBoundsJ(interior);
  auto ib = cellbounds.GetBoundsI(interior);

  const int nbuffers = info.extent_int(0);
  const int scratch_level = 1; // 0 is actual scratch (tiny); 1 is HBM
  size_t scratch_size_in_bytes = 1;

  if (cellbounds.ncellsk(entire) > 1) { // 3D
    par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, "ProlongateCellCenteredValues3d", DevExecSpace(),
        scratch_size_in_bytes, scratch_level, 0, nbuffers - 1,
        KOKKOS_LAMBDA(team_mbr_t team_member, const int buf) {
          if (info(buf).allocated &&
              info(buf).refinement_op == RefinementOp_t::Prolongation) {
            par_for_inner(inner_loop_pattern_ttr_tag, team_member, 0, info(buf).Nv - 1,
                          info(buf).sk, info(buf).ek, info(buf).sj, info(buf).ej,
                          info(buf).si, info(buf).ei,
                          [&](const int n, const int k, const int j, const int i) {
                            // x3 direction
                            int fk = (k - ckb.s) * 2 + kb.s;
                            const Real x3m = info(buf).coarse_coords.x3v(k - 1);
                            const Real x3c = info(buf).coarse_coords.x3v(k);
                            const Real x3p = info(buf).coarse_coords.x3v(k + 1);
                            Real dx3m = x3c - x3m;
                            Real dx3p = x3p - x3c;
                            const Real fx3m = info(buf).coords.x3v(fk);
                            const Real fx3p = info(buf).coords.x3v(fk + 1);
                            Real dx3fm = x3c - fx3m;
                            Real dx3fp = fx3p - x3c;

                            // x2 direction
                            int fj = (j - cjb.s) * 2 + jb.s;
                            const Real x2m = info(buf).coarse_coords.x2v(j - 1);
                            const Real x2c = info(buf).coarse_coords.x2v(j);
                            const Real x2p = info(buf).coarse_coords.x2v(j + 1);
                            Real dx2m = x2c - x2m;
                            Real dx2p = x2p - x2c;
                            const Real fx2m = info(buf).coords.x2v(fj);
                            const Real fx2p = info(buf).coords.x2v(fj + 1);
                            Real dx2fm = x2c - fx2m;
                            Real dx2fp = fx2p - x2c;

                            // x1 direction
                            int fi = (i - cib.s) * 2 + ib.s;
                            const Real x1m = info(buf).coarse_coords.x1v(i - 1);
                            const Real x1c = info(buf).coarse_coords.x1v(i);
                            const Real x1p = info(buf).coarse_coords.x1v(i + 1);
                            Real dx1m = x1c - x1m;
                            Real dx1p = x1p - x1c;
                            const Real fx1m = info(buf).coords.x1v(fi);
                            const Real fx1p = info(buf).coords.x1v(fi + 1);
                            Real dx1fm = x1c - fx1m;
                            Real dx1fp = fx1p - x1c;

                            auto &coarse = info(buf).coarse;
                            auto &fine = info(buf).fine;

                            Real ccval = coarse(n, k, j, i);

                            // calculate 3D gradients using the minmod limiter
                            Real gx1m = (ccval - coarse(n, k, j, i - 1)) / dx1m;
                            Real gx1p = (coarse(n, k, j, i + 1) - ccval) / dx1p;
                            Real gx1c = 0.5 * (SIGN(gx1m) + SIGN(gx1p)) *
                                        std::min(std::abs(gx1m), std::abs(gx1p));
                            Real gx2m = (ccval - coarse(n, k, j - 1, i)) / dx2m;
                            Real gx2p = (coarse(n, k, j + 1, i) - ccval) / dx2p;
                            Real gx2c = 0.5 * (SIGN(gx2m) + SIGN(gx2p)) *
                                        std::min(std::abs(gx2m), std::abs(gx2p));
                            Real gx3m = (ccval - coarse(n, k - 1, j, i)) / dx3m;
                            Real gx3p = (coarse(n, k + 1, j, i) - ccval) / dx3p;
                            Real gx3c = 0.5 * (SIGN(gx3m) + SIGN(gx3p)) *
                                        std::min(std::abs(gx3m), std::abs(gx3p));

                            // KGF: add the off-centered quantities first to preserve FP
                            // symmetry interpolate onto the finer grid
                            fine(n, fk, fj, fi) =
                                ccval - (gx1c * dx1fm + gx2c * dx2fm + gx3c * dx3fm);
                            fine(n, fk, fj, fi + 1) =
                                ccval + (gx1c * dx1fp - gx2c * dx2fm - gx3c * dx3fm);
                            fine(n, fk, fj + 1, fi) =
                                ccval - (gx1c * dx1fm - gx2c * dx2fp + gx3c * dx3fm);
                            fine(n, fk, fj + 1, fi + 1) =
                                ccval + (gx1c * dx1fp + gx2c * dx2fp - gx3c * dx3fm);
                            fine(n, fk + 1, fj, fi) =
                                ccval - (gx1c * dx1fm + gx2c * dx2fm - gx3c * dx3fp);
                            fine(n, fk + 1, fj, fi + 1) =
                                ccval + (gx1c * dx1fp - gx2c * dx2fm + gx3c * dx3fp);
                            fine(n, fk + 1, fj + 1, fi) =
                                ccval - (gx1c * dx1fm - gx2c * dx2fp - gx3c * dx3fp);
                            fine(n, fk + 1, fj + 1, fi + 1) =
                                ccval + (gx1c * dx1fp + gx2c * dx2fp + gx3c * dx3fp);
                          });
          }
        });
  } else if (cellbounds.ncellsj(entire) > 1) { // 2D
    par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, "ProlongateCellCenteredValues2d", DevExecSpace(),
        scratch_size_in_bytes, scratch_level, 0, nbuffers - 1,
        KOKKOS_LAMBDA(team_mbr_t team_member, const int buf) {
          if (info(buf).allocated &&
              info(buf).refinement_op == RefinementOp_t::Prolongation) {
            const int k = ckb.s;
            const int fk = kb.s;
            par_for_inner(
                inner_loop_pattern_ttr_tag, team_member, 0, info(buf).Nv - 1,
                info(buf).sj, info(buf).ej, info(buf).si, info(buf).ei,
                [&](const int n, const int j, const int i) {
                  // x2 direction
                  int fj = (j - cjb.s) * 2 + jb.s;
                  const Real x2m = info(buf).coarse_coords.x2v(j - 1);
                  const Real x2c = info(buf).coarse_coords.x2v(j);
                  const Real x2p = info(buf).coarse_coords.x2v(j + 1);
                  Real dx2m = x2c - x2m;
                  Real dx2p = x2p - x2c;
                  const Real fx2m = info(buf).coords.x2v(fj);
                  const Real fx2p = info(buf).coords.x2v(fj + 1);
                  Real dx2fm = x2c - fx2m;
                  Real dx2fp = fx2p - x2c;

                  // x1 direction
                  int fi = (i - cib.s) * 2 + ib.s;
                  const Real x1m = info(buf).coarse_coords.x1v(i - 1);
                  const Real x1c = info(buf).coarse_coords.x1v(i);
                  const Real x1p = info(buf).coarse_coords.x1v(i + 1);
                  Real dx1m = x1c - x1m;
                  Real dx1p = x1p - x1c;
                  const Real fx1m = info(buf).coords.x1v(fi);
                  const Real fx1p = info(buf).coords.x1v(fi + 1);
                  Real dx1fm = x1c - fx1m;
                  Real dx1fp = fx1p - x1c;

                  auto &coarse = info(buf).coarse;
                  auto &fine = info(buf).fine;

                  Real ccval = coarse(n, k, j, i);

                  // calculate 2D gradients using the minmod limiter
                  Real gx1m = (ccval - coarse(n, k, j, i - 1)) / dx1m;
                  Real gx1p = (coarse(n, k, j, i + 1) - ccval) / dx1p;
                  Real gx1c = 0.5 * (SIGN(gx1m) + SIGN(gx1p)) *
                              std::min(std::abs(gx1m), std::abs(gx1p));
                  Real gx2m = (ccval - coarse(n, k, j - 1, i)) / dx2m;
                  Real gx2p = (coarse(n, k, j + 1, i) - ccval) / dx2p;
                  Real gx2c = 0.5 * (SIGN(gx2m) + SIGN(gx2p)) *
                              std::min(std::abs(gx2m), std::abs(gx2p));

                  // KGF: add the off-centered quantities first to preserve FP symmetry
                  // interpolate onto the finer grid
                  fine(n, fk, fj, fi) = ccval - (gx1c * dx1fm + gx2c * dx2fm);
                  fine(n, fk, fj, fi + 1) = ccval + (gx1c * dx1fp - gx2c * dx2fm);
                  fine(n, fk, fj + 1, fi) = ccval - (gx1c * dx1fm - gx2c * dx2fp);
                  fine(n, fk, fj + 1, fi + 1) = ccval + (gx1c * dx1fp + gx2c * dx2fp);
                });
          }
        });
  } else { // 1D
    par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, "ProlongateCellCenteredValues1d", DevExecSpace(),
        scratch_size_in_bytes, scratch_level, 0, nbuffers - 1,
        KOKKOS_LAMBDA(team_mbr_t team_member, const int buf) {
          if (info(buf).allocated &&
              info(buf).refinement_op == RefinementOp_t::Prolongation) {
            const int k = ckb.s;
            const int fk = kb.s;
            const int j = cjb.s;
            const int fj = jb.s;
            par_for_inner(inner_loop_pattern_ttr_tag, team_member, 0, info(buf).Nv - 1,
                          info(buf).si, info(buf).ei, [&](const int n, const int i) {
                            int fi = (i - cib.s) * 2 + ib.s;
                            const Real x1m = info(buf).coarse_coords.x1v(i - 1);
                            const Real x1c = info(buf).coarse_coords.x1v(i);
                            const Real x1p = info(buf).coarse_coords.x1v(i + 1);
                            Real dx1m = x1c - x1m;
                            Real dx1p = x1p - x1c;
                            const Real fx1m = info(buf).coords.x1v(fi);
                            const Real fx1p = info(buf).coords.x1v(fi + 1);
                            Real dx1fm = x1c - fx1m;
                            Real dx1fp = fx1p - x1c;

                            auto &coarse = info(buf).coarse;
                            auto &fine = info(buf).fine;

                            Real ccval = coarse(n, k, j, i);

                            // calculate 1D gradient using the min-mod limiter
                            Real gx1m = (ccval - coarse(n, k, j, i - 1)) / dx1m;
                            Real gx1p = (coarse(n, k, j, i + 1) - ccval) / dx1p;
                            Real gx1c = 0.5 * (SIGN(gx1m) + SIGN(gx1p)) *
                                        std::min(std::abs(gx1m), std::abs(gx1p));

                            // interpolate on to the finer grid
                            fine(n, fk, fj, fi) = ccval - gx1c * dx1fm;
                            fine(n, fk, fj, fi + 1) = ccval + gx1c * dx1fp;
                          });
          }
        });
  }
}

} // namespace cell_centered_refinement
} // namespace parthenon
