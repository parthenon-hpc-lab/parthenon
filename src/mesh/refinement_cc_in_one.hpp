//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
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

#ifndef MESH_REFINEMENT_CC_IN_ONE_HPP_
#define MESH_REFINEMEN_CC_IN_ONE_HPP_

#include "coordinates/coordinates.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh_refinement.hpp" // for RefinementInfo

namespace parthenon {
namespace cell_centered_refinement {
template <typename Pack>
void Restrict(const Pack &fine, Pack &coarse, ParArray1D<RefinementInfo> &info) {
  int nblocks = fine.GetDim(5);
  int nvars = fine.GetDim(4);
  const int scratch_level = 0; // 0 is actual scratch (tiny); 1 is HBM
  // parthenon::ScratchPad2D<Real>::shmem_size(nblocks, nvars);
  size_t scratch_size_in_bytes = 0;
  if (fine.GetDim(3) > 1) { // 3D
    par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, "RestrictCellCenteredValues3d", DevExecSpace(),
        scratch_size_in_bytes, scratch_level, 0, nblocks - 1, 0, nvars - 1,
        KOKKOS_LAMBDA(team_mbr_t team_member, const int b, const int n) {
          par_for_inner(
              DEFAULT_INNER_LOOP_PATTERN, team_member, info(b).sk, info(b).ek, info(b).sj,
              info(b).ej, info(b).si, info(b).ei,
              KOKKOS_LAMBDA(const int ck, const int cj, const int ci) {
                const int k = (ck - info(b).ckb.s) * 2 + info(b).kb.s;
                const int j = (cj - info(b).cjb.s) * 2 + info(b).jb.s;
                const int i = (ci - info(b).cib.s) * 2 + info(b).ib.s;
                // KGF: add the off-centered quantities first to preserve FP symmetry
                const Real vol000 = info(b).coords.Volume(k, j, i);
                const Real vol001 = info(b).coords.Volume(k, j, i + 1);
                const Real vol010 = info(b).coords.Volume(k, j + 1, i);
                const Real vol011 = info(b).coords.Volume(k, j + 1, i + 1);
                const Real vol100 = info(b).coords.Volume(k + 1, j, i);
                const Real vol101 = info(b).coords.Volume(k + 1, j, i + 1);
                const Real vol110 = info(b).coords.Volume(k + 1, j + 1, i);
                const Real vol111 = info(b).coords.Volume(k + 1, j + 1, i + 1);
                Real tvol = ((vol000 + vol010) + (vol001 + vol011)) +
                            ((vol100 + vol110) + (vol101 + vol111));
                // KGF: add the off-centered quantities first to preserve FP symmetry
                coarse(b, n, ck, cj, ci) =
                    (((fine(b, n, k, j, i) * vol000 + fine(b, n, k, j + 1, i) * vol010) +
                      (fine(b, n, k, j, i + 1) * vol001 +
                       fine(b, n, k, j + 1, i + 1) * vol011)) +
                     ((fine(b, n, k + 1, j, i) * vol100 +
                       fine(b, n, k + 1, j + 1, i) * vol110) +
                      (fine(b, n, k + 1, j, i + 1) * vol101 +
                       fine(b, n, k + 1, j + 1, i + 1) * vol111))) /
                    tvol;
              });
        });
  } else if (fine.GetDim(2) > 1) { // 2D
    par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, "RestrictCellCenteredValues2d", DevExecSpace(),
        scratch_size_in_bytes, scratch_level, 0, nblocks - 1, 0, nvars - 1,
        KOKKOS_LAMBDA(team_mbr_t team_member, const int b, const int n) {
          const int k = info(b).kb.s;
          par_for_inner(
              DEFAULT_INNER_LOOP_PATTERN, team_member, info(b).sj, info(b).ej, info(b).si,
              info(b).ei, KOKKOS_LAMBDA(const int cj, const int ci) {
                const int j = (cj - info(b).cjb.s) * 2 + info(b).jb.s;
                const int i = (ci - info(b).cib.s) * 2 + info(b).ib.s;
                // KGF: add the off-centered quantities first to preserve FP symmetry
                const Real vol00 = info(b).coords.Volume(k, j, i);
                const Real vol10 = info(b).coords.Volume(k, j + 1, i);
                const Real vol01 = info(b).coords.Volume(k, j, i + 1);
                const Real vol11 = info(b).coords.Volume(k, j + 1, i + 1);
                Real tvol = (vol00 + vol10) + (vol01 + vol11);

                // KGF: add the off-centered quantities first to preserve FP symmetry
                coarse(b, n, 0, cj, ci) =
                    ((fine(b, n, 0, j, i) * vol00 + fine(b, n, 0, j + 1, i) * vol10) +
                     (fine(b, n, 0, j, i + 1) * vol01 +
                      fine(b, n, 0, j + 1, i + 1) * vol11)) /
                    tvol;
              });
        });
  } else if (fine.GetDim(1) > 1) { // 1D
    par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, "RestrictCellCenteredValues1d", DevExecSpace(),
        scratch_size_in_bytes, scratch_level, 0, nblocks - 1, 0, nvars - 1,
        KOKKOS_LAMBDA(team_mbr_t team_member, const int b, const int n) {
          const int ck = info(b).ckb.s;
          const int cj = info(b).cjb.s;
          const int k = info(b).kb.s;
          const int j = info(b).jb.s;
          par_for_inner(
              DEFAULT_INNER_LOOP_PATTERN, team_member, info(b).si, info(b).ei,
              KOKKOS_LAMBDA(const int ci) {
                const int i = (ci - info(b).cib.s) * 2 + info(b).ib.s;
                const Real vol0 = info(b).coords.Volume(k, j, i);
                const Real vol1 = info(b).coords.Volume(k, j, i + 1);
                Real tvol = vol0 + vol1;
                coarse(b, n, ck, cj, ci) =
                    (fine(b, n, k, j, i) * vol0 + fine(b, n, k, j, i + 1) * vol1) / tvol;
              });
        });
  }
}

} // namespace cell_centered_refinement

} // namespace parthenon

#endif // MESH_REFINEMENT_CC_IN_ONE_HPP_
