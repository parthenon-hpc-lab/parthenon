//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2022 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
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

#ifndef MESH_REFINEMENT_CC_IN_ONE_HPP_
#define MESH_REFINEMENT_CC_IN_ONE_HPP_

#include <algorithm>
#include <utility> // std::forward
#include <vector>

#include "bvals/cc/bvals_cc_in_one.hpp" // for buffercache_t
#include "coordinates/coordinates.hpp"  // for coordinates
#include "globals.hpp"                  // for Globals
#include "interface/mesh_data.hpp"
#include "mesh/domain.hpp" // for IndexShape

namespace parthenon {
namespace cell_centered_refinement {
// The existence of this overload allows us to avoid a deep-copy in
// the per-meshblock calls
// TODO(JMM): I don't love having two overloads here.  However when
// we shift entirely to in-one machinery, the info_h only overload
// will go away.
void Restrict(const cell_centered_bvars::BufferCache_t &info,
              const cell_centered_bvars::BufferCacheHost_t &info_h,
              const IndexShape &cellbounds, const IndexShape &c_cellbounds);
// The existence of this overload allows us to avoid a deep-copy in
// the per-meshblock calls
void Restrict(const cell_centered_bvars::BufferCacheHost_t &info_h,
              const IndexShape &cellbounds, const IndexShape &c_cellbounds);

TaskStatus RestrictPhysicalBounds(MeshData<Real> *md);

std::vector<bool> ComputePhysicalRestrictBoundsAllocStatus(MeshData<Real> *md);
void ComputePhysicalRestrictBounds(MeshData<Real> *md);

// TODO(JMM): I don't love having two overloads here.  However when
// we shift entirely to in-one machinery, the info_h only overload
// will go away.
void Prolongate(const cell_centered_bvars::BufferCache_t &info,
                const cell_centered_bvars::BufferCacheHost_t &info_h,
                const IndexShape &cellbounds, const IndexShape &c_cellbounds);
// The existence of this overload allows us to avoid a deep-copy in
// the per-meshblock calls
void Prolongate(const cell_centered_bvars::BufferCacheHost_t &info_h,
                const IndexShape &cellbounds, const IndexShape &c_cellbounds);

// TODO(JMM): We may wish to expose some of these impl functions eventually.
namespace impl {

template <typename Info_t>
KOKKOS_FORCEINLINE_FUNCTION bool DoRefinementOp(const Info_t &info,
                                                const RefinementOp_t op) {
  return (info.allocated && info.refinement_op == op);
}

template <int DIM, typename Info_t>
KOKKOS_FORCEINLINE_FUNCTION void
GetLoopBoundsFromBndInfo(const Info_t &info, const int ckbs, const int cjbs, int &sk,
                         int &ek, int &sj, int &ej, int &si, int &ei) {
  sk = info.sk;
  ek = info.ek;
  sj = info.sj;
  ej = info.ej;
  si = info.si;
  ei = info.ei;
  if (DIM < 3) sk = ek = ckbs; // TODO(C++17) make constexpr
  if (DIM < 2) sj = ej = cjbs;
}

// JMM: A single prolongation/restriction loop template without
// specializations is possible, if we're willing to always do the 6D
// loop with different specialized loop bounds. The danger of that
// approach is that if, e.g., a TVVR loop pattern is utilized at lower
// dimensionality but not higher-dimensionality, the pattern may not
// work out optimally. I have implemented it here, but we may wish to
// revert to separate loops per dimension, if the performance hit is
// too large.
//
// There's a host version of the loop, which only requires buffer cache host,
// a device version, which requires the buffer cache device only,
// and a version that automatically swaps between them depending on
// the size of the buffer cache.
template <int DIM, template <int> class Stencil>
inline void ProlongationRestrictionLoop(const cell_centered_bvars::BufferCache_t &info,
                                        const IndexShape &cellbounds,
                                        const IndexShape &c_cellbounds,
                                        const RefinementOp_t op) {
  const IndexDomain interior = IndexDomain::interior;
  auto ckb = c_cellbounds.GetBoundsK(interior);
  auto cjb = c_cellbounds.GetBoundsJ(interior);
  auto cib = c_cellbounds.GetBoundsI(interior);
  auto kb = cellbounds.GetBoundsK(interior);
  auto jb = cellbounds.GetBoundsJ(interior);
  auto ib = cellbounds.GetBoundsI(interior);
  const int nbuffers = info.extent_int(0);
  const int scratch_level = 1; // 0 is actual scratch (tiny); 1 is HBM
  size_t scratch_size_in_bytes = 1;
  par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "ProlongateOrRestrictCellCenteredValues",
      DevExecSpace(), scratch_size_in_bytes, scratch_level, 0, nbuffers - 1,
      KOKKOS_LAMBDA(team_mbr_t team_member, const int buf) {
        if (DoRefinementOp(info(buf), op)) {
          int sk, ek, sj, ej, si, ei;
          GetLoopBoundsFromBndInfo<DIM>(info(buf), ckb.s, cjb.s, sk, ek, sj, ej, si, ei);
          par_for_inner(inner_loop_pattern_ttr_tag, team_member, 0, info(buf).Nt - 1, 0,
                        info(buf).Nu - 1, 0, info(buf).Nv - 1, sk, ek, sj, ej, si, ei,
                        [&](const int t, const int u, const int v, const int k,
                            const int j, const int i) {
                          Stencil<DIM>::Do(t, u, v, k, j, i, ckb, cjb, cib, kb, jb, ib,
                                           info(buf).coords, info(buf).coarse_coords,
                                           &(info(buf).coarse), &(info(buf).fine));
                        });
        }
      });
}
template <int DIM, template <int> class Stencil>
inline void
ProlongationRestrictionLoop(const cell_centered_bvars::BufferCacheHost_t &info_h,
                            const IndexShape &cellbounds, const IndexShape &c_cellbounds,
                            const RefinementOp_t op) {
  const IndexDomain interior = IndexDomain::interior;
  auto ckb =
      c_cellbounds.GetBoundsK(interior); // TODO(JMM): This may need some additional
  auto cjb = c_cellbounds.GetBoundsJ(interior); // logic for different field centers
  auto cib =
      c_cellbounds.GetBoundsI(interior);     // perhaps the solution is to pass IndexShape
  auto kb = cellbounds.GetBoundsK(interior); // into the stencil directly.
  auto jb = cellbounds.GetBoundsJ(interior);
  auto ib = cellbounds.GetBoundsI(interior);
  const int nbuffers = info_h.extent_int(0);
  for (int buf = 0; buf < nbuffers; ++buf) {
    if (DoRefinementOp(info_h(buf), op)) {
      int sk, ek, sj, ej, si, ei;
      GetLoopBoundsFromBndInfo<DIM>(info_h(buf), ckb.s, cjb.s, sk, ek, sj, ej, si, ei);
      auto coords = info_h(buf).coords;
      auto coarse_coords = info_h(buf).coarse_coords;
      auto coarse = info_h(buf).coarse;
      auto fine = info_h(buf).fine;
      par_for(
          DEFAULT_LOOP_PATTERN, "ProlongateOrRestrictCellCenteredValues", DevExecSpace(),
          0, info_h(buf).Nt - 1, 0, info_h(buf).Nu - 1, 0, info_h(buf).Nv - 1, sk, ek, sj,
          ej, si, ei,
          KOKKOS_LAMBDA(const int t, const int u, const int v, const int k, const int j,
                        const int i) {
            Stencil<DIM>::Do(t, u, v, k, j, i, ckb, cjb, cib, kb, jb, ib, coords,
                             coarse_coords, &coarse, &fine);
          });
    }
  }
}
template <int DIM, template <int> class Stencil>
inline void
ProlongationRestrictionLoop(const cell_centered_bvars::BufferCache_t &info,
                            const cell_centered_bvars::BufferCacheHost_t &info_h,
                            const IndexShape &cellbounds, const IndexShape &c_cellbounds,
                            const RefinementOp_t op) {
  const int nbuffers = info_h.extent_int(0);
  if (nbuffers > Globals::cell_centered_refinement::min_num_bufs) {
    ProlongationRestrictionLoop<DIM, Stencil>(info, cellbounds, c_cellbounds, op);
  } else {
    ProlongationRestrictionLoop<DIM, Stencil>(info_h, cellbounds, c_cellbounds, op);
  }
}

template <template <int> class Stencil, typename... Args>
inline void DoProlongationRestrictionOp(const IndexShape &cellbnds, Args &&...args) {
  const IndexDomain entire = IndexDomain::entire;
  if (cellbnds.ncellsk(entire) > 1) { // 3D
    ProlongationRestrictionLoop<3, Stencil>(std::forward<Args>(args)...);
  } else if (cellbnds.ncellsj(entire) > 1) { // 2D
    ProlongationRestrictionLoop<2, Stencil>(std::forward<Args>(args)...);
  } else if (cellbnds.ncellsi(entire) > 1) { // 1D
    ProlongationRestrictionLoop<1, Stencil>(std::forward<Args>(args)...);
  }
}

} // namespace impl

} // namespace cell_centered_refinement
} // namespace parthenon

#endif // MESH_REFINEMENT_CC_IN_ONE_HPP_
