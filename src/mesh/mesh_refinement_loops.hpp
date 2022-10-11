//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2022 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2022. Triad National Security, LLC. All rights reserved.
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

#ifndef MESH_MESH_REFINEMENT_LOOPS_HPP_
#define MESH_MESH_REFINEMENT_LOOPS_HPP_

#include <algorithm>
#include <utility> // std::forward
#include <vector>

#include "bvals/cc/bnd_info.hpp"       // for buffercache_t
#include "coordinates/coordinates.hpp" // for coordinates
#include "globals.hpp"                 // for Globals
#include "kokkos_abstraction.hpp"      // for ParArray
#include "mesh/domain.hpp"             // for IndexShape

namespace parthenon {
namespace refinement {
namespace loops {

// TODO(JMM) if LayoutLeft is ever relaxed, these might need to become
// template parameters
using Idx_t = ParArray1D<std::size_t>;
using IdxHost_t = typename ParArray1D<std::size_t>::HostMirror;

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
inline void
ProlongationRestrictionLoop(const cell_centered_bvars::BufferCache_t &info,
                            const Idx_t &buffer_idxs, const IndexShape &cellbounds,
                            const IndexShape &c_cellbounds, const RefinementOp_t op,
                            const std::size_t nbuffers) {
  const IndexDomain interior = IndexDomain::interior;
  auto ckb = c_cellbounds.GetBoundsK(interior);
  auto cjb = c_cellbounds.GetBoundsJ(interior);
  auto cib = c_cellbounds.GetBoundsI(interior);
  auto kb = cellbounds.GetBoundsK(interior);
  auto jb = cellbounds.GetBoundsJ(interior);
  auto ib = cellbounds.GetBoundsI(interior);
  const int scratch_level = 1; // 0 is actual scratch (tiny); 1 is HBM
  size_t scratch_size_in_bytes = 1;
  par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "ProlongateOrRestrictCellCenteredValues",
      DevExecSpace(), scratch_size_in_bytes, scratch_level, 0, nbuffers - 1,
      KOKKOS_LAMBDA(team_mbr_t team_member, const int sub_idx) {
        const std::size_t buf = buffer_idxs(sub_idx);
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
                            const IdxHost_t &buffer_idxs_h, const IndexShape &cellbounds,
                            const IndexShape &c_cellbounds, const RefinementOp_t op,
                            const std::size_t nbuffers) {
  const IndexDomain interior = IndexDomain::interior;
  auto ckb =
      c_cellbounds.GetBoundsK(interior); // TODO(JMM): This may need some additional
  auto cjb = c_cellbounds.GetBoundsJ(interior); // logic for different field centers
  auto cib =
      c_cellbounds.GetBoundsI(interior);     // perhaps the solution is to pass IndexShape
  auto kb = cellbounds.GetBoundsK(interior); // into the stencil directly.
  auto jb = cellbounds.GetBoundsJ(interior);
  auto ib = cellbounds.GetBoundsI(interior);
  for (int sub_idx = 0; sub_idx < nbuffers; ++sub_idx) {
    const std::size_t buf = buffer_idxs_h(sub_idx);
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
                            const Idx_t &buffer_idxs, const IdxHost_t &buffer_idxs_h,
                            const IndexShape &cellbounds, const IndexShape &c_cellbounds,
                            const RefinementOp_t op, const std::size_t nbuffers) {
  if (nbuffers > Globals::refinement::min_num_bufs) {
    ProlongationRestrictionLoop<DIM, Stencil>(info, buffer_idxs, cellbounds, c_cellbounds,
                                              op, nbuffers);
  } else {
    ProlongationRestrictionLoop<DIM, Stencil>(info_h, buffer_idxs_h, cellbounds,
                                              c_cellbounds, op, nbuffers);
  }
}

template <template <int> class Stencil, class... Args>
inline void DoProlongationRestrictionOp(const IndexShape &cellbnds, Args &&...args) {
  if (cellbnds.ncellsk(IndexDomain::entire) > 1) { // 3D
    ProlongationRestrictionLoop<3, Stencil>(std::forward<Args>(args)...);
  } else if (cellbnds.ncellsj(IndexDomain::entire) > 1) { // 2D
    ProlongationRestrictionLoop<2, Stencil>(std::forward<Args>(args)...);
  } else if (cellbnds.ncellsi(IndexDomain::entire) > 1) { // 1D
    ProlongationRestrictionLoop<1, Stencil>(std::forward<Args>(args)...);
  }
}

} // namespace loops
} // namespace refinement
} // namespace parthenon

#endif // MESH_MESH_REFINEMENT_LOOPS_HPP_
