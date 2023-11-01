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

#ifndef PROLONG_RESTRICT_PR_LOOPS_HPP_
#define PROLONG_RESTRICT_PR_LOOPS_HPP_

#include <algorithm>
#include <utility> // std::forward
#include <vector>

#include "bvals/comms/bnd_info.hpp"    // for buffercache_t
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

template <int DIM, class Stencil, TopologicalElement FEL, TopologicalElement CEL>
KOKKOS_INLINE_FUNCTION void InnerProlongationRestrictionLoop(
    team_mbr_t &team_member, std::size_t buf, const ProResInfoArr_t &info,
    const IndexRange &ckb, const IndexRange &cjb, const IndexRange &cib,
    const IndexRange &kb, const IndexRange &jb, const IndexRange &ib) {
  PARTHENON_INSTRUMENT
  const auto &idxer = info(buf).idxer[static_cast<int>(CEL)];
  par_for_inner(
      inner_loop_pattern_tvr_tag, team_member, 0, idxer.size() - 1, [&](const int ii) {
        const auto [t, u, v, k, j, i] = idxer(ii);
        if (idxer.IsActive(k, j, i)) {
          Stencil::template Do<DIM, FEL, CEL>(t, u, v, k, j, i, ckb, cjb, cib, kb, jb, ib,
                                              info(buf).coords, info(buf).coarse_coords,
                                              &(info(buf).coarse), &(info(buf).fine));
        }
      });
}

template <int DIM, class Stencil, TopologicalElement... ELs, class... Args>
KOKKOS_INLINE_FUNCTION void IterateInnerProlongationRestrictionLoop(Args &&...args) {
  (
      [&] {
        if constexpr (Stencil::OperationRequired(ELs, TE::NN))
          InnerProlongationRestrictionLoop<DIM, Stencil, ELs, TE::NN>(
              std::forward<Args>(args)...);
        if constexpr (Stencil::OperationRequired(ELs, TE::E3))
          InnerProlongationRestrictionLoop<DIM, Stencil, ELs, TE::E3>(
              std::forward<Args>(args)...);
        if constexpr (Stencil::OperationRequired(ELs, TE::E2))
          InnerProlongationRestrictionLoop<DIM, Stencil, ELs, TE::E2>(
              std::forward<Args>(args)...);
        if constexpr (Stencil::OperationRequired(ELs, TE::E1))
          InnerProlongationRestrictionLoop<DIM, Stencil, ELs, TE::E1>(
              std::forward<Args>(args)...);
        if constexpr (Stencil::OperationRequired(ELs, TE::F1))
          InnerProlongationRestrictionLoop<DIM, Stencil, ELs, TE::F1>(
              std::forward<Args>(args)...);
        if constexpr (Stencil::OperationRequired(ELs, TE::F2))
          InnerProlongationRestrictionLoop<DIM, Stencil, ELs, TE::F2>(
              std::forward<Args>(args)...);
        if constexpr (Stencil::OperationRequired(ELs, TE::F3))
          InnerProlongationRestrictionLoop<DIM, Stencil, ELs, TE::F3>(
              std::forward<Args>(args)...);
        if constexpr (Stencil::OperationRequired(ELs, TE::CC))
          InnerProlongationRestrictionLoop<DIM, Stencil, ELs, TE::CC>(
              std::forward<Args>(args)...);
      }(),
      ...);
}

template <int DIM, class Stencil>
inline void
ProlongationRestrictionLoop(const ProResInfoArr_t &info, const Idx_t &buffer_idxs,
                            const IndexShape &cellbounds, const IndexShape &c_cellbounds,
                            const RefinementOp_t op, const std::size_t nbuffers) {
  PARTHENON_INSTRUMENT
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
      DEFAULT_OUTER_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(),
      scratch_size_in_bytes, scratch_level, 0, nbuffers - 1,
      KOKKOS_LAMBDA(team_mbr_t team_member, const int sub_idx) {
        const std::size_t buf = buffer_idxs(sub_idx);
        if (DoRefinementOp(info(buf), op)) {
          using TE = TopologicalElement;
          if (info(buf).fine.topological_type == TopologicalType::Cell)
            IterateInnerProlongationRestrictionLoop<DIM, Stencil, TE::CC>(
                team_member, buf, info, ckb, cjb, cib, kb, jb, ib);
          if (info(buf).fine.topological_type == TopologicalType::Face)
            IterateInnerProlongationRestrictionLoop<DIM, Stencil, TE::F1, TE::F2, TE::F3>(
                team_member, buf, info, ckb, cjb, cib, kb, jb, ib);
          if (info(buf).fine.topological_type == TopologicalType::Edge)
            IterateInnerProlongationRestrictionLoop<DIM, Stencil, TE::E3, TE::E2, TE::E1>(
                team_member, buf, info, ckb, cjb, cib, kb, jb, ib);
          if (info(buf).fine.topological_type == TopologicalType::Node)
            IterateInnerProlongationRestrictionLoop<DIM, Stencil, TE::NN>(
                team_member, buf, info, ckb, cjb, cib, kb, jb, ib);
        }
      });
}

template <int DIM, class Stencil, TopologicalElement FEL, TopologicalElement CEL>
inline void
InnerHostProlongationRestrictionLoop(std::size_t buf, const ProResInfoArrHost_t &info,
                                     const IndexRange &ckb, const IndexRange &cjb,
                                     const IndexRange &cib, const IndexRange &kb,
                                     const IndexRange &jb, const IndexRange &ib) {
  PARTHENON_INSTRUMENT
  const auto &idxer = info(buf).idxer[static_cast<int>(CEL)];
  auto coords = info(buf).coords;
  auto coarse_coords = info(buf).coarse_coords;
  auto coarse = info(buf).coarse;
  auto fine = info(buf).fine;
  par_for(
      DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), 0, 0, 0, 0, 0,
      idxer.size() - 1, KOKKOS_LAMBDA(const int, const int, const int ii) {
        const auto [t, u, v, k, j, i] = idxer(ii);
        if (idxer.IsActive(k, j, i)) {
          Stencil::template Do<DIM, FEL, CEL>(t, u, v, k, j, i, ckb, cjb, cib, kb, jb, ib,
                                              coords, coarse_coords, &coarse, &fine);
        }
      });
}

template <int DIM, class Stencil, TopologicalElement... ELs, class... Args>
inline void IterateInnerHostProlongationRestrictionLoop(Args &&...args) {
  (
      [&] {
        if constexpr (Stencil::OperationRequired(ELs, TE::NN))
          InnerHostProlongationRestrictionLoop<DIM, Stencil, ELs, TE::NN>(
              std::forward<Args>(args)...);
        if constexpr (Stencil::OperationRequired(ELs, TE::E3))
          InnerHostProlongationRestrictionLoop<DIM, Stencil, ELs, TE::E3>(
              std::forward<Args>(args)...);
        if constexpr (Stencil::OperationRequired(ELs, TE::E2))
          InnerHostProlongationRestrictionLoop<DIM, Stencil, ELs, TE::E2>(
              std::forward<Args>(args)...);
        if constexpr (Stencil::OperationRequired(ELs, TE::E1))
          InnerHostProlongationRestrictionLoop<DIM, Stencil, ELs, TE::E1>(
              std::forward<Args>(args)...);
        if constexpr (Stencil::OperationRequired(ELs, TE::F1))
          InnerHostProlongationRestrictionLoop<DIM, Stencil, ELs, TE::F1>(
              std::forward<Args>(args)...);
        if constexpr (Stencil::OperationRequired(ELs, TE::F2))
          InnerHostProlongationRestrictionLoop<DIM, Stencil, ELs, TE::F2>(
              std::forward<Args>(args)...);
        if constexpr (Stencil::OperationRequired(ELs, TE::F3))
          InnerHostProlongationRestrictionLoop<DIM, Stencil, ELs, TE::F3>(
              std::forward<Args>(args)...);
        if constexpr (Stencil::OperationRequired(ELs, TE::CC))
          InnerHostProlongationRestrictionLoop<DIM, Stencil, ELs, TE::CC>(
              std::forward<Args>(args)...);
      }(),
      ...);
}

template <int DIM, class Stencil>
inline void
ProlongationRestrictionLoop(const ProResInfoArrHost_t &info_h,
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
      using TE = TopologicalElement;
      if (info_h(buf).fine.topological_type == TopologicalType::Cell)
        IterateInnerHostProlongationRestrictionLoop<DIM, Stencil, TE::CC>(
            buf, info_h, ckb, cjb, cib, kb, jb, ib);
      if (info_h(buf).fine.topological_type == TopologicalType::Face)
        IterateInnerHostProlongationRestrictionLoop<DIM, Stencil, TE::F1, TE::F2, TE::F3>(
            buf, info_h, ckb, cjb, cib, kb, jb, ib);
      if (info_h(buf).fine.topological_type == TopologicalType::Edge)
        IterateInnerHostProlongationRestrictionLoop<DIM, Stencil, TE::E3, TE::E2, TE::E1>(
            buf, info_h, ckb, cjb, cib, kb, jb, ib);
      if (info_h(buf).fine.topological_type == TopologicalType::Node)
        IterateInnerHostProlongationRestrictionLoop<DIM, Stencil, TE::NN>(
            buf, info_h, ckb, cjb, cib, kb, jb, ib);
    }
  }
}
template <int DIM, class Stencil>
inline void
ProlongationRestrictionLoop(const ProResInfoArr_t &info,
                            const ProResInfoArrHost_t &info_h, const Idx_t &buffer_idxs,
                            const IdxHost_t &buffer_idxs_h, const IndexShape &cellbounds,
                            const IndexShape &c_cellbounds, const RefinementOp_t op,
                            const std::size_t nbuffers) {
  if (nbuffers > Globals::refinement::min_num_bufs) {
    ProlongationRestrictionLoop<DIM, Stencil>(info, buffer_idxs, cellbounds, c_cellbounds,
                                              op, nbuffers);
  } else {
    ProlongationRestrictionLoop<DIM, Stencil>(info_h, buffer_idxs_h, cellbounds,
                                              c_cellbounds, op, nbuffers);
  }
}

template <class Stencil, class... Args>
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

#endif // PROLONG_RESTRICT_PR_LOOPS_HPP_
