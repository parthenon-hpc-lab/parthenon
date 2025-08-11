//========================================================================================
// (C) (or copyright) 2020-2024. Triad National Security, LLC. All rights
// reserved.
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

#ifndef SRC_UTILITIES_RECONSTRUCT_HPP_
#define SRC_UTILITIES_RECONSTRUCT_HPP_

#include <memory>
#include <vector>

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <utils/indexer.hpp>

#include "utilities/scratch_pack.hpp"

namespace scalar_imex {
using namespace parthenon::driver::prelude;

KOKKOS_FORCEINLINE_FUNCTION
Real mc(const Real dm, const Real dp, const Real alpha = 1.99) {
  const Real dc = (dm * dp > 0.0) * 0.5 * (dm + dp);
  return std::copysign(
      std::min(std::fabs(dc), alpha * std::min(std::fabs(dm), std::fabs(dp))), dc);
}

KOKKOS_INLINE_FUNCTION
void PiecewiseLinear(const Real qm, const Real q0, const Real qp, Real &p, Real &m,
                     const Real slope_limit = 1.99) {
  Real dq = qp - q0;
  // const Real slope_limit = 0.99 / ((wgt <= 0.5)*wgt + (wgt > 0.5)*(1.0-wgt));
  dq = 0.5 * mc(q0 - qm, dq, slope_limit);
  p = q0 + dq; // wgt *dq;
  m = q0 - dq; //(1.0 - wgt) * dq;
}

template <class pack_t>
KOKKOS_INLINE_FUNCTION void
Reconstruct(parthenon::team_mbr_t member, const int b, const int k,
            parthenon::TopologicalElement flux_te,
            const parthenon::utils::IndexingData &cellbounds, const pack_t &pack,
            parthenon::utils::ScratchPack<pack_t> &wm,
            parthenon::utils::ScratchPack<pack_t> &wp) {
  const auto [kb, jb, ib] = cellbounds.GetReconstructionRange(flux_te);
  const auto [kbe, jbe, ibe] = cellbounds.Get3DIndexRange(IndexDomain::entire);
  const parthenon::Indexer2D idxer_full({jbe.s, jbe.e}, {ibe.s, ibe.e});
  const int npoints =
      idxer_full.GetFlatIdx(jb.e, ib.e) - idxer_full.GetFlatIdx(jb.s, ib.s) + 1;
  parthenon::Indexer2D idxer_recon({jb.s, jb.e}, {ib.s, ib.e});
  const auto [koff, joff, ioff] = cellbounds.GetOffsetArray(flux_te);
  // Do L/R reconstruction across this slab
  for (int l = pack.GetLowerBound(b); l <= pack.GetUpperBound(b); ++l) {
    Real *pl = &pack(b, l, k - koff, jb.s - joff, ib.s - ioff);
    Real *pm = &pack(b, l, k, jb.s, ib.s);
    Real *pu = &pack(b, l, k + koff, jb.s + joff, ib.s + ioff);
    Real *pwp = &wp(l, jb.s, ib.s);
    Real *pwm = &wm(l, jb.s, ib.s);
    parthenon::par_for_inner(member, 0, npoints - 1, [&](const int idx) {
      PiecewiseLinear(pl[idx], pm[idx], pu[idx], pwp[idx], pwm[idx]);
    });
  }
}

} // namespace scalar_imex

#endif // SRC_UTILITIES_RECONSTRUCT_HPP_
