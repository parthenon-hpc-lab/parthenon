//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2023 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2023. Triad National Security, LLC. All rights reserved.
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

#ifndef PROLONG_RESTRICT_PR_OPS_HPP_
#define PROLONG_RESTRICT_PR_OPS_HPP_

#include <algorithm>
#include <cstring>

#include "coordinates/coordinates.hpp"  // for coordinates
#include "interface/variable_state.hpp" // For variable state in ParArray
#include "kokkos_abstraction.hpp"       // ParArray
#include "mesh/domain.hpp"              // for IndesShape

/*
 * Explanation---To be added to docs, when pulled through as a user
 * interface.
 *
 * Each refinement op is defined as a templated struct (templated on
 * dimension) that defines the static "Do" operation.
 *
 * We do this because you can template on a templated class, but you
 * can't template on a templated function. I.e., the following works
 * with the struct formulation:
 *
 * template<template<int> typename F>
 * void DoSomething(const int i) {
 *     if (i == 1) F<1>::Do();
 *     if (i == 2) F<2>::Do();
 *     if (i == 3) F<3>::Do();
 * }
 *
 * DoSomething<RestrictCellAverage>(3);
 *
 * However the same call pattern would NOT work with a templated function.
 *
 * TODO(JMM): To enable custom prolongation/restriction operations, we
 * will need to provide (likely in state descriptor so it can be
 * per-variable) a templated function that registers the
 * prolongation/restriction LOOP functions, templated on custom user
 * classes that look like the ones below. The LOOP must be the thing
 * we register, to avoid having to deal with relocatable device code,
 * for performance reasons, as this bit of code is performance
 * critical.
 *
 * I thought of adding the registration machinery now, but I figured
 * smaller changes are probably better. That said, I wanted to put the
 * bones of the machinery in, because it reduces code duplication for,
 * e.g., the loop switching tooling.

 * TODO(JMM): Function signatures currently are real gross. Might be
 * worth coalescing some of this stuff into structs, rather than
 * unpacking it and then passing it in.
 */

namespace parthenon {
namespace refinement_ops {

namespace util {
// compute distances from cell center to the nearest center in the + or -
// coordinate direction. Do so for both coarse and fine grids.
template <int DIM, TopologicalElement EL>
KOKKOS_FORCEINLINE_FUNCTION void
GetGridSpacings(const Coordinates_t &coords, const Coordinates_t &coarse_coords,
                const IndexRange &cib, const IndexRange &ib, int i, int fi, Real *dxm,
                Real *dxp, Real *dxfm, Real *dxfp) {
  // here "f" signifies the fine grid, not face locations.
  const Real xm = coarse_coords.X<DIM, EL>(i - 1);
  const Real xc = coarse_coords.X<DIM, EL>(i);
  const Real xp = coarse_coords.X<DIM, EL>(i + 1);
  *dxm = xc - xm;
  *dxp = xp - xc;
  const Real fxm = coords.X<DIM, EL>(fi);
  const Real fxp = coords.X<DIM, EL>(fi + 1);
  *dxfm = xc - fxm;
  *dxfp = fxp - xc;
}

KOKKOS_FORCEINLINE_FUNCTION
Real GradMinMod(const Real fc, const Real fm, const Real fp, const Real dxm,
                const Real dxp, Real &gxm, Real &gxp) {
  gxm = (fc - fm) / dxm;
  gxp = (fp - fc) / dxp;
  return 0.5 * (SIGN(gxm) + SIGN(gxp)) * std::min(std::abs(gxm), std::abs(gxp));
}

} // namespace util

struct RestrictAverage {
  static constexpr bool OperationRequired(TopologicalElement fel,
                                          TopologicalElement cel) {
    return fel == cel;
  }

  template <int DIM, TopologicalElement el = TopologicalElement::CC,
            TopologicalElement /*cel*/ = TopologicalElement::CC>
  KOKKOS_FORCEINLINE_FUNCTION static void
  Do(const int l, const int m, const int n, const int ck, const int cj, const int ci,
     const IndexRange &ckb, const IndexRange &cjb, const IndexRange &cib,
     const IndexRange &kb, const IndexRange &jb, const IndexRange &ib,
     const Coordinates_t &coords, const Coordinates_t &coarse_coords,
     const ParArrayND<Real, VariableState> *pcoarse,
     const ParArrayND<Real, VariableState> *pfine) {
    constexpr bool INCLUDE_X1 =
        (DIM > 0) && (el == TE::CC || el == TE::F2 || el == TE::F3 || el == TE::E1);
    constexpr bool INCLUDE_X2 =
        (DIM > 1) && (el == TE::CC || el == TE::F3 || el == TE::F1 || el == TE::E2);
    constexpr bool INCLUDE_X3 =
        (DIM > 2) && (el == TE::CC || el == TE::F1 || el == TE::F2 || el == TE::E3);
    constexpr int element_idx = static_cast<int>(el) % 3;

    auto &coarse = *pcoarse;
    auto &fine = *pfine;

    const int i = (DIM > 0) ? (ci - cib.s) * 2 + ib.s : ib.s;
    const int j = (DIM > 1) ? (cj - cjb.s) * 2 + jb.s : jb.s;
    const int k = (DIM > 2) ? (ck - ckb.s) * 2 + kb.s : kb.s;

    // JMM: If dimensionality is wrong, accesses are out of bounds. Only
    // access cells if dimensionality is correct.
    Real vol[2][2][2], terms[2][2][2]; // memset not available on all accelerators
    for (int ok = 0; ok < 2; ++ok) {
      for (int oj = 0; oj < 2; ++oj) {
        for (int oi = 0; oi < 2; ++oi) {
          vol[ok][oj][oi] = terms[ok][oj][oi] = 0;
        }
      }
    }

    for (int ok = 0; ok < 1 + INCLUDE_X3; ++ok) {
      for (int oj = 0; oj < 1 + INCLUDE_X2; ++oj) {
        for (int oi = 0; oi < 1 + INCLUDE_X1; ++oi) {
          vol[ok][oj][oi] = coords.Volume<el>(k + ok, j + oj, i + oi);
          terms[ok][oj][oi] =
              vol[ok][oj][oi] * fine(element_idx, l, m, n, k + ok, j + oj, i + oi);
        }
      }
    }
    // KGF: add the off-centered quantities first to preserve FP
    // symmetry
    const Real tvol = ((vol[0][0][0] + vol[0][1][0]) + (vol[0][0][1] + vol[0][1][1])) +
                      ((vol[1][0][0] + vol[1][1][0]) + (vol[1][0][1] + vol[1][1][1]));
    coarse(element_idx, l, m, n, ck, cj, ci) =
        (((terms[0][0][0] + terms[0][1][0]) + (terms[0][0][1] + terms[0][1][1])) +
         ((terms[1][0][0] + terms[1][1][0]) + (terms[1][0][1] + terms[1][1][1]))) /
        tvol;
  }
};

template <bool use_minmod_slope>
struct ProlongateSharedGeneral {
  static constexpr bool OperationRequired(TopologicalElement fel,
                                          TopologicalElement cel) {
    return fel == cel;
  }

  template <int DIM, TopologicalElement el = TopologicalElement::CC,
            TopologicalElement /*cel*/ = TopologicalElement::CC>
  KOKKOS_FORCEINLINE_FUNCTION static void
  Do(const int l, const int m, const int n, const int k, const int j, const int i,
     const IndexRange &ckb, const IndexRange &cjb, const IndexRange &cib,
     const IndexRange &kb, const IndexRange &jb, const IndexRange &ib,
     const Coordinates_t &coords, const Coordinates_t &coarse_coords,
     const ParArrayND<Real, VariableState> *pcoarse,
     const ParArrayND<Real, VariableState> *pfine) {
    using namespace util;
    auto &coarse = *pcoarse;
    auto &fine = *pfine;

    constexpr int element_idx = static_cast<int>(el) % 3;

    const int fi = (DIM > 0) ? (i - cib.s) * 2 + ib.s : ib.s;
    const int fj = (DIM > 1) ? (j - cjb.s) * 2 + jb.s : jb.s;
    const int fk = (DIM > 2) ? (k - ckb.s) * 2 + kb.s : kb.s;

    constexpr bool INCLUDE_X1 =
        (DIM > 0) && (el == TE::CC || el == TE::F2 || el == TE::F3 || el == TE::E1);
    constexpr bool INCLUDE_X2 =
        (DIM > 1) && (el == TE::CC || el == TE::F3 || el == TE::F1 || el == TE::E2);
    constexpr bool INCLUDE_X3 =
        (DIM > 2) && (el == TE::CC || el == TE::F1 || el == TE::F2 || el == TE::E3);

    const Real fc = coarse(element_idx, l, m, n, k, j, i);

    Real dx1fm = 0;
    [[maybe_unused]] Real dx1fp = 0;
    Real gx1m = 0, gx1p = 0;
    if constexpr (INCLUDE_X1) {
      Real dx1m, dx1p;
      GetGridSpacings<1, el>(coords, coarse_coords, cib, ib, i, fi, &dx1m, &dx1p, &dx1fm,
                             &dx1fp);

      Real gx1c =
          GradMinMod(fc, coarse(element_idx, l, m, n, k, j, i - 1),
                     coarse(element_idx, l, m, n, k, j, i + 1), dx1m, dx1p, gx1m, gx1p);
      if constexpr (use_minmod_slope) {
        gx1m = gx1c;
        gx1p = gx1c;
      }
    }

    Real dx2fm = 0;
    [[maybe_unused]] Real dx2fp = 0;
    Real gx2m = 0, gx2p = 0;
    if constexpr (INCLUDE_X2) {
      Real dx2m, dx2p;
      GetGridSpacings<2, el>(coords, coarse_coords, cjb, jb, j, fj, &dx2m, &dx2p, &dx2fm,
                             &dx2fp);
      Real gx2c =
          GradMinMod(fc, coarse(element_idx, l, m, n, k, j - 1, i),
                     coarse(element_idx, l, m, n, k, j + 1, i), dx2m, dx2p, gx2m, gx2p);
      if constexpr (use_minmod_slope) {
        gx2m = gx2c;
        gx2p = gx2c;
      }
    }

    Real dx3fm = 0;
    [[maybe_unused]] Real dx3fp = 0;
    Real gx3c = 0;
    Real gx3m = 0, gx3p = 0;
    if constexpr (INCLUDE_X3) {
      Real dx3m, dx3p;
      GetGridSpacings<3, el>(coords, coarse_coords, ckb, kb, k, fk, &dx3m, &dx3p, &dx3fm,
                             &dx3fp);
      Real gx3c =
          GradMinMod(fc, coarse(element_idx, l, m, n, k - 1, j, i),
                     coarse(element_idx, l, m, n, k + 1, j, i), dx3m, dx3p, gx3m, gx3p);
      if constexpr (use_minmod_slope) {
        gx3m = gx3c;
        gx3p = gx3c;
      }
    }

    // KGF: add the off-centered quantities first to preserve FP symmetry
    // JMM: Extraneous quantities are zero
    fine(element_idx, l, m, n, fk, fj, fi) =
        fc - (gx1m * dx1fm + gx2m * dx2fm + gx3m * dx3fm);
    if constexpr (INCLUDE_X1)
      fine(element_idx, l, m, n, fk, fj, fi + 1) =
          fc + (gx1p * dx1fp - gx2m * dx2fm - gx3m * dx3fm);
    if constexpr (INCLUDE_X2)
      fine(element_idx, l, m, n, fk, fj + 1, fi) =
          fc - (gx1m * dx1fm - gx2p * dx2fp + gx3m * dx3fm);
    if constexpr (INCLUDE_X2 && INCLUDE_X1)
      fine(element_idx, l, m, n, fk, fj + 1, fi + 1) =
          fc + (gx1p * dx1fp + gx2p * dx2fp - gx3m * dx3fm);
    if constexpr (INCLUDE_X3)
      fine(element_idx, l, m, n, fk + 1, fj, fi) =
          fc - (gx1m * dx1fm + gx2m * dx2fm - gx3p * dx3fp);
    if constexpr (INCLUDE_X3 && INCLUDE_X1)
      fine(element_idx, l, m, n, fk + 1, fj, fi + 1) =
          fc + (gx1p * dx1fp - gx2m * dx2fm + gx3p * dx3fp);
    if constexpr (INCLUDE_X3 && INCLUDE_X2)
      fine(element_idx, l, m, n, fk + 1, fj + 1, fi) =
          fc - (gx1m * dx1fm - gx2p * dx2fp - gx3p * dx3fp);
    if constexpr (INCLUDE_X3 && INCLUDE_X2 && INCLUDE_X1)
      fine(element_idx, l, m, n, fk + 1, fj + 1, fi + 1) =
          fc + (gx1p * dx1fp + gx2p * dx2fp + gx3p * dx3fp);
  }
};

using ProlongateSharedMinMod = ProlongateSharedGeneral<true>;
using ProlongateSharedLinear = ProlongateSharedGeneral<false>;

struct ProlongateInternalAverage {
  static constexpr bool OperationRequired(TopologicalElement fel,
                                          TopologicalElement cel) {
    return IsSubmanifold(fel, cel);
  }
  // Here, fel is the topological element on which the field is defined and
  // cel is the topological element on which we are filling the internal values
  // of the field. So, for instance, we could fill the fine cell values of an
  // x-face field within the volume of a coarse cell. This is assumes that the
  // values of the fine cells on the elements corresponding with the coarse cell
  // have been filled.
  template <int DIM, TopologicalElement fel = TopologicalElement::CC,
            TopologicalElement cel = TopologicalElement::CC>
  KOKKOS_FORCEINLINE_FUNCTION static void
  Do(const int l, const int m, const int n, const int k, const int j, const int i,
     const IndexRange &ckb, const IndexRange &cjb, const IndexRange &cib,
     const IndexRange &kb, const IndexRange &jb, const IndexRange &ib,
     const Coordinates_t &coords, const Coordinates_t &coarse_coords,
     const ParArrayND<Real, VariableState> *,
     const ParArrayND<Real, VariableState> *pfine) {
    using namespace util;

    if constexpr (!IsSubmanifold(fel, cel)) {
      return;
    } else {
      auto &fine = *pfine;

      constexpr int element_idx = static_cast<int>(fel) % 3;

      // The incoming {k, j, i} coordinates should be thought of as the coordinates
      // of the topological element cel on the coarse grid, while the fine grid
      // coordinate range defined by {[fk, fk + CENTER_X2], [fj, fj + CENTER_X2],
      // [fi, fi + CENTER_X1]} for the topological element fel is what gets filled.
      // Therefore, this method should be interated over an Indexer defined on the
      // coarse grid for the topological element cel. This will result in the
      // correct set of internal points being set.
      const int fi = (DIM > 0) ? (i - cib.s) * 2 + ib.s : ib.s;
      const int fj = (DIM > 1) ? (j - cjb.s) * 2 + jb.s : jb.s;
      const int fk = (DIM > 2) ? (k - ckb.s) * 2 + kb.s : kb.s;

      // Determine wether or not the fields coordinates are on coordinate centers (i.e.
      // same coordinate position as a zone center)
      constexpr bool CENTER_X1 =
          (DIM > 0) && (fel == TE::CC || fel == TE::F2 || fel == TE::F3 || fel == TE::E1);
      constexpr bool CENTER_X2 =
          (DIM > 1) && (fel == TE::CC || fel == TE::F3 || fel == TE::F1 || fel == TE::E2);
      constexpr bool CENTER_X3 =
          (DIM > 2) && (fel == TE::CC || fel == TE::F1 || fel == TE::F2 || fel == TE::E3);

      // Determine the directions we want our averaging stencil to extend in
      constexpr bool STENCIL_X1 =
          (DIM > 0) && !CENTER_X1 &&
          (cel == TE::CC || cel == TE::F2 || cel == TE::F3 || cel == TE::E1);
      constexpr bool STENCIL_X2 =
          (DIM > 1) && !CENTER_X2 &&
          (cel == TE::CC || cel == TE::F3 || cel == TE::F1 || cel == TE::E2);
      constexpr bool STENCIL_X3 =
          (DIM > 2) && !CENTER_X3 &&
          (cel == TE::CC || cel == TE::F1 || cel == TE::F2 || cel == TE::E3);

      // Prolongate elements internal to topological element el_avg by averaging over
      // coarse region defined by {cel, k, j, i}
      constexpr Real w =
          1.0 / ((1.0 + STENCIL_X3) * (1.0 + STENCIL_X2) * (1.0 + STENCIL_X1));

      // Iterate over all interior fine elements on {cel, k, j, i}
      for (int ok = 0; ok < 1 + CENTER_X3; ++ok) {
        for (int oj = 0; oj < 1 + CENTER_X2; ++oj) {
          for (int oi = 0; oi < 1 + CENTER_X1; ++oi) {
            // Indices of the fine element that is currently getting filled
            const int tk = fk + ok + STENCIL_X3;
            const int tj = fj + oj + STENCIL_X2;
            const int ti = fi + oi + STENCIL_X1;
            Real f = 0.0;
            // Iterate over the appropriate stencil of fine shared elements
            // for the current interior fine element
            for (int stk = -STENCIL_X3; stk <= STENCIL_X3; stk += 2) {
              for (int stj = -STENCIL_X2; stj <= STENCIL_X2; stj += 2) {
                for (int sti = -STENCIL_X1; sti <= STENCIL_X1; sti += 2) {
                  // LFR: Obviously, you could generalize this to more complicated
                  // stencils with a weight array and a larger range.
                  f += w * fine(element_idx, l, m, n, tk + stk, tj + stj, ti + sti);
                }
              }
            }
            fine(element_idx, l, m, n, tk, tj, ti) = f;
          }
        }
      }
    }
  }
};

} // namespace refinement_ops
} // namespace parthenon

#endif // PROLONG_RESTRICT_PR_OPS_HPP_
