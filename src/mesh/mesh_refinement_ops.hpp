//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2022 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
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

#ifndef MESH_MESH_REFINEMENT_OPS_HPP_
#define MESH_MESH_REFINEMENT_OPS_HPP_

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
 * Note that inspection of assembly indicates that for if statements
 * that depend on a templated function, the if is resolved at compile
 * time EVEN WITHOUT CONSTEXPR FROM C++17. The compiler is smart
 * enough to optimize out the branching. This is why the template
 * machinery is used, rather than a simple run-time parameter for the
 * dimensionality.
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
 *
 * TODO(JMM): Compared to the previous version of the code, this one
 * multiplies by zero sometimes.
 */

namespace parthenon {
namespace refinement_ops {

namespace util {
// TODO(JMM): this could be simplified if grid spacing was always uniform
template <int DIM>
KOKKOS_INLINE_FUNCTION Real GetXCC(const Coordinates_t &coords, int i);
template <>
KOKKOS_INLINE_FUNCTION Real GetXCC<1>(const Coordinates_t &coords, int i) {
  return coords.x1v(i);
}
template <>
KOKKOS_INLINE_FUNCTION Real GetXCC<2>(const Coordinates_t &coords, int i) {
  return coords.x2v(i);
}
template <>
KOKKOS_INLINE_FUNCTION Real GetXCC<3>(const Coordinates_t &coords, int i) {
  return coords.x3v(i);
}
// compute distances from cell center to the nearest center in the + or -
// coordinate direction. Do so for both coarse and fine grids.
template <int DIM>
KOKKOS_FORCEINLINE_FUNCTION void
GetGridSpacings(const Coordinates_t &coords, const Coordinates_t &coarse_coords,
                const IndexRange &cib, const IndexRange &ib, int i, int *fi, Real *dxm,
                Real *dxp, Real *dxfm, Real *dxfp) {
  // here "f" signifies the fine grid, not face locations.
  *fi = (i - cib.s) * 2 + ib.s;
  const Real xm = GetXCC<DIM>(coarse_coords, i - 1);
  const Real xc = GetXCC<DIM>(coarse_coords, i);
  const Real xp = GetXCC<DIM>(coarse_coords, i + 1);
  *dxm = xc - xm;
  *dxp = xp - xc;
  const Real fxm = GetXCC<DIM>(coords, *fi);
  const Real fxp = GetXCC<DIM>(coords, *fi + 1);
  *dxfm = xc - fxm;
  *dxfp = fxp - xc;
}
KOKKOS_FORCEINLINE_FUNCTION
Real GradMinMod(const Real fc, const Real fm, const Real fp, const Real dxm,
                const Real dxp) {
  const Real gxm = (fc - fm) / dxm;
  const Real gxp = (fp - fc) / dxp;
  return 0.5 * (SIGN(gxm) + SIGN(gxp)) * std::min(std::abs(gxm), std::abs(gxp));
}

KOKKOS_FORCEINLINE_FUNCTION
Real GradFiniteDiff(const Real fc, const Real fm, const Real fp, const Real dxm,
                    const Real dxp) {
  return (fp - fm) / (dxm + dxp);
}

} // namespace util

template <int DIM>
struct RestrictCellAverage {
  KOKKOS_FORCEINLINE_FUNCTION static void
  Do(const int l, const int m, const int n, const int ck, const int cj, const int ci,
     const IndexRange &ckb, const IndexRange &cjb, const IndexRange &cib,
     const IndexRange &kb, const IndexRange &jb, const IndexRange &ib,
     const Coordinates_t &coords, const Coordinates_t &coarse_coords,
     const ParArray6D<Real, VariableState> *pcoarse,
     const ParArray6D<Real, VariableState> *pfine) {
    auto &coarse = *pcoarse;
    auto &fine = *pfine;
    const int i = (ci - cib.s) * 2 + ib.s;
    int j = jb.s;
    if (DIM > 1) {
      j = (cj - cjb.s) * 2 + jb.s;
    }
    int k = kb.s;
    if (DIM > 2) {
      k = (ck - ckb.s) * 2 + kb.s;
    }
    // JMM: If dimensionality is wrong, accesses are out of bounds. Only
    // access cells if dimensionality is correct.
    Real vol[2][2][2], terms[2][2][2];
    std::memset(&vol[0][0][0], 0., 8 * sizeof(Real));
    std::memset(&terms[0][0][0], 0., 8 * sizeof(Real));
    for (int ok = 0; ok < 1 + (DIM > 2); ++ok) {
      for (int oj = 0; oj < 1 + (DIM > 1); ++oj) {
        for (int oi = 0; oi < 1 + 1; ++oi) {
          vol[ok][oj][oi] = coords.Volume(k + ok, j + oj, i + oi);
          terms[ok][oj][oi] = vol[ok][oj][oi] * fine(l, m, n, k + ok, j + oj, i + oi);
        }
      }
    }
    // KGF: add the off-centered quantities first to preserve FP
    // symmetry
    const Real tvol = ((vol[0][0][0] + vol[0][1][0]) + (vol[0][0][1] + vol[0][1][1])) +
                      ((vol[1][0][0] + vol[1][1][0]) + (vol[1][0][1] + vol[1][1][1]));
    coarse(l, m, n, ck, cj, ci) =
        (((terms[0][0][0] + terms[0][1][0]) + (terms[0][0][1] + terms[0][1][1])) +
         ((terms[1][0][0] + terms[1][1][0]) + (terms[1][0][1] + terms[1][1][1]))) /
        tvol;
  }
};

template <int DIM>
struct ProlongateCellMinMod {
  KOKKOS_FORCEINLINE_FUNCTION static void
  Do(const int l, const int m, const int n, const int k, const int j, const int i,
     const IndexRange &ckb, const IndexRange &cjb, const IndexRange &cib,
     const IndexRange &kb, const IndexRange &jb, const IndexRange &ib,
     const Coordinates_t &coords, const Coordinates_t &coarse_coords,
     const ParArray6D<Real, VariableState> *pcoarse,
     const ParArray6D<Real, VariableState> *pfine) {
    using namespace util;
    auto &coarse = *pcoarse;
    auto &fine = *pfine;

    const Real fc = coarse(l, m, n, k, j, i);

    int fi;
    Real dx1fm, dx1fp, dx1m, dx1p;
    GetGridSpacings<1>(coords, coarse_coords, cib, ib, i, &fi, &dx1m, &dx1p, &dx1fm,
                       &dx1fp);
    const Real gx1c = GradFiniteDiff(fc, coarse(l, m, n, k, j, i - 1),
                                 coarse(l, m, n, k, j, i + 1), dx1m, dx1p);

    int fj = jb.s; // overwritten as needed
    Real dx2fm = 0;
    Real dx2fp = 0;
    Real gx2c = 0;
    if (DIM > 1) { // TODO(c++17) make constexpr
      Real dx2m, dx2p;
      GetGridSpacings<2>(coords, coarse_coords, cjb, jb, j, &fj, &dx2m, &dx2p, &dx2fm,
                         &dx2fp);
      gx2c = GradFiniteDiff(fc, coarse(l, m, n, k, j - 1, i), coarse(l, m, n, k, j + 1, i),
                        dx2m, dx2p);
    }
    int fk = kb.s;
    Real dx3fm = 0;
    Real dx3fp = 0;
    Real gx3c = 0;
    if (DIM > 2) { // TODO(c++17) make constexpr
      Real dx3m, dx3p;
      GetGridSpacings<3>(coords, coarse_coords, ckb, kb, k, &fk, &dx3m, &dx3p, &dx3fm,
                         &dx3fp);
      gx3c = GradFiniteDiff(fc, coarse(l, m, n, k - 1, j, i), coarse(l, m, n, k + 1, j, i),
                        dx3m, dx3p);
    }

    // KGF: add the off-centered quantities first to preserve FP symmetry
    // JMM: Extraneous quantities are zero
    fine(l, m, n, fk, fj, fi) = fc - (gx1c * dx1fm + gx2c * dx2fm + gx3c * dx3fm);
    fine(l, m, n, fk, fj, fi + 1) = fc + (gx1c * dx1fp - gx2c * dx2fm - gx3c * dx3fm);
    if (DIM > 1) { // TODO(c++17) make constexpr
      fine(l, m, n, fk, fj + 1, fi) = fc - (gx1c * dx1fm - gx2c * dx2fp + gx3c * dx3fm);
      fine(l, m, n, fk, fj + 1, fi + 1) =
          fc + (gx1c * dx1fp + gx2c * dx2fp - gx3c * dx3fm);
    }
    if (DIM > 2) { // TODO(c++17) make constexpr
      fine(l, m, n, fk + 1, fj, fi) = fc - (gx1c * dx1fm + gx2c * dx2fm - gx3c * dx3fp);
      fine(l, m, n, fk + 1, fj, fi + 1) =
          fc + (gx1c * dx1fp - gx2c * dx2fm + gx3c * dx3fp);
      fine(l, m, n, fk + 1, fj + 1, fi) =
          fc - (gx1c * dx1fm - gx2c * dx2fp - gx3c * dx3fp);
      fine(l, m, n, fk + 1, fj + 1, fi + 1) =
          fc + (gx1c * dx1fp + gx2c * dx2fp + gx3c * dx3fp);
    }
  }
};
} // namespace refinement_ops
} // namespace parthenon

#endif // MESH_MESH_REFINEMENT_OPS_HPP_
