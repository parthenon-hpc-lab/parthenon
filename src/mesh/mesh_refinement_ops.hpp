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

#ifndef MESH_MESH_REFINEMENT_OPS_HPP_
#define MESH_MESH_REFINEMENT_OPS_HPP_

#include <algorithm>

#include "coordinates/coordinates.hpp" // for coordinates
#include "kokkos_abstraction.hpp"      // ParArray
#include "mesh/domain.hpp"             // for IndesShape

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
KOKKOS_INLINE_FUNCTION Real GetX(const Coordinates_t &coords, int i);
template <>
KOKKOS_INLINE_FUNCTION Real GetX<1>(const Coordinates_t &coords, int i) {
  return coords.x1v(i);
}
template <>
KOKKOS_INLINE_FUNCTION Real GetX<2>(const Coordinates_t &coords, int i) {
  return coords.x2v(i);
}
template <>
KOKKOS_INLINE_FUNCTION Real GetX<3>(const Coordinates_t &coords, int i) {
  return coords.x3v(i);
}
template <int DIM>
KOKKOS_FORCEINLINE_FUNCTION void
GetSlopes(const Coordinates_t &coords, const Coordinates_t &coarse_coords,
          const IndexRange &cib, const IndexRange &ib, int i, int &fi, Real &dxm,
          Real &dxp, Real &dxfm, Real &dxfp) {
  fi = (i - cib.s) * 2 + ib.s;
  const Real xm = GetX<DIM>(coarse_coords, i - 1);
  const Real xc = GetX<DIM>(coarse_coords, i);
  const Real xp = GetX<DIM>(coarse_coords, i + 1);
  dxm = xc - xm;
  dxp = xp - xc;
  const Real fxm = GetX<DIM>(coords, fi);
  const Real fxp = GetX<DIM>(coords, fi + 1);
  dxfm = xc - fxm;
  dxfp = fxp - xc;
}
KOKKOS_FORCEINLINE_FUNCTION
Real GradMinMod(const Real ccval, const Real fm, const Real fp, const Real dxm,
                const Real dxp) {
  const Real gxm = (ccval - fm) / dxm;
  const Real gxp = (fp - ccval) / dxp;
  return 0.5 * (SIGN(gxm) + SIGN(gxp)) * std::min(std::abs(gxm), std::abs(gxp));
}
} // namespace util

template <int DIM>
struct RestrictCellAverage {

  KOKKOS_FORCEINLINE_FUNCTION static void
  Do(const int l, const int m, const int n, const int ck, const int cj, const int ci,
     const IndexRange &ckb, const IndexRange &cjb, const IndexRange &cib,
     const IndexRange &kb, const IndexRange &jb, const IndexRange &ib,
     const Coordinates_t &coords, const Coordinates_t &coarse_coords,
     const ParArray6D<Real> &coarse, const ParArray6D<Real> &fine) {
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
    const Real vol000 = coords.Volume(k, j, i);
    const Real vol001 = coords.Volume(k, j, i + 1);
    const Real fine000 = fine(l, m, n, k, j, i);
    const Real fine001 = fine(l, m, n, k, j, i + 1);
    Real vol010, vol011, vol100, vol101, vol110, vol111;
    Real fine010, fine011, fine100, fine101, fine110, fine111;
    vol010 = vol011 = vol100 = vol101 = vol110 = vol111 = 0;
    fine010 = fine011 = fine100 = fine101 = fine110 = fine111 = 0;
    if (DIM > 1) { // TODO(c++17) make constexpr
      vol010 = coords.Volume(k, j + 1, i);
      vol011 = coords.Volume(k, j + 1, i + 1);
      fine010 = fine(l, m, n, k, j + 1, i);
      fine011 = fine(l, m, n, k, j + 1, i + 1);
    }
    if (DIM > 2) { // TODO(c++17) make constexpr
      vol100 = coords.Volume(k + 1, j, i);
      vol101 = coords.Volume(k + 1, j, i + 1);
      vol110 = coords.Volume(k + 1, j + 1, i);
      vol111 = coords.Volume(k + 1, j + 1, i + 1);
      fine100 = fine(l, m, n, k + 1, j, i);
      fine101 = fine(l, m, n, k + 1, j, i + 1);
      fine110 = fine(l, m, n, k + 1, j + 1, i);
      fine111 = fine(l, m, n, k + 1, j + 1, i + 1);
    }
    // KGF: add the off-centered quantities first to preserve FP
    // symmetry
    const Real tvol =
        ((vol000 + vol010) + (vol001 + vol011)) + ((vol100 + vol110) + (vol101 + vol111));
    coarse(l, m, n, ck, cj, ci) =
        (((fine000 * vol000 + fine010 * vol010) + (fine001 * vol001 + fine011 * vol011)) +
         ((fine100 * vol100 + fine110 * vol110) +
          (fine101 * vol101 + fine111 * vol111))) /
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
     const ParArray6D<Real> &coarse, const ParArray6D<Real> &fine) {
    using namespace util;

    const Real ccval = coarse(l, m, n, k, j, i);

    int fi;
    Real dx1fm, dx1fp, dx1m, dx1p;
    GetSlopes<1>(coords, coarse_coords, cib, ib, i, fi, dx1m, dx1p, dx1fm, dx1fp);
    const Real gx1c = GradMinMod(ccval, coarse(l, m, n, k, j, i - 1),
                                 coarse(l, m, n, k, j, i + 1), dx1m, dx1p);

    int fj = jb.s; // overwritten as needed
    Real dx2fm = 0;
    Real dx2fp = 0;
    Real gx2c = 0;
    if (DIM > 1) { // TODO(c++17) make constexpr
      Real dx2m, dx2p;
      GetSlopes<2>(coords, coarse_coords, cjb, jb, j, fj, dx2m, dx2p, dx2fm, dx2fp);
      gx2c = GradMinMod(ccval, coarse(l, m, n, k, j - 1, i), coarse(l, m, n, k, j + 1, i),
                        dx2m, dx2p);
    }
    int fk = kb.s;
    Real dx3fm = 0;
    Real dx3fp = 0;
    Real gx3c = 0;
    if (DIM > 2) { // TODO(c++17) make constexpr
      Real dx3m, dx3p;
      GetSlopes<3>(coords, coarse_coords, ckb, kb, k, fk, dx3m, dx3p, dx3fm, dx3fp);
      gx3c = GradMinMod(ccval, coarse(l, m, n, k - 1, j, i), coarse(l, m, n, k + 1, j, i),
                        dx3m, dx3p);
    }

    // KGF: add the off-centered quantities first to preserve FP symmetry
    // JMM: Extraneous quantities are zero
    fine(l, m, n, fk, fj, fi) = ccval - (gx1c * dx1fm + gx2c * dx2fm + gx3c * dx3fm);
    fine(l, m, n, fk, fj, fi + 1) = ccval + (gx1c * dx1fp - gx2c * dx2fm - gx3c * dx3fm);
    if (DIM > 1) { // TODO(c++17) make constexpr
      fine(l, m, n, fk, fj + 1, fi) =
          ccval - (gx1c * dx1fm - gx2c * dx2fp + gx3c * dx3fm);
      fine(l, m, n, fk, fj + 1, fi + 1) =
          ccval + (gx1c * dx1fp + gx2c * dx2fp - gx3c * dx3fm);
    }
    if (DIM > 2) { // TODO(c++17) make constexpr
      fine(l, m, n, fk + 1, fj, fi) =
          ccval - (gx1c * dx1fm + gx2c * dx2fm - gx3c * dx3fp);
      fine(l, m, n, fk + 1, fj, fi + 1) =
          ccval + (gx1c * dx1fp - gx2c * dx2fm + gx3c * dx3fp);
      fine(l, m, n, fk + 1, fj + 1, fi) =
          ccval - (gx1c * dx1fm - gx2c * dx2fp - gx3c * dx3fp);
      fine(l, m, n, fk + 1, fj + 1, fi + 1) =
          ccval + (gx1c * dx1fp + gx2c * dx2fp + gx3c * dx3fp);
    }
  }
};
} // namespace refinement_ops
} // namespace parthenon

#endif // MESH_MESH_REFINEMENT_OPS_HPP_
