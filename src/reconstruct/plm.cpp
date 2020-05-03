//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
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
//! \file plm_simple.cpp
//  \brief  piecewise linear reconstruction for both uniform and non-uniform meshes
//  Operates on the entire nx4 range of a single ParArrayND<Real> input (no MHD).

// REFERENCES:
// (Mignone) A. Mignone, "High-order conservative reconstruction schemes for finite volume
// methods in cylindrical and spherical coordinates", JCP, 270, 784 (2014)
//========================================================================================

#include "mesh/mesh.hpp"
#include "reconstruct/reconstruction.hpp"

namespace parthenon {

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::PiecewiseLinearX1()
//  \brief

void Reconstruction::PiecewiseLinearX1(const int k, const int j, const int il,
                                       const int iu, const ParArrayND<Real> &q,
                                       ParArrayND<Real> &ql, ParArrayND<Real> &qr) {
  // set work arrays to shallow copies of scratch arrays
  ParArrayND<Real> &qc = scr1_ni_, &dql = scr2_ni_, &dqr = scr3_ni_, &dqm = scr4_ni_;
  const int nu = q.GetDim(4) - 1;

  // compute L/R slopes for each variable
  for (int n = 0; n <= nu; ++n) {
#pragma omp simd
    for (int i = il; i <= iu; ++i) {
      // renamed dw* -> dq* from plm.cpp
      dql(n, i) = (q(n, k, j, i) - q(n, k, j, i - 1));
      dqr(n, i) = (q(n, k, j, i + 1) - q(n, k, j, i));
      qc(n, i) = q(n, k, j, i);
    }
  }

  // Apply simplified van Leer (VL) limiter expression for a Cartesian-like coordinate
  // with uniform mesh spacing
  for (int n = 0; n <= nu; ++n) {
#pragma omp simd simdlen(SIMD_WIDTH)
    for (int i = il; i <= iu; ++i) {
      Real dq2 = dql(n, i) * dqr(n, i);
      dqm(n, i) = 2.0 * dq2 / (dql(n, i) + dqr(n, i));
      if (dq2 <= 0.0) dqm(n, i) = 0.0;
    }
  }

  // compute ql_(i+1/2) and qr_(i-1/2) using limited slopes
  for (int n = 0; n <= nu; ++n) {
#pragma omp simd simdlen(SIMD_WIDTH)
    for (int i = il; i <= iu; ++i) {
      // Mignone equation 30
      ql(n, i + 1) = qc(n, i) + 0.5 * dqm(n, i);
      qr(n, i) = qc(n, i) - 0.5 * dqm(n, i);
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::PiecewiseLinearX2()
//  \brief

void Reconstruction::PiecewiseLinearX2(const int k, const int j, const int il,
                                       const int iu, const ParArrayND<Real> &q,
                                       ParArrayND<Real> &ql, ParArrayND<Real> &qr) {
  // set work arrays to shallow copies of scratch arrays
  ParArrayND<Real> &qc = scr1_ni_, &dql = scr2_ni_, &dqr = scr3_ni_, &dqm = scr4_ni_;
  const int nu = q.GetDim(4) - 1;

  // compute L/R slopes for each variable
  for (int n = 0; n <= nu; ++n) {
#pragma omp simd
    for (int i = il; i <= iu; ++i) {
      // renamed dw* -> dq* from plm.cpp
      dql(n, i) = (q(n, k, j, i) - q(n, k, j - 1, i));
      dqr(n, i) = (q(n, k, j + 1, i) - q(n, k, j, i));
      qc(n, i) = q(n, k, j, i);
    }
  }

  // Apply simplified van Leer (VL) limiter expression for a Cartesian-like coordinate
  // with uniform mesh spacing
  for (int n = 0; n <= nu; ++n) {
#pragma omp simd simdlen(SIMD_WIDTH)
    for (int i = il; i <= iu; ++i) {
      Real dq2 = dql(n, i) * dqr(n, i);
      dqm(n, i) = 2.0 * dq2 / (dql(n, i) + dqr(n, i));
      if (dq2 <= 0.0) dqm(n, i) = 0.0;
    }
  }

  // compute ql_(j+1/2) and qr_(j-1/2) using limited slopes
  // dimensionless, not technically a "dx" quantity
  for (int n = 0; n <= nu; ++n) {
#pragma omp simd simdlen(SIMD_WIDTH)
    for (int i = il; i <= iu; ++i) {
      ql(n, i) = qc(n, i) + 0.5 * dqm(n, i);
      qr(n, i) = qc(n, i) - 0.5 * dqm(n, i);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::PiecewiseLinearX3()
//  \brief

void Reconstruction::PiecewiseLinearX3(const int k, const int j, const int il,
                                       const int iu, const ParArrayND<Real> &q,
                                       ParArrayND<Real> &ql, ParArrayND<Real> &qr) {
  // set work arrays to shallow copies of scratch arrays
  ParArrayND<Real> &qc = scr1_ni_, &dql = scr2_ni_, &dqr = scr3_ni_, &dqm = scr4_ni_;
  const int nu = q.GetDim(4) - 1;

  // compute L/R slopes for each variable
  for (int n = 0; n <= nu; ++n) {
#pragma omp simd
    for (int i = il; i <= iu; ++i) {
      // renamed dw* -> dq* from plm.cpp
      dql(n, i) = (q(n, k, j, i) - q(n, k - 1, j, i));
      dqr(n, i) = (q(n, k + 1, j, i) - q(n, k, j, i));
      qc(n, i) = q(n, k, j, i);
    }
  }

  // Apply simplified van Leer (VL) limiter expression for a Cartesian-like coordinate
  // with uniform mesh spacing
  for (int n = 0; n <= nu; ++n) {
#pragma omp simd simdlen(SIMD_WIDTH)
    for (int i = il; i <= iu; ++i) {
      Real dq2 = dql(n, i) * dqr(n, i);
      dqm(n, i) = 2.0 * dq2 / (dql(n, i) + dqr(n, i));
      if (dq2 <= 0.0) dqm(n, i) = 0.0;
    }
  }

  // compute ql_(k+1/2) and qr_(k-1/2) using limited slopes
  for (int n = 0; n <= nu; ++n) {
#pragma omp simd simdlen(SIMD_WIDTH)
    for (int i = il; i <= iu; ++i) {
      ql(n, i) = qc(n, i) + 0.5 * dqm(n, i);
      qr(n, i) = qc(n, i) - 0.5 * dqm(n, i);
    }
  }
}

} // namespace parthenon
