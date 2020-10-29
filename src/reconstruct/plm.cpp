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

#include "reconstruct/reconstruction.hpp"

#include "defs.hpp"
#include "mesh/mesh.hpp"

namespace parthenon {

namespace {

struct DataInternal {
  const int k, j, il, iu, nu;
  const ParArrayND<Real> &q;
  ParArrayND<Real> &qc;
  ParArrayND<Real> &ql;
  ParArrayND<Real> &qr;

  ParArrayND<Real> &dql;
  ParArrayND<Real> &dqr;
  ParArrayND<Real> &dqm;
  const Coordinates_t &coords;
  DataInternal(const int k, const int j, const int il, const int iu, const int nu,
               const ParArrayND<Real> &q, ParArrayND<Real> &qc, ParArrayND<Real> &ql,
               ParArrayND<Real> &qr, ParArrayND<Real> &dql, ParArrayND<Real> &dqr,
               ParArrayND<Real> &dqm, const Coordinates_t &coords)
      : k(k), j(j), il(il), iu(iu), nu(nu), q(q), qc(qc), ql(ql), qr(qr), dql(dql),
        dqr(dqr), dqm(dqm), coords(coords) {}
};

void init_dql_and_dqr_and_qc_(DataInternal &data, const CoordinateDirection direction) {
  int delta_i = 0;
  int delta_j = 0;
  int delta_k = 0;
  if (direction == X1DIR ) {
    delta_i = 1;
  } else if (direction == X2DIR) {
    delta_j = 1;
  } else {
    delta_k = 1;
  }

  // compute L/R slopes for each variable
  const int j = data.j;
  const int k = data.k;
  for (int n = 0; n <= data.nu; ++n) {
#pragma omp simd
    for (int i = data.il; i <= data.iu; ++i) {
      // renamed dw* -> dq* from plm.cpp
      data.dql(n, i) =
          (data.q(n, k, j, i) - data.q(n, k - delta_k, j - delta_j, i - delta_i));
      data.dqr(n, i) =
          (data.q(n, k + delta_k, j + delta_j, i + delta_i) - data.q(n, k, j, i));
      data.qc(n, i) = data.q(n, k, j, i);
    }
  }
}

void apply_simplified_van_leer_(DataInternal &data) {
  for (int n = 0; n <= data.nu; ++n) {
#pragma omp simd simdlen(SIMD_WIDTH)
    for (int i = data.il; i <= data.iu; ++i) {
      const Real dq2 = data.dql(n, i) * data.dqr(n, i);
      data.dqm(n, i) = 2.0 * dq2 / (data.dql(n, i) + data.dqr(n, i));
      if (dq2 <= 0.0) data.dqm(n, i) = 0.0;
    }
  }
}

void apply_general_van_leer_(DataInternal &data, const CoordinateDirection direction) {
  const int j = data.j;
  const int k = data.k;
  if (direction == X1DIR) {
    for (int n = 0; n <= data.nu; ++n) {
#pragma omp simd simdlen(SIMD_WIDTH)
      for (int i = data.il; i <= data.iu; ++i) {
        const Real dqF = data.dqr(n, i) * data.coords.dx1f(i) / data.coords.dx1v(i);
        const Real dqB = data.dql(n, i) * data.coords.dx1f(i) / data.coords.dx1v(i - 1);
        const Real dq2 = dqF * dqB;
        // cf, cb -> 2 (uniform Cartesian mesh / original VL value) w/ vanishing curvature
        // (may not exactly hold for nonuniform meshes, but converges w/ smooth
        // nonuniformity)
        const Real cf = data.coords.dx1v(i) /
                        (data.coords.x1f(i + 1) - data.coords.x1v(i)); // (Mignone eq 33)
        const Real cb =
            data.coords.dx1v(i - 1) / (data.coords.x1v(i) - data.coords.x1f(i));
        // (modified) VL limiter (Mignone eq 37)
        // (dQ^F term from eq 31 pulled into eq 37, then multiply by (dQ^F/dQ^F)^2)
        data.dqm(n, i) =
            (dq2 * (cf * dqB + cb * dqF) / (SQR(dqB) + SQR(dqF) + dq2 * (cf + cb - 2.0)));
        if (dq2 <= 0.0)
          data.dqm(n, i) = 0.0; // ---> no concern for divide-by-0 in above line

        // Real v = dqB/dqF;
        // monotoniced central (MC) limiter (Mignone eq 38)
        // (std::min calls should avoid issue if divide-by-zero causes v=Inf)
        // dqm(n,i) = dqF*std::max(0.0, std::min(0.5*(1.0 + v), std::min(cf, cb*v)));
      }
    }
  } else if (direction == X2DIR) {
    const Real cf = data.coords.dx2v(j) / (data.coords.x2f(j + 1) - data.coords.x2v(j));
    const Real cb = data.coords.dx2v(j - 1) / (data.coords.x2v(j) - data.coords.x2f(j));
    const Real dxF = data.coords.dx2f(j) /
                     data.coords.dx2v(j); // directionless, not technically a dx quantity
    const Real dxB = data.coords.dx2f(j) / data.coords.dx2v(j - 1);
    for (int n = 0; n <= data.nu; ++n) {
#pragma omp simd simdlen(SIMD_WIDTH)
      for (int i = data.il; i <= data.iu; ++i) {
        const Real dqF = data.dqr(n, i) * dxF;
        const Real dqB = data.dql(n, i) * dxB;
        const Real dq2 = dqF * dqB;
        // (modified) VL limiter (Mignone eq 37)
        data.dqm(n, i) =
            (dq2 * (cf * dqB + cb * dqF) / (SQR(dqB) + SQR(dqF) + dq2 * (cf + cb - 2.0)));
        if (dq2 <= 0.0)
          data.dqm(n, i) = 0.0; // ---> no concern for divide-by-0 in above line

        // Real v = dqB/dqF;
        // // monotoniced central (MC) limiter (Mignone eq 38)
        // // (std::min calls should avoid issue if divide-by-zero causes v=Inf)
        // dqm(n,i) = dqF*std::max(0.0, std::min(0.5*(1.0 + v), std::min(cf, cb*v)));
      }
    }
  } else {
    const Real dxF = data.coords.dx3f(k) / data.coords.dx3v(k);
    const Real dxB = data.coords.dx3f(k) / data.coords.dx3v(k - 1);
    for (int n = 0; n <= data.nu; ++n) {
#pragma omp simd simdlen(SIMD_WIDTH)
      for (int i = data.il; i <= data.iu; ++i) {
        const Real dqF = data.dqr(n, i) * dxF;
        const Real dqB = data.dql(n, i) * dxB;
        const Real dq2 = dqF * dqB;
        // original VL limiter (Mignone eq 36)
        data.dqm(n, i) = 2.0 * dq2 / (dqF + dqB);
        // dq2 > 0 ---> dqF, dqB are nonzero and have the same sign ----> no risk for
        // (dqF + dqB) = 0 cancellation causing a divide-by-0 in the above line
        if (dq2 <= 0.0) data.dqm(n, i) = 0.0;
      }
    }
  }
}

void compute_ql_and_qr_using_limited_slopes_(DataInternal &data,
                                             const CoordinateDirection direction) {
  if (direction == X1DIR) {
    // compute ql_(i+1/2) and qr_(i-1/2) using limited slopes
    for (int n = 0; n <= data.nu; ++n) {         // Same
#pragma omp simd simdlen(SIMD_WIDTH)             // Same
      for (int i = data.il; i <= data.iu; ++i) { // Same
        // Mignone equation 30
        data.ql(n, i + 1) =
            data.qc(n, i) +
            ((data.coords.x1f(i + 1) - data.coords.x1v(i)) / data.coords.dx1f(i)) *
                data.dqm(n, i); // Different 1D
        data.qr(n, i) = data.qc(n, i) - ((data.coords.x1v(i) - data.coords.x1f(i)) /
                                         data.coords.dx1f(i)) *
                                            data.dqm(n, i); // Different 1D
      }
    }
  } else { // If X2 or X3
    const int j = data.j;
    const int k = data.k;
    // compute ql_(j+1/2) and qr_(j-1/2) using limited slopes or
    // compute ql_(k+1/2) and qr_(k-1/2) using limited slopes
    Real dxp = (data.coords.x2f(j + 1) - data.coords.x2v(j)) / data.coords.dx2f(j);
    Real dxm = (data.coords.x2v(j) - data.coords.x2f(j)) / data.coords.dx2f(j);
    if (direction == X3DIR) {
      dxp = (data.coords.x3f(k + 1) - data.coords.x3v(k)) / data.coords.dx3f(k);
      dxm = (data.coords.x3v(k) - data.coords.x3f(k)) / data.coords.dx3f(k);
    }

    for (int n = 0; n <= data.nu; ++n) {
#pragma omp simd simdlen(SIMD_WIDTH)
      for (int i = data.il; i <= data.iu; ++i) {
        data.ql(n, i) = data.qc(n, i) + dxp * data.dqm(n, i);
        data.qr(n, i) = data.qc(n, i) - dxm * data.dqm(n, i);
      }
    }
  }
  return;
}

void Reconstruction::PiecewiseLinear_(const int k, const int j, const int il, const int iu,
    const ParArrayND<Real> &q, ParArrayND<Real> &ql, ParArrayND<Real> &qr, 
    const CoordinateDirection direction) {
  auto pmb = GetBlockPointer();
  const int nu = q.GetDim(4) - 1;
  ParArrayND<Real> &qc = scr1_ni_, &dql = scr2_ni_, &dqr = scr3_ni_, &dqm = scr4_ni_;
  DataInternal data(k, j, il, iu, nu, q, qc, ql, qr, dql, dqr, dqm, pmb->coords);

  init_dql_and_dqr_and_qc_(data, direction);

  if (uniform[direction]) {
    // Apply simplified van Leer (VL) limiter expression for a Cartesian-like coordinate
    // with uniform mesh spacing
    apply_simplified_van_leer_(data);
  } else {
    // Apply general VL limiter expression w/ the Mignone correction for a Cartesian-like
    // coordinate with nonuniform mesh spacing
    apply_general_van_leer_(data, direction);
  }
  compute_ql_and_qr_using_limited_slopes_(data, direction);
}
} // namespace

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::PiecewiseLinearX1()
//  \brief
void Reconstruction::PiecewiseLinearX1(const int k, const int j, const int il,
                                       const int iu, const ParArrayND<Real> &q,
                                       ParArrayND<Real> &ql, ParArrayND<Real> &qr) {
  PiecewiseLinear_(k, j, il, iu, q, ql, qr, X1DIR);
}

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::PiecewiseLinearX2()
//  \brief
void Reconstruction::PiecewiseLinearX2(const int k, const int j, const int il,
                                       const int iu, const ParArrayND<Real> &q,
                                       ParArrayND<Real> &ql, ParArrayND<Real> &qr) {
  PiecewiseLinear_(k, j, il, iu, q, ql, qr, X2DIR);
}

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::PiecewiseLinearX2()
//  \brief
void Reconstruction::PiecewiseLinearX3(const int k, const int j, const int il,
                                       const int iu, const ParArrayND<Real> &q,
                                       ParArrayND<Real> &ql, ParArrayND<Real> &qr) {
  PiecewiseLinear_(k, j, il, iu, q, ql, qr, X3DIR);
}

} // namespace parthenon
