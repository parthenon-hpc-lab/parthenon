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
#ifndef RECONSTRUCT_PLM_INLINE_HPP_
#define RECONSTRUCT_PLM_INLINE_HPP_
//! \file plm.hpp
//  \brief implements piecewise linear reconstruction

#include "coordinates/coordinates.hpp"
#include "mesh/mesh.hpp"

namespace parthenon {

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::PiecewiseLinearX1()
//  \brief
template <typename T>
KOKKOS_INLINE_FUNCTION void
PiecewiseLinearX1(parthenon::team_mbr_t const &member, const int k, const int j,
                  const int il, const int iu, const Coordinates_t &coords, const T &q,
                  ScratchPad2D<Real> &ql, ScratchPad2D<Real> &qr, ScratchPad2D<Real> &qc,
                  ScratchPad2D<Real> &dql, ScratchPad2D<Real> &dqr,
                  ScratchPad2D<Real> &dqm) {
  const int nu = q.GetDim(4) - 1;

  // compute L/R slopes for each variable
  for (int n = 0; n <= nu; ++n) {
    parthenon::par_for_inner(member, il, iu, [&](const int i) {
      // renamed dw* -> dq* from plm.cpp
      dql(n, i) = (q(n, k, j, i) - q(n, k, j, i - 1));
      dqr(n, i) = (q(n, k, j, i + 1) - q(n, k, j, i));
      qc(n, i) = q(n, k, j, i);
    });
  }
  member.team_barrier();

  // Apply simplified van Leer (VL) limiter expression for a Cartesian-like coordinate
  // with uniform mesh spacing
  //  if (uniform[X1DIR]) {
  if (true) { // TODO(pgrete) make this work again
    for (int n = 0; n <= nu; ++n) {
      parthenon::par_for_inner(member, il, iu, [&](const int i) {
        Real dq2 = dql(n, i) * dqr(n, i);
        dqm(n, i) = 2.0 * dq2 / (dql(n, i) + dqr(n, i));
        if (dq2 <= 0.0) dqm(n, i) = 0.0;
      });
    }
    member.team_barrier();

    // Apply general VL limiter expression w/ the Mignone correction for a Cartesian-like
    // coordinate with nonuniform mesh spacing
  } else {
    for (int n = 0; n <= nu; ++n) {
      parthenon::par_for_inner(member, il, iu, [&](const int i) {
        Real dqF = dqr(n, i) * coords.Dxf<1,1>(i) / coords.Dxc<1>(i);
        Real dqB = dql(n, i) * coords.Dxf<1,1>(i) / coords.Dxc<1>(i - 1);
        Real dq2 = dqF * dqB;
        // cf, cb -> 2 (uniform Cartesian mesh / original VL value) w/ vanishing
        // curvature (may not exactly hold for nonuniform meshes, but converges w/
        // smooth nonuniformity)
        Real cf = coords.Dxc<1>(i) / (coords.Xf<1,1>(i + 1) - coords.Xc<1>(i)); // (Mignone eq 33)
        Real cb = coords.Dxc<1>(i - 1) / (coords.Xc<1>(i) - coords.Xf<1,1>(i));
        // (modified) VL limiter (Mignone eq 37)
        // (dQ^F term from eq 31 pulled into eq 37, then multiply by (dQ^F/dQ^F)^2)
        dqm(n, i) =
            (dq2 * (cf * dqB + cb * dqF) / (SQR(dqB) + SQR(dqF) + dq2 * (cf + cb - 2.0)));
        if (dq2 <= 0.0) dqm(n, i) = 0.0; // ---> no concern for divide-by-0 in above line

        // Real v = dqB/dqF;
        // monotoniced central (MC) limiter (Mignone eq 38)
        // (std::min calls should avoid issue if divide-by-zero causes v=Inf)
        // dqm(n,i) = dqF*std::max(0.0, std::min(0.5*(1.0 + v), std::min(cf, cb*v)));
      });
    }
    member.team_barrier();
  }

  // compute ql_(i+1/2) and qr_(i-1/2) using limited slopes
  for (int n = 0; n <= nu; ++n) {
    parthenon::par_for_inner(member, il, iu, [&](const int i) {
      // Mignone equation 30
      ql(n, i + 1) =
          qc(n, i) + ((coords.Xf<1,1>(i + 1) - coords.Xc<1>(i)) / coords.Dxf<1,1>(i)) * dqm(n, i);
      qr(n, i) =
          qc(n, i) - ((coords.Xc<1>(i) - coords.Xf<1,1>(i)) / coords.Dxf<1,1>(i)) * dqm(n, i);
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::PiecewiseLinearX2()
//  \brief
template <typename T>
KOKKOS_INLINE_FUNCTION void
PiecewiseLinearX2(parthenon::team_mbr_t const &member, const int k, const int j,
                  const int il, const int iu, const Coordinates_t &coords, const T &q,
                  ScratchPad2D<Real> &ql, ScratchPad2D<Real> &qr, ScratchPad2D<Real> &qc,
                  ScratchPad2D<Real> &dql, ScratchPad2D<Real> &dqr,
                  ScratchPad2D<Real> &dqm) {
  const int nu = q.GetDim(4) - 1;

  // compute L/R slopes for each variable
  for (int n = 0; n <= nu; ++n) {
    parthenon::par_for_inner(member, il, iu, [&](const int i) {
      // renamed dw* -> dq* from plm.cpp
      dql(n, i) = (q(n, k, j, i) - q(n, k, j - 1, i));
      dqr(n, i) = (q(n, k, j + 1, i) - q(n, k, j, i));
      qc(n, i) = q(n, k, j, i);
    });
  }
  member.team_barrier();

  // Apply simplified van Leer (VL) limiter expression for a Cartesian-like coordinate
  // with uniform mesh spacing
  // if (uniform[X2DIR]) {
  if (true) { // TODO(pgrete) make work again
    for (int n = 0; n <= nu; ++n) {
      parthenon::par_for_inner(member, il, iu, [&](const int i) {
        Real dq2 = dql(n, i) * dqr(n, i);
        dqm(n, i) = 2.0 * dq2 / (dql(n, i) + dqr(n, i));
        if (dq2 <= 0.0) dqm(n, i) = 0.0;
      });
    }
    member.team_barrier();

    // Apply general VL limiter expression w/ the Mignone correction for a Cartesian-like
    // coordinate with nonuniform mesh spacing
  } else {
    Real cf = coords.Dxc<2>(j) / (coords.Xf<2,2>(j + 1) - coords.Xc<2>(j));
    Real cb = coords.Dxc<2>(j - 1) / (coords.Xc<2>(j) - coords.Xf<2,2>(j));
    Real dxF =
        coords.Dxf<2,2>(j) / coords.Dxc<2>(j); // dimensionless, not technically a dx quantity
    Real dxB = coords.Dxf<2,2>(j) / coords.Dxc<2>(j - 1);
    for (int n = 0; n <= nu; ++n) {
      parthenon::par_for_inner(member, il, iu, [&](const int i) {
        Real dqF = dqr(n, i) * dxF;
        Real dqB = dql(n, i) * dxB;
        Real dq2 = dqF * dqB;
        // (modified) VL limiter (Mignone eq 37)
        dqm(n, i) =
            (dq2 * (cf * dqB + cb * dqF) / (SQR(dqB) + SQR(dqF) + dq2 * (cf + cb - 2.0)));
        if (dq2 <= 0.0) dqm(n, i) = 0.0; // ---> no concern for divide-by-0 in above line

        // Real v = dqB/dqF;
        // // monotoniced central (MC) limiter (Mignone eq 38)
        // // (std::min calls should avoid issue if divide-by-zero causes v=Inf)
        // dqm(n,i) = dqF*std::max(0.0, std::min(0.5*(1.0 + v), std::min(cf, cb*v)));
      });
    }
    member.team_barrier();
  }

  // compute ql_(j+1/2) and qr_(j-1/2) using limited slopes
  // dimensionless, not technically a "dx" quantity
  Real dxp = (coords.Xf<2,2>(j + 1) - coords.Xc<2>(j)) / coords.Dxf<2,2>(j);
  Real dxm = (coords.Xc<2>(j) - coords.Xf<2,2>(j)) / coords.Dxf<2,2>(j);
  for (int n = 0; n <= nu; ++n) {
    parthenon::par_for_inner(member, il, iu, [&](const int i) {
      ql(n, i) = qc(n, i) + dxp * dqm(n, i);
      qr(n, i) = qc(n, i) - dxm * dqm(n, i);
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::PiecewiseLinearX3()
//  \brief
template <typename T>
KOKKOS_INLINE_FUNCTION void
PiecewiseLinearX3(parthenon::team_mbr_t const &member, const int k, const int j,
                  const int il, const int iu, const Coordinates_t &coords, const T &q,
                  ScratchPad2D<Real> &ql, ScratchPad2D<Real> &qr, ScratchPad2D<Real> &qc,
                  ScratchPad2D<Real> &dql, ScratchPad2D<Real> &dqr,
                  ScratchPad2D<Real> &dqm) {
  const int nu = q.GetDim(4) - 1;

  // compute L/R slopes for each variable
  for (int n = 0; n <= nu; ++n) {
    parthenon::par_for_inner(member, il, iu, [&](const int i) {
      // renamed dw* -> dq* from plm.cpp
      dql(n, i) = (q(n, k, j, i) - q(n, k - 1, j, i));
      dqr(n, i) = (q(n, k + 1, j, i) - q(n, k, j, i));
      qc(n, i) = q(n, k, j, i);
    });
  }
  member.team_barrier();

  // Apply simplified van Leer (VL) limiter expression for a Cartesian-like coordinate
  // with uniform mesh spacing
  // if (uniform[X3DIR]) {
  if (true) { // TODO(pgrete) make conditional work again
    for (int n = 0; n <= nu; ++n) {
      parthenon::par_for_inner(member, il, iu, [&](const int i) {
        Real dq2 = dql(n, i) * dqr(n, i);
        dqm(n, i) = 2.0 * dq2 / (dql(n, i) + dqr(n, i));
        if (dq2 <= 0.0) dqm(n, i) = 0.0;
      });
    }
    member.team_barrier();

    // Apply original VL limiter's general expression for a Cartesian-like coordinate with
    // nonuniform mesh spacing
  } else {
    Real dxF = coords.Dxf<3,3>(k) / coords.Dxc<3>(k);
    Real dxB = coords.Dxf<3,3>(k) / coords.Dxc<3>(k - 1);
    for (int n = 0; n <= nu; ++n) {
      parthenon::par_for_inner(member, il, iu, [&](const int i) {
        Real dqF = dqr(n, i) * dxF;
        Real dqB = dql(n, i) * dxB;
        Real dq2 = dqF * dqB;
        // original VL limiter (Mignone eq 36)
        dqm(n, i) = 2.0 * dq2 / (dqF + dqB);
        // dq2 > 0 ---> dqF, dqB are nonzero and have the same
        // sign ----> no risk for (dqF + dqB) = 0 cancellation
        // causing a divide-by-0 in the above line
        if (dq2 <= 0.0) dqm(n, i) = 0.0;
      });
    }
    member.team_barrier();
  }

  // compute ql_(k+1/2) and qr_(k-1/2) using limited slopes
  Real dxp = (coords.Xf<3,3>(k + 1) - coords.Xc<3>(k)) / coords.Dxf<3,3>(k);
  Real dxm = (coords.Xc<3>(k) - coords.Xf<3,3>(k)) / coords.Dxf<3,3>(k);
  for (int n = 0; n <= nu; ++n) {
    parthenon::par_for_inner(member, il, iu, [&](const int i) {
      ql(n, i) = qc(n, i) + dxp * dqm(n, i);
      qr(n, i) = qc(n, i) - dxm * dqm(n, i);
    });
  }
}

} // namespace parthenon

#endif // RECONSTRUCT_PLM_INLINE_HPP_
