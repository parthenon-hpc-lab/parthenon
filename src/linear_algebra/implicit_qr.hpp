//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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

#ifndef IMPLICIT_QR_HPP
#define IMPLICIT_QR_HPP

#include "parthenon/parthenon.hpp"

#include "givens.hpp"
#include "householder.hpp"
#include "matrix.hpp"

KOKKOS_FORCEINLINE_FUNCTION
double WilkinsonShift(double a00, double a11, double a01) {
  const double a01_sqr = a01 * a01;
  const double delta = 0.5 * (a00 - a11);
  return a11 - sign_of(delta) * a01_sqr /
                   (std::abs(delta) + std::sqrt(delta * delta + a01_sqr));
}

template <class tm_t>
KOKKOS_FORCEINLINE_FUNCTION int Partition(tm_t tm, double *d, double *b,
                                          std::size_t *start, std::size_t *end,
                                          int nrows) {
  int partition{0};
  bool collecting{false};

  const double eps = std::numeric_limits<double>::epsilon();
  const double tol = 100.0 * eps;

  // Global scale for the tridiagonal
  double scale = 0.0;
  find_maximum(
      tm, 0, nrows - 2,
      KOKKOS_LAMBDA(const int i, double &scale) {
        scale = std::max(scale, std::abs(d[i]));
        scale = std::max(scale, std::abs(b[i]));
      },
      scale);
  scale = std::max(scale, std::abs(d[nrows - 1]));
  if (scale == 0.0) scale = 1.0; // avoid all-zero edge case

  auto should_split = [&](int i) -> bool {
    const double bi = std::abs(b[i]);
    const double di = std::abs(d[i]);
    const double dj = std::abs(d[i + 1]);

    // 1) floating-point "negligible" test
    if (di + bi == di) return true;
    if (dj + bi == dj) return true;

    // 2) scale-aware relative test with a floor
    const double thresh = tol * (di + dj + scale);
    return bi <= thresh;
  };

  sequential_loop(0, nrows - 2, [&](const int i) {
    if (should_split(i)) {
      if (collecting) {
        once_per_team(tm, [&]() { end[partition] = i + 1; });
        collecting = false;
        partition++;
      }
    } else {
      if (!collecting) {
        once_per_team(tm, [&]() { start[partition] = i; });
        collecting = true;
      }
    }
  });
  if (collecting) {
    once_per_team(tm, [&]() { end[partition] = nrows; });
    partition++;
  }
  return partition;
}

// Perform implicit QR (Francis algorithm) on a symmetric, tridiagonal
// matrix A with diagonals d_i = A_{i,i} and off-diagonals b_i = A_{i, i+1}
// in place. On return, all elements of b should be ~zero and d should
// contain the eigenvalues of A.
template <class tm_t, class matrix_t>
KOKKOS_FORCEINLINE_FUNCTION int
ImplicitQRTridiag(tm_t tm, double *d, double *b, matrix_t *pQ, std::size_t *start,
                  std::size_t *end, const int nrows, const int max_iters) {
  int iter{0};
  for (iter = 0; iter < max_iters; ++iter) {
    // Collect decoupled regions of the matrix
    std::size_t npartitions = Partition(tm, d, b, start, end, nrows);
    // If all off diagonal elements are close to zero, we are done
    if (npartitions == 0) break;

    // Partitions are independent of each other, so could distribute this over
    // team members, not obvious how to cleanly abstract this second level of
    // parallelism though, so ignore for now
    for (int partition = 0; partition < npartitions; ++partition) {
      double bulge{0.0};
      const int sp = start[partition];
      const int ep = end[partition];

      // size one partition is already by definition diagonal
      if (ep - sp == 2) {
        // Directly calculate Givens rotation required to diagonalize 2x2 matrix
        const auto [c, s] = ComputeGivensDiagonalize2by2(d[sp], d[sp + 1], b[sp]);
        bulge = ApplyGivensLeftRight<true, true>(tm, sp, c, s, bulge, d, b);
        if (pQ) ApplyGivensRight(tm, sp, c, s, *pQ);
      } else if (ep - sp > 2) {
        const double mu = WilkinsonShift(d[ep - 2], d[ep - 1], b[ep - 2]);
        const auto [c, s] = ComputeGivensZeroSecond(d[sp] - mu, b[sp]);
        bulge = ApplyGivensLeftRight<true, false>(tm, sp, c, s, bulge, d, b);
        if (pQ) ApplyGivensRight(tm, sp, c, s, *pQ);
        sequential_loop(sp + 1, ep - 3, [&](const int i) {
          const auto [c, s] = ComputeGivensZeroSecond(b[i - 1], bulge);
          bulge = ApplyGivensLeftRight<false, false>(tm, i, c, s, bulge, d, b);
          if (pQ) ApplyGivensRight(tm, i, c, s, *pQ);
        });
        const auto [c2, s2] = ComputeGivensZeroSecond(b[ep - 3], bulge);
        ApplyGivensLeftRight<false, true>(tm, ep - 2, c2, s2, bulge, d, b);
        if (pQ) ApplyGivensRight(tm, ep - 2, c2, s2, *pQ);
      }
    }
  }
  return iter;
}

// Perform implicit QR (Francis algorithm) on Gram matrix of a symmetric, bidiagonal
// matrix A with diagonals d_i = A_{i,i} and off-diagonals b_i = A_{i, i+1}
// in place. On return, all elements of b should be ~zero and d should
// contain the singular values of A. The Gram matrix B = A^T A is never explicitly 
// formed.
template <class tm_t, class matrix_u_t, class matrix_v_t>
KOKKOS_FORCEINLINE_FUNCTION int ImplicitQRBidiag(tm_t tm, double *d, double *b,
                                                 matrix_u_t *pU, matrix_v_t *pV,
                                                 std::size_t *start, std::size_t *end,
                                                 const int nrows, const int max_iters) {
  int iter{0};
  for (iter = 0; iter < max_iters; ++iter) {
    // Collect decoupled regions of the matrix
    std::size_t npartitions = Partition(tm, d, b, start, end, nrows);
    // If all off diagonal elements are close to zero, we are done
    if (npartitions == 0) break;

    // Partitions are independent of each other, so could distribute this over
    // team members, not obvious how to cleanly abstract this second level of
    // parallelism though, so ignore for now
    for (int partition = 0; partition < npartitions; ++partition) {
      double bulge{0.0};
      const int sp = start[partition];
      const int ep = end[partition];

      // size one partition is already by definition diagonal
      if (ep - sp == 2) {
        auto result = ComputeSVD2by2UpperTriangular(d[sp], b[sp], d[sp + 1]);
        d[sp] = result.smax;
        d[sp+1] = result.smin;
        b[sp] = 0.0;
        if (pV) ApplyGivensRight(tm, sp, result.cr, result.sr, *pV);
        if (pU) ApplyGivensRight(tm, sp, result.cl, result.sl, *pU);
      } else if (ep - sp > 2) {
        // Compute shift and initial Given's rotation (which must satisfy
        // implicit QR) for the Gram matrix T = A^T A
        const int ii = ep - 2;
        const double te00 = d[ii] * d[ii] + b[ii - 1] * b[ii - 1];
        const double te01 = b[ii] * d[ii];
        const double te11 = d[ii + 1] * d[ii + 1] + b[ii] * b[ii];
        const double mu = WilkinsonShift(te00, te11, te01);

        const double t00 = d[sp] * d[sp];
        const double t01 = b[sp] * d[sp];
        const auto [c1, s1] = ComputeGivensZeroSecond(t00 - mu, t01);
        bulge = ApplyGivensRight<true, false>(tm, sp, c1, s1, bulge, d, b);
        if (pV) ApplyGivensRight(tm, sp, c1, s1, *pV);

        const auto [c2, s2] = ComputeGivensZeroSecond(d[sp], bulge);
        bulge = ApplyGivensLeft<true, false>(tm, sp, c2, s2, bulge, d, b);
        if (pU) ApplyGivensRight(tm, sp, c2, s2, *pU);
        sequential_loop(sp + 1, ep - 3, [&](const int i) {
          const auto [c1, s1] = ComputeGivensZeroSecond(b[i - 1], bulge);
          bulge = ApplyGivensRight<false, false>(tm, i, c1, s1, bulge, d, b);
          if (pV) ApplyGivensRight(tm, i, c1, s1, *pV);
          const auto [c2, s2] = ComputeGivensZeroSecond(d[i], bulge);
          bulge = ApplyGivensLeft<false, false>(tm, i, c2, s2, bulge, d, b);
          if (pU) ApplyGivensRight(tm, i, c2, s2, *pU);
        });
        const auto [c3, s3] = ComputeGivensZeroSecond(b[ep - 3], bulge);
        bulge = ApplyGivensRight<false, true>(tm, ep - 2, c3, s3, bulge, d, b);
        if (pV) ApplyGivensRight(tm, ep - 2, c3, s3, *pV);
        const auto [c4, s4] = ComputeGivensZeroSecond(d[ep - 2], bulge);
        bulge = ApplyGivensLeft<false, true>(tm, ep - 2, c4, s4, bulge, d, b);
        if (pU) ApplyGivensRight(tm, ep - 2, c4, s4, *pU);
      }
    }
  }
  return iter;
}

#endif // IMPLICIT_QR_HPP
