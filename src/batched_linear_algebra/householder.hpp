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

#ifndef BATCHED_LINEAR_ALGEBRA_HOUSEHOLDER_HPP_
#define BATCHED_LINEAR_ALGEBRA_HOUSEHOLDER_HPP_

// This file was made in part with generative AI.

#include <algorithm>
#include <cmath>

#include "batched_linear_algebra/execution_utils.hpp"
#include "batched_linear_algebra/matrix_utils.hpp"
#include "utils/robust.hpp"

namespace parthenon {
namespace batched_linear_algebra {

/// [This documentation was generated with assistance from generative AI]
/// Construct a normalized Householder vector for a column transformation.
///
/// Given the column segment
///   x = A(row:nrows-1, col),
/// this routine constructs a normalized Householder vector v such that
/// the reflector
///   H = I - 2 v vᵀ
/// satisfies
///   H x = ±‖x‖ e₁.
///
/// On exit:
///   - v[i] = 0 for i < row
///   - v[row:nrows-1] contains the normalized Householder vector
///   - If ‖x‖ = 0, v is set to zero and the reflector is the identity
///
/// This reflector is intended for left application: A ← H A.
///
/// x is scaled by max|x_i| before forming any sums of squares (the reflector is
/// invariant under this scaling), which avoids overflow/underflow for entries
/// outside roughly [1e-154, 1e154].
template <class tm_t, class matrix_t>
KOKKOS_FORCEINLINE_FUNCTION void
build_householder_vector_col(tm_t tm, int row, int col, const matrix_t &A, double *v) {
  const int nrows = GetNrows(A);

  double xmax{0.0};
  find_maximum(
      tm, row, nrows - 1,
      [&](const int r, double &mx) { mx = std::max(mx, std::abs(A(r, col))); }, xmax);
  if (xmax == 0.0) {
    parallel_loop(tm, 0, nrows - 1, [&](const int i) { v[i] = 0.0; });
    return;
  }

  double norm_tail{0.0};
  summation(
      tm, row + 1, nrows - 1,
      [&](const int i, double &norm) {
        v[i] = A(i, col) / xmax;
        norm += v[i] * v[i];
      },
      norm_tail);

  const double x0 = A(row, col) / xmax;
  const double norm_x = safe_sqrt(norm_tail + x0 * x0);
  // Keep the head in a local so v is not written by every thread before the
  // final loop
  const double vh = x0 + sign_of(x0) * norm_x;
  const double norm_v = safe_sqrt(norm_tail + vh * vh);

  double inv_norm_v = parthenon::robust::ratio(1.0, norm_v);
  parallel_loop(tm, 0, nrows - 1, [&](const int i) {
    v[i] = ((i == row) * vh + (i > row) * v[i]) * inv_norm_v;
  });
}

template <class tm_t, class matrix_t>
KOKKOS_FORCEINLINE_FUNCTION void
build_householder_vector_row(tm_t tm, int row, int col, const matrix_t &A, double *v) {
  const int ncols = GetNcols(A);

  // Scale x = A(row, col:ncols-1) by max|x_j| to avoid overflow/underflow.
  // If the row segment is already zero, the reflector is identity
  double xmax{0.0};
  find_maximum(
      tm, col, ncols - 1,
      [&](const int c, double &mx) { mx = std::max(mx, std::abs(A(row, c))); }, xmax);
  if (xmax == 0.0) {
    parallel_loop(tm, 0, ncols - 1, [&](const int j) { v[j] = 0.0; });
    return;
  }

  // Copy the remainder of the scaled row segment into v
  double norm_tail{0.0};
  summation(
      tm, col + 1, ncols - 1,
      [&](const int j, double &norm) {
        v[j] = A(row, j) / xmax;
        norm += v[j] * v[j];
      },
      norm_tail);

  // v[col] = x₀ + sign(x₀) * ||x||, with x = A(row, col:ncols-1) / xmax
  const double x0 = A(row, col) / xmax;
  const double norm_x = safe_sqrt(norm_tail + x0 * x0);
  // Keep the head in a local so v is not written by every thread before the
  // final loop
  const double vh = x0 + sign_of(x0) * norm_x;
  const double norm_v = safe_sqrt(norm_tail + vh * vh);

  const double inv_norm_v = parthenon::robust::ratio(1.0, norm_v);

  // Zero entries before col, set the head, and normalize the active part
  parallel_loop(tm, 0, ncols - 1, [&](const int j) {
    v[j] = ((j == col) * vh + (j > col) * v[j]) * inv_norm_v;
  });
}

// Apply the Householder transformation H = I - 2 v^T v to A in place,
// i.e. A <- H A. Here v is assumed to be normalized.
// The parameter start_idx specifies the first non-zero entry in v.
template <class tm_t, class matrix_t>
KOKKOS_FORCEINLINE_FUNCTION void
apply_left_householder_transformation(tm_t tm, const double *const v, double *scratch,
                                      matrix_t &A, int row_start_idx = 0,
                                      int col_start_idx = 0) {
  const int nrows = GetNrows(A);
  const int ncols = GetNcols(A);
  for (int c = col_start_idx; c < ncols; ++c) {
    double w{0.0};
    summation(
        tm, row_start_idx, nrows - 1,
        [&](int r, double &ww) { ww += 2.0 * v[r] * A(r, c); }, w);
    once_per_team(tm, [&]() { scratch[c] = w; });
  }
  barrier(tm);
  parallel_loop(tm, col_start_idx, ncols - 1, row_start_idx, nrows - 1,
                [&](int c, int r) { A(r, c) -= scratch[c] * v[r]; });
}

// Apply the Householder transformation H = I - 2 v^T v to A from the left in
// place, i.e. A <- A H. Here v is assumed to be normalized.
// The parameter start_idx specifies the first non-zero entry in v.
template <class tm_t, class matrix_t>
KOKKOS_FORCEINLINE_FUNCTION void
apply_right_householder_transformation(tm_t tm, const double *const v, double *scratch,
                                       matrix_t &A, int col_start_idx = 0,
                                       int row_start_idx = 0) {
  const int nrows = GetNrows(A);
  const int ncols = GetNcols(A);
  for (int r = row_start_idx; r < nrows; ++r) {
    double w{0.0};
    summation(
        tm, col_start_idx, ncols - 1,
        [&](int c, double &ww) { ww += 2.0 * v[c] * A(r, c); }, w);
    once_per_team(tm, [&]() { scratch[r] = w; });
  }
  barrier(tm);
  parallel_loop(tm, col_start_idx, ncols - 1, row_start_idx, nrows - 1,
                [&](int c, int r) { A(r, c) -= scratch[r] * v[c]; });
}

} // namespace batched_linear_algebra
} // namespace parthenon

#endif // BATCHED_LINEAR_ALGEBRA_HOUSEHOLDER_HPP_
