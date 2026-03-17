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

#ifndef HOUSEHOLDER_HPP
#define HOUSEHOLDER_HPP

#include "matrix.hpp"
#include "utils/robust.hpp"

#include <cmath>

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
template <class tm_t, class matrix_t>
KOKKOS_FORCEINLINE_FUNCTION void
build_householder_vector_col(tm_t tm, int row, int col, const matrix_t &A, double *v) {
  const int nrows = GetNrows(A);
  const int ncols = GetNcols(A);
  double norm_x{0.0};

  summation(
      tm, row, nrows - 1,
      KOKKOS_LAMBDA(const int r, double &norm) { norm += A(r, col) * A(r, col); },
      norm_x);

  norm_x = safe_sqrt(norm_x);
  if (norm_x == 0.0) {
    parallel_loop(
        tm, 0, nrows - 1, KOKKOS_LAMBDA(const int i) { v[i] = 0.0; });
    return;
  }

  v[row] = A(row, col) + sign_of(A(row, col)) * norm_x;
  double norm_v{0.0};
  summation(
      tm, row + 1, nrows - 1,
      KOKKOS_LAMBDA(const int i, double &norm) {
        v[i] = A(i, col);
        norm += v[i] * v[i];
      },
      norm_v);
  norm_v += v[row] * v[row];
  norm_v = safe_sqrt(norm_v);

  double inv_norm_v = parthenon::robust::ratio(1.0, norm_v);
  parallel_loop(
      tm, 0, nrows - 1, KOKKOS_LAMBDA(const int i) { v[i] *= (i >= row) * inv_norm_v; });
}

template <class tm_t, class matrix_t>
KOKKOS_FORCEINLINE_FUNCTION void
build_householder_vector_row(tm_t tm, int row, int col, const matrix_t &A, double *v) {
  const int nrows = GetNrows(A);
  const int ncols = GetNcols(A);

  double norm_x{0.0};

  // Compute ||x|| where x = A(row, col:ncols-1)
  summation(
      tm, col, ncols - 1,
      KOKKOS_LAMBDA(const int c, double &norm) { norm += A(row, c) * A(row, c); },
      norm_x);

  norm_x = safe_sqrt(norm_x);

  // If the row segment is already zero, the reflector is identity
  if (norm_x == 0.0) {
    parallel_loop(
        tm, 0, ncols - 1, KOKKOS_LAMBDA(const int j) { v[j] = 0.0; });
    return;
  }

  // v[col] = x₀ + sign(x₀) * ||x||
  v[col] = A(row, col) + sign_of(A(row, col)) * norm_x;

  double norm_v{0.0};

  // Copy the remainder of the row segment into v
  summation(
      tm, col + 1, ncols - 1,
      KOKKOS_LAMBDA(const int j, double &norm) {
        v[j] = A(row, j);
        norm += v[j] * v[j];
      },
      norm_v);

  norm_v += v[col] * v[col];
  norm_v = safe_sqrt(norm_v);

  const double inv_norm_v = parthenon::robust::ratio(1.0, norm_v);

  // Zero entries before col and normalize the active part
  parallel_loop(
      tm, 0, ncols - 1, KOKKOS_LAMBDA(const int j) { v[j] *= (j >= col) * inv_norm_v; });
}

// Apply the Householder transformation H = I - 2 v^T v to A in place,
// i.e. A <- H A. Here v is assumed to be normalized.
template <class tm_t, class matrix_t>
KOKKOS_FORCEINLINE_FUNCTION void
apply_left_householder_transformation(tm_t tm, const double *const v, double *scratch,
                                      matrix_t &A) {
  const int nrows = GetNrows(A);
  const int ncols = GetNcols(A);
  for (int c = 0; c < ncols; ++c) {
    double w{0.0};
    summation(
        tm, 0, nrows - 1,
        KOKKOS_LAMBDA(int r, double &ww) { ww += 2.0 * v[r] * A(r, c); }, w);
    once_per_team(
        tm, KOKKOS_LAMBDA() { scratch[c] = w; });
  }
  barrier(tm);
  parallel_loop(
      tm, 0, ncols - 1, 0, nrows - 1,
      KOKKOS_LAMBDA(int c, int r) { A(r, c) -= scratch[c] * v[r]; });
}

// Apply the Householder transformation H = I - 2 v^T v to A from the left in
// place, i.e. A <- A H. Here v is assumed to be normalized.
template <class tm_t, class matrix_t>
KOKKOS_FORCEINLINE_FUNCTION void
apply_right_householder_transformation(tm_t tm, const double *const v, double *scratch,
                                       matrix_t &A) {
  const int nrows = GetNrows(A);
  const int ncols = GetNcols(A);
  for (int r = 0; r < nrows; ++r) {
    double w{0.0};
    summation(
        tm, 0, ncols - 1,
        KOKKOS_LAMBDA(int c, double &ww) { ww += 2.0 * v[c] * A(r, c); }, w);
    once_per_team(
        tm, KOKKOS_LAMBDA() { scratch[r] = w; });
  }
  barrier(tm);
  parallel_loop(
      tm, 0, ncols - 1, 0, nrows - 1,
      KOKKOS_LAMBDA(int c, int r) { A(r, c) -= scratch[r] * v[c]; });
}

#endif // HOUSEHOLDER_HPP
