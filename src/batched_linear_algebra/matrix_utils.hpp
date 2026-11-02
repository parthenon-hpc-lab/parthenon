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

#ifndef BATCHED_LINEAR_ALGEBRA_MATRIX_UTILS_HPP_
#define BATCHED_LINEAR_ALGEBRA_MATRIX_UTILS_HPP_

// This file was made in part with generative AI.

#include "kokkos_abstraction.hpp"

namespace parthenon {
namespace batched_linear_algebra {

// Lightweight wrappers that let raw pointers, Kokkos views, and permuted or
// transposed versions of them be used as matrices by the algorithms in this
// library. Any type used as a matrix must provide operator()(int row, int col)
// and overloads of GetNrows/GetNcols findable by argument-dependent lookup.

struct unity_vector_t {
  KOKKOS_INLINE_FUNCTION
  constexpr double operator()(int) const { return 1.0; }
  constexpr double operator[](int) const { return 1.0; }
};

template <class Vec, class PermVec>
struct vector_permuted_wrapper_t {
  KOKKOS_INLINE_FUNCTION
  vector_permuted_wrapper_t(const Vec &vec_in, const PermVec &perm_in)
      : vec(vec_in), perm(perm_in) {}

  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(int i) const { return vec(perm(i)); }

  Vec vec;
  PermVec perm;
};

template <class Vec, class PermVec>
KOKKOS_INLINE_FUNCTION auto GetPermuted(const Vec &vec, const PermVec &perm,
                                        int n_active) {
  return vector_permuted_wrapper_t<Vec, PermVec>(vec, perm);
}

template <class Mat, class PermVec>
struct matrix_permuted_cols_wrapper_t {
  Mat mat;
  PermVec perm;
  int ncols_active;

  KOKKOS_INLINE_FUNCTION
  matrix_permuted_cols_wrapper_t(const Mat &mat_in, const PermVec &perm_in,
                                 int ncols_active_in)
      : mat(mat_in), perm(perm_in), ncols_active(ncols_active_in) {}

  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(int r, int c) const { return mat(r, perm(c)); }
};

template <class Mat, class PermVec>
struct matrix_permuted_rows_wrapper_t {
  Mat mat;
  PermVec perm;
  int nrows_active;

  KOKKOS_INLINE_FUNCTION
  matrix_permuted_rows_wrapper_t(const Mat &mat_in, const PermVec &perm_in,
                                 int nrows_active_in)
      : mat(mat_in), perm(perm_in), nrows_active(nrows_active_in) {}

  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(int r, int c) const { return mat(perm(r), c); }
};

template <class T>
struct matrix_transpose_wrapper_t {
  KOKKOS_INLINE_FUNCTION
  matrix_transpose_wrapper_t(T *data, int orig_nrows, int orig_ncols)
      : orig_nrows(orig_nrows), orig_ncols(orig_ncols), data(data) {}

  KOKKOS_INLINE_FUNCTION
  T &operator()(int r, int c) { return data[c * orig_ncols + r]; }

  KOKKOS_INLINE_FUNCTION
  T &operator()(int r, int c) const { return data[c * orig_ncols + r]; }

  template <class PermVec>
  KOKKOS_INLINE_FUNCTION auto GetPermutedRows(const PermVec &perm,
                                              int nrows_active) const {
    return matrix_permuted_rows_wrapper_t<matrix_transpose_wrapper_t<T>, PermVec>(
        *this, perm, nrows_active);
  }

  int orig_nrows, orig_ncols;
  T *data;
};

template <class T>
struct matrix_wrapper_t {
  KOKKOS_INLINE_FUNCTION
  matrix_wrapper_t(T *data, int nrows, int ncols)
      : nrows(nrows), ncols(ncols), data(data) {}

  KOKKOS_INLINE_FUNCTION
  T &operator()(int r, int c) { return data[r * ncols + c]; }

  KOKKOS_INLINE_FUNCTION
  T &operator()(int r, int c) const { return data[r * ncols + c]; }

  KOKKOS_INLINE_FUNCTION
  auto GetTranspose() const { return matrix_transpose_wrapper_t<T>(data, nrows, ncols); }

  template <class PermVec>
  KOKKOS_INLINE_FUNCTION auto GetPermutedCols(const PermVec &perm,
                                              int ncols_active) const {
    return matrix_permuted_cols_wrapper_t<matrix_wrapper_t<T>, PermVec>(*this, perm,
                                                                        ncols_active);
  }

  int nrows, ncols;
  T *data;
};

template <class T>
KOKKOS_FORCEINLINE_FUNCTION int GetNrows(const matrix_transpose_wrapper_t<T> &m) {
  return m.orig_ncols;
}
template <class T>
KOKKOS_FORCEINLINE_FUNCTION int GetNcols(const matrix_transpose_wrapper_t<T> &m) {
  return m.orig_nrows;
}

template <class T>
KOKKOS_FORCEINLINE_FUNCTION int GetNrows(const matrix_wrapper_t<T> &m) {
  return m.nrows;
}
template <class T>
KOKKOS_FORCEINLINE_FUNCTION int GetNcols(const matrix_wrapper_t<T> &m) {
  return m.ncols;
}

// Fallback for Kokkos views (and anything else with extent_int)
template <class par_array_t>
KOKKOS_FORCEINLINE_FUNCTION int GetNrows(const par_array_t &m) {
  return m.extent_int(0);
}
template <class par_array_t>
KOKKOS_FORCEINLINE_FUNCTION int GetNcols(const par_array_t &m) {
  return m.extent_int(1);
}

template <class Mat, class PermVec>
KOKKOS_FORCEINLINE_FUNCTION int
GetNrows(const matrix_permuted_cols_wrapper_t<Mat, PermVec> &m) {
  return GetNrows(m.mat);
}

template <class Mat, class PermVec>
KOKKOS_FORCEINLINE_FUNCTION int
GetNcols(const matrix_permuted_cols_wrapper_t<Mat, PermVec> &m) {
  return m.ncols_active;
}

template <class Mat, class PermVec>
KOKKOS_FORCEINLINE_FUNCTION int
GetNrows(const matrix_permuted_rows_wrapper_t<Mat, PermVec> &m) {
  return m.nrows_active;
}

template <class Mat, class PermVec>
KOKKOS_FORCEINLINE_FUNCTION int
GetNcols(const matrix_permuted_rows_wrapper_t<Mat, PermVec> &m) {
  return GetNcols(m.mat);
}

} // namespace batched_linear_algebra
} // namespace parthenon

#endif // BATCHED_LINEAR_ALGEBRA_MATRIX_UTILS_HPP_
