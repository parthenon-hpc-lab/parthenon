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

#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iosfwd>
#include <vector>

#include "parthenon/parthenon.hpp"

using Vector = std::vector<double>;
using serial_tm_t = int; // TODO: parthenon::team_mbr_t;

KOKKOS_INLINE_FUNCTION
double safe_sqrt(const double a) { return std::sqrt(std::max(a, 0.)); }

class Matrix {
 public:
  Matrix(int nrows, int ncols);
  Matrix(int nrows) : Matrix(nrows, nrows) {}

  KOKKOS_FORCEINLINE_FUNCTION
  double &operator()(int row, int col) const { return data_(row, col); }

  static Matrix Transpose(const Matrix &A);
  static Matrix Identity(const Matrix &A);
  static Matrix Identity(int nrows, int ncols);

  static Matrix FromDiagonal(const std::vector<double> &diag);

  // Generate n×n matrix with i.i.d. Gaussian N(0,1) entries.
  static Matrix RandomGaussian(int n, unsigned seed = 12345) {
    return RandomGaussian(n, n, seed);
  }
  static Matrix RandomGaussian(int m, int n, unsigned seed = 12345);

  static Matrix RandomOrthogonal(int n, unsigned seed);

  // Build symmetric matrix A = Q Λ Qᵀ
  // where Λ holds the given eigenvalues.
  static Matrix FromSpectrum(const std::vector<double> &lambda, unsigned seed = 12345);

  static Matrix FromSingularValues(const std::vector<double> &lambda,
                                   unsigned seed = 12345);

  void SetRow(int row, std::vector<double> vals);

  KOKKOS_FORCEINLINE_FUNCTION
  int nrows() const { return nrows_; }

  KOKKOS_FORCEINLINE_FUNCTION
  int ncols() const { return ncols_; }

  KOKKOS_FORCEINLINE_FUNCTION
  bool IsSquare() const { return nrows_ == ncols_; }

  auto &GetData() { return data_; }

  double FrobeniusNorm() const;

  Matrix GetDeepCopy() const {
    Matrix other(nrows_, ncols_);
    Kokkos::deep_copy(other.data_, data_);
    return other;
  }

 private:
  parthenon::ParArray2D<double>::HostMirror data_;
  int ncols_, nrows_;
};

struct unity_vector_t {
  KOKKOS_INLINE_FUNCTION
  constexpr double operator()(int) const { return 1.0; }
  constexpr double operator[](int) const { return 1.0; }
};

template <class Vec, class PermVec>
struct vector_permuted_wrapper_t {
  KOKKOS_INLINE_FUNCTION
  vector_permuted_wrapper_t(const Vec &vec_in, const PermVec &perm_in)
      : vec(vec_in), perm(perm_in){}

  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(int i) const {
    return vec(perm(i));
  }

  Vec vec;
  PermVec perm;
};

template <class Vec, class PermVec>
KOKKOS_INLINE_FUNCTION
auto GetPermuted(const Vec &vec, const PermVec &perm, int n_active) {
  return vector_permuted_wrapper_t<Vec, PermVec>(vec, perm);
}

template <class Mat, class PermVec>
struct matrix_permuted_cols_wrapper_t {
  Mat mat;
  PermVec perm;
  int ncols_active;

  KOKKOS_INLINE_FUNCTION
  matrix_permuted_cols_wrapper_t(const Mat &mat_in, const PermVec &perm_in, int ncols_active_in)
      : mat(mat_in), perm(perm_in), ncols_active(ncols_active_in) {}

  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(int r, int c) const {
    return mat(r, perm(c));
  }
};

template <class Mat, class PermVec>
struct matrix_permuted_rows_wrapper_t {
  Mat mat;
  PermVec perm;
  int nrows_active;

  KOKKOS_INLINE_FUNCTION
  matrix_permuted_rows_wrapper_t(const Mat &mat_in, const PermVec &perm_in, int nrows_active_in)
      : mat(mat_in), perm(perm_in), nrows_active(nrows_active_in) {}

  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(int r, int c) const {
    return mat(perm(r), c);
  }
};

template <class T>
struct matrix_transpose_wrapper_t {
  KOKKOS_INLINE_FUNCTION
  matrix_transpose_wrapper_t(T *data, int orig_nrows, int orig_ncols)
      : data(data), orig_nrows(orig_nrows), orig_ncols(orig_ncols) {}

  KOKKOS_INLINE_FUNCTION
  T &operator()(int r, int c) {
    return data[c * orig_ncols + r];
  }

  KOKKOS_INLINE_FUNCTION
  T &operator()(int r, int c) const {
    return data[c * orig_ncols + r];
  }

  template <class PermVec>
  KOKKOS_INLINE_FUNCTION
  auto GetPermutedRows(const PermVec &perm, int nrows_active) const {
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
    : data(data), nrows(nrows), ncols(ncols) {}
  
  KOKKOS_INLINE_FUNCTION
  T &operator()(int r, int c) {
    return data[r * ncols + c];
  }

  KOKKOS_INLINE_FUNCTION
  T &operator()(int r, int c) const {
    return data[r * ncols + c];
  }

  KOKKOS_INLINE_FUNCTION
  auto GetTranspose() const {
    return matrix_transpose_wrapper_t<T>(data, nrows, ncols);
  }
  
  template <class PermVec>
  KOKKOS_INLINE_FUNCTION
  auto GetPermutedCols(const PermVec &perm, int ncols_active) const {
    return matrix_permuted_cols_wrapper_t<matrix_wrapper_t<T>, PermVec>(
        *this, perm, ncols_active);
  }

  int nrows, ncols;
  T *data;
};

template <class T>
KOKKOS_FORCEINLINE_FUNCTION
int GetNrows(const matrix_transpose_wrapper_t<T> &m) { return m.orig_ncols; }
template <class T>
KOKKOS_FORCEINLINE_FUNCTION
int GetNcols(const matrix_transpose_wrapper_t<T> &m) { return m.orig_nrows; }

template <class T>
KOKKOS_FORCEINLINE_FUNCTION
int GetNrows(const matrix_wrapper_t<T> &m) { return m.nrows; }
template <class T>
KOKKOS_FORCEINLINE_FUNCTION
int GetNcols(const matrix_wrapper_t<T> &m) { return m.ncols; }


KOKKOS_FORCEINLINE_FUNCTION
int GetNrows(const Matrix &m) { return m.nrows(); }
KOKKOS_FORCEINLINE_FUNCTION
int GetNcols(const Matrix &m) { return m.ncols(); }

template <class par_array_t>
KOKKOS_FORCEINLINE_FUNCTION int GetNrows(const par_array_t &m) {
  return m.extent_int(0);
}
template <class par_array_t>
KOKKOS_FORCEINLINE_FUNCTION int GetNcols(const par_array_t &m) {
  return m.extent_int(1);
}

template <class Mat, class PermVec>
KOKKOS_FORCEINLINE_FUNCTION
int GetNrows(const matrix_permuted_cols_wrapper_t<Mat, PermVec> &m) {
  return GetNrows(m.mat);
}

template <class Mat, class PermVec>
KOKKOS_FORCEINLINE_FUNCTION
int GetNcols(const matrix_permuted_cols_wrapper_t<Mat, PermVec> &m) {
  return m.ncols_active;
}

template <class Mat, class PermVec>
KOKKOS_FORCEINLINE_FUNCTION
int GetNrows(const matrix_permuted_rows_wrapper_t<Mat, PermVec> &m) {
  return m.nrows_active;
}

template <class Mat, class PermVec>
KOKKOS_FORCEINLINE_FUNCTION
int GetNcols(const matrix_permuted_rows_wrapper_t<Mat, PermVec> &m) {
  return GetNcols(m.mat);
}


// Stream output
std::ostream &operator<<(std::ostream &os, const Matrix &m);

// Matrix–matrix multiply
void Multiply(const Matrix &A, const Matrix &B, Matrix &C);

// TODO(LFR): Move all the stuff below to a new header

// Template must stay in the header so callers can instantiate it.
template <typename T>
KOKKOS_INLINE_FUNCTION int sign_of(T val) {
  constexpr T zero{0};
  // Zero is counted as positive for Householder reflector stability
  return (zero <= val) - (val < zero);
}

template <class tm_t>
KOKKOS_FORCEINLINE_FUNCTION void barrier(tm_t tm) {
  if constexpr (std::is_same_v<tm_t, parthenon::team_mbr_t>) {
    tm.team_barrier();
  }
}

template <class tm_t>
KOKKOS_FORCEINLINE_FUNCTION int rank(tm_t tm) {
  if constexpr (std::is_same_v<tm_t, parthenon::team_mbr_t>) {
    return tm.team_rank();
  } else {
    return 0;
  }
}

template <class tm_t, class F>
KOKKOS_FORCEINLINE_FUNCTION void once_per_team(tm_t tm, const F &func) {
  if constexpr (std::is_same_v<tm_t, serial_tm_t>) {
    func();
  } else if constexpr (std::is_same_v<tm_t, parthenon::team_mbr_t>) {
    Kokkos::single(Kokkos::PerTeam(tm), func);
  }
}

template <class F>
KOKKOS_FORCEINLINE_FUNCTION void sequential_loop(const int il, const int iu,
                                                 const F &func) {
  for (int i = il; i <= iu; ++i)
    func(i);
}

template <class tm_t, class F>
KOKKOS_FORCEINLINE_FUNCTION void parallel_loop(tm_t tm, const int il, const int iu,
                                               const F &func) {
  if constexpr (std::is_same_v<tm_t, serial_tm_t>) {
    for (int i = il; i <= iu; ++i)
      func(i);
  } else if constexpr (std::is_same_v<tm_t, parthenon::team_mbr_t>) {
    parthenon::par_for_inner(tm, il, iu, func);
  }
}

template <class tm_t, class F>
KOKKOS_FORCEINLINE_FUNCTION void parallel_loop(tm_t tm, const int jl, const int ju,
                                               const int il, const int iu,
                                               const F &func) {
  if constexpr (std::is_same_v<tm_t, serial_tm_t>) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        func(j, i);
      }
    }
  } else if constexpr (std::is_same_v<tm_t, parthenon::team_mbr_t>) {
    parthenon::par_for_inner(tm, jl, ju, il, iu, func);
  }
}

template <class tm_t, class F>
KOKKOS_FORCEINLINE_FUNCTION void summation(tm_t tm, const int il, const int iu,
                                           const F &func, double &sum) {
  if constexpr (std::is_same_v<tm_t, serial_tm_t>) {
    for (int i = il; i <= iu; ++i)
      func(i, sum);
  } else if constexpr (std::is_same_v<tm_t, parthenon::team_mbr_t>) {
    parthenon::par_reduce_inner(parthenon::inner_loop_pattern_ttr_tag, tm, il, iu, func,
                                Kokkos::Sum<double>(sum));
  }
}

template <class tm_t, class F>
KOKKOS_FORCEINLINE_FUNCTION void find_maximum(tm_t tm, const int il, const int iu,
                                              const F &func, double &mx) {
  if constexpr (std::is_same_v<tm_t, serial_tm_t>) {
    for (int i = il; i <= iu; ++i)
      func(i, mx);
  } else if constexpr (std::is_same_v<tm_t, parthenon::team_mbr_t>) {
    parthenon::par_reduce_inner(parthenon::inner_loop_pattern_ttr_tag, tm, il, iu, func,
                                Kokkos::Max<double>(mx));
  }
}

template <class tm_t, class F>
KOKKOS_FORCEINLINE_FUNCTION void summation(tm_t tm, const int jl, const int ju,
                                           const int il, const int iu, const F &func,
                                           double *sum) {
  if constexpr (std::is_same_v<tm_t, serial_tm_t>) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        func(j, i, sum);
      }
    }
  } else if constexpr (std::is_same_v<tm_t, parthenon::team_mbr_t>) {
    PARTHENON_FAIL("Shit, this doesn't work!");
  }
}

#endif // MATRIX_HPP
