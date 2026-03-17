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

// Stream output
std::ostream &operator<<(std::ostream &os, const Matrix &m);

// Matrix–matrix multiply
void Multiply(const Matrix &A, const Matrix &B, Matrix &C);

// TODO(LFR): Move all the stuff below to a new header

// Template must stay in the header so callers can instantiate it.
template <typename T>
KOKKOS_INLINE_FUNCTION int sign_of(T val) {
  constexpr T zero{0};
  return (zero < val) - (val < zero);
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
