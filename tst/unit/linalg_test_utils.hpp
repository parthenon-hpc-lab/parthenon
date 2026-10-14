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

// Host-only dense matrix used to build inputs and check results in the
// batched_linear_algebra unit tests.

#ifndef TST_UNIT_LINALG_TEST_UTILS_HPP_
#define TST_UNIT_LINALG_TEST_UTILS_HPP_

// This file was made in part with generative AI.

#include <iosfwd>
#include <limits>
#include <type_traits>
#include <vector>

#include "basic_types.hpp"
#include "kokkos_abstraction.hpp"
#include "parthenon_arrays.hpp"

using parthenon::Real;
using Vector = std::vector<Real>;

// Converts a tolerance chosen for double precision into one for Real. The one-argument
// form keeps the same number of machine epsilons, which suits backward-error checks
// (reconstruction, orthogonality, residuals). Where that rescaling would be uselessly
// loose, the two-argument form gives the single-precision tolerance explicitly.
constexpr Real PrecisionTol(double double_tol) {
  return static_cast<Real>(double_tol * (std::numeric_limits<Real>::epsilon() /
                                         std::numeric_limits<double>::epsilon()));
}
constexpr Real PrecisionTol(double double_tol, double single_tol) {
  return static_cast<Real>(std::is_same_v<Real, double> ? double_tol : single_tol);
}

class Matrix {
 public:
  Matrix(int nrows, int ncols);
  explicit Matrix(int nrows) : Matrix(nrows, nrows) {}

  KOKKOS_FORCEINLINE_FUNCTION
  Real &operator()(int row, int col) const { return data_(row, col); }

  static Matrix Transpose(const Matrix &A);
  static Matrix Identity(const Matrix &A);
  static Matrix Identity(int nrows, int ncols);

  static Matrix FromDiagonal(const std::vector<Real> &diag);

  // Generate n×n matrix with i.i.d. Gaussian N(0,1) entries.
  static Matrix RandomGaussian(int n, unsigned seed = 12345) {
    return RandomGaussian(n, n, seed);
  }
  static Matrix RandomGaussian(int m, int n, unsigned seed = 12345);

  static Matrix RandomOrthogonal(int n, unsigned seed);

  // Build symmetric matrix A = Q Λ Qᵀ
  // where Λ holds the given eigenvalues.
  static Matrix FromSpectrum(const std::vector<Real> &lambda, unsigned seed = 12345);

  static Matrix FromSingularValues(const std::vector<Real> &lambda,
                                   unsigned seed = 12345);

  void SetRow(int row, std::vector<Real> vals);

  KOKKOS_FORCEINLINE_FUNCTION
  int nrows() const { return nrows_; }

  KOKKOS_FORCEINLINE_FUNCTION
  int ncols() const { return ncols_; }

  KOKKOS_FORCEINLINE_FUNCTION
  bool IsSquare() const { return nrows_ == ncols_; }

  auto &GetData() { return data_; }

  Real FrobeniusNorm() const;

  Matrix GetDeepCopy() const {
    Matrix other(nrows_, ncols_);
    Kokkos::deep_copy(other.data_, data_);
    return other;
  }

 private:
  parthenon::ParArray2D<Real>::HostMirror data_;
  int ncols_, nrows_;
};

KOKKOS_FORCEINLINE_FUNCTION
int GetNrows(const Matrix &m) { return m.nrows(); }
KOKKOS_FORCEINLINE_FUNCTION
int GetNcols(const Matrix &m) { return m.ncols(); }

// Stream output
std::ostream &operator<<(std::ostream &os, const Matrix &m);

// Matrix–matrix multiply
void Multiply(const Matrix &A, const Matrix &B, Matrix &C);

#endif // TST_UNIT_LINALG_TEST_UTILS_HPP_
