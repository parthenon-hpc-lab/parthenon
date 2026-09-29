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
#include <vector>

#include "kokkos_abstraction.hpp"
#include "parthenon_arrays.hpp"

using Vector = std::vector<double>;

class Matrix {
 public:
  Matrix(int nrows, int ncols);
  explicit Matrix(int nrows) : Matrix(nrows, nrows) {}

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

// Stream output
std::ostream &operator<<(std::ostream &os, const Matrix &m);

// Matrix–matrix multiply
void Multiply(const Matrix &A, const Matrix &B, Matrix &C);

#endif // TST_UNIT_LINALG_TEST_UTILS_HPP_
