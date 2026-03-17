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

// TODO(LFR): Maybe remove this or just move it to the tests

#include "matrix.hpp"

#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

#include "qr_decomposition.hpp"

Matrix::Matrix(int nrows, int ncols)
    : data_("matrix data", nrows, ncols), ncols_(ncols), nrows_(nrows) {}

Matrix Matrix::Transpose(const Matrix &A) {
  Matrix AT(A.ncols(), A.nrows());
  for (int i = 0; i < A.nrows(); ++i) {
    for (int j = 0; j < A.ncols(); ++j) {
      AT(j, i) = A(i, j);
    }
  }
  return AT;
}

Matrix Matrix::Identity(const Matrix &A) { return Identity(A.nrows(), A.ncols()); }

Matrix Matrix::Identity(int nrows, int ncols) {
  Matrix I(nrows, ncols);
  const int ones = std::min(nrows, ncols);
  for (int diag = 0; diag < ones; ++diag)
    I(diag, diag) = 1.0;
  return I;
}

Matrix Matrix::FromDiagonal(const std::vector<double> &diag) {
  Matrix A(diag.size());
  for (int i = 0; i < diag.size(); ++i)
    A(i, i) = diag[i];
  return A;
}

// Generate n×n matrix with i.i.d. Gaussian N(0,1) entries.
Matrix Matrix::RandomGaussian(int m, int n, unsigned seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> dist(0.0, 1.0);

  Matrix M(m, n);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      M(i, j) = dist(rng);

  return M;
}

Matrix Matrix::RandomOrthogonal(int n, unsigned seed) {
  Matrix A = Matrix::RandomGaussian(n, n, seed);
  Matrix Q = Matrix::Identity(n, n);
  QRDecomposition(A, Q);
  return Q;
}

// Build symmetric matrix A = Q Λ Qᵀ
// where Λ holds the given eigenvalues.
Matrix Matrix::FromSpectrum(const std::vector<double> &lambda, unsigned seed) {
  const int n = static_cast<int>(lambda.size());

  // Step 1: random Orthogongal matrix
  Matrix Q = Matrix::RandomOrthogonal(n, seed);

  Matrix Lambda = Matrix::FromDiagonal(lambda);

  // Step 4: A = Q Λ Qᵀ
  Matrix QT = Matrix::Transpose(Q);
  Matrix QL(n, n), A(n, n);
  Multiply(Q, Lambda, QL);
  Multiply(QL, QT, A);

  return A;
}

Matrix Matrix::FromSingularValues(const std::vector<double> &lambda, unsigned seed) {
  const int n = static_cast<int>(lambda.size());

  Matrix U = Matrix::RandomOrthogonal(n, seed * 17 + 1);
  Matrix V = Matrix::RandomOrthogonal(n, seed * 19 + 3);
  Matrix Lambda = Matrix::FromDiagonal(lambda);

  Matrix VT = Matrix::Transpose(V);
  Matrix temp(n, n);
  Multiply(U, Lambda, temp);
  Matrix out(n, n);
  Multiply(temp, VT, out);

  return out;
}

void Matrix::SetRow(int row, std::vector<double> vals) {
  for (int c = 0; c < ncols_; ++c)
    (*this)(row, c) = vals[c];
}

double Matrix::FrobeniusNorm() const {
  double s = 0.0;
  for (int r = 0; r < nrows_; ++r)
    for (int c = 0; c < ncols_; ++c)
      s += data_(r, c) * data_(r, c);
  return std::sqrt(s);
}

std::ostream &operator<<(std::ostream &os, const Matrix &m) {
  const int nr = m.nrows();
  const int nc = m.ncols();

  const int width = 9; // room for "+1.23e-04"
  const int precision = 2;

  for (int r = 0; r < nr; ++r) {
    os << "[ ";
    for (int c = 0; c < nc; ++c) {
      if (std::abs(m(r, c)) < 1.e-12) {
        os << "   ~0    ";
      } else {
        os << std::setw(width) << std::scientific << std::setprecision(precision)
           << std::showpos << m(r, c);
      }
      if (c + 1 < nc) os << " ";
    }
    os << " ]";
    if (r + 1 < nr) os << "\n";
  }

  os << std::noshowpos;
  return os;
}

void Multiply(const Matrix &A, const Matrix &B, Matrix &C) {
  assert(A.ncols() == B.nrows());
  assert(C.nrows() == A.nrows());
  assert(C.ncols() == B.ncols());

  for (int r = 0; r < C.nrows(); ++r) {
    for (int c = 0; c < C.ncols(); ++c) {
      C(r, c) = 0.0;
      for (int i = 0; i < A.ncols(); ++i) {
        C(r, c) += A(r, i) * B(i, c);
      }
    }
  }
}
