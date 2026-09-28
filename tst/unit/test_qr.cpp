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

// This file was made in part with generative AI.

#include <algorithm>
#include <cmath>
#include <vector>

#include <catch2/catch.hpp>

#include "batched_linear_algebra/qr_decomposition.hpp"
#include "linalg_test_utils.hpp"

using namespace parthenon::batched_linear_algebra; // NOLINT(build/namespaces)

static Matrix Multiply2(const Matrix &A, const Matrix &B) {
  Matrix C(A.nrows(), B.ncols());
  Multiply(A, B, C);
  return C;
}

static double OrthoError(const Matrix &Q) {
  const int n = Q.ncols();
  Matrix Qt = Matrix::Transpose(Q);
  Matrix QtQ = Multiply2(Qt, Q);

  double s = 0.0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      const double e = QtQ(i, j) - (i == j ? 1.0 : 0.0);
      s += e * e;
    }
  }
  return std::sqrt(s);
}

static double RowOrthoError(const Matrix &Q) {
  const int m = Q.nrows();
  Matrix Qt = Matrix::Transpose(Q);
  Matrix QQt = Multiply2(Q, Qt);

  double s = 0.0;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < m; ++j) {
      const double e = QQt(i, j) - (i == j ? 1.0 : 0.0);
      s += e * e;
    }
  }
  return std::sqrt(s);
}

static double ReconstructionError(const Matrix &A0, const Matrix &Q, const Matrix &R) {
  Matrix QR = Multiply2(Q, R);

  double s = 0.0;
  for (int r = 0; r < A0.nrows(); ++r) {
    for (int c = 0; c < A0.ncols(); ++c) {
      const double e = A0(r, c) - QR(r, c);
      s += e * e;
    }
  }
  return std::sqrt(s);
}

static double UpperTrapezoidError(const Matrix &R) {
  const int m = R.nrows();
  const int n = R.ncols();
  double s = 0.0;
  for (int c = 0; c < n; ++c) {
    for (int r = c + 1; r < m; ++r) {
      s += R(r, c) * R(r, c);
    }
  }
  return std::sqrt(s);
}

static double AboveDiagonalError(const Matrix &A) {
  const int m = A.nrows();
  const int n = A.ncols();
  double s = 0.0;
  for (int r = 0; r < m; ++r) {
    for (int c = r + 1; c < n; ++c) {
      s += A(r, c) * A(r, c);
    }
  }
  return std::sqrt(s);
}

static double LQReconstructionError(const Matrix &A0, const Matrix &L, const Matrix &Q) {
  Matrix LQ = Multiply2(L, Q);

  double s = 0.0;
  for (int r = 0; r < A0.nrows(); ++r) {
    for (int c = 0; c < A0.ncols(); ++c) {
      const double e = A0(r, c) - LQ(r, c);
      s += e * e;
    }
  }
  return std::sqrt(s);
}

// Largest |R(r, c)| strictly below the diagonal.
static double MaxBelowDiagonal(const Matrix &R) {
  double mx = 0.0;
  for (int c = 0; c < R.ncols(); ++c) {
    for (int r = c + 1; r < R.nrows(); ++r) {
      mx = std::max(mx, std::abs(R(r, c)));
    }
  }
  return mx;
}

// Largest per-column relative reconstruction error ||A0(:,c) - (QR)(:,c)|| /
// ||A0(:,c)||, assuming R is upper trapezoidal. col_scale[c] is a known scale for
// column c, divided out before squaring so the check itself cannot under/overflow.
// Zero columns fall back to the absolute (scaled) error.
static double ColumnRelativeReconstructionError(const Matrix &A0, const Matrix &Q,
                                                const Matrix &R,
                                                const std::vector<double> &col_scale) {
  double mx = 0.0;
  for (int c = 0; c < A0.ncols(); ++c) {
    double e = 0.0;
    double nrm = 0.0;
    for (int i = 0; i < A0.nrows(); ++i) {
      double s = 0.0;
      for (int k = 0; k <= c; ++k) {
        s += Q(i, k) * R(k, c);
      }
      const double d = (A0(i, c) - s) / col_scale[c];
      const double a = A0(i, c) / col_scale[c];
      e += d * d;
      nrm += a * a;
    }
    mx = std::max(mx, nrm > 0.0 ? std::sqrt(e / nrm) : std::sqrt(e));
  }
  return mx;
}

// Factor A0 with a full Q, a thin Q, and no Q. Checks that R is exactly upper
// trapezoidal in every case, that Q is orthogonal, that every column of A0 is
// reconstructed to within tol relative to its own norm, and that the no-Q R is
// identical to the with-Q R.
static void CheckQR(const Matrix &A0, const std::vector<double> &col_scale,
                    const double tol) {
  const int m = A0.nrows();
  const int n = A0.ncols();

  Matrix R_full = A0.GetDeepCopy();
  for (const int qcols : {m, n}) {
    Matrix A = A0.GetDeepCopy();
    Matrix Q(m, qcols);
    REQUIRE(QRDecomposition::execute(&A, &Q) == 0);

    REQUIRE(MaxBelowDiagonal(A) == 0.0);
    REQUIRE(OrthoError(Q) / std::sqrt(static_cast<double>(qcols)) < 1e-12);
    REQUIRE(ColumnRelativeReconstructionError(A0, Q, A, col_scale) < tol);
    if (qcols == m) R_full = A;
  }

  Matrix R = A0.GetDeepCopy();
  REQUIRE(QRDecomposition::execute(&R) == 0);
  REQUIRE(MaxBelowDiagonal(R) == 0.0);
  for (int r = 0; r < m; ++r) {
    for (int c = 0; c < n; ++c) {
      REQUIRE(R(r, c) == R_full(r, c));
    }
  }
}

TEST_CASE("QR decomposition robustness", "[qr][rect][robust]") {
  const int m = 12;
  const int n = 6;
  const double tol = 1e-13;

  SECTION("Uniformly scaled matrices near and beyond the sum-of-squares range") {
    for (const double scale : {1e-200, 1e-160, 1e-150, 1e150, 1e160, 1e200}) {
      Matrix A = Matrix::RandomGaussian(m, n, 4234u);
      for (int r = 0; r < m; ++r) {
        for (int c = 0; c < n; ++c) {
          A(r, c) *= scale;
        }
      }
      CheckQR(A, std::vector<double>(n, scale), tol);
    }
  }

  SECTION("Graded columns spanning 1 to 1e-200") {
    Matrix A = Matrix::RandomGaussian(m, n, 5234u);
    std::vector<double> col_scale(n);
    for (int c = 0; c < n; ++c) {
      col_scale[c] = std::pow(10.0, -40.0 * c);
      for (int r = 0; r < m; ++r) {
        A(r, c) *= col_scale[c];
      }
    }
    CheckQR(A, col_scale, tol);
  }

  SECTION("Nearly dependent column") {
    Matrix A = Matrix::RandomGaussian(m, n, 6234u);
    Matrix noise = Matrix::RandomGaussian(m, 1, 6235u);
    for (int r = 0; r < m; ++r) {
      A(r, 3) = A(r, 0) + 1e-14 * noise(r, 0);
    }
    CheckQR(A, std::vector<double>(n, 1.0), tol);
  }

  SECTION("Exactly dependent column") {
    Matrix A = Matrix::RandomGaussian(m, n, 7234u);
    for (int r = 0; r < m; ++r) {
      A(r, 3) = A(r, 0);
    }
    CheckQR(A, std::vector<double>(n, 1.0), tol);

    // The dependent column should produce a negligible diagonal entry in R.
    Matrix R = A.GetDeepCopy();
    QRDecomposition::execute(&R);
    REQUIRE(std::abs(R(3, 3)) < 1e-13 * A.FrobeniusNorm());
  }

  SECTION("Zero columns") {
    for (const int zero_col : {0, 2, n - 1}) {
      Matrix A = Matrix::RandomGaussian(m, n, 8234u + zero_col);
      for (int r = 0; r < m; ++r) {
        A(r, zero_col) = 0.0;
      }
      CheckQR(A, std::vector<double>(n, 1.0), tol);
    }
  }

  SECTION("Zero matrix") {
    Matrix A(m, n);
    CheckQR(A, std::vector<double>(n, 1.0), tol);
  }

  SECTION("Wide matrix LQ without Q gives exactly lower-trapezoidal L") {
    Matrix A = Matrix::RandomGaussian(n, m, 9234u);
    Matrix L_withQ = A.GetDeepCopy();
    Matrix Q(n, m);
    REQUIRE(LQDecomposition::execute(&L_withQ, &Q) == 0);

    Matrix L = A.GetDeepCopy();
    REQUIRE(LQDecomposition::execute(&L) == 0);
    REQUIRE(MaxBelowDiagonal(Matrix::Transpose(L)) == 0.0);
    REQUIRE(MaxBelowDiagonal(Matrix::Transpose(L_withQ)) == 0.0);
    for (int r = 0; r < n; ++r) {
      for (int c = 0; c < m; ++c) {
        REQUIRE(L(r, c) == L_withQ(r, c));
      }
    }
  }
}

TEST_CASE("Tall-skinny QR decomposition", "[qr][rect][thin]") {
  SECTION("Random tall-skinny matrices, full Q") {
    for (const auto [m, n] : {std::pair<int, int>{8, 3}, {16, 5}, {30, 8}}) {
      for (unsigned seed = 0; seed < 10; ++seed) {
        Matrix A = Matrix::RandomGaussian(m, n, seed + 1234u);
        Matrix A0 = A.GetDeepCopy();
        Matrix Q(m, m);

        const int iters = QRDecomposition::execute(&A, &Q);
        REQUIRE(iters == 0);

        const double scale = std::max(1.0, A0.FrobeniusNorm());
        REQUIRE(UpperTrapezoidError(A) / scale < 1e-12);
        REQUIRE(OrthoError(Q) / std::max(1.0, std::sqrt(static_cast<double>(m))) < 1e-12);
        REQUIRE(ReconstructionError(A0, Q, A) / scale < 1e-11);
      }
    }
  }

  SECTION("Random tall-skinny matrices, thin Q") {
    for (const auto [m, n] : {std::pair<int, int>{8, 3}, {16, 5}, {30, 8}}) {
      for (unsigned seed = 0; seed < 10; ++seed) {
        Matrix A = Matrix::RandomGaussian(m, n, seed + 2234u);
        Matrix A0 = A.GetDeepCopy();
        Matrix Q(m, n);

        const int iters = QRDecomposition::execute(&A, &Q);
        REQUIRE(iters == 0);

        const double scale = std::max(1.0, A0.FrobeniusNorm());
        REQUIRE(UpperTrapezoidError(A) / scale < 1e-12);
        REQUIRE(OrthoError(Q) / std::max(1.0, std::sqrt(static_cast<double>(n))) < 1e-12);

        // With a thin Q (m x n), R is the top n x n block of the returned A;
        // reconstruct A0 = Q * R from that block.
        Matrix R(n, n);
        for (int r = 0; r < n; ++r) {
          for (int c = 0; c < n; ++c) {
            R(r, c) = A(r, c);
          }
        }
        REQUIRE(ReconstructionError(A0, Q, R) / scale < 1e-11);
      }
    }
  }

  SECTION("Single column") {
    Matrix A(7, 1);
    for (int i = 0; i < 7; ++i) {
      A(i, 0) = 1.0 + 0.5 * i;
    }

    Matrix A0 = A.GetDeepCopy();
    Matrix Q(7, 7);

    const int iters = QRDecomposition::execute(&A, &Q);
    REQUIRE(iters == 0);

    const double scale = std::max(1.0, A0.FrobeniusNorm());
    REQUIRE(UpperTrapezoidError(A) / scale < 1e-12);
    REQUIRE(OrthoError(Q) / std::max(1.0, std::sqrt(7.0)) < 1e-12);
    REQUIRE(ReconstructionError(A0, Q, A) / scale < 1e-12);
  }

  SECTION("Wide matrix LQ decomposition") {
    for (const auto [m, n] : {std::pair<int, int>{3, 8}, {5, 16}}) {
      for (unsigned seed = 0; seed < 10; ++seed) {
        Matrix A = Matrix::RandomGaussian(m, n, seed + 3234u);
        Matrix A0 = A.GetDeepCopy();
        Matrix Q(m, n);

        const int iters = LQDecomposition::execute(&A, &Q);
        REQUIRE(iters == 0);

        const double scale = std::max(1.0, A0.FrobeniusNorm());
        REQUIRE(AboveDiagonalError(A) / scale < 1e-12);
        REQUIRE(RowOrthoError(Q) / std::max(1.0, std::sqrt(static_cast<double>(m))) <
                1e-12);

        Matrix L(m, m);
        for (int r = 0; r < m; ++r) {
          for (int c = 0; c < m; ++c) {
            L(r, c) = A(r, c);
          }
        }

        REQUIRE(LQReconstructionError(A0, L, Q) / scale < 1e-11);
      }
    }
  }
}
