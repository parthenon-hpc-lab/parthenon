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

#include <cmath>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "batched_linear_algebra/qr_decomposition.hpp"
#include "batched_linear_algebra/qr_solve.hpp"
#include "linalg_test_utils.hpp"

using namespace parthenon::batched_linear_algebra; // NOLINT(build/namespaces)

namespace qr_solve_test {

Matrix Multiply2(const Matrix &A, const Matrix &B) {
  Matrix C(A.nrows(), B.ncols());
  Multiply(A, B, C);
  return C;
}

// Leading nrows x ncols block of A.
Matrix Block(const Matrix &A, int nrows, int ncols) {
  Matrix out(nrows, ncols);
  for (int r = 0; r < nrows; ++r) {
    for (int c = 0; c < ncols; ++c) {
      out(r, c) = A(r, c);
    }
  }
  return out;
}

double DiffNorm(const Matrix &A, const Matrix &B) {
  double s = 0.0;
  for (int r = 0; r < A.nrows(); ++r) {
    for (int c = 0; c < A.ncols(); ++c) {
      const double e = A(r, c) - B(r, c);
      s += e * e;
    }
  }
  return std::sqrt(s);
}

double RelativeError(const Matrix &X, const Matrix &X0) {
  return DiffNorm(X, X0) / X0.FrobeniusNorm();
}

// Solve A0 X = B0 with QRSolve and return X (the leading rows of the
// overwritten B).
Matrix Solve(const Matrix &A0, const Matrix &B0) {
  Matrix A = A0.GetDeepCopy();
  Matrix B = B0.GetDeepCopy();
  REQUIRE(QRSolve::execute(&A, &B) == 0);
  return Block(B, A0.ncols(), B0.ncols());
}

} // namespace qr_solve_test

using namespace qr_solve_test; // NOLINT(build/namespaces)

TEST_CASE("Upper triangular solve", "[qr_solve][triangular]") {
  for (const int n : {1, 4, 9}) {
    for (const int k : {1, 3, 8}) {
      // Store R in a taller matrix with garbage below the diagonal, which the
      // solve must ignore
      const int m = n + 3;
      Matrix R = Matrix::RandomGaussian(m, n, 100u + n + k);
      Matrix R_upper(n, n);
      for (int r = 0; r < m; ++r) {
        for (int c = 0; c < n; ++c) {
          if (r == c) R(r, c) = 5.0 + std::abs(R(r, c));
          if (r > c) R(r, c) = 1e30;
          if (r < n && r <= c) R_upper(r, c) = R(r, c);
        }
      }

      // B has extra rows below n, which must be left untouched
      const Matrix X0 = Matrix::RandomGaussian(n, k, 200u + n + k);
      const Matrix RX0 = Multiply2(R_upper, X0);
      Matrix B(n + 2, k);
      for (int r = 0; r < n + 2; ++r) {
        for (int c = 0; c < k; ++c) {
          B(r, c) = r < n ? RX0(r, c) : -7.0;
        }
      }

      REQUIRE(TriangularSolve::execute(R, &B) == 0);
      const Matrix X = Block(B, n, k);
      REQUIRE(RelativeError(X, X0) < 1e-14);
      REQUIRE(DiffNorm(Multiply2(R_upper, X), RX0) / RX0.FrobeniusNorm() < 1e-14);
      for (int r = n; r < n + 2; ++r) {
        for (int c = 0; c < k; ++c) {
          REQUIRE(B(r, c) == -7.0);
        }
      }
    }
  }
}

TEST_CASE("QR solve of square systems", "[qr_solve][square]") {
  for (const int n : {1, 2, 5, 12}) {
    for (const int k : {1, 3, 2 * n}) {
      for (unsigned seed = 0; seed < 5; ++seed) {
        const Matrix A0 = Matrix::RandomGaussian(n, n, 1000u + 31 * seed + n);
        const Matrix X0 = Matrix::RandomGaussian(n, k, 2000u + 31 * seed + k);
        const Matrix B0 = Multiply2(A0, X0);

        const Matrix X = Solve(A0, B0);
        REQUIRE(DiffNorm(Multiply2(A0, X), B0) /
                    (A0.FrobeniusNorm() * X.FrobeniusNorm()) <
                1e-14);
        REQUIRE(RelativeError(X, X0) < 1e-11);
      }
    }
  }
}

TEST_CASE("QR solve of least-squares systems", "[qr_solve][lsq]") {
  for (const auto [m, n] : std::vector<std::pair<int, int>>{{6, 3}, {16, 5}, {30, 8}}) {
    for (const int k : {1, 4, 2 * m}) {
      const Matrix A0 = Matrix::RandomGaussian(m, n, 3000u + m + k);
      const Matrix X0 = Matrix::RandomGaussian(n, k, 4000u + m + k);

      // r = Q(:, n:m) g is orthogonal to range(A0), so X0 is the least-squares
      // solution of A0 X = A0 X0 + r and r(:,c) is the residual of column c
      Matrix R = A0.GetDeepCopy();
      Matrix Q(m, m);
      REQUIRE(QRDecomposition::execute(&R, &Q) == 0);
      const Matrix g = Matrix::RandomGaussian(m - n, k, 5000u + m + k);
      Matrix r(m, k);
      for (int i = 0; i < m; ++i) {
        for (int c = 0; c < k; ++c) {
          for (int j = 0; j < m - n; ++j) {
            r(i, c) += Q(i, n + j) * g(j, c);
          }
        }
      }
      Matrix B0 = Multiply2(A0, X0);
      for (int i = 0; i < m; ++i) {
        for (int c = 0; c < k; ++c) {
          B0(i, c) += r(i, c);
        }
      }

      Matrix A = A0.GetDeepCopy();
      Matrix B = B0.GetDeepCopy();
      REQUIRE(QRSolve::execute(&A, &B) == 0);
      REQUIRE(RelativeError(Block(B, n, k), X0) < 1e-12);

      // The trailing rows of Qᵀ B carry the residual norms
      for (int c = 0; c < k; ++c) {
        double tail = 0.0;
        double res = 0.0;
        for (int i = 0; i < m; ++i) {
          if (i >= n) tail += B(i, c) * B(i, c);
          res += r(i, c) * r(i, c);
        }
        REQUIRE(std::abs(std::sqrt(tail) - std::sqrt(res)) < 1e-13 * B0.FrobeniusNorm());
      }
    }
  }
}

TEST_CASE("QR solve robustness", "[qr_solve][robust]") {
  const int n = 8;
  const int k = 3;

  SECTION("Ill-conditioned matrix is solved backward stably") {
    std::vector<double> sings(n);
    for (int i = 0; i < n; ++i) {
      sings[i] = std::pow(10.0, -10.0 * i / (n - 1));
    }
    const Matrix A0 = Matrix::FromSingularValues(sings, 6000u);
    const Matrix X0 = Matrix::RandomGaussian(n, k, 6001u);
    const Matrix B0 = Multiply2(A0, X0);

    const Matrix X = Solve(A0, B0);
    REQUIRE(DiffNorm(Multiply2(A0, X), B0) / (A0.FrobeniusNorm() * X.FrobeniusNorm()) <
            1e-14);
    // Forward error is bounded by roughly cond(A) * eps
    REQUIRE(RelativeError(X, X0) < 1e-4);
  }

  SECTION("Uniformly scaled matrices near and beyond the sum-of-squares range") {
    for (const double scale : {1e-200, 1e-150, 1e150, 1e200}) {
      // Scale A but not B, so the solution scales by 1 / scale
      const Matrix A1 = Matrix::RandomGaussian(n, n, 7000u);
      const Matrix X1 = Matrix::RandomGaussian(n, k, 7001u);
      const Matrix B0 = Multiply2(A1, X1);
      Matrix A0 = A1.GetDeepCopy();
      Matrix X0 = X1.GetDeepCopy();
      for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) {
          A0(r, c) *= scale;
        }
        for (int c = 0; c < k; ++c) {
          X0(r, c) /= scale;
        }
      }

      const Matrix X = Solve(A0, B0);
      double err = 0.0;
      double nrm = 0.0;
      for (int r = 0; r < n; ++r) {
        for (int c = 0; c < k; ++c) {
          // Compare in unscaled units so the check itself cannot overflow
          const double d = (X(r, c) - X0(r, c)) * scale;
          err += d * d;
          nrm += X1(r, c) * X1(r, c);
        }
      }
      REQUIRE(std::sqrt(err / nrm) < 1e-11);
    }
  }
}

TEST_CASE("QR right solve", "[qr_solve][right]") {
  for (const auto [n, m] : std::vector<std::pair<int, int>>{{5, 5}, {12, 12}, {4, 9}}) {
    for (const int k : {1, 3, 2 * m}) {
      const Matrix A0 = Matrix::RandomGaussian(n, m, 8000u + n + m + k);
      const Matrix X0 = Matrix::RandomGaussian(k, n, 9000u + n + m + k);
      const Matrix B0 = Multiply2(X0, A0);

      Matrix A = A0.GetDeepCopy();
      Matrix B = B0.GetDeepCopy();
      REQUIRE(QRSolveRight::execute(&A, &B) == 0);
      const Matrix X = Block(B, k, n);
      REQUIRE(RelativeError(X, X0) < 1e-11);

      // Same algorithm as QRSolve on explicitly transposed copies. The two
      // instantiations may contract floating-point operations differently, so
      // compare to rounding rather than bitwise.
      Matrix AT = Matrix::Transpose(A0);
      Matrix BT = Matrix::Transpose(B0);
      REQUIRE(QRSolve::execute(&AT, &BT) == 0);
      REQUIRE(DiffNorm(B, Matrix::Transpose(BT)) < 1e-13 * B.FrobeniusNorm());
    }
  }
}
