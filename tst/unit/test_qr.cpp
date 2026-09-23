#include <algorithm>
#include <cmath>
#include <vector>

#include <catch2/catch.hpp>

#include "linear_algebra/matrix.hpp"
#include "linear_algebra/qr_decomposition.hpp"

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

static double LQReconstructionError(const Matrix &A0, const Matrix &L,
                                    const Matrix &Q) {
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
        REQUIRE(OrthoError(Q) / std::max(1.0, std::sqrt(double(m))) < 1e-12);
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
        REQUIRE(OrthoError(Q) / std::max(1.0, std::sqrt(double(n))) < 1e-12);
        REQUIRE(ReconstructionError(A0, Q, A) / scale < 1e-11);
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
        REQUIRE(RowOrthoError(Q) / std::max(1.0, std::sqrt(double(m))) < 1e-12);

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
