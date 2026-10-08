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
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "batched_linear_algebra/execution_utils.hpp"
#include "batched_linear_algebra/maxvol.hpp"
#include "batched_linear_algebra/qr_decomposition.hpp"
#include "linalg_test_utils.hpp"

using namespace parthenon::batched_linear_algebra; // NOLINT(build/namespaces)

namespace maxvol_test {

constexpr double kTau = 1.05;

// A(I,:) as an r x r matrix.
Matrix SubRows(const Matrix &A, const std::vector<int> &I) {
  const int r = static_cast<int>(I.size());
  Matrix S(r, A.ncols());
  for (int a = 0; a < r; ++a) {
    for (int c = 0; c < A.ncols(); ++c) {
      S(a, c) = A(I[a], c);
    }
  }
  return S;
}

// |det S| from the diagonal of R in S = Q R.
double AbsDet(const Matrix &S) {
  Matrix R = S.GetDeepCopy();
  QRDecomposition::execute(&R);
  double det = 1.0;
  for (int i = 0; i < R.nrows(); ++i) {
    det *= std::abs(R(i, i));
  }
  return det;
}

double MaxAbs(const Matrix &A) {
  double mx = 0.0;
  for (int r = 0; r < A.nrows(); ++r) {
    for (int c = 0; c < A.ncols(); ++c) {
      mx = std::max(mx, std::abs(A(r, c)));
    }
  }
  return mx;
}

// Checks that I is a valid set of distinct rows and that B = A A(I,:)⁻¹, and
// optionally that B is tau-dominant.
void CheckMaxvol(const Matrix &A, const Matrix &B, const std::vector<int> &I,
                 const bool check_dominance = true) {
  const int n = A.nrows();
  const int r = A.ncols();
  REQUIRE(static_cast<int>(I.size()) == r);
  std::vector<int> sorted = I;
  std::sort(sorted.begin(), sorted.end());
  REQUIRE(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end());
  REQUIRE(sorted.front() >= 0);
  REQUIRE(sorted.back() < n);

  // B(I,:) is the identity
  for (int a = 0; a < r; ++a) {
    for (int c = 0; c < r; ++c) {
      REQUIRE(std::abs(B(I[a], c) - (a == c)) < 1e-12);
    }
  }

  // B A(I,:) reproduces A
  const Matrix S = SubRows(A, I);
  Matrix BS(n, r);
  Multiply(B, S, BS);
  double err = 0.0;
  for (int i = 0; i < n; ++i) {
    for (int c = 0; c < r; ++c) {
      const double d = BS(i, c) - A(i, c);
      err += d * d;
    }
  }
  REQUIRE(std::sqrt(err) < 1e-12 * A.FrobeniusNorm());

  if (check_dominance) REQUIRE(MaxAbs(B) <= kTau * (1.0 + 1e-12));
}

// Run maxvol on A with the greedy start and check the result.
std::pair<Matrix, std::vector<int>> RunMaxvol(const Matrix &A) {
  Matrix B(A.nrows(), A.ncols());
  std::vector<int> I(A.ncols());
  const int iters = Maxvol::execute(A, &B, I.data(), true, kTau);
  REQUIRE(iters >= 0);
  REQUIRE(iters < 100);
  CheckMaxvol(A, B, I);
  return {B, I};
}

} // namespace maxvol_test

using namespace maxvol_test; // NOLINT(build/namespaces)

TEST_CASE("Maximum location helper", "[maxvol][utils]") {
  const std::vector<double> vals{0.5, -3.0, 2.0, 7.5, -7.0, 1.0};
  maxloc_value_t res;
  find_maximum_location(
      serial_tm_t(), 0, static_cast<int>(vals.size()) - 1,
      [&](const int i, maxloc_value_t &m) {
        if (vals[i] > m.val) {
          m.val = vals[i];
          m.loc = i;
        }
      },
      res);
  REQUIRE(res.val == 7.5);
  REQUIRE(res.loc == 3);

  // Flattened 2D index, as used for argmax |B(i,j)|
  const Matrix A = Matrix::RandomGaussian(7, 3, 11u);
  find_maximum_location(
      serial_tm_t(), 0, 7 * 3 - 1,
      [&](const int idx, maxloc_value_t &m) {
        const double v = std::abs(A(idx / 3, idx % 3));
        if (v > m.val) {
          m.val = v;
          m.loc = idx;
        }
      },
      res);
  REQUIRE(res.val == MaxAbs(A));
  REQUIRE(std::abs(A(res.loc / 3, res.loc % 3)) == res.val);
}

TEST_CASE("Maxvol on random matrices", "[maxvol][random]") {
  for (const auto [n, r] :
       std::vector<std::pair<int, int>>{{10, 1}, {20, 4}, {50, 8}, {8, 8}}) {
    for (unsigned seed = 0; seed < 5; ++seed) {
      const Matrix A = Matrix::RandomGaussian(n, r, 100u + 17 * seed + n);
      const auto [B, I] = RunMaxvol(A);
      if (n == r) {
        // Every row is selected, so B is the identity
        REQUIRE(MaxAbs(B) <= 1.0 + 1e-12);
      }
    }
  }
}

TEST_CASE("Maxvol is close to the maximum volume", "[maxvol][volume]") {
  const int n = 9;
  const int r = 3;
  for (unsigned seed = 0; seed < 5; ++seed) {
    const Matrix A = Matrix::RandomGaussian(n, r, 200u + seed);
    const auto [B, I] = RunMaxvol(A);
    const double det = AbsDet(SubRows(A, I));

    double max_det = 0.0;
    for (int a = 0; a < n; ++a) {
      for (int b = a + 1; b < n; ++b) {
        for (int c = b + 1; c < n; ++c) {
          max_det = std::max(max_det, AbsDet(SubRows(A, {a, b, c})));
        }
      }
    }
    // A(J,:) = B(J,:) A(I,:) and Hadamard's inequality give
    // |det A(J,:)| <= (tau sqrt(r))^r |det A(I,:)| for every row set J
    REQUIRE(det >= max_det / std::pow(kTau * std::sqrt(r), r) * (1.0 - 1e-12));

    // No single row swap increases the volume by more than tau
    for (int i = 0; i < n; ++i) {
      if (std::find(I.begin(), I.end(), i) != I.end()) continue;
      for (int j = 0; j < r; ++j) {
        std::vector<int> J = I;
        J[j] = i;
        REQUIRE(AbsDet(SubRows(A, J)) <= kTau * det * (1.0 + 1e-12));
      }
    }
  }
}

TEST_CASE("Maxvol finds planted dominant rows", "[maxvol][planted]") {
  const int n = 30;
  const int r = 4;
  const std::vector<int> planted{3, 11, 17, 25};
  Matrix A = Matrix::RandomGaussian(n, r, 300u);
  for (const int p : planted) {
    for (int c = 0; c < r; ++c) {
      A(p, c) *= 1e3;
    }
  }
  auto [B, I] = RunMaxvol(A);
  std::sort(I.begin(), I.end());
  REQUIRE(I == planted);
}

TEST_CASE("Maxvol with a single column", "[maxvol][rank1]") {
  const Matrix A = Matrix::RandomGaussian(15, 1, 400u);
  const auto [B, I] = RunMaxvol(A);
  int imax = 0;
  for (int i = 1; i < A.nrows(); ++i) {
    if (std::abs(A(i, 0)) > std::abs(A(imax, 0))) imax = i;
  }
  REQUIRE(I[0] == imax);
}

TEST_CASE("Maxvol warm start", "[maxvol][warm]") {
  const int n = 40;
  const int r = 6;
  const Matrix A = Matrix::RandomGaussian(n, r, 500u);

  // A poor starting set still converges
  Matrix B(n, r);
  std::vector<int> I(r);
  for (int j = 0; j < r; ++j) {
    I[j] = j;
  }
  const int iters = Maxvol::execute(A, &B, I.data(), false, kTau);
  REQUIRE(iters > 0);
  REQUIRE(iters < 100);
  CheckMaxvol(A, B, I);

  // Restarting from a converged set does no swaps and gives the same B
  Matrix B2(n, r);
  std::vector<int> I2 = I;
  REQUIRE(Maxvol::execute(A, &B2, I2.data(), false, kTau) == 0);
  REQUIRE(I2 == I);
  for (int i = 0; i < n; ++i) {
    for (int c = 0; c < r; ++c) {
      REQUIRE(std::abs(B2(i, c) - B(i, c)) < 1e-12 * std::max(1.0, MaxAbs(B)));
    }
  }

  // With max_iters = 0 only B = A A(I,:)⁻¹ is formed
  Matrix B3(n, r);
  std::vector<int> I3{5, 1, 30, 12, 7, 22};
  const std::vector<int> I3_start = I3;
  REQUIRE(Maxvol::execute(A, &B3, I3.data(), false, kTau, 0) == 0);
  REQUIRE(I3 == I3_start);
  CheckMaxvol(A, B3, I3, false);
}

TEST_CASE("Maxvol on a rank deficient matrix stays in bounds", "[maxvol][deficient]") {
  // A zero column makes the elimination produce NaN, after which no pivot
  // location is found. The result is meaningless but the indices stay valid.
  const int n = 12;
  const int r = 4;
  Matrix A = Matrix::RandomGaussian(n, r, 700u);
  for (int i = 0; i < n; ++i) {
    A(i, 1) = 0.0;
  }
  Matrix B(n, r);
  std::vector<int> I(r);
  Maxvol::execute(A, &B, I.data(), true, kTau);
  for (const int i : I) {
    REQUIRE(i >= 0);
    REQUIRE(i < n);
  }
}

TEST_CASE("Maxvol column selection through a transpose view", "[maxvol][columns]") {
  const int r = 4;
  const int n = 20;
  Matrix W = Matrix::RandomGaussian(r, n, 600u);
  const Matrix WT = Matrix::Transpose(W);

  matrix_transpose_view_t<Matrix> W_view(W);
  Matrix B(n, r);
  std::vector<int> J(r);
  REQUIRE(Maxvol::execute(W_view, &B, J.data(), true, kTau) < 100);
  CheckMaxvol(WT, B, J);

  // Same selection as running on an explicit transpose
  const auto [B_ref, J_ref] = RunMaxvol(WT);
  REQUIRE(J == J_ref);
}
