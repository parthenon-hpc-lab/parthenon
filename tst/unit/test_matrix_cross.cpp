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
#include <functional>
#include <tuple>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "batched_linear_algebra/matrix_cross.hpp"
#include "batched_linear_algebra/square_svd.hpp"
#include "linalg_test_utils.hpp"

using namespace parthenon::batched_linear_algebra; // NOLINT(build/namespaces)

namespace matrix_cross_test {

constexpr int kMaxSweeps = 10;

// Entries 1/(i + j + 1) computed on demand, never stored
struct HilbertLike {
  int nrows, ncols;
  KOKKOS_INLINE_FUNCTION double operator()(const int i, const int j) const {
    return 1.0 / (i + j + 1);
  }
};
KOKKOS_INLINE_FUNCTION int GetNrows(const HilbertLike &h) { return h.nrows; }
KOKKOS_INLINE_FUNCTION int GetNcols(const HilbertLike &h) { return h.ncols; }

// Wraps a matrix and counts how many entries are read (host only; annotated so
// the library's host-device templates can call it without warnings)
struct CountingMatrix {
  const Matrix *A;
  int *count;
  KOKKOS_INLINE_FUNCTION double operator()(const int i, const int j) const {
    ++(*count);
    return (*A)(i, j);
  }
};
KOKKOS_INLINE_FUNCTION int GetNrows(const CountingMatrix &m) { return m.A->nrows(); }
KOKKOS_INLINE_FUNCTION int GetNcols(const CountingMatrix &m) { return m.A->ncols(); }

// U Vᵀ with random U (n×r) and V (m×r), so rank r
Matrix LowRank(const int n, const int m, const int r, const unsigned seed) {
  const Matrix U = Matrix::RandomGaussian(n, r, seed);
  const Matrix VT = Matrix::RandomGaussian(r, m, seed + 1);
  Matrix A(n, m);
  Multiply(U, VT, A);
  return A;
}

struct CrossResult {
  Matrix C, R;
  std::vector<int> I, J;
  int sweeps;
};

void CheckDistinct(const std::vector<int> &idx, const int n) {
  std::vector<int> sorted = idx;
  std::sort(sorted.begin(), sorted.end());
  REQUIRE(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end());
  REQUIRE(sorted.front() >= 0);
  REQUIRE(sorted.back() < n);
}

// Runs the cross and checks that I and J are valid, C(I,:) is the identity and
// R = A(I,:). If warm is non-null, its I and J are the starting indices.
template <class matrix_t>
CrossResult RunCross(const matrix_t &A, const int r, const CrossResult *warm = nullptr,
                     const int max_sweeps = kMaxSweeps) {
  const int n = GetNrows(A);
  const int m = GetNcols(A);
  CrossResult res{Matrix(n, r), Matrix(r, m), std::vector<int>(r), std::vector<int>(r),
                  0};
  if (warm) {
    res.I = warm->I;
    res.J = warm->J;
  }
  res.sweeps = MatrixCross::execute(A, &res.C, &res.R, res.I.data(), res.J.data(), r,
                                    warm == nullptr, max_sweeps);
  REQUIRE(res.sweeps >= 0);
  REQUIRE(res.sweeps <= max_sweeps);

  CheckDistinct(res.I, n);
  CheckDistinct(res.J, m);
  for (int a = 0; a < r; ++a) {
    for (int k = 0; k < r; ++k) {
      REQUIRE(std::abs(res.C(res.I[a], k) - (a == k)) < 1e-12);
    }
    for (int b = 0; b < m; ++b) {
      REQUIRE(res.R(a, b) == A(res.I[a], b));
    }
  }
  return res;
}

// Max-norm and Frobenius-norm errors of C R against A
template <class matrix_t>
std::pair<double, double> CrossError(const matrix_t &A, const CrossResult &res) {
  const int n = GetNrows(A);
  const int m = GetNcols(A);
  Matrix CR(n, m);
  Multiply(res.C, res.R, CR);
  double max_err = 0.0;
  double frob_err = 0.0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      const double d = CR(i, j) - A(i, j);
      max_err = std::max(max_err, std::abs(d));
      frob_err += d * d;
    }
  }
  return {max_err, std::sqrt(frob_err)};
}

} // namespace matrix_cross_test

using namespace matrix_cross_test; // NOLINT(build/namespaces)

TEST_CASE("Matrix cross recovers exactly low-rank matrices", "[matrix_cross][exact]") {
  for (const auto [n, m, r] : std::vector<std::tuple<int, int, int>>{
           {30, 20, 4}, {20, 30, 4}, {12, 12, 12}, {25, 1, 1}}) {
    for (unsigned seed = 0; seed < 3; ++seed) {
      const Matrix A = LowRank(n, m, r, 100u + 10 * seed + n);
      const auto res = RunCross(A, r);
      REQUIRE(res.sweeps < kMaxSweeps);
      REQUIRE(CrossError(A, res).second < 1e-12 * A.FrobeniusNorm());
    }
  }
}

TEST_CASE("Matrix cross of a lazily evaluated matrix", "[matrix_cross][lazy]") {
  const HilbertLike H{40, 30};
  Matrix dense(H.nrows, H.ncols);
  for (int i = 0; i < H.nrows; ++i) {
    for (int j = 0; j < H.ncols; ++j) {
      dense(i, j) = H(i, j);
    }
  }
  std::vector<double> sings(H.ncols);
  SquareSVD::execute(&dense, sings.data());
  std::sort(sings.begin(), sings.end(), std::greater<double>());

  double prev_err = 1.0;
  for (const int r : {2, 4, 6, 8, 10}) {
    const auto res = RunCross(H, r);
    REQUIRE(res.sweeps < kMaxSweeps);
    const double err = CrossError(H, res).first;
    REQUIRE(err < prev_err);
    // A maximum-volume cross has max-norm error at most (r + 1) sigma_{r+1}
    REQUIRE(err <= (r + 1) * sings[r]);
    prev_err = err;
  }
}

TEST_CASE("Matrix cross warm start", "[matrix_cross][warm]") {
  const HilbertLike H{40, 30};
  const int r = 6;
  const auto first = RunCross(H, r);

  // Restarting from converged indices makes one sweep with no changes
  const auto again = RunCross(H, r, &first);
  REQUIRE(again.sweeps == 1);
  REQUIRE(again.I == first.I);
  REQUIRE(again.J == first.J);
  for (int i = 0; i < H.nrows; ++i) {
    for (int k = 0; k < r; ++k) {
      REQUIRE(std::abs(again.C(i, k) - first.C(i, k)) < 1e-10);
    }
  }

  // A perturbed start still converges to a good approximation
  const Matrix A = LowRank(30, 20, 4, 900u);
  const auto exact = RunCross(A, 4);
  CrossResult start = exact;
  for (int b = 0; b < A.ncols(); ++b) {
    if (std::find(start.J.begin(), start.J.end(), b) == start.J.end()) {
      start.J[0] = b;
      break;
    }
  }
  const auto perturbed = RunCross(A, 4, &start);
  REQUIRE(perturbed.sweeps < kMaxSweeps);
  REQUIRE(CrossError(A, perturbed).second < 1e-12 * A.FrobeniusNorm());
}

TEST_CASE("Matrix cross with no sweeps", "[matrix_cross][sweeps]") {
  // Only the first column step runs, which RunCross checks for consistency
  const HilbertLike H{25, 20};
  const auto res = RunCross(H, 5, nullptr, 0);
  REQUIRE(res.sweeps == 0);
  for (int k = 0; k < 5; ++k) {
    REQUIRE(res.J[k] == (k * H.ncols) / 5);
  }
}

TEST_CASE("Matrix cross only reads the fibers", "[matrix_cross][evaluations]") {
  const int n = 30;
  const int m = 20;
  const int r = 4;
  const Matrix A = LowRank(n, m, r, 1000u);
  int count = 0;
  const CountingMatrix CA{&A, &count};
  Matrix C(n, r);
  Matrix R(r, m);
  std::vector<int> I(r), J(r);
  const int sweeps = MatrixCross::execute(CA, &C, &R, I.data(), J.data(), r);
  REQUIRE(sweeps < kMaxSweeps);
  // One column step, then a row and a column step per sweep, then R
  REQUIRE(count == n * r + sweeps * (m + n) * r + r * m);

  // A null R skips the final evaluation
  count = 0;
  const int sweeps_no_r =
      MatrixCross::execute(CA, &C, static_cast<Matrix *>(nullptr), I.data(), J.data(), r);
  REQUIRE(count == n * r + sweeps_no_r * (m + n) * r);
}
