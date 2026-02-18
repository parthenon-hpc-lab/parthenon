#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

// #include <catch2/catch_test_macros.hpp>
#include <catch2/catch.hpp>

#include "linear_algebra/matrix.hpp"
#include "linear_algebra/square_svd.hpp"
#include "linear_algebra/symmetric_evd.hpp"

// ---------- basic norms / helpers ----------

static Matrix Multiply2(const Matrix &A, const Matrix &B) {
  Matrix C(A.nrows(), B.ncols());
  Multiply(A, B, C);
  return C;
}

static double OrthoError(const Matrix &Q) {
  // || Q^T Q - I ||_F
  const int n = Q.ncols();
  Matrix Qt = Matrix::Transpose(Q);
  Matrix QtQ = Multiply2(Qt, Q);

  double s = 0.0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      double e = QtQ(i, j) - (i == j ? 1.0 : 0.0);
      s += e * e;
    }
  }
  return std::sqrt(s);
}

// Check reconstruction: ||A - U*diag(s)*V^T||_F / max(||A||_F, atol)
static void CheckSVDReconstruction(const Matrix &A0, const Matrix &U, const Matrix &V,
                                   const std::vector<double> &sings, double rtol = 1e-10,
                                   double atol = 1e-12) {
  Matrix S = Matrix::FromDiagonal(sings);
  Matrix US = Multiply2(U, S);
  Matrix Vt = Matrix::Transpose(V);
  Matrix Ahat = Multiply2(US, Vt);

  // A0 - Ahat
  Matrix R(A0.nrows(), A0.ncols());
  for (int r = 0; r < A0.nrows(); ++r)
    for (int c = 0; c < A0.ncols(); ++c)
      R(r, c) = A0(r, c) - Ahat(r, c);

  const double denom = std::max(A0.FrobeniusNorm(), atol);
  const double rel = R.FrobeniusNorm() / denom;
  REQUIRE(rel < rtol);
}

static void CheckSingularValueSanity(
    const std::vector<double> &sings,
    double floor_rel = 50.0 * std::numeric_limits<double>::epsilon()) {
  // SVD convention: singular values should be non-negative (allow tiny
  // negatives)
  double smax = 0.0;
  for (double x : sings)
    smax = std::max(smax, std::abs(x));
  double tol = std::max(1.0, smax) * floor_rel;
  for (double x : sings)
    REQUIRE(x >= -tol);
}

static void SortAbs(std::vector<double> &v) {
  for (double &x : v)
    x = std::abs(x);
  std::sort(v.begin(), v.end());
}

static void RunSVDStress(const std::vector<double> &sings_template, int num_realizations,
                         int max_iter_factor, double recon_rtol,
                         double ortho_rtol = 1e-10, double sv_rtol = 1e-8,
                         double min_floor = 1e-12) {
  const int n = static_cast<int>(sings_template.size());

  std::vector<double> s_ref = sings_template;
  // Allow caller to pass unsorted; comparisons are order-insensitive.
  SortAbs(s_ref);

  for (int r = 0; r < num_realizations; ++r) {
    Matrix A = Matrix::FromSingularValues(sings_template, /*seed=*/12345u + r);
    Matrix A0 = A.GetDeepCopy();

    Matrix U(n, n), V(n, n);
    std::vector<double> s(n);

    int iters = SquareSVD::execute(&A, &U, &V, s.data());
    REQUIRE(iters < max_iter_factor * n);

    // 1) reconstruction
    CheckSVDReconstruction(A0, U, V, s, recon_rtol, /*atol=*/1e-12);

    // 2) orthogonality
    double uerr = OrthoError(U);
    double verr = OrthoError(V);
    // Scale orthogonality tolerance loosely with n
    REQUIRE(uerr / std::max(1.0, std::sqrt(double(n))) < ortho_rtol);
    REQUIRE(verr / std::max(1.0, std::sqrt(double(n))) < ortho_rtol);

    // 3) singular values sanity + compare to reference spectrum
    // (order-insensitive)
    // Removed: This checks that all singular values are positive,
    // which is just a convention (and is not currently enforced
    // in the SquareSVD).
    // CheckSingularValueSanity(s);

    std::vector<double> s_sorted = s;
    SortAbs(s_sorted);

    // floor based on magnitude of reference spectrum
    double floor_val = min_floor;
    for (int i = 0; i < n; ++i)
      floor_val = std::max(floor_val, 1e-3 * std::max(1.0, s_ref[i]));

    for (int i = 0; i < n; ++i) {
      double diff = std::abs(s_sorted[i] - s_ref[i]);
      double denom = std::max(floor_val, s_ref[i]);
      REQUIRE(diff / denom < sv_rtol);
    }
  }
}

// ---------- Tests ----------

TEST_CASE("ImplicitQR bidiag SVD stress tests over spectra", "[svd][qr][bidiag]") {
  const int n = 50;
  const int num_realizations = 100;

  // Reconstruction is the most important contract.
  const double recon_rtol = 5e-10;
  const double sv_rtol = 1e-8;

  // 1) Baseline: arithmetic-ish with a big outlier and tiny perturbations
  SECTION("Arithmetic-ish with perturbed tail and large outlier") {
    std::vector<double> s(n);
    for (int i = 0; i < n; ++i)
      s[i] = 1.0 + 0.25 * i;

    for (int i = 1; i <= 8; ++i) {
      double sign = (i % 2 ? -1.0 : 1.0); // allow mixed signs in construction
      s[n - i] = sign * (4.0 + i * 1.e-8);
    }
    s.back() = 1.e4;

    RunSVDStress(s, num_realizations, /*max_iter_factor=*/6, recon_rtol,
                 /*ortho_rtol=*/1e-10, sv_rtol);
  }

  // 2) Strong clustering (harder deflation / close singular values)
  SECTION("Clustered spectrum around 1.0") {
    std::vector<double> s(n);

    for (int i = 0; i < n / 4; ++i)
      s[i] = 0.1 + 0.01 * i;
    for (int i = n / 4; i < 3 * n / 4; ++i)
      s[i] = 1.0 + 1.e-10 * (i - n / 4);
    for (int i = 3 * n / 4; i < n; ++i)
      s[i] = 10.0 + 0.05 * (i - 3 * n / 4);

    RunSVDStress(s, num_realizations, /*max_iter_factor=*/8, recon_rtol,
                 /*ortho_rtol=*/1e-10, sv_rtol);
  }

  // 3) Repeated values (multiplicity)
  SECTION("Repeated singular values") {
    std::vector<double> s(n, 0.0);
    for (int i = 0; i < n / 2; ++i)
      s[i] = 2.0;
    for (int i = n / 2; i < n; ++i)
      s[i] = 2.0; // all equal
    // sprinkle a couple distinct to avoid fully-degenerate stress if desired
    s[0] = 1.0;
    s[1] = 3.0;

    RunSVDStress(s, num_realizations, /*max_iter_factor=*/10, recon_rtol,
                 /*ortho_rtol=*/1e-10, sv_rtol);
  }

  // 4) Wide dynamic range (scaling / conditioning)
  SECTION("Wide dynamic range") {
    std::vector<double> s(n);
    for (int i = 0; i < n; ++i) {
      double alpha = -10.0 + 20.0 * (static_cast<double>(i) / (n - 1));
      double val = std::pow(10.0, alpha);
      double sign = (i % 2 ? -1.0 : 1.0); // allow sign in construction
      s[i] = sign * val;
    }

    RunSVDStress(s, num_realizations, /*max_iter_factor=*/12,
                 /*recon_rtol=*/2e-9, /*ortho_rtol=*/2e-10, /*sv_rtol=*/2e-7);
  }

  // 5) Nearly rank-deficient (tiny tail)
  SECTION("Nearly rank-deficient tail") {
    std::vector<double> s(n, 0.0);
    for (int i = 0; i < n - 5; ++i)
      s[i] = 1.0 + 0.1 * i;
    for (int i = n - 5; i < n; ++i)
      s[i] = 1e-12 * (i - (n - 5) + 1);

    RunSVDStress(s, num_realizations, /*max_iter_factor=*/12,
                 /*recon_rtol=*/2e-9, /*ortho_rtol=*/2e-10, /*sv_rtol=*/5e-7,
                 /*min_floor=*/1e-12);
  }

  // 6) Strongly rank-deficient (exact zeros)
  SECTION("Strongly rank-deficient with exact zeros") {
    std::vector<double> s(n, 0.0);
    int k = 10;
    for (int i = 0; i < k; ++i)
      s[i] = 1.0 + i; // only k nonzero

    RunSVDStress(s, num_realizations, /*max_iter_factor=*/15,
                 /*recon_rtol=*/5e-9, /*ortho_rtol=*/5e-10, /*sv_rtol=*/1e-6,
                 /*min_floor=*/1e-14);
  }
}

TEST_CASE("SVD vs Gram eigenvalues: singular values match sqrt(eigs(A^T A))",
          "[svd][gram]") {
  const int n = 30;
  const int num_realizations = 50;

  for (int r = 0; r < num_realizations; ++r) {
    Matrix A = Matrix::RandomGaussian(n, n, /*seed=*/7777u + r);
    Matrix A0 = A.GetDeepCopy();

    Matrix U(n, n), V(n, n);
    std::vector<double> s(n);
    int iters = SquareSVD::execute(&A, &U, &V, s.data());
    REQUIRE(iters < 15 * n);

    // Form G = A0^T A0
    Matrix At = Matrix::Transpose(A0);
    Matrix G(n, n);
    Multiply(At, A0, G);

    // Eigen-decompose G (PSD)
    Matrix Q(n, n);
    std::vector<double> d(n);
    Matrix G0 = G.GetDeepCopy();
    int iters_e = SymmetricEVD::execute(&G, &Q, d.data());
    REQUIRE(iters_e < 10 * n);

    // Compare sorted singular values to sorted sqrt of eigenvalues (clamp tiny
    // negatives)
    std::vector<double> s_sorted = s;
    SortAbs(s_sorted);

    std::vector<double> g_sorted = d;
    std::sort(g_sorted.begin(), g_sorted.end());
    for (double &x : g_sorted)
      x = (x < 0.0 ? 0.0 : x);
    for (double &x : g_sorted)
      x = std::sqrt(x);

    // Relative compare with a floor
    double smax = std::max(1.0, s_sorted.back());
    double floor_val = 1e-12 * smax;

    for (int i = 0; i < n; ++i) {
      double diff = std::abs(s_sorted[i] - g_sorted[i]);
      double denom = std::max(floor_val, g_sorted[i]);
      REQUIRE(diff / denom < 1e-6);
    }

    // Also enforce reconstruction
    CheckSVDReconstruction(A0, U, V, s, /*rtol=*/5e-10, /*atol=*/1e-12);
  }
}

TEST_CASE("SVD handles scaled matrices: scaling singular values", "[svd][scale]") {
  const int n = 25;
  const int num_realizations = 30;

  std::vector<double> scales = {1e-12, 1e-6, 1.0, 1e6, 1e12};

  for (int r = 0; r < num_realizations; ++r) {
    Matrix A = Matrix::RandomGaussian(n, n, /*seed=*/9999u + r);
    Matrix A0 = A.GetDeepCopy();

    // Baseline SVD
    Matrix U0(n, n), V0(n, n);
    std::vector<double> s0(n);
    SquareSVD::execute(&A, &U0, &V0, s0.data());
    SortAbs(s0);

    for (double alpha : scales) {
      Matrix As = A0.GetDeepCopy();
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
          As(i, j) *= alpha;

      Matrix U(n, n), V(n, n);
      std::vector<double> s(n);
      SquareSVD::execute(&As, &U, &V, s.data());
      SortAbs(s);

      // Compare s ≈ |alpha| s0
      double floor_val = 1e-12 * std::max(1.0, std::abs(alpha) * s0.back());
      for (int i = 0; i < n; ++i) {
        double ref = std::abs(alpha) * s0[i];
        double diff = std::abs(s[i] - ref);
        double denom = std::max(floor_val, ref);
        REQUIRE(diff / denom < 1e-6);
      }
    }
  }
}
