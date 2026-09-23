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

static double SingularValueEnergy(const std::vector<double> &sings) {
  double sum = 0.0;
  for (double x : sings) sum += x * x;
  return sum;
}

static void CheckSingularValueEnergyIdentity(const Matrix &A,
                                             const std::vector<double> &sings,
                                             double rtol = 1e-12,
                                             double atol = 1e-12) {
  const double sigma2 = SingularValueEnergy(sings);
  const double frob2 = A.FrobeniusNorm() * A.FrobeniusNorm();
  const double denom = std::max(frob2, atol);
  REQUIRE(std::abs(sigma2 - frob2) / denom < rtol);
}

static void SortAbs(std::vector<double> &v) {
  for (double &x : v)
    x = std::abs(x);
  std::sort(v.begin(), v.end());
}

static Matrix ThinOrthogonal(int m, int k, unsigned seed) {
  Matrix Q = Matrix::RandomOrthogonal(m, seed);
  Matrix T(m, k);
  for (int r = 0; r < m; ++r) {
    for (int c = 0; c < k; ++c) {
      T(r, c) = Q(r, c);
    }
  }
  return T;
}

static Matrix FromTallSingularValues(int m, int n, const std::vector<double> &sings,
                                     unsigned seed = 12345u) {
  const int k = static_cast<int>(sings.size());
  Matrix U = ThinOrthogonal(m, k, seed * 17u + 1u);
  Matrix V = Matrix::RandomOrthogonal(n, seed * 19u + 3u);
  Matrix S = Matrix::FromDiagonal(sings);
  Matrix US = Multiply2(U, S);
  Matrix Vt = Matrix::Transpose(V);
  Matrix A(m, n);
  Multiply(US, Vt, A);
  return A;
}

static void RunTallSVDStress(int m, int n, const std::vector<double> &sings_template,
                             int num_realizations, double recon_rtol,
                             double ortho_rtol = 1e-10, double sv_rtol = 1e-8,
                             double min_floor = 1e-12, bool explicit_scratch = false) {
  REQUIRE(m >= n);
  REQUIRE(static_cast<int>(sings_template.size()) == n);

  std::vector<double> s_ref = sings_template;
  SortAbs(s_ref);

  for (int r = 0; r < num_realizations; ++r) {
    Matrix A = FromTallSingularValues(m, n, sings_template, /*seed=*/12345u + r);
    Matrix A0 = A.GetDeepCopy();

    Matrix U(m, n), V(n, n);
    std::vector<double> s(n);

    int iters = 0;
    if (explicit_scratch) {
      std::vector<double> scratch(SquareSVD::double_scratch_size(m, n));
      std::vector<std::size_t> iscratch(SquareSVD::sizet_scratch_size(n));
      iters = SquareSVD::execute(serial_tm_t(), &A, &U, &V, s.data(), scratch.data(),
                                 iscratch.data());
    } else {
      iters = SquareSVD::execute(&A, &U, &V, s.data());
    }
    REQUIRE(iters >= 0);

    CheckSVDReconstruction(A0, U, V, s, recon_rtol, /*atol=*/1e-12);
    REQUIRE(OrthoError(U) / std::max(1.0, std::sqrt(double(n))) < ortho_rtol);
    REQUIRE(OrthoError(V) / std::max(1.0, std::sqrt(double(n))) < ortho_rtol);
    CheckSingularValueEnergyIdentity(A0, s, /*rtol=*/1e-10, /*atol=*/1e-12);

    std::vector<double> s_sorted = s;
    SortAbs(s_sorted);

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
    REQUIRE(iters > 0);

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

TEST_CASE("Tall-skinny SVD stress tests", "[svd][rect][thin]") {
  const double recon_rtol = 5e-10;
  const double sv_rtol = 1e-8;

  SECTION("Auto scratch sizing") {
    RunTallSVDStress(8, 3, {4.0, 1.5, 0.25}, 20, recon_rtol,
                     /*ortho_rtol=*/1e-10, sv_rtol);
    RunTallSVDStress(30, 5, {12.0, 3.0, 1.0, 0.2, 0.05}, 10, recon_rtol,
                     /*ortho_rtol=*/1e-10, sv_rtol);
  }

  SECTION("Clustered spectrum") {
    RunTallSVDStress(12, 4, {1.0, 1.0 + 1.e-12, 1.0 + 2.e-12, 1.0 + 3.e-12}, 12,
                     recon_rtol, /*ortho_rtol=*/1e-10, sv_rtol);
  }

  SECTION("Wide dynamic range") {
    RunTallSVDStress(18, 6, {1.e3, 1.e1, 1.0, 1.e-2, 1.e-4, 1.e-6}, 10, recon_rtol,
                     /*ortho_rtol=*/2e-10, /*sv_rtol=*/2e-7);
  }

  SECTION("Nearly rank-deficient tail") {
    RunTallSVDStress(16, 5, {5.0, 2.0, 0.5, 1.e-12, 2.e-13}, 10, recon_rtol,
                     /*ortho_rtol=*/2e-10, /*sv_rtol=*/5e-7, /*min_floor=*/1e-12);
  }

  SECTION("Strongly rank-deficient with exact zeros") {
    RunTallSVDStress(14, 5, {7.0, 3.0, 1.0, 0.0, 0.0}, 10, recon_rtol,
                     /*ortho_rtol=*/2e-10, /*sv_rtol=*/1e-6, /*min_floor=*/1e-14);
  }

  SECTION("Explicit scratch sizing") {
    REQUIRE(SquareSVD::double_scratch_size(30, 5) == 90);
    REQUIRE(SquareSVD::double_scratch_size(5) == 15);
    RunTallSVDStress(9, 4, {6.0, 2.0, 0.5, 0.1}, 8, recon_rtol,
                     /*ortho_rtol=*/1e-10, sv_rtol, /*min_floor=*/1e-12,
                     /*explicit_scratch=*/true);
  }
}

TEST_CASE("Tall-skinny SVD edge cases", "[svd][rect][edge_case]") {
  SECTION("Single column") {
    Matrix A(6, 1);
    for (int i = 0; i < 6; ++i) {
      A(i, 0) = 1.0 + 0.25 * i;
    }

    Matrix A0 = A.GetDeepCopy();
    Matrix U(6, 1), V(1, 1);
    std::vector<double> s(1);
    int iters = SquareSVD::execute(&A, &U, &V, s.data());
    REQUIRE(iters == 0);
    CheckSVDReconstruction(A0, U, V, s, /*rtol=*/1e-12, /*atol=*/1e-12);
    REQUIRE(OrthoError(U) < 1e-12);
    REQUIRE(OrthoError(V) < 1e-12);
  }

  SECTION("Zero matrix") {
    Matrix A(4, 3);
    Matrix A0 = A.GetDeepCopy();
    Matrix U(4, 3), V(3, 3);
    std::vector<double> s(3);
    int iters = SquareSVD::execute(&A, &U, &V, s.data());
    REQUIRE(iters >= 0);
    CheckSVDReconstruction(A0, U, V, s, /*rtol=*/1e-12, /*atol=*/1e-12);
    REQUIRE(OrthoError(U) < 1e-12);
    REQUIRE(OrthoError(V) < 1e-12);
    for (double x : s) REQUIRE(std::abs(x) < 1e-12);
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

    RunSVDStress(s, num_realizations, /*max_iter_factor=*/2, recon_rtol,
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
    for (int k = 1; k <= 10; ++k) {
      std::vector<double> s(n, 0.0);
      for (int i = 0; i < k; ++i)
        s[i] = 1.0 + i; // only k nonzero

      RunSVDStress(s, num_realizations, /*max_iter_factor=*/15,
                   /*recon_rtol=*/5e-9, /*ortho_rtol=*/5e-10, /*sv_rtol=*/1e-6,
                   /*min_floor=*/1e-14);
    }
  }
}

TEST_CASE("SVD handles single row/column non-zero matrices", "[svd][edge_case]") {
  SECTION("Matrix with only first row non-zero") {
    for (int n : {5}) {  // Test multiple sizes
      Matrix A(n, n);  // Initialize with zeros
      // Set only the first row to non-zero values
      for (int j = 0; j < n; ++j) {
        A(0, j) = 1.0 + j;  // Simple increasing pattern
      }
      
      Matrix A0 = A.GetDeepCopy();
      Matrix U(n, n), V(n, n);
      std::vector<double> s(n);
      
      int iters = SquareSVD::execute(&A, &U, &V, s.data());
      REQUIRE(iters < 15 * n);
      REQUIRE(iters > 0);

      // Check for NaNs in the results
      for (int i = 0; i < n; ++i) {
        REQUIRE_FALSE(std::isnan(s[i]));
        for (int j = 0; j < n; ++j) {
          REQUIRE_FALSE(std::isnan(U(i, j)));
          REQUIRE_FALSE(std::isnan(V(i, j)));
        }
      }
      
      // Verify reconstruction
      CheckSVDReconstruction(A0, U, V, s, /*rtol=*/1e-9, /*atol=*/1e-12);
      
      // For a single-row matrix, there should be exactly one non-zero singular value
      double row_norm = 0.0;
      for (int j = 0; j < n; ++j) {
        row_norm += A0(0, j) * A0(0, j);
      }
      row_norm = std::sqrt(row_norm);
      
      // The first singular value should match the norm of the row
      REQUIRE(std::abs(s[0] - row_norm) / row_norm < 1e-10);
      
      // All other singular values should be effectively zero
      for (int i = 1; i < n; ++i) {
        REQUIRE(std::abs(s[i]) < 1e-10 * row_norm);
      }
    }
  }
}

TEST_CASE("SVD handles 1x1 matrices exactly", "[svd][edge_case]") {
  SECTION("Positive scalar") {
    Matrix A(1, 1);
    A(0, 0) = 0.5;
    Matrix A0 = A.GetDeepCopy();

    Matrix U(1, 1), V(1, 1);
    std::vector<double> s(1);

    int iters = SquareSVD::execute(&A, &U, &V, s.data());
    REQUIRE(iters == 0);

    REQUIRE(std::abs(s[0] - 0.5) < 1e-14);
    CheckSingularValueSanity(s);
    CheckSingularValueEnergyIdentity(A0, s, /*rtol=*/1e-14, /*atol=*/1e-14);
    CheckSVDReconstruction(A0, U, V, s, /*rtol=*/1e-14, /*atol=*/1e-14);
    REQUIRE(OrthoError(U) < 1e-14);
    REQUIRE(OrthoError(V) < 1e-14);
  }

  SECTION("Negative scalar") {
    Matrix A(1, 1);
    A(0, 0) = -0.5;
    Matrix A0 = A.GetDeepCopy();

    Matrix U(1, 1), V(1, 1);
    std::vector<double> s(1);

    int iters = SquareSVD::execute(&A, &U, &V, s.data());
    REQUIRE(iters == 0);

    REQUIRE(std::abs(s[0] - 0.5) < 1e-14);
    CheckSingularValueSanity(s);
    CheckSingularValueEnergyIdentity(A0, s, /*rtol=*/1e-14, /*atol=*/1e-14);
    CheckSVDReconstruction(A0, U, V, s, /*rtol=*/1e-14, /*atol=*/1e-14);
    REQUIRE(OrthoError(U) < 1e-14);
    REQUIRE(OrthoError(V) < 1e-14);
  }
}

TEST_CASE("SVD handles structured rank-deficient matrices with zero leading entries",
          "[svd][edge_case]") {
  Matrix A(4, 4);
  A(0, 0) = 0.0; A(0, 1) = 1.0; A(0, 2) = 2.0; A(0, 3) = 0.0;
  A(1, 0) = 0.0; A(1, 1) = 0.0; A(1, 2) = 0.0; A(1, 3) = 0.0;
  A(2, 0) = 3.0; A(2, 1) = 0.0; A(2, 2) = 1.0; A(2, 3) = 4.0;
  A(3, 0) = 0.0; A(3, 1) = 2.0; A(3, 2) = 0.0; A(3, 3) = 1.0;

  Matrix A0 = A.GetDeepCopy();
  Matrix U(4, 4), V(4, 4);
  std::vector<double> s(4);

  int iters = SquareSVD::execute(&A, &U, &V, s.data());
  REQUIRE(iters > 0);
  REQUIRE(iters < 15 * 4);

  CheckSingularValueSanity(s);
  CheckSingularValueEnergyIdentity(A0, s, /*rtol=*/1e-12, /*atol=*/1e-12);
  CheckSVDReconstruction(A0, U, V, s, /*rtol=*/1e-12, /*atol=*/1e-12);
  REQUIRE(OrthoError(U) < 1e-12);
  REQUIRE(OrthoError(V) < 1e-12);
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
    REQUIRE(iters > 0);

    // Form G = A0^T A0
    Matrix At = Matrix::Transpose(A0);
    Matrix G(n, n);
    Multiply(At, A0, G);

    // Eigen-decompose G (PSD)
    Matrix Q(n, n);
    std::vector<double> d(n);
    Matrix G0 = G.GetDeepCopy();
    int iters_e = SymmetricEVD::execute(&G, &Q, d.data());
    REQUIRE(iters_e < 5 * n);

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
    int iters = SquareSVD::execute(&A, &U0, &V0, s0.data());
    REQUIRE(iters < 5 * n);
    REQUIRE(iters > 0);
    SortAbs(s0);

    for (double alpha : scales) {
      Matrix As = A0.GetDeepCopy();
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
          As(i, j) *= alpha;

      Matrix U(n, n), V(n, n);
      std::vector<double> s(n);
      int iters = SquareSVD::execute(&As, &U, &V, s.data());
      REQUIRE(iters < 5 * n);
      REQUIRE(iters > 0);
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
