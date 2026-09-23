#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

// #include <catch2/catch_test_macros.hpp>
#include <catch2/catch.hpp>

#include "linear_algebra/matrix.hpp"
#include "linear_algebra/symmetric_evd.hpp"

static double FrobeniusNorm(const Matrix &A) {
  double s = 0.0;
  for (int r = 0; r < A.nrows(); ++r)
    for (int c = 0; c < A.ncols(); ++c)
      s += A(r, c) * A(r, c);
  return std::sqrt(s);
}

static double Vector2Norm(const std::vector<double> &v) {
  double s = 0.0;
  for (double x : v)
    s += x * x;
  return std::sqrt(s);
}

// Extract column j of Q as a std::vector<double>
static std::vector<double> GetColumn(const Matrix &Q, int j) {
  std::vector<double> v(Q.nrows());
  for (int i = 0; i < Q.nrows(); ++i)
    v[i] = Q(i, j);
  return v;
}

// Compute residual ||A v - lambda v||_2
static double EigenResidual2Norm(const Matrix &A, const std::vector<double> &v,
                                 double lambda) {
  const int n = A.nrows();
  double s = 0.0;
  for (int i = 0; i < n; ++i) {
    double Avi = 0.0;
    for (int j = 0; j < n; ++j)
      Avi += A(i, j) * v[j];
    double ri = Avi - lambda * v[i];
    s += ri * ri;
  }
  return std::sqrt(s);
}

// Check all eigenpairs: columns of Q are eigenvectors, eigs are eigenvalues
static void CheckEigenpairsResidual(const Matrix &A_orig, const Matrix &Q,
                                    const std::vector<double> &eigs, double rtol = 1e-10,
                                    double atol = 1e-12) {
  const double An = FrobeniusNorm(A_orig);
  const double denomA = std::max(An, atol);

  for (int i = 0; i < Q.ncols(); ++i) {
    auto v = GetColumn(Q, i);
    const double vn = Vector2Norm(v);
    const double denom = denomA * std::max(vn, atol);

    const double res = EigenResidual2Norm(A_orig, v, eigs[i]);
    REQUIRE(res / denom < rtol);
    REQUIRE(std::isfinite(eigs[i]));
  }
}

// Helper: run many random rotations of a given spectrum through the pipeline
void RunSpectrumTest(const Vector &eigs_template, int num_realizations,
                     int max_iter_factor, double tol, double min_val = 1.e-10,
                     double eigvec_rtol = 1e-10) {
  std::size_t n = eigs_template.size();

  Vector eigen_values = eigs_template;
  std::sort(eigen_values.begin(), eigen_values.end());

  for (int realization = 0; realization < num_realizations; ++realization) {
    Matrix A = Matrix::FromSpectrum(eigen_values, realization + 12345);
    Matrix A0 = A.GetDeepCopy(); // keep original for residual checks

    Matrix Q(n, n);
    std::vector<double> d(n);

    int iters = SymmetricEVD::execute(&A, &Q, d.data());

    REQUIRE(iters < max_iter_factor * n);

    // Eigenvalue check (order-insensitive): sort both
    std::vector<double> d_sorted = d;
    std::sort(d_sorted.begin(), d_sorted.end());

    double scale = 1.e-3;
    double floor_val = min_val;
    for (std::size_t i = 0; i < n; ++i)
      floor_val = std::max(floor_val, scale * std::abs(eigen_values[i]));

    for (std::size_t i = 0; i < n; ++i) {
      const double diff = std::abs(d_sorted[i] - eigen_values[i]);
      const double denom = std::max(floor_val, std::abs(eigen_values[i]));
      REQUIRE(diff / denom < tol);
    }

    // Eigenvector residual check (order-sensitive to your returned pairing).
    // Assumes d[i] corresponds to column i of Q.
    CheckEigenpairsResidual(A0, Q, d, eigvec_rtol, /*atol=*/1e-12);
  }
}

TEST_CASE("ImplicitQR stress tests over spectra", "[eig][qr][tridiag]") {
  const int n = 50;
  const int num_realizations = 100;
  const double tol = 1.e-8;

  // 1) Baseline: arithmetic spectrum with a big outlier and tiny perturbations
  SECTION("Arithmetic plus perturbed tail and large outlier") {
    Vector eigs(n);
    std::iota(eigs.begin(), eigs.end(), -5); // -5, -4, ..., 44

    // Perturb the largest 8 eigenvalues with tiny shifts and sign flips
    for (int i = 1; i <= 8; ++i) {
      double sign = (i % 2 ? -1.0 : 1.0);
      eigs[eigs.size() - i] = sign * (4.0 + i * 1.e-8);
    }
    // Add one big eigenvalue
    eigs.back() = 1.e4;

    RunSpectrumTest(eigs, num_realizations, /*max_iter_factor=*/3, tol);
  }

  // 2) Strongly clustered spectrum in the middle
  SECTION("Clustered spectrum around 1.0") {
    Vector eigs(n);

    // Left cluster near -10
    for (int i = 0; i < n / 4; ++i)
      eigs[i] = -10.0 + 0.1 * i;

    // Tight cluster near 1.0
    for (int i = n / 4; i < 3 * n / 4; ++i)
      eigs[i] = 1.0 + 1.e-8 * (i - n / 4);

    // Right cluster near +10
    for (int i = 3 * n / 4; i < n; ++i)
      eigs[i] = 10.0 + 0.1 * (i - 3 * n / 4);

    RunSpectrumTest(eigs, num_realizations, /*max_iter_factor=*/5, tol);
  }

  // 3) Repeated eigenvalues (tests multiplicity / near-multiplicity)
  SECTION("Repeated eigenvalues") {
    Vector eigs(n);

    // Half at -1, half at +2
    for (int i = 0; i < n / 2; ++i)
      eigs[i] = -1.0;
    for (int i = n / 2; i < n; ++i)
      eigs[i] = 2.0;

    RunSpectrumTest(eigs, num_realizations, /*max_iter_factor=*/5, tol);
  }

  // 4) Wide dynamic range spectrum (conditioning / scaling stress)
  SECTION("Wide dynamic range") {
    Vector eigs(n);
    // Log-spaced from 1e-6 to 1e6, with alternating sign
    for (int i = 0; i < n; ++i) {
      double alpha = -6.0 + 12.0 * (static_cast<double>(i) / (n - 1));
      double val = std::pow(10.0, alpha);
      double sign = (i % 2 ? -1.0 : 1.0);
      eigs[i] = sign * val;
    }

    RunSpectrumTest(eigs, num_realizations, /*max_iter_factor=*/5, tol);
  }

  // 5) Symmetric +/- pairs with one large outlier (tests your ±λ corner case)
  SECTION("Symmetric plus/minus pairs and large outlier") {
    Vector eigs(n);

    // ± pairs
    int half = (n - 1) / 2;
    for (int k = 0; k < half; ++k) {
      double val = 1.0 + 0.1 * k;
      eigs[2 * k] = -val;
      eigs[2 * k + 1] = val;
    }
    // Put a large eigenvalue at the end
    eigs.back() = 1.e4;

    RunSpectrumTest(eigs, num_realizations, /*max_iter_factor=*/5, tol);
  }

  // 6) Nearly reducible / block-structured spectrum
  SECTION("Block-structured spectrum") {
    Vector eigs(n);

    // Block 1: around -5
    for (int i = 0; i < n / 3; ++i)
      eigs[i] = -5.0 + 0.01 * i;

    // Block 2: tight cluster near 0
    for (int i = n / 3; i < 2 * n / 3; ++i)
      eigs[i] = 1.e-6 * (i - n / 3 + 1);

    // Block 3: around +5
    for (int i = 2 * n / 3; i < n; ++i)
      eigs[i] = 5.0 + 0.01 * (i - 2 * n / 3);

    RunSpectrumTest(eigs, num_realizations, /*max_iter_factor=*/5, tol);
  }
}

TEST_CASE("Gram matrix eigenvalues: full-rank PSD", "[gram][eig]") {
  const int n = 30; // columns -> Gram size n x n
  const int m = 60; // rows (m >= n for full rank with high prob)
  const int num_realizations = 50;

  const double eps = std::numeric_limits<double>::epsilon();

  for (int r = 0; r < num_realizations; ++r) {
    // Build random X and Gram G = X^T X
    Matrix X = Matrix::RandomGaussian(m, n, /*seed=*/12345u + r);
    Matrix Xt = Matrix::Transpose(X);
    Matrix G = Matrix(n, n);

    Multiply(Xt, X, G);

    // Reference trace of G (sum of diagonal) before tridiagonalization
    double traceG = 0.0;
    for (int i = 0; i < n; ++i)
      traceG += G(i, i);

    // Perform the eigenvalue decomposition
    Matrix Q(n, n);
    std::vector<double> d(n);
    Matrix G0 = G.GetDeepCopy();
    int iters = SymmetricEVD::execute(&G, &Q, d.data());

    CheckEigenpairsResidual(G0, Q, d, /*rtol=*/1e-10, /*atol=*/1e-12);

    REQUIRE(iters < 5 * n); // loose iteration bound for safety

    // Sort eigenvalues
    std::sort(d.begin(), d.end());

    // 1) PSD check: eigenvalues should be >= 0 up to small negative noise
    // Bound: allow small negative values on order of O(n * eps * ||G||)
    double lambda_max = std::abs(d.back());
    double psd_tol = 100.0 * n * eps * std::max(1.0, lambda_max);
    REQUIRE(d.front() >= -psd_tol);

    // 2) Trace check: sum of eigenvalues ≈ trace(G)
    double sum_lambda = 0.0;
    for (double v : d)
      sum_lambda += v;

    double trace_err = std::abs(sum_lambda - traceG);
    double trace_scale = std::max(std::abs(traceG), 1.0);
    REQUIRE(trace_err / trace_scale < 1e-10);
  }
}

TEST_CASE("Gram matrix eigenvalues: nearly rank-deficient", "[gram][eig]") {
  const int n = 30;
  const int m = 40;
  const int num_realizations = 50;

  const double eps = std::numeric_limits<double>::epsilon();

  for (int r = 0; r < num_realizations; ++r) {
    // Start with random X
    Matrix X = Matrix::RandomGaussian(m, n, /*seed=*/22222u + r);

    // Make last column nearly a linear combination of the others
    // x_n ≈ sum_j alpha_j x_j, with small noise
    std::mt19937_64 rng(9000u + r);
    std::normal_distribution<double> dist(0.0, 1.0);
    std::vector<double> alpha(n - 1);
    for (int j = 0; j < n - 1; ++j)
      alpha[j] = dist(rng);

    for (int i = 0; i < m; ++i) {
      double val = 0.0;
      for (int j = 0; j < n - 1; ++j)
        val += alpha[j] * X(i, j);
      // Add a bit of noise so it's "nearly" dependent
      val += 1e-10 * dist(rng);
      X(i, n - 1) = val;
    }

    Matrix Xt = Matrix::Transpose(X);
    Matrix G = Matrix(n, n);
    Multiply(Xt, X, G);

    Matrix Q(n, n);
    std::vector<double> d(n);
    Matrix G0 = G.GetDeepCopy();
    int iters = SymmetricEVD::execute(&G, &Q, d.data());

    CheckEigenpairsResidual(G0, Q, d, /*rtol=*/1e-10, /*atol=*/1e-12);

    REQUIRE(iters < 6 * n);

    std::sort(d.begin(), d.end());
    double lambda_max = d.back();

    // PSD-ish: smallest eigenvalue should not be significantly negative
    double psd_tol = 100.0 * n * eps * std::max(1.0, std::abs(lambda_max));
    REQUIRE(d.front() >= -psd_tol);

    // Near-rank-deficient: smallest eigenvalue should be "small"
    // relative to largest
    double rel_small = std::abs(d.front()) / std::max(std::abs(lambda_max), 1.0);
    REQUIRE(rel_small < 1e-6);
  }
}

TEST_CASE("Gram matrix eigenvalues: strongly rank-deficient", "[gram][eig]") {
  const int n = 30;
  const int k = 10; // rank k < n
  const int m = 40;
  const int num_realizations = 50;

  const double eps = std::numeric_limits<double>::epsilon();

  for (int r = 0; r < num_realizations; ++r) {
    // X has only k independent columns, rest are exact copies
    Matrix X = Matrix::RandomGaussian(m, k, /*seed=*/33333u + r);
    Matrix Xfull(m, n);

    // First k columns: independent
    for (int j = 0; j < k; ++j)
      for (int i = 0; i < m; ++i)
        Xfull(i, j) = X(i, j);

    // Remaining n-k columns: copy of column 0 (exact dependence)
    for (int j = k; j < n; ++j)
      for (int i = 0; i < m; ++i)
        Xfull(i, j) = X(i, 0);

    Matrix Xt = Matrix::Transpose(Xfull);
    Matrix G = Matrix(n, n);
    Multiply(Xt, Xfull, G);

    Matrix Q(n, n);
    std::vector<double> d(n);
    Matrix G0 = G.GetDeepCopy();
    int iters = SymmetricEVD::execute(&G, &Q, d.data());

    CheckEigenpairsResidual(G0, Q, d, /*rtol=*/1e-10, /*atol=*/1e-12);

    REQUIRE(iters < 6 * n);

    std::sort(d.begin(), d.end());

    double lambda_max = d.back();
    double psd_tol = 200.0 * n * eps * std::max(1.0, std::abs(lambda_max));
    REQUIRE(d.front() >= -psd_tol);

    // Expect roughly n-k eigenvalues close to zero
    // We check that at least (n-k)/2 of the smallest are very small.
    int num_small = 0;
    for (int i = 0; i < n; ++i) {
      double rel = std::abs(d[i]) / std::max(std::abs(lambda_max), 1.0);
      if (rel < 1e-8) ++num_small;
    }
    REQUIRE(num_small >= (n - k) / 2);
  }
}
