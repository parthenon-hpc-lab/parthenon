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

#ifndef BATCHED_LINEAR_ALGEBRA_QR_SOLVE_HPP_
#define BATCHED_LINEAR_ALGEBRA_QR_SOLVE_HPP_

// This file was made in part with generative AI.

#include <algorithm>
#include <vector>

#include "batched_linear_algebra/execution_utils.hpp"
#include "batched_linear_algebra/householder.hpp"
#include "batched_linear_algebra/matrix_utils.hpp"
#include "batched_linear_algebra/qr_decomposition.hpp"
#include "kokkos_abstraction.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {
namespace batched_linear_algebra {

class TriangularSolve {
 public:
  /// Solve the upper triangular system R X = B in place by back substitution.
  ///
  /// On entry:
  ///   - R is an m×n matrix with m >= n. Only the upper triangle of its leading
  ///     n×n block is read, so the entries below the diagonal may hold anything.
  ///   - pB must be non-null and point to a matrix with at least n rows.
  ///
  /// On exit:
  ///   - The leading n rows of *pB are overwritten with X. Any rows below n are
  ///     left untouched.
  ///
  /// Notes:
  ///   - The diagonal of R is not checked for zeros. A singular R produces
  ///     inf/NaN in X.
  ///   - No workspace is needed.
  ///   - With a team handle, R and B must be complete before the call (add a
  ///     team barrier after whatever wrote them); the routine does not start
  ///     with one.
  ///
  /// Returns:
  ///   - 0.
  template <class tm_t, class matrix_r_t, class matrix_b_t>
  KOKKOS_INLINE_FUNCTION static int execute(tm_t tm, const matrix_r_t &R,
                                            matrix_b_t *pB) {
    PARTHENON_REQUIRE(pB, "B must not be null.");
    auto &B = *pB;
    const int n = GetNcols(R);
    const int nrhs = GetNcols(B);
    PARTHENON_REQUIRE(GetNrows(R) >= n, "TriangularSolve requires nrows >= ncols.");
    PARTHENON_REQUIRE(GetNrows(B) >= n, "B must have at least as many rows as R has columns.");

    // Column-oriented: once X(i,:) is known, eliminate it from every row above
    // at once, which exposes i * nrhs independent updates per step
    sequential_loop(0, n - 1, [&](const int inv_i) {
      const int i = n - 1 - inv_i;
      const double rii = R(i, i);
      parallel_loop(tm, 0, nrhs - 1, [&](const int c) { B(i, c) /= rii; });
      barrier(tm);
      parallel_loop(tm, 0, i - 1, 0, nrhs - 1,
                    [&](const int r, const int c) { B(r, c) -= R(r, i) * B(i, c); });
      barrier(tm);
    });
    return 0;
  }

  template <class matrix_r_t, class matrix_b_t>
  KOKKOS_INLINE_FUNCTION static int execute(const matrix_r_t &R, matrix_b_t *pB) {
    return execute(serial_tm_t(), R, pB);
  }
};

class QRSolve {
 public:
  /// Solve A X = B for a real square or tall matrix A using Householder QR. For
  /// a tall A this gives the least-squares solution minimizing ||A X - B||_F.
  ///
  /// Each Householder reflector is applied to B as soon as it is built, so B is
  /// overwritten with Qᵀ B without ever forming Q, and the system R X = (Qᵀ B)
  /// is then solved by back substitution.
  ///
  /// On entry:
  ///   - pA must be non-null and point to an m×n matrix with m >= n.
  ///   - pB must be non-null and point to an m×k matrix of right-hand sides.
  ///
  /// On exit:
  ///   - The upper triangle of *pA holds the R factor. The entries below the
  ///     diagonal are unspecified.
  ///   - The first n rows of *pB hold the solution X.
  ///   - Rows n to m-1 of *pB hold the trailing rows of Qᵀ B, so the norm of
  ///     column c of that block is the least-squares residual ||A X(:,c) - B(:,c)||.
  ///
  /// Notes:
  ///   - A is assumed to have full column rank. This is not checked, and a
  ///     rank-deficient A produces inf/NaN in X.
  ///
  /// Returns:
  ///   - 0.
  template <class tm_t, class matrix_a_t, class matrix_b_t>
  KOKKOS_INLINE_FUNCTION static int execute(tm_t tm, matrix_a_t *pA, matrix_b_t *pB,
                                            double *scratch) {
    PARTHENON_REQUIRE(pA, "A must not be null.");
    PARTHENON_REQUIRE(pB, "B must not be null.");
    auto &A = *pA;
    auto &B = *pB;
    const int nrows = GetNrows(A);
    const int ncols = GetNcols(A);
    PARTHENON_REQUIRE(nrows >= ncols, "QRSolve requires nrows >= ncols.");
    PARTHENON_REQUIRE(GetNrows(B) == nrows, "B must have the same number of rows as A.");

    double *v = &(scratch[0]);
    // Shared by the A and B updates, so it is sized for the wider of the two
    double *s = &(scratch[nrows]);

    sequential_loop(0, ncols - 1, [&](const int col) {
      build_householder_vector_col(tm, col, col, A, v);
      barrier(tm);

      apply_left_householder_transformation(tm, v, s, A, col, col);
      barrier(tm);

      apply_left_householder_transformation(tm, v, s, B, col, 0);
      barrier(tm);
    });

    return TriangularSolve::execute(tm, A, pB);
  }

  template <class matrix_a_t, class matrix_b_t>
  KOKKOS_INLINE_FUNCTION static int execute(matrix_a_t *pA, matrix_b_t *pB,
                                            double *scratch) {
    return execute(serial_tm_t(), pA, pB, scratch);
  }

  template <class matrix_a_t, class matrix_b_t>
  static int execute(matrix_a_t *pA, matrix_b_t *pB) {
    std::vector<double> scratch(
        double_scratch_size(GetNrows(*pA), GetNcols(*pA), GetNcols(*pB)));
    return execute(serial_tm_t(), pA, pB, scratch.data());
  }

  // nrows and ncols are the dimensions of A, nrhs the number of columns of B.
  KOKKOS_INLINE_FUNCTION static constexpr std::size_t
  double_scratch_size(std::size_t nrows, std::size_t ncols, std::size_t nrhs) {
    // v + per-column reflector coefficients
    return nrows + std::max(ncols, nrhs);
  }

  static std::size_t total_shmem_scratch_size(std::size_t nrows, std::size_t ncols,
                                              std::size_t nrhs) {
    return parthenon::ScratchPad1D<double>::shmem_size(
        double_scratch_size(nrows, ncols, nrhs));
  }
};

class QRSolveRight {
 public:
  /// Solve X A = B for a real square or wide matrix A by transposing the problem
  /// to Aᵀ Xᵀ = Bᵀ and reusing QRSolve. For a wide A this gives the
  /// least-squares solution minimizing ||X A - B||_F.
  ///
  /// On entry:
  ///   - pA must be non-null and point to an n×m matrix with n <= m.
  ///   - pB must be non-null and point to a k×m matrix of right-hand sides.
  ///
  /// On exit:
  ///   - The lower triangle of *pA holds the L factor of A = L Q. The entries
  ///     above the diagonal are unspecified.
  ///   - The first n columns of *pB hold the solution X.
  ///   - Columns n to m-1 of *pB hold the corresponding least-squares residual
  ///     information, as described for QRSolve.
  ///
  /// Returns:
  ///   - 0.
  template <class tm_t, class matrix_a_t, class matrix_b_t>
  KOKKOS_INLINE_FUNCTION static int execute(tm_t tm, matrix_a_t *pA, matrix_b_t *pB,
                                            double *scratch) {
    PARTHENON_REQUIRE(pA, "A must not be null.");
    PARTHENON_REQUIRE(pB, "B must not be null.");
    matrix_transpose_view_t<matrix_a_t> AT(*pA);
    matrix_transpose_view_t<matrix_b_t> BT(*pB);
    return QRSolve::execute(tm, &AT, &BT, scratch);
  }

  template <class matrix_a_t, class matrix_b_t>
  KOKKOS_INLINE_FUNCTION static int execute(matrix_a_t *pA, matrix_b_t *pB,
                                            double *scratch) {
    return execute(serial_tm_t(), pA, pB, scratch);
  }

  template <class matrix_a_t, class matrix_b_t>
  static int execute(matrix_a_t *pA, matrix_b_t *pB) {
    std::vector<double> scratch(
        double_scratch_size(GetNrows(*pA), GetNcols(*pA), GetNrows(*pB)));
    return execute(serial_tm_t(), pA, pB, scratch.data());
  }

  // nrows and ncols are the dimensions of A, nrhs the number of rows of B.
  KOKKOS_INLINE_FUNCTION static constexpr std::size_t
  double_scratch_size(std::size_t nrows, std::size_t ncols, std::size_t nrhs) {
    return QRSolve::double_scratch_size(ncols, nrows, nrhs);
  }

  static std::size_t total_shmem_scratch_size(std::size_t nrows, std::size_t ncols,
                                              std::size_t nrhs) {
    return QRSolve::total_shmem_scratch_size(ncols, nrows, nrhs);
  }
};

} // namespace batched_linear_algebra
} // namespace parthenon

#endif // BATCHED_LINEAR_ALGEBRA_QR_SOLVE_HPP_
