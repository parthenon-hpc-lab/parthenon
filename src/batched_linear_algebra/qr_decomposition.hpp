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

#ifndef BATCHED_LINEAR_ALGEBRA_QR_DECOMPOSITION_HPP_
#define BATCHED_LINEAR_ALGEBRA_QR_DECOMPOSITION_HPP_

// This file was made in part with generative AI.

#include <algorithm>
#include <vector>

#include "batched_linear_algebra/execution_utils.hpp"
#include "batched_linear_algebra/householder.hpp"
#include "batched_linear_algebra/matrix_utils.hpp"
#include "kokkos_abstraction.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {
namespace batched_linear_algebra {

template <class matrix_t>
struct matrix_transpose_view_t {
  KOKKOS_INLINE_FUNCTION explicit matrix_transpose_view_t(matrix_t &mat_in)
      : mat(&mat_in) {}

  KOKKOS_INLINE_FUNCTION decltype(auto) operator()(int r, int c) { return (*mat)(c, r); }

  KOKKOS_INLINE_FUNCTION decltype(auto) operator()(int r, int c) const {
    return (*mat)(c, r);
  }

  matrix_t *mat;
};

template <class matrix_t>
KOKKOS_FORCEINLINE_FUNCTION int GetNrows(const matrix_transpose_view_t<matrix_t> &m) {
  return GetNcols(*m.mat);
}

template <class matrix_t>
KOKKOS_FORCEINLINE_FUNCTION int GetNcols(const matrix_transpose_view_t<matrix_t> &m) {
  return GetNrows(*m.mat);
}

class QRDecomposition {
 public:
  /// [This documentation was generated with assistance from generative AI]
  /// Compute a Householder QR decomposition of a real tall-skinny matrix A in place.
  ///
  /// On entry:
  ///   - pA must be non-null and point to an m×n matrix with m >= n.
  ///   - pQ may be null. If non-null, it must point to either an m×m matrix
  ///     (full Q) or an m×n matrix (thin Q).
  ///
  /// On exit:
  ///   - *pA is overwritten with the upper trapezoidal R factor.
  ///   - If pQ != nullptr, *pQ is overwritten with the orthogonal Q factor.
  ///   - The factorization satisfies A_original = Q R.
  ///
  /// Notes:
  ///   - tm is an execution context handle (templated) so this routine can be
  ///     invoked from Kokkos kernels or from host code.
  ///   - The input Q is initialized to the identity before the Householder
  ///     transformations are applied.
  ///
  /// Returns:
  ///   - 0 on success.

  template <class tm_t, class matrix_a_t, class matrix_q_t>
  KOKKOS_INLINE_FUNCTION static int execute(tm_t tm, matrix_a_t *pA, matrix_q_t *pQ,
                                            double *scratch) {
    PARTHENON_REQUIRE(pA, "A must not be null.");
    auto &A = *pA;
    const int nrows = GetNrows(A);
    const int ncols = GetNcols(A);
    PARTHENON_REQUIRE(nrows >= ncols, "QRDecomposition requires nrows >= ncols.");

    const int max_dim = std::max(nrows, ncols);
    double *v = &(scratch[0]);
    double *s = &(scratch[max_dim]);
    // Store Householder heads here; tails live in the lower triangle of A.
    double *vhead = &(scratch[2 * max_dim]);

    if (pQ) {
      auto &Q = *pQ;
      const int qrows = GetNrows(Q);
      const int qcols = GetNcols(Q);
      PARTHENON_REQUIRE(qrows == nrows, "Q must have the same number of rows as A.");
      PARTHENON_REQUIRE(qcols == nrows || qcols == ncols,
                        "Q must be either full (m x m) or thin (m x n).");
    }

    sequential_loop(0, ncols - 1, [&](const int col) {
      build_householder_vector_col(tm, col, col, A, v);
      barrier(tm);

      apply_left_householder_transformation(tm, v, s, A, col, col);
      barrier(tm);

      if (pQ) {
        once_per_team(tm, [&]() { vhead[col] = v[col]; });
        parallel_loop(tm, col + 1, nrows - 1, [&](int r) { A(r, col) = v[r]; });
      } else {
        // The reflector leaves O(eps ||x||) roundoff below the diagonal; these
        // entries are zero in exact arithmetic, so set them explicitly.
        parallel_loop(tm, col + 1, nrows - 1, [&](int r) { A(r, col) = 0.0; });
      }
    });

    if (pQ) {
      auto &Q = *pQ;
      if (GetNcols(Q) == nrows) {
        parallel_loop(tm, 0, nrows - 1, 0, nrows - 1,
                      [&](int r, int c) { Q(r, c) = (r == c); });
      } else {
        parallel_loop(tm, 0, nrows - 1, 0, ncols - 1,
                      [&](int r, int c) { Q(r, c) = (r == c); });
      }
      barrier(tm);

      sequential_loop(0, ncols - 1, [&](const int inv_col) {
        const int col = ncols - 1 - inv_col;
        parallel_loop(tm, 0, col - 1, [&](int r) { v[r] = 0.0; });
        once_per_team(tm, [&]() { v[col] = vhead[col]; });
        parallel_loop(tm, col + 1, nrows - 1, [&](int r) {
          v[r] = A(r, col);
          A(r, col) = 0.0;
        });
        barrier(tm);
        apply_left_householder_transformation(tm, v, s, Q, col, 0);
        barrier(tm);
      });
    }

    return 0;
  }

  template <class tm_t, class matrix_t>
  KOKKOS_INLINE_FUNCTION static int execute(tm_t tm, matrix_t *pA, double *scratch) {
    matrix_t *pQ = nullptr;
    return execute(tm, pA, pQ, scratch);
  }

  template <class matrix_a_t, class matrix_q_t>
  KOKKOS_INLINE_FUNCTION static int execute(matrix_a_t *pA, matrix_q_t *pQ,
                                            double *scratch) {
    return execute(serial_tm_t(), pA, pQ, scratch);
  }

  template <class matrix_t>
  KOKKOS_INLINE_FUNCTION static int execute(matrix_t *pA, double *scratch) {
    matrix_t *pQ = nullptr;
    return execute(serial_tm_t(), pA, pQ, scratch);
  }

  template <class matrix_t>
  static int execute(matrix_t *pA, matrix_t *pQ) {
    const int nrows = GetNrows(*pA);
    const int ncols = GetNcols(*pA);
    std::vector<double> scratch(double_scratch_size(nrows, ncols));
    return execute(serial_tm_t(), pA, pQ, scratch.data());
  }

  template <class matrix_t>
  static int execute(matrix_t *pA) {
    const int nrows = GetNrows(*pA);
    const int ncols = GetNcols(*pA);
    std::vector<double> scratch(double_scratch_size(nrows, ncols));
    matrix_t *pQ = nullptr;
    return execute(serial_tm_t(), pA, pQ, scratch.data());
  }

  KOKKOS_INLINE_FUNCTION static constexpr std::size_t
  double_scratch_size(std::size_t nrows, std::size_t ncols) {
    // v + scratch scalars + Householder heads
    return 2 * std::max(nrows, ncols) + ncols;
  }

  KOKKOS_INLINE_FUNCTION static constexpr std::size_t
  double_scratch_size(std::size_t nrows) {
    return double_scratch_size(nrows, nrows);
  }

  static std::size_t total_shmem_scratch_size(std::size_t nrows, std::size_t ncols) {
    return parthenon::ScratchPad1D<double>::shmem_size(double_scratch_size(nrows, ncols));
  }

  static std::size_t total_shmem_scratch_size(std::size_t nrows) {
    return total_shmem_scratch_size(nrows, nrows);
  }

  /// Solve a square linear system A x = b using a previously computed
  /// QR factorization A = Q R.
  ///
  /// The solution is obtained by first computing x = Q^T b, followed
  /// by back substitution to solve R x = Q^T b. The factorization
  /// matrices Q and R are not modified and may be reused for multiple
  /// right-hand sides without repeating the QR decomposition.
  ///
  /// On entry:
  ///   - R must be an n x n upper triangular matrix produced by
  ///     QRDecomposition::execute().
  ///   - Q must be the corresponding n x n orthogonal matrix.
  ///   - b must contain the n-element right-hand-side vector.
  ///   - x must provide storage for the n-element solution vector.
  ///   - b and x must not alias.
  ///
  /// On exit:
  ///   - x contains the solution if the routine succeeds.
  ///   - Q, R, and b are unchanged.
  ///   - If the solve fails, x may contain a partial result.
  ///
  /// Numerical singularity is detected when a diagonal entry of R
  /// satisfies |R(i,i)| <= rtol * max_j |R(j,j)|.
  ///
  /// The routine performs no dynamic allocations and is callable
  /// from host code or Kokkos device kernels.
  ///
  /// Returns:
  ///   - 0 on success.
  ///   - 1 if R is singular or numerically singular.
  template <class matrix_r_t, class matrix_q_t, class vector_b_t, class vector_x_t>
  KOKKOS_INLINE_FUNCTION static int solve(const matrix_r_t &R, const matrix_q_t &Q,
                                          const vector_b_t &b, vector_x_t &x,
                                          const double rtol = 1e-14) {
    const int n = GetNrows(R);

    PARTHENON_REQUIRE(GetNcols(R) == n, "QR solve requires square R.");
    PARTHENON_REQUIRE(GetNrows(Q) == n && GetNcols(Q) == n,
                      "QR solve requires square Q.");

    // Find a scale for detecting a numerically singular R.
    double rscale = 0.0;
    for (int i = 0; i < n; ++i) {
      const double rii = R(i, i);
      const double arii = (rii >= 0.0) ? rii : -rii;
      rscale = std::max(rscale, arii);
    }

    if (rscale == 0.0) return 1;

    // x <- Q^T b
    //
    // We use x itself as temporary storage for Q^T b, so b and x
    // should not alias.
    for (int i = 0; i < n; ++i) {
      double sum = 0.0;

      for (int j = 0; j < n; ++j) {
        sum += Q(j, i) * b[j];
      }

      x[i] = sum;
    }

    // Back substitution:
    //
    //   R x = Q^T b
    //
    for (int i = n - 1; i >= 0; --i) {
      double rhs = x[i];

      for (int j = i + 1; j < n; ++j) {
        rhs -= R(i, j) * x[j];
      }

      const double rii = R(i, i);
      const double arii = (rii >= 0.0) ? rii : -rii;

      if (arii <= rtol * rscale) return 1;

      x[i] = rhs / rii;
    }

    return 0;
  }
};

class LQDecomposition {
 public:
  /// [This documentation was generated with assistance from generative AI]
  /// Compute an LQ decomposition of a real wide matrix A in place by
  /// transposing the problem and reusing QRDecomposition.
  ///
  /// On entry:
  ///   - pA must be non-null and point to an m×n matrix with m <= n.
  ///   - pQ may be null. If non-null, it may point to an m×n matrix (thin Q)
  ///     or an n×n matrix (full Q).
  ///
  /// On exit:
  ///   - *pA is overwritten with the L factor, stored in the same shape as A.
  ///   - If pQ != nullptr, *pQ is overwritten with the orthogonal Q factor.
  ///   - The factorization satisfies A_original = L Q.

  template <class tm_t, class matrix_a_t, class matrix_q_t>
  KOKKOS_INLINE_FUNCTION static int execute(tm_t tm, matrix_a_t *pA, matrix_q_t *pQ,
                                            double *scratch) {
    PARTHENON_REQUIRE(pA, "A must not be null.");
    auto &A = *pA;
    const int nrows = GetNrows(A);
    const int ncols = GetNcols(A);
    PARTHENON_REQUIRE(nrows <= ncols, "LQDecomposition requires nrows <= ncols.");

    matrix_transpose_view_t<matrix_a_t> AT(A);
    if (pQ) {
      matrix_transpose_view_t<matrix_q_t> QT(*pQ);
      return QRDecomposition::execute(tm, &AT, &QT, scratch);
    }
    return QRDecomposition::execute(tm, &AT, scratch);
  }

  template <class tm_t, class matrix_t>
  KOKKOS_INLINE_FUNCTION static int execute(tm_t tm, matrix_t *pA, double *scratch) {
    matrix_t *pQ = nullptr;
    return execute(tm, pA, pQ, scratch);
  }

  template <class matrix_a_t, class matrix_q_t>
  KOKKOS_INLINE_FUNCTION static int execute(matrix_a_t *pA, matrix_q_t *pQ,
                                            double *scratch) {
    return execute(serial_tm_t(), pA, pQ, scratch);
  }

  template <class matrix_t>
  KOKKOS_INLINE_FUNCTION static int execute(matrix_t *pA, double *scratch) {
    matrix_t *pQ = nullptr;
    return execute(serial_tm_t(), pA, pQ, scratch);
  }

  template <class matrix_a_t, class matrix_q_t>
  static int execute(matrix_a_t *pA, matrix_q_t *pQ) {
    const int nrows = GetNrows(*pA);
    const int ncols = GetNcols(*pA);
    std::vector<double> scratch(QRDecomposition::double_scratch_size(ncols, nrows));
    return execute(serial_tm_t(), pA, pQ, scratch.data());
  }

  template <class matrix_t>
  static int execute(matrix_t *pA) {
    const int nrows = GetNrows(*pA);
    const int ncols = GetNcols(*pA);
    std::vector<double> scratch(QRDecomposition::double_scratch_size(ncols, nrows));
    matrix_t *pQ = nullptr;
    return execute(serial_tm_t(), pA, pQ, scratch.data());
  }

  KOKKOS_INLINE_FUNCTION static constexpr std::size_t
  double_scratch_size(std::size_t nrows, std::size_t ncols) {
    return QRDecomposition::double_scratch_size(ncols, nrows);
  }

  KOKKOS_INLINE_FUNCTION static constexpr std::size_t
  double_scratch_size(std::size_t nrows) {
    return double_scratch_size(nrows, nrows);
  }

  static std::size_t total_shmem_scratch_size(std::size_t nrows, std::size_t ncols) {
    return QRDecomposition::total_shmem_scratch_size(ncols, nrows);
  }

  static std::size_t total_shmem_scratch_size(std::size_t nrows) {
    return total_shmem_scratch_size(nrows, nrows);
  }
};

} // namespace batched_linear_algebra
} // namespace parthenon

#endif // BATCHED_LINEAR_ALGEBRA_QR_DECOMPOSITION_HPP_
