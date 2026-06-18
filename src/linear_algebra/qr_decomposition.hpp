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

#ifndef QR_DECOMPOSITION_HPP
#define QR_DECOMPOSITION_HPP

#include "parthenon/parthenon.hpp"

#include "householder.hpp"
#include "matrix.hpp"

template <class matrix_t>
struct matrix_transpose_view_t {
  KOKKOS_INLINE_FUNCTION explicit matrix_transpose_view_t(matrix_t &mat_in)
      : mat(&mat_in) {}

  KOKKOS_INLINE_FUNCTION decltype(auto) operator()(int r, int c) {
    return (*mat)(c, r);
  }

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

  template <class tm_t, class matrix_t>
  KOKKOS_INLINE_FUNCTION static int execute(tm_t tm, matrix_t *pA, matrix_t *pQ,
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
      PARTHENON_REQUIRE(qrows == nrows,
                        "Q must have the same number of rows as A.");
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
      });
    }

    return 0;
  }

  template <class tm_t, class matrix_t>
  KOKKOS_INLINE_FUNCTION static int execute(tm_t tm, matrix_t *pA, double *scratch) {
    matrix_t *pQ = nullptr;
    return execute(tm, pA, pQ, scratch);
  }

  template <class matrix_t>
  KOKKOS_INLINE_FUNCTION static int execute(matrix_t *pA, matrix_t *pQ,
                                            double *scratch) {
    return execute(serial_tm_t(), pA, pQ, scratch);
  }

  template <class matrix_t>
  KOKKOS_INLINE_FUNCTION static int execute(matrix_t *pA, double *scratch) {
    matrix_t *pQ = nullptr;
    return execute(serial_tm_t(), pA, pQ, scratch);
  }

  template <class matrix_t>
  KOKKOS_INLINE_FUNCTION static int execute(matrix_t *pA, matrix_t *pQ) {
    const int nrows = GetNrows(*pA);
    const int ncols = GetNcols(*pA);
    std::vector<double> scratch(double_scratch_size(nrows, ncols));
    return execute(serial_tm_t(), pA, pQ, scratch.data());
  }

  template <class matrix_t>
  KOKKOS_INLINE_FUNCTION static int execute(matrix_t *pA) {
    const int nrows = GetNrows(*pA);
    const int ncols = GetNcols(*pA);
    std::vector<double> scratch(double_scratch_size(nrows, ncols));
    matrix_t *pQ = nullptr;
    return execute(serial_tm_t(), pA, pQ, scratch.data());
  }

  KOKKOS_INLINE_FUNCTION static std::size_t double_scratch_size(std::size_t nrows,
                                                                std::size_t ncols) {
    // v + scratch scalars + Householder heads
    return 2 * std::max(nrows, ncols) + ncols;
  }

  KOKKOS_INLINE_FUNCTION static std::size_t double_scratch_size(std::size_t nrows) {
    return double_scratch_size(nrows, nrows);
  }

  static std::size_t total_shmem_scratch_size(std::size_t nrows, std::size_t ncols) {
    return parthenon::ScratchPad1D<double>::shmem_size(double_scratch_size(nrows, ncols));
  }

  static std::size_t total_shmem_scratch_size(std::size_t nrows) {
    return total_shmem_scratch_size(nrows, nrows);
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

  template <class tm_t, class matrix_t>
  KOKKOS_INLINE_FUNCTION static int execute(tm_t tm, matrix_t *pA, matrix_t *pQ,
                                            double *scratch) {
    PARTHENON_REQUIRE(pA, "A must not be null.");
    auto &A = *pA;
    const int nrows = GetNrows(A);
    const int ncols = GetNcols(A);
    PARTHENON_REQUIRE(nrows <= ncols, "LQDecomposition requires nrows <= ncols.");

    matrix_transpose_view_t<matrix_t> AT(A);
    if (pQ) {
      matrix_transpose_view_t<matrix_t> QT(*pQ);
      return QRDecomposition::execute(tm, &AT, &QT, scratch);
    }
    return QRDecomposition::execute(tm, &AT, scratch);
  }

  template <class tm_t, class matrix_t>
  KOKKOS_INLINE_FUNCTION static int execute(tm_t tm, matrix_t *pA, double *scratch) {
    matrix_t *pQ = nullptr;
    return execute(tm, pA, pQ, scratch);
  }

  template <class matrix_t>
  KOKKOS_INLINE_FUNCTION static int execute(matrix_t *pA, matrix_t *pQ,
                                            double *scratch) {
    return execute(serial_tm_t(), pA, pQ, scratch);
  }

  template <class matrix_t>
  KOKKOS_INLINE_FUNCTION static int execute(matrix_t *pA, double *scratch) {
    matrix_t *pQ = nullptr;
    return execute(serial_tm_t(), pA, pQ, scratch);
  }

  template <class matrix_t>
  KOKKOS_INLINE_FUNCTION static int execute(matrix_t *pA, matrix_t *pQ) {
    const int nrows = GetNrows(*pA);
    const int ncols = GetNcols(*pA);
    std::vector<double> scratch(QRDecomposition::double_scratch_size(ncols, nrows));
    return execute(serial_tm_t(), pA, pQ, scratch.data());
  }

  template <class matrix_t>
  KOKKOS_INLINE_FUNCTION static int execute(matrix_t *pA) {
    const int nrows = GetNrows(*pA);
    const int ncols = GetNcols(*pA);
    std::vector<double> scratch(QRDecomposition::double_scratch_size(ncols, nrows));
    matrix_t *pQ = nullptr;
    return execute(serial_tm_t(), pA, pQ, scratch.data());
  }

  KOKKOS_INLINE_FUNCTION static std::size_t double_scratch_size(std::size_t nrows,
                                                                std::size_t ncols) {
    return QRDecomposition::double_scratch_size(ncols, nrows);
  }

  KOKKOS_INLINE_FUNCTION static std::size_t double_scratch_size(std::size_t nrows) {
    return double_scratch_size(nrows, nrows);
  }

  static std::size_t total_shmem_scratch_size(std::size_t nrows, std::size_t ncols) {
    return QRDecomposition::total_shmem_scratch_size(ncols, nrows);
  }

  static std::size_t total_shmem_scratch_size(std::size_t nrows) {
    return total_shmem_scratch_size(nrows, nrows);
  }
};

template <class matrix_t>
KOKKOS_FORCEINLINE_FUNCTION int QRDecomposition(matrix_t &A, matrix_t &Q) {
  return ::QRDecomposition::execute(&A, &Q);
}

template <class matrix_t>
KOKKOS_FORCEINLINE_FUNCTION int LQDecomposition(matrix_t &A, matrix_t &Q) {
  return ::LQDecomposition::execute(&A, &Q);
}

#endif // QR_DECOMPOSITION_HPP
