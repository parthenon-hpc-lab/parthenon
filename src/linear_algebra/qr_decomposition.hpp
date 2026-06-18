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

class QRDecomposition {
 public:
  /// [This documentation was generated with assistance from generative AI]
  /// Compute a Householder QR decomposition of a real tall-skinny matrix A in place.
  ///
  /// On entry:
  ///   - pA must be non-null and point to an m×n matrix with m >= n.
  ///   - pQ may be null. If non-null, it must point to an m×m matrix.
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

    if (pQ) {
      auto &Q = *pQ;
      PARTHENON_REQUIRE(GetNrows(Q) == nrows,
                        "Q must have the same number of rows as A.");
      PARTHENON_REQUIRE(GetNcols(Q) == nrows, "Q must be square with size A.nrows().");

      // Start from identity so the right-applied Householder reflectors build Q.
      parallel_loop(tm, 0, nrows - 1, 0, nrows - 1,
                    [&](int r, int c) { Q(r, c) = (r == c); });
      barrier(tm);
    }

    sequential_loop(0, ncols - 1, [&](const int col) {
      build_householder_vector_col(tm, col, col, A, v);
      barrier(tm);

      apply_left_householder_transformation(tm, v, s, A, col, col);
      barrier(tm);

      if (pQ) {
        apply_right_householder_transformation(tm, v, s, *pQ, col);
        barrier(tm);
      }
    });

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
    return 2 * std::max(nrows, ncols);
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

template <class matrix_t>
KOKKOS_FORCEINLINE_FUNCTION int QRDecomposition(matrix_t &A, matrix_t &Q) {
  return ::QRDecomposition::execute(&A, &Q);
}

#endif // QR_DECOMPOSITION_HPP
