#ifndef SYMMETRIC_EVD_HPP
#define SYMMETRIC_EVD_HPP

#include "parthenon/parthenon.hpp"

#include "householder.hpp"
#include "implicit_qr.hpp"
#include "matrix.hpp"

class SymmetricEVD {
 public:
  /// Compute the eigen-decomposition of a real symmetric matrix A in place.
  ///
  /// On entry:
  ///   - pA must be non-null and point to an n×n symmetric matrix A.
  ///   - pQ may be null. If non-null, it must point to an n×n matrix.
  ///
  /// On exit:
  ///   - *pA is overwritten (original contents are destroyed). Internally A is
  ///     reduced to tridiagonal form via Householder similarity transforms and
  ///     then diagonalized by an implicit QR iteration.
  ///   - eigs[0..n-1] contains the computed eigenvalues (not guaranteed
  ///     sorted).
  ///   - If pQ != nullptr, *pQ is overwritten and its *columns* contain the
  ///     eigenvectors of A: column i of Q corresponds to eigs[i].
  ///
  /// Mathematical summary:
  ///   - The routine computes A = Q Λ Qᵀ, where Λ = diag(eigs) and Q is
  ///     orthogonal.
  ///   - Equivalently, A Q = Q Λ.
  ///
  /// Scratch:
  ///   - scratch: double workspace used for Householder vectors and tridiagonal
  ///     off-diagonals.
  ///   - iscratch: integer workspace used for block partitioning in the QR
  ///     stage.
  ///
  /// Notes:
  ///   - tm is an execution context handle (templated) so this routine can be
  ///     invoked from Kokkos kernels (with or without hierarchical parallelism)
  ///     or from host code; synchronization/loop semantics are provided by the
  ///     tm_t implementation.
  ///
  /// Returns:
  ///   - The number of implicit QR iterations performed (hitting the internal
  ///     limit indicates non-convergence).

  template <class tm_t, class matrix_t>
  KOKKOS_INLINE_FUNCTION static int execute(tm_t tm, matrix_t *pA, matrix_t *pQ,
                                            double *eigs, double *scratch,
                                            std::size_t *iscratch) {
    PARTHENON_REQUIRE(pA, "A must not be null.");
    auto &A = *pA;
    const int ncols = GetNcols(A);
    // Tridiagonalize the symmetric matrix via Householder transformations
    double *v = &(scratch[0]);
    double *s = &(scratch[ncols]);
    if (pQ) {
      auto &Q = *pQ;
      // Set Q to the identity
      parallel_loop(tm, 0, ncols - 1, 0, ncols - 1,
                    [&](int r, int c) { Q(r, c) = (r == c); });
    }

    sequential_loop(0, ncols - 3, [&](const int col) {
      int row = col + 1;
      build_householder_vector_col(tm, row, col, A, v);
      barrier(tm);
      apply_left_householder_transformation(tm, v, s, A);
      barrier(tm);
      apply_right_householder_transformation(tm, v, s, A);
      barrier(tm);
      if (pQ) apply_right_householder_transformation(tm, v, s, *pQ);
    });

    // Move to tridiagonal storage
    barrier(tm);
    once_per_team(
        tm, KOKKOS_LAMBDA() { eigs[0] = A(0, 0); });
    parallel_loop(
        tm, 0, ncols - 2, KOKKOS_LAMBDA(int i) {
          eigs[i + 1] = A(i + 1, i + 1);
          scratch[i] = A(i, i + 1);
        });

    barrier(tm);
    std::size_t *start = &(iscratch[0]);
    std::size_t *end = &(iscratch[ncols / 2 + 1]);
    const int status = ImplicitQRTridiag(tm, eigs, scratch, pQ, start, end, ncols, 10 * ncols);
    if (status == 10 * ncols) return -status;
    return status;
  }

  template <class tm_t, class matrix_t>
  KOKKOS_INLINE_FUNCTION static int execute(tm_t tm, matrix_t *pA, double *eigs,
                                            double *scratch, std::size_t *iscratch) {
    matrix_t *pQ = nullptr;
    return execute(tm, pA, pQ, eigs, scratch, iscratch);
  }

  template <class matrix_t>
  KOKKOS_INLINE_FUNCTION static int execute(matrix_t *pA, double *eigs, double *scratch,
                                            std::size_t *iscratch) {
    matrix_t *pQ = nullptr;
    return execute(serial_tm_t(), pA, pQ, eigs, scratch, iscratch);
  }

  // Version that is only callable on host and allocates its own
  // scratch space
  template <class matrix_t>
  KOKKOS_INLINE_FUNCTION static int execute(matrix_t *pA, matrix_t *pQ, double *eigs) {
    const int ncols = GetNcols(*pA);
    std::vector<double> scratch(double_scratch_size(ncols));
    std::vector<std::size_t> iscratch(sizet_scratch_size(ncols));
    return execute(serial_tm_t(), pA, pQ, eigs, scratch.data(), iscratch.data());
  }

  template <class matrix_t>
  KOKKOS_INLINE_FUNCTION static int execute(matrix_t *pA, double *eigs) {
    const int ncols = GetNcols(*pA);
    std::vector<double> scratch(double_scratch_size(ncols));
    std::vector<std::size_t> iscratch(sizet_scratch_size(ncols));
    matrix_t *pQ = nullptr;
    return execute(serial_tm_t(), pA, pQ, eigs, scratch.data(), iscratch.data());
  }

  KOKKOS_INLINE_FUNCTION
  static std::size_t double_scratch_size(std::size_t ncols) { return 2 * ncols; }

  KOKKOS_INLINE_FUNCTION
  static std::size_t sizet_scratch_size(std::size_t ncols) { return ncols + 2; }

  static std::size_t total_shmem_scratch_size(std::size_t ncols) {
    return parthenon::ScratchPad1D<double>::shmem_size(double_scratch_size(ncols)) +
           parthenon::ScratchPad1D<std::size_t>::shmem_size(sizet_scratch_size(ncols));
  }
};

#endif // SYMMETRIC_EVD_HPP
