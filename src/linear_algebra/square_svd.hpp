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

#ifndef SQUARE_SVD_HPP
#define SQUARE_SVD_HPP

#include "parthenon/parthenon.hpp"

#include "householder.hpp"
#include "implicit_qr.hpp"
#include "matrix.hpp"

class SquareSVD {
 public:
  /// [This documentation was generated with assistance from generative AI]
  /// Compute the singular value decomposition (SVD) of a real square matrix A
  /// in place.
  ///
  /// On entry:
  ///   - pA must be non-null and point to an n×n real matrix A.
  ///   - pU and pV may be null. If non-null, they must point to n×n matrices.
  ///
  /// On exit:
  ///   - *pA is overwritten (original contents are destroyed). Internally A is
  ///     reduced to bidiagonal form via Householder transformations and then
  ///     diagonalized by an implicit QR iteration.
  ///   - sings[0..n-1] contains the computed singular values (not guaranteed
  ///     sorted but guaranteed positive).
  ///   - If pU != nullptr, *pU is overwritten and its *columns* contain the
  ///   left
  ///     singular vectors.
  ///   - If pV != nullptr, *pV is overwritten and its *columns* contain the
  ///   right
  ///     singular vectors.
  ///
  /// Mathematical summary:
  ///   - The routine computes A = U Σ Vᵀ, where Σ = diag(sings) and U and V are
  ///     orthogonal.
  ///   - Equivalently, Uᵀ A V = Σ.
  ///
  /// Scratch:
  ///   - scratch: double workspace used for Householder vectors and bidiagonal
  ///     off-diagonals.
  ///   - iscratch: integer workspace used for block partitioning in the QR
  ///   stage.
  ///
  /// Notes:
  ///   - tm is an execution context handle (templated) so this routine can be
  ///     invoked from Kokkos kernels or from host code.
  ///
  /// Returns:
  ///   - The number of implicit QR iterations performed (hitting the internal
  ///     limit indicates non-convergence).

  template <class tm_t, class matrix_t>
  KOKKOS_INLINE_FUNCTION static int execute(tm_t tm, matrix_t *pA, matrix_t *pU,
                                            matrix_t *pV, double *sings, double *scratch,
                                            std::size_t *iscratch) {
    PARTHENON_REQUIRE(pA, "A must not be null.");
    auto &A = *pA;
    const int ncols = GetNcols(A);
    const int nrows = GetNrows(A);
    PARTHENON_REQUIRE(nrows >= ncols,
                    "Tall-skinny SVD requires nrows >= ncols.");
    const int max_dim = (nrows > ncols ? nrows : ncols);

    if (ncols == 1) {
      double sigma2{0.0};

      summation(
          tm, 0, nrows - 1,
          [&](const int r, double &sum) {
            sum += A(r, 0) * A(r, 0);
          },
          sigma2);

      const double sigma = safe_sqrt(sigma2);

      once_per_team(tm, [&]() { sings[0] = sigma; });

      if (pV) {
        once_per_team(tm, [&]() { (*pV)(0, 0) = 1.0; });
      }

      if (pU) {
        if (sigma > 0.0) {
          parallel_loop(
              tm, 0, nrows - 1,
              [&](const int r) { (*pU)(r, 0) = A(r, 0) / sigma; });
        } else {
          parallel_loop(
              tm, 0, nrows - 1,
              [&](const int r) { (*pU)(r, 0) = (r == 0 ? 1.0 : 0.0); });
        }
      }
      barrier(tm);
      return 0;
    }

    // Bidiagonalize the tall-skinny matrix via Householder transformations
    double *v = &(scratch[0]);
    double *s = &(scratch[max_dim]);
    double *vhead = &(scratch[2 * max_dim]);
    if (pU) {
      auto &U = *pU;
      parallel_loop(tm, 0, nrows - 1, 0, ncols - 1,
                    [&](int r, int c) { U(r, c) = (r == c); });
    }
    if (pV) {
      auto &V = *pV;
      parallel_loop(tm, 0, ncols - 1, 0, ncols - 1,
                    [&](int r, int c) { V(r, c) = (r == c); });
    }
    
    const int upper_lim = (nrows == ncols ? ncols - 2 : ncols - 1);
    sequential_loop(0, upper_lim, [&](const int col) {
      build_householder_vector_col(tm, col, col, A, v);
      barrier(tm);
      apply_left_householder_transformation(tm, v, s, A, col, col);
      
      once_per_team(
          tm, [&]() { vhead[col] = v[col]; });
      parallel_loop(tm, col + 1, nrows - 1, [&](int r) { 
        A(r, col) = v[r];
      });
      barrier(tm);

      if (col < upper_lim) {
        barrier(tm);
        build_householder_vector_row(tm, col, col + 1, A, v);
        barrier(tm);
        apply_right_householder_transformation(tm, v, s, A, col + 1, col);
        if (pV) apply_right_householder_transformation(tm, v, s, *pV, col + 1);
      }
      barrier(tm);
      
    });
    
    // Move to bidiagonal storage
    barrier(tm);
    once_per_team(
        tm, [&]() { sings[0] = A(0, 0); });
    parallel_loop(
        tm, 0, ncols - 2, [&](int i) {
          sings[i + 1] = A(i + 1, i + 1);
          scratch[i] = A(i, i + 1);
        });

    barrier(tm);
    std::size_t *start = &(iscratch[0]);
    std::size_t *end = &(iscratch[ncols / 2 + 1]);
    const int status =
        ImplicitQRBidiag(tm, sings, scratch, pU, pV, start, end, ncols, 10 * ncols);
    if (status == 10 * ncols) return -status;

    // Apply the Householder vectors to pU 
    if (pU) {
      sequential_loop(0, upper_lim, [&](const int inv_col) {
        int col = upper_lim - inv_col;
        // Reconstruct householder vector
        parallel_loop(tm, 0, col - 1, [&](int r) { 
          v[r] = 0.0;
        });
        once_per_team(
            tm, [&]() { v[col] = vhead[col]; });
        parallel_loop(tm, col + 1, nrows - 1, [&](int r) { 
          v[r] = A(r, col);
        });
        barrier(tm);
        apply_left_householder_transformation(tm, v, s, *pU, col, 0);
      });
    }

    // Ensure singular values are positive
    parallel_loop(
        tm, 0, ncols - 1, [&](int col) {
          if (sings[col] < 0.) {
            sings[col] *= -1.;
            if (pU) {
              for (int row = 0; row < nrows; row++) {
                (*pU)(row, col) *= -1.;
              }
            }
          }
        });
    barrier(tm);

    return status;
  }

  template <class tm_t, class matrix_t>
  KOKKOS_INLINE_FUNCTION static int execute(tm_t tm, matrix_t *pA, double *eigs,
                                            double *scratch, std::size_t *iscratch) {
    matrix_t *pU = nullptr;
    matrix_t *pV = nullptr;
    return execute(tm, pA, pU, pV, eigs, scratch, iscratch);
  }

  template <class matrix_t>
  KOKKOS_INLINE_FUNCTION static int execute(matrix_t *pA, double *eigs, double *scratch,
                                            std::size_t *iscratch) {
    matrix_t *pU = nullptr;
    matrix_t *pV = nullptr;
    return execute(serial_tm_t(), pA, pU, pV, eigs, scratch, iscratch);
  }

  // Version that is only callable on host and allocates its own
  // scratch space
  template <class matrix_t>
  KOKKOS_INLINE_FUNCTION static int execute(matrix_t *pA, matrix_t *pU, matrix_t *pV,
                                            double *eigs) {
    const int nrows = GetNrows(*pA);
    const int ncols = GetNcols(*pA);
    std::vector<double> scratch(double_scratch_size(nrows, ncols));
    std::vector<std::size_t> iscratch(sizet_scratch_size(ncols));
    return execute(serial_tm_t(), pA, pU, pV, eigs, scratch.data(), iscratch.data());
  }

  template <class matrix_t>
  KOKKOS_INLINE_FUNCTION static int execute(matrix_t *pA, double *eigs) {
    const int nrows = GetNrows(*pA);
    const int ncols = GetNcols(*pA);
    std::vector<double> scratch(double_scratch_size(nrows, ncols));
    std::vector<std::size_t> iscratch(sizet_scratch_size(ncols));
    matrix_t *pU = nullptr;
    matrix_t *pV = nullptr;
    return execute(serial_tm_t(), pA, pU, pV, eigs, scratch.data(), iscratch.data());
  }

  KOKKOS_INLINE_FUNCTION
  static std::size_t double_scratch_size(std::size_t nrows, std::size_t ncols) {
    return 3 * std::max(nrows, ncols);
  }

  KOKKOS_INLINE_FUNCTION
  static std::size_t double_scratch_size(std::size_t ncols) {
    return double_scratch_size(ncols, ncols);
  }

  KOKKOS_INLINE_FUNCTION
  static std::size_t sizet_scratch_size(std::size_t ncols) { return ncols + 2; }

  static std::size_t total_shmem_scratch_size(std::size_t nrows, std::size_t ncols) {
    return parthenon::ScratchPad1D<double>::shmem_size(double_scratch_size(nrows, ncols)) +
           parthenon::ScratchPad1D<std::size_t>::shmem_size(sizet_scratch_size(ncols));
  }

  static std::size_t total_shmem_scratch_size(std::size_t ncols) {
    return total_shmem_scratch_size(ncols, ncols);
  }
};

#endif // SQUARE_SVD_HPP
