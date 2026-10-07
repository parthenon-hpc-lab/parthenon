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

#ifndef BATCHED_LINEAR_ALGEBRA_MAXVOL_HPP_
#define BATCHED_LINEAR_ALGEBRA_MAXVOL_HPP_

// This file was made in part with generative AI.

#include <cmath>
#include <vector>

#include "batched_linear_algebra/execution_utils.hpp"
#include "batched_linear_algebra/matrix_utils.hpp"
#include "batched_linear_algebra/qr_solve.hpp"
#include "kokkos_abstraction.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {
namespace batched_linear_algebra {

class Maxvol {
 public:
  /// Find r rows I of a tall n×r matrix A whose r×r submatrix A(I,:) is
  /// tau-dominant, i.e. every entry of B = A A(I,:)⁻¹ satisfies |B(i,j)| <= tau.
  /// A dominant submatrix has nearly maximal |det| among all r×r submatrices.
  ///
  /// The rows are improved one at a time: while the largest |B(i,j)| exceeds
  /// tau, row I[j] is replaced by row i, which increases |det A(I,:)| by a factor
  /// |B(i,j)|, and B is updated by a rank-1 correction.
  ///
  /// On entry:
  ///   - A is an n×r matrix with n >= r. It is only read and must not alias *pB.
  ///   - pB must be non-null and point to an n×r matrix.
  ///   - I has length r. If initialize_indices is false, it must hold r distinct
  ///     row indices with A(I,:) nonsingular. Otherwise its contents are ignored
  ///     and the starting rows are chosen greedily: for each column in turn, pick
  ///     the row with the largest entry in that column and eliminate it from the
  ///     remaining rows.
  ///
  /// On exit:
  ///   - I holds the selected rows.
  ///   - *pB holds B = A A(I,:)⁻¹, so B(I,:) is the identity up to rounding.
  ///
  /// Notes:
  ///   - tau should be greater than 1. With tau = 1, rounding can make the swaps
  ///     cycle, and only max_iters stops the loop.
  ///   - A is assumed to have full column rank. This is not checked. For a rank
  ///     deficient A the greedy start can repeat rows and B contains inf/NaN.
  ///   - A column selection of a wide matrix W is the row selection of Wᵀ, which
  ///     can be passed through matrix_transpose_view_t.
  ///   - With a team handle and initialize_indices false, add a team barrier
  ///     between filling A and I and calling execute.
  ///
  /// Returns:
  ///   - The number of row swaps performed. A value equal to max_iters means the
  ///     final B may still have entries larger than tau.
  template <class tm_t, class matrix_a_t, class matrix_b_t>
  KOKKOS_INLINE_FUNCTION static int
  execute(tm_t tm, const matrix_a_t &A, matrix_b_t *pB, int *I, double *scratch,
          const bool initialize_indices = true, const double tau = 1.05,
          const int max_iters = 100) {
    PARTHENON_REQUIRE(pB, "B must not be null.");
    auto &B = *pB;
    const int nrows = GetNrows(A);
    const int rank = GetNcols(A);
    PARTHENON_REQUIRE(nrows >= rank, "Maxvol requires nrows >= ncols.");
    PARTHENON_REQUIRE(GetNrows(B) == nrows && GetNcols(B) == rank,
                      "B must have the same shape as A.");

    matrix_wrapper_t<double> M(scratch, rank, rank);
    // Shared by the greedy start, the solve and the rank-1 updates
    double *work = &(scratch[rank * rank]);

    if (initialize_indices) {
      parallel_loop(tm, 0, nrows - 1, 0, rank - 1,
                    [&](const int r, const int c) { B(r, c) = A(r, c); });
      barrier(tm);

      double *pivot_row = work;
      double *factors = &(work[rank]);
      sequential_loop(0, rank - 1, [&](const int k) {
        maxloc_value_t piv;
        find_maximum_location(
            tm, 0, nrows - 1,
            [&](const int i, maxloc_value_t &mx) {
              const double val = std::abs(B(i, k));
              if (val > mx.val) {
                mx.val = val;
                mx.loc = i;
              }
            },
            piv);
        // If the remaining columns are all NaN (rank deficient A), no location is
        // found and loc is left at INT_MAX; fall back to k to stay in bounds
        const int p = (piv.loc < nrows) ? piv.loc : k;
        once_per_team(tm, [&]() { I[k] = p; });

        const double inv_pivot = 1.0 / B(p, k);
        parallel_loop(tm, k + 1, rank - 1, [&](const int c) { pivot_row[c] = B(p, c); });
        parallel_loop(tm, 0, nrows - 1, [&](const int i) {
          // x * (1/x) need not round to 1, so set the pivot's factor
          factors[i] = (i == p) ? 1.0 : B(i, k) * inv_pivot;
        });
        barrier(tm);

        // Row p has factor exactly 1, so this zeroes it (and every earlier pivot
        // row stays zero), which keeps chosen rows out of later pivot searches
        parallel_loop(tm, 0, nrows - 1, k + 1, rank - 1, [&](const int i, const int c) {
          B(i, c) -= factors[i] * pivot_row[c];
        });
        barrier(tm);
      });
    }

    parallel_loop(tm, 0, rank - 1, 0, rank - 1,
                  [&](const int r, const int c) { M(r, c) = A(I[r], c); });
    parallel_loop(tm, 0, nrows - 1, 0, rank - 1,
                  [&](const int r, const int c) { B(r, c) = A(r, c); });
    barrier(tm);
    QRSolveRight::execute(tm, &M, pB, work);

    double *col = work;
    double *row = &(work[nrows]);
    int iter = 0;
    for (; iter < max_iters; ++iter) {
      maxloc_value_t mx;
      find_maximum_location(
          tm, 0, nrows * rank - 1,
          [&](const int idx, maxloc_value_t &m) {
            const double val = std::abs(B(idx / rank, idx % rank));
            if (val > m.val) {
              m.val = val;
              m.loc = idx;
            }
          },
          mx);
      if (mx.val <= tau) break;

      const int i = mx.loc / rank;
      const int j = mx.loc % rank;
      once_per_team(tm, [&]() { I[j] = i; });

      // B <- B - B(:,j) (B(i,:) - e_jᵀ) / B(i,j), which makes row i equal e_jᵀ
      const double inv_bij = 1.0 / B(i, j);
      parallel_loop(tm, 0, nrows - 1, [&](const int a) { col[a] = B(a, j); });
      parallel_loop(tm, 0, rank - 1,
                    [&](const int b) { row[b] = (B(i, b) - (b == j)) * inv_bij; });
      barrier(tm);
      parallel_loop(tm, 0, nrows - 1, 0, rank - 1,
                    [&](const int a, const int b) { B(a, b) -= col[a] * row[b]; });
      barrier(tm);
    }

    return iter;
  }

  template <class matrix_a_t, class matrix_b_t>
  KOKKOS_INLINE_FUNCTION static int
  execute(const matrix_a_t &A, matrix_b_t *pB, int *I, double *scratch,
          const bool initialize_indices = true, const double tau = 1.05,
          const int max_iters = 100) {
    return execute(serial_tm_t(), A, pB, I, scratch, initialize_indices, tau, max_iters);
  }

  // Version that is only callable on host and allocates its own scratch space
  template <class matrix_a_t, class matrix_b_t>
  static int execute(const matrix_a_t &A, matrix_b_t *pB, int *I,
                     const bool initialize_indices = true, const double tau = 1.05,
                     const int max_iters = 100) {
    std::vector<double> scratch(double_scratch_size(GetNrows(A), GetNcols(A)));
    return execute(serial_tm_t(), A, pB, I, scratch.data(), initialize_indices, tau,
                   max_iters);
  }

  // nrows and rank are the dimensions of A.
  KOKKOS_INLINE_FUNCTION static constexpr std::size_t
  double_scratch_size(std::size_t nrows, std::size_t rank) {
    // M = A(I,:), then a work area that holds the QRSolveRight workspace
    // (rank + nrows), the greedy-start pivot row and factors, or the rank-1
    // update vectors
    return rank * rank + nrows + rank;
  }

  static std::size_t total_shmem_scratch_size(std::size_t nrows, std::size_t rank) {
    return parthenon::ScratchPad1D<double>::shmem_size(double_scratch_size(nrows, rank));
  }
};

} // namespace batched_linear_algebra
} // namespace parthenon

#endif // BATCHED_LINEAR_ALGEBRA_MAXVOL_HPP_
