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

#ifndef BATCHED_LINEAR_ALGEBRA_MATRIX_CROSS_HPP_
#define BATCHED_LINEAR_ALGEBRA_MATRIX_CROSS_HPP_

// This file was made in part with generative AI.

#include <algorithm>
#include <vector>

#include "batched_linear_algebra/execution_utils.hpp"
#include "batched_linear_algebra/matrix_utils.hpp"
#include "batched_linear_algebra/maxvol.hpp"
#include "batched_linear_algebra/qr_decomposition.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {
namespace batched_linear_algebra {

class MatrixCross {
 public:
  /// Rank-r cross (skeleton) approximation of an n×m matrix A,
  ///
  ///   A ≈ C A(I,:),   C = A(:,J) A(I,J)⁻¹,
  ///
  /// which is exact if rank A = r. Only the entries of A(:,J) and A(I,:) are
  /// read, (n + m) r per sweep, so A may be a functor that computes its entries
  /// on demand.
  ///
  /// The index sets are improved by alternating sweeps. With J fixed, the
  /// column fiber A(:,J) is orthogonalized by QR and Maxvol on its Q selects I.
  /// With I fixed, the same is done on A(I,:)ᵀ to select J. Each Maxvol call is
  /// warm-started from the current indices, and the sweeps stop once a sweep
  /// changes neither I nor J. |det A(I,J)| never decreases.
  ///
  /// On entry:
  ///   - A is an n×m matrix. It is only read.
  ///   - pC must be non-null and point to an n×rank matrix.
  ///   - pR may be null. If non-null, it must point to a rank×m matrix. A null
  ///     pR still needs a pointer type, e.g. static_cast<decltype(pC)>(nullptr).
  ///   - I and J have length rank, with 1 <= rank <= min(n, m).
  ///   - If initialize_indices is true, I and J are ignored. The starting J is
  ///     rank evenly spaced columns, and the starting I is chosen by Maxvol's
  ///     greedy start. Otherwise I and J are a warm start (e.g. from a previous
  ///     call) and A(I,J) must be nonsingular.
  ///
  /// On exit:
  ///   - I and J hold the selected rows and columns.
  ///   - *pC holds C = A(:,J) A(I,J)⁻¹, so C(I,:) is the identity up to rounding.
  ///   - If pR != nullptr, *pR holds A(I,:).
  ///
  /// Notes:
  ///   - With a team handle, add a team barrier between preparing A, I and J and
  ///     calling execute, and before reading the results.
  ///   - tau is passed to Maxvol and should be greater than 1.
  ///   - The rank is fixed. No error estimate is made and rank deficiency of
  ///     the fibers is not checked.
  ///
  /// Returns:
  ///   - The number of sweeps, counting the final sweep that made no changes. A
  ///     value equal to max_sweeps means the index sets may still be changing.
  template <class tm_t, class matrix_a_t, class matrix_c_t, class matrix_r_t>
  KOKKOS_INLINE_FUNCTION static int
  execute(tm_t tm, const matrix_a_t &A, matrix_c_t *pC, matrix_r_t *pR, int *I, int *J,
          const int rank, double *scratch, const bool initialize_indices = true,
          const int max_sweeps = 10, const double tau = 1.05) {
    PARTHENON_REQUIRE(pC, "C must not be null.");
    const int nrows = GetNrows(A);
    const int ncols = GetNcols(A);
    PARTHENON_REQUIRE(rank >= 1 && rank <= std::min(nrows, ncols),
                      "MatrixCross requires 1 <= rank <= min(nrows, ncols).");
    PARTHENON_REQUIRE(GetNrows(*pC) == nrows && GetNcols(*pC) == rank,
                      "C must be nrows x rank.");
    if (pR) {
      PARTHENON_REQUIRE(GetNrows(*pR) == rank && GetNcols(*pR) == ncols,
                        "R must be rank x ncols.");
    }

    const int max_dim = std::max(nrows, ncols);
    double *fiber_data = scratch;
    double *q_data = &(scratch[max_dim * rank]);
    double *work = &(scratch[2 * max_dim * rank]);

    // Rows I from the column fiber A(:,J). Maxvol's B is C.
    auto column_step = [&](const bool greedy) {
      matrix_wrapper_t<double> F(fiber_data, nrows, rank);
      matrix_wrapper_t<double> Q(q_data, nrows, rank);
      parallel_loop(tm, 0, nrows - 1, 0, rank - 1,
                    [&](const int a, const int k) { F(a, k) = A(a, J[k]); });
      barrier(tm);
      QRDecomposition::execute(tm, &F, &Q, work);
      const int swaps = Maxvol::execute(tm, Q, pC, I, work, greedy, tau);
      barrier(tm);
      return swaps;
    };

    // Columns J from the row fiber A(I,:)ᵀ. Maxvol's B goes into the dead fiber.
    auto row_step = [&]() {
      matrix_wrapper_t<double> F(fiber_data, ncols, rank);
      matrix_wrapper_t<double> Q(q_data, ncols, rank);
      parallel_loop(tm, 0, ncols - 1, 0, rank - 1,
                    [&](const int b, const int k) { F(b, k) = A(I[k], b); });
      barrier(tm);
      QRDecomposition::execute(tm, &F, &Q, work);
      const int swaps = Maxvol::execute(tm, Q, &F, J, work, false, tau);
      barrier(tm);
      return swaps;
    };

    if (initialize_indices) {
      parallel_loop(tm, 0, rank - 1, [&](const int k) { J[k] = (k * ncols) / rank; });
      barrier(tm);
    }
    column_step(initialize_indices);

    int sweep = 0;
    while (sweep < max_sweeps) {
      const int row_swaps = row_step();
      const int col_swaps = column_step(false);
      ++sweep;
      if (row_swaps == 0 && col_swaps == 0) break;
    }

    if (pR) {
      auto &R = *pR;
      parallel_loop(tm, 0, rank - 1, 0, ncols - 1,
                    [&](const int k, const int b) { R(k, b) = A(I[k], b); });
    }
    return sweep;
  }

  template <class matrix_a_t, class matrix_c_t, class matrix_r_t>
  KOKKOS_INLINE_FUNCTION static int
  execute(const matrix_a_t &A, matrix_c_t *pC, matrix_r_t *pR, int *I, int *J,
          const int rank, double *scratch, const bool initialize_indices = true,
          const int max_sweeps = 10, const double tau = 1.05) {
    return execute(serial_tm_t(), A, pC, pR, I, J, rank, scratch, initialize_indices,
                   max_sweeps, tau);
  }

  // Version that is only callable on host and allocates its own scratch space
  template <class matrix_a_t, class matrix_c_t, class matrix_r_t>
  static int execute(const matrix_a_t &A, matrix_c_t *pC, matrix_r_t *pR, int *I, int *J,
                     const int rank, const bool initialize_indices = true,
                     const int max_sweeps = 10, const double tau = 1.05) {
    std::vector<double> scratch(double_scratch_size(GetNrows(A), GetNcols(A), rank));
    return execute(serial_tm_t(), A, pC, pR, I, J, rank, scratch.data(),
                   initialize_indices, max_sweeps, tau);
  }

  // nrows and ncols are the dimensions of A.
  KOKKOS_INLINE_FUNCTION static constexpr std::size_t
  double_scratch_size(std::size_t nrows, std::size_t ncols, std::size_t rank) {
    // Fiber and its thin Q, then a work area shared by QR and Maxvol
    const std::size_t max_dim = std::max(nrows, ncols);
    return 2 * max_dim * rank +
           std::max(QRDecomposition::double_scratch_size(max_dim, rank),
                    Maxvol::double_scratch_size(max_dim, rank));
  }

  static std::size_t total_shmem_scratch_size(std::size_t nrows, std::size_t ncols,
                                              std::size_t rank) {
    return parthenon::ScratchPad1D<double>::shmem_size(
        double_scratch_size(nrows, ncols, rank));
  }
};

} // namespace batched_linear_algebra
} // namespace parthenon

#endif // BATCHED_LINEAR_ALGEBRA_MATRIX_CROSS_HPP_
