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

#ifndef BATCHED_LINEAR_ALGEBRA_MATMUL_HPP_
#define BATCHED_LINEAR_ALGEBRA_MATMUL_HPP_

// This file was made in part with generative AI.

#include "batched_linear_algebra/execution_utils.hpp"
#include "batched_linear_algebra/matrix_utils.hpp"
#include "kokkos_abstraction.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {
namespace batched_linear_algebra {

template <int MB = 8, int NB = 8, int KB = 8, bool SymmetricOutput = false, class tm_t,
          class A_t, class B_t, class C_t>
KOKKOS_FORCEINLINE_FUNCTION void
MatMulPacked(tm_t tm, const A_t &A, const B_t &B, C_t &C,
             parthenon::ScratchPad1D<double> &a_scratch,
             parthenon::ScratchPad1D<double> &b_scratch,
             parthenon::ScratchPad1D<double> &c_scratch) {
  const int m = GetNrows(A);
  const int k = GetNcols(A);
  const int n = GetNcols(B);

  PARTHENON_REQUIRE(GetNrows(B) == k, "Inner dimensions must match.");
  PARTHENON_REQUIRE(GetNrows(C) == m, "C row dimension mismatch.");
  PARTHENON_REQUIRE(GetNcols(C) == n, "C col dimension mismatch.");
  if constexpr (SymmetricOutput) {
    PARTHENON_REQUIRE(m == n, "Output matrix must be square for symmetric result.");
  }

  const int mb_size = MB > 0 ? MB : m;
  const int kb_size = KB > 0 ? KB : k;
  const int nb_size = NB > 0 ? NB : n;
  constexpr bool full_output_tile = (MB <= 0 && NB <= 0);

  PARTHENON_REQUIRE(a_scratch.extent_int(0) >= mb_size * kb_size, "A scratch too small.");
  PARTHENON_REQUIRE(b_scratch.extent_int(0) >= nb_size * kb_size, "B scratch too small.");
  if (!full_output_tile)
    PARTHENON_REQUIRE(c_scratch.extent_int(0) >= mb_size * nb_size,
                      "C scratch too small.");
  auto AT = matrix_wrapper_t<double>(a_scratch.data(), mb_size, kb_size);
  auto BT = matrix_wrapper_t<double>(b_scratch.data(), kb_size, nb_size);
  auto CT = full_output_tile
                ? matrix_wrapper_t<double>(c_scratch.data(), 1, 1)
                : matrix_wrapper_t<double>(c_scratch.data(), mb_size, nb_size);
  for (int i0 = 0; i0 < m; i0 += mb_size) {
    const int mb = (i0 + mb_size <= m) ? mb_size : (m - i0);
    const int jstart = SymmetricOutput ? i0 : 0;
    for (int j0 = jstart; j0 < n; j0 += nb_size) {
      const int nb = (j0 + nb_size <= n) ? nb_size : (n - j0);
      const bool diagonal_tile = i0 == j0;

      // Zero local C tile
      parallel_loop(tm, 0, mb - 1, 0, nb - 1, [&](const int ii, const int jj) {
        if constexpr (full_output_tile) {
          C(i0 + ii, j0 + jj) = 0.0;
        } else {
          CT(ii, jj) = 0.0;
        }
      });
      barrier(tm);

      for (int k0 = 0; k0 < k; k0 += kb_size) {
        const int kb = (k0 + kb_size <= k) ? kb_size : (k - k0);

        // Pack A tile: A(i0:i0+mb, k0:k0+kb)
        parallel_loop(tm, 0, mb - 1, 0, kb - 1, [&](const int ii, const int kk) {
          AT(ii, kk) = A(i0 + ii, k0 + kk);
        });
        barrier(tm);

        // Pack B tile: B(k0:k0+kb, j0:j0+nb)
        parallel_loop(tm, 0, kb - 1, 0, nb - 1, [&](const int kk, const int jj) {
          BT(kk, jj) = B(k0 + kk, j0 + jj);
        });
        barrier(tm);

        // CT += AT * BT
        parallel_loop(tm, 0, mb - 1, 0, nb - 1, [&](const int ii, const int jj) {
          if constexpr (SymmetricOutput) {
            if (diagonal_tile && jj < ii) return;
          }
          double sum = 0.0;
          for (int kk = 0; kk < kb; ++kk) {
            sum += AT(ii, kk) * BT(kk, jj);
          }
          if constexpr (full_output_tile) {
            C(i0 + ii, j0 + jj) += sum;
          } else {
            CT(ii, jj) += sum;
          }
        });
        barrier(tm);
      }

      // Write back C tile
      if constexpr (!full_output_tile) {
        parallel_loop(tm, 0, mb - 1, 0, nb - 1, [&](const int ii, const int jj) {
          C(i0 + ii, j0 + jj) = CT(ii, jj);
        });
        barrier(tm);
      }
    }
  }
  if constexpr (SymmetricOutput) {
    parallel_loop(tm, 0, m - 1, 0, m - 1, [&](const int i, const int j) {
      if (j > i) C(j, i) = C(i, j);
    });
    barrier(tm);
  }
}

} // namespace batched_linear_algebra
} // namespace parthenon

#endif // BATCHED_LINEAR_ALGEBRA_MATMUL_HPP_
