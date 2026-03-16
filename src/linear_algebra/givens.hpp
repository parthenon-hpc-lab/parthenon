#ifndef GIVENS_HPP
#define GIVENS_HPP

#include <cmath>
#include <utility>

#include "matrix.hpp"

// Returns the 2x2 unitary matrix components
// sigma = [c -s]
//         [s  c]
// such that sigma . (a1, a2)^T = (alpha, 0)^T
KOKKOS_FORCEINLINE_FUNCTION
auto ComputeGivensZeroSecond(double a1, double a2) {
  // s * a1 + c * a2 = 0
  // s = -c a2 / a1
  // s^2 + c^2 = 1
  // c^2 (1 - a2^2 / a1^2) = 1
  // c^2 = a1^2 / (a1^2 + a2^2)
  // s^2 = a2^2 / (a1^2 + a2^2)
  if (a2 == 0.0) return std::make_pair(1.0, 0.0);
  double norm = 1.0 / (std::hypot(a1, a2) + 1e-15);
  // (c, s)
  return std::make_pair(norm * a1, -norm * a2);
}

// Returns the 2x2 unitary matrix components
// sigma = [c -s]
//         [s  c]
// such that sigma . (a1, a2)^T = (0, alpha)^T
KOKKOS_FORCEINLINE_FUNCTION
auto ComputeGivensZeroFirst(double a1, double a2) {
  if (a2 == 0.0) return std::make_pair(1.0, 0.0);
  double norm = 1.0 / (std::hypot(a1, a2) + 1e-15);
  // (c, s)
  return std::make_pair(norm * a2, norm * a1);
}

KOKKOS_FORCEINLINE_FUNCTION
auto ComputeGivensDiagonalize2by2(double d1, double d2, double b1) {
  // Computes a Givens rotation to diagonalize a symmetric 2x2 matrix
  // Fixes the pi/2 ambiguity in the diagonalizing rotation by selecting
  // the eigenvector of the larger eigenvalue as the first rotated basis
  // vector, ensuring the first diagonal entry of Q A Q^T is the larger
  // eigenvalue.
  const double half_diff = 0.5 * (d1 - d2);
  const double r = sqrt(half_diff * half_diff + b1 * b1);
  const double y = 0.5 * (d2 - d1) + r;
  const double norm = std::hypot(b1, y);
  return std::make_pair(b1 / norm, -y / norm);
}

template <bool return_zero>
KOKKOS_FORCEINLINE_FUNCTION double ValueAtIndexOrZero(int idx, double *arr) {
  if constexpr (return_zero) {
    return 0.0;
  } else {
    return arr[idx];
  }
}

// Applies a Givens rotation G (defined by the cosine c on the diagonal in
// indices gidx and gidx + 1 and the sine \pm s on the diagonal off) and its
// transpose G^T to a symmetric tri-diagonal matrix with a bulge. Defined by
// diagonal elements d and off-diagonal elements b, and bulge element bulge at
// (gidx - 1, gidx + 1) and (gidx + 1, gidx - 1).
template <bool first, bool last, class tm_t>
KOKKOS_FORCEINLINE_FUNCTION double ApplyGivensLeftRight(tm_t tm, int gidx, double c,
                                                        double s, double bulge, double *d,
                                                        double *b) {
  const double d1 = d[gidx];
  const double d2 = d[gidx + 1];
  const double b0 = ValueAtIndexOrZero<first>(gidx - 1, b);
  const double b1 = b[gidx];
  const double b2 = ValueAtIndexOrZero<last>(gidx + 1, b);

  barrier(tm);
  once_per_team(
      tm, KOKKOS_LAMBDA() {
        const double c2 = c * c;
        const double s2 = s * s;
        const double cs = c * s;

        // Apply G A G^T to the tridiagonal elements
        d[gidx] = c2 * d1 + s2 * d2 - 2 * cs * b1;
        d[gidx + 1] = s2 * d1 + c2 * d2 + 2 * cs * b1;
        if constexpr (!first) b[gidx - 1] = c * b0 - s * bulge;
        b[gidx] = cs * (d1 - d2) + (c2 - s2) * b1;
        if constexpr (!last) b[gidx + 1] = c * b2;
      });
  barrier(tm);

  // Return the value of the bulge element at the new position (gidx, gidx + 2)
  return -s * b2;
}

// A <- G A
// with G_{gidx, gidx} = G_{gidx + 1, gidx + 1} = c
// and G_{gidx, gidx + 1} = -G_{gidx + 1, gidx} = -s
// Bulge starts at element (gidx + 1, gidx)
template <bool first, bool last, class tm_t>
KOKKOS_FORCEINLINE_FUNCTION double ApplyGivensLeft(tm_t tm, int gidx, double c, double s,
                                                   double bulge, double *d, double *b) {
  const double d1 = d[gidx];
  const double d2 = d[gidx + 1];
  const double b1 = b[gidx];
  const double b2 = ValueAtIndexOrZero<last>(gidx + 1, b);

  barrier(tm);
  once_per_team(
      tm, KOKKOS_LAMBDA() {
        // Apply G A to the upper bidiagonal elements
        d[gidx] = c * d1 - bulge * s;
        d[gidx + 1] = c * d2 + s * b1;
        b[gidx] = c * b1 - s * d2;
        if constexpr (!last) b[gidx + 1] = c * b2;
      });
  barrier(tm);

  // Return the value of the bulge element at the new position (gidx, gidx + 2)
  return -s * b2;
}

// A <- A G^T
// with G_{gidx, gidx} = G_{gidx + 1, gidx + 1} = c
// and G_{gidx, gidx + 1} = -G_{gidx + 1, gidx} = -s
// Bulge starts at element (gidx - 1, gidx + 1)
template <bool first, bool last, class tm_t>
KOKKOS_FORCEINLINE_FUNCTION double ApplyGivensRight(tm_t tm, int gidx, double c, double s,
                                                    double bulge, double *d, double *b) {
  const double d1 = d[gidx];
  const double d2 = d[gidx + 1];
  const double b0 = ValueAtIndexOrZero<first>(gidx - 1, b);
  const double b1 = b[gidx];

  barrier(tm);
  once_per_team(
      tm, KOKKOS_LAMBDA() {
        // Apply A G to the upper bidiagonal elements
        d[gidx] = c * d1 - s * b1;
        d[gidx + 1] = c * d2;
        if constexpr (!first) b[gidx - 1] = c * b0 - s * bulge;
        b[gidx] = c * b1 + s * d1;
        // If first = false, this assumes that c and s are chosen such that
        // A' = A G element
        // A'(gidx - 1, gidx + 1) = c * bulge + s * b0 = 0
      });
  barrier(tm);

  // Return the value of the bulge element at the new position (gidx + 1, gidx)
  return -s * d2;
}

// A <- G A
// with G_{gidx, gidx} = G_{gidx + 1, gidx + 1} = c
// and G_{gidx, gidx + 1} = -G_{gidx + 1, gidx} = -s
template <class tm_t, class matrix_t>
KOKKOS_INLINE_FUNCTION void ApplyGivensLeft(tm_t tm, int gidx, double c, double s,
                                            matrix_t &A) {
  const int ncols = GetNcols(A);
  parallel_loop(
      tm, 0, ncols - 1, KOKKOS_LAMBDA(const int col) {
        double a1 = A(gidx, col);
        double a2 = A(gidx + 1, col);
        A(gidx, col) = c * a1 - s * a2;
        A(gidx + 1, col) = s * a1 + c * a2;
      });
}

// A <- A G^T
// with G_{gidx, gidx} = G_{gidx + 1, gidx + 1} = c
// and G_{gidx, gidx + 1} = -G_{gidx + 1, gidx} = -s
template <class tm_t, class matrix_t>
KOKKOS_INLINE_FUNCTION void ApplyGivensRight(tm_t tm, int gidx, double c, double s,
                                             matrix_t &A) {
  const int nrows = GetNrows(A);
  parallel_loop(
      tm, 0, nrows - 1, KOKKOS_LAMBDA(const int row) {
        double a1 = A(row, gidx);
        double a2 = A(row, gidx + 1);
        A(row, gidx) = c * a1 - s * a2;
        A(row, gidx + 1) = s * a1 + c * a2;
      });
}

// A <- G A G^T
// with G_{gidx, gidx} = G_{gidx + 1, gidx + 1} = c
// and G_{gidx, gidx + 1} = -G_{gidx + 1, gidx} = -s
template <class tm_t, class matrix_t>
KOKKOS_INLINE_FUNCTION void ApplyGivensLeftRight(tm_t tm, int gidx, double c, double s,
                                                 matrix_t &A) {
  ApplyGivensLeft(tm, gidx, c, s, A);
  ApplyGivensRight(tm, gidx, c, s, A);
}
#endif // GIVENS_HPP