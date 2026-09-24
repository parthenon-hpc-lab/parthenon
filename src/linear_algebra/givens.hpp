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

#ifndef GIVENS_HPP
#define GIVENS_HPP

#include <cmath>
#include <utility>

#include "matrix.hpp"
#include "utils/robust.hpp"

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
  double norm = parthenon::robust::ratio(1.0, std::hypot(a1, a2));
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
  double norm = parthenon::robust::ratio(1.0, std::hypot(a1, a2));
  // (c, s)
  return std::make_pair(norm * a2, norm * a1);
}

KOKKOS_FORCEINLINE_FUNCTION
auto ComputeGivensDiagonalize2by2(double d1, double d2, double b1) {
  const double t = (d1 - d2) / (2.0 * b1);
  const double tau = t + sign_of(t) * std::sqrt(1.0 + t * t);
  const double s = -1.0 / std::sqrt(1.0 + tau * tau);
  const double c = -tau * s;
  return std::make_pair(c, s);
} 

struct SVD2by2Result {
  double smin;
  double smax;
  double sr;
  double cr;
  double sl;
  double cl;
};

KOKKOS_FORCEINLINE_FUNCTION
SVD2by2Result ComputeSVD2by2UpperTriangular(double a11, double a12, double a22) {
  using namespace parthenon::robust;
  // Scale the matrix to order unity 
  double scale = std::max(std::max(std::abs(a11), std::abs(a22)), std::abs(a12));
  if (scale == 0.0) return SVD2by2Result{0, 0, 0, 1, 0, 1};

  // and make the dominant diagonal positive (if there is one)
  if (a11 != 0.0 || a22 != 0.0)
    scale *= std::abs(a11) > std::abs(a22) ? sign_of(a11) : sign_of(a22);
  a11 = ratio(a11, scale);
  a12 = ratio(a12, scale);
  a22 = ratio(a22, scale);

  // If the trailing diagonal is dominant in the working frame, swap the
  // diagonal entries so the formulas act on a matrix with dominant leading
  // diagonal
  const bool swap = a22 > a11;
  if (swap) {
    const double temp = a22;
    a22 = a11;
    a11 = temp;
  }

  // Singular values of [a11 a12; 0 a22] in the LAPACK 2x2 form
  const double sp = safe_sqrt((a11 + a22) * (a11 + a22) + a12 * a12);
  const double sm = safe_sqrt((a11 - a22) * (a11 - a22) + a12 * a12);
  const double aa = 0.5 * (sp + sm);

  SVD2by2Result out;
  out.smax = scale * aa;
  out.smin = ratio(scale * a11 * a22, aa);
  
  // First right singular vector v1 is proportional to
  // [a11 * a12, -(sigma_max^2 - a11^2)]
  double xr = a11 * a12;
  // Evaluate -(sigma_max^2 - a11^2) in a form that avoids cancellation when
  // sigma_max is close to a11 by rationalizing the relevant differences
  double yr = ratio(a12 * a12, sp + (a11 + a22));
  yr += ratio(a12 * a12, sm + (a11 - a22));
  yr *= -0.5 * (aa + a11);
  const double normr = std::hypot(xr, yr);
  const double cr = ratio(xr, normr);
  const double sr = ratio(yr, normr);

  // Recover the first left singular vector from A v1 = sigma_max u1
  const double cl = ratio(a11 * cr - a12 * sr, aa);
  const double sl = ratio(a22 * sr, aa);
  // Renormalize for floating-point safety
  const double norml = std::hypot(cl, sl);
  
  if (swap) {
    // Swapping the working-frame diagonal entries corresponds to a transpose-
    // permutation relation, so the left and right singular vectors map back
    // with exchanged roles and permuted components.
    out.cr = ratio(sl, norml);
    out.sr = ratio(cl, norml);
    out.cl = sr;
    out.sl = cr;
  } else {
    out.cr = cr;
    out.sr = sr;
    out.cl = ratio(cl, norml);
    out.sl = ratio(sl, norml);
  }

  return out;
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