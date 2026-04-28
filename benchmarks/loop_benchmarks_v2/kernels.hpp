#pragma once

#include <array>
#include <cmath>

#include <Kokkos_Core.hpp>

namespace plb2 {

template <int NITER, int SX, int SY, int SZ>
KOKKOS_INLINE_FUNCTION double ComputeUnifiedCellHoisted(
    double center, const std::array<const double *, SX> &x_ptrs,
    const std::array<const double *, SY> &y_ptrs, const std::array<const double *, SZ> &z_ptrs,
    const std::array<double, NITER> &alpha, const std::array<double, NITER> &beta) {
  double value = center;

  for (int ix = 0; ix < SX; ++ix) {
    value += *x_ptrs[ix];
  }
  for (int iy = 0; iy < SY; ++iy) {
    value += *y_ptrs[iy];
  }
  for (int iz = 0; iz < SZ; ++iz) {
    value += *z_ptrs[iz];
  }

  for (int iter = 0; iter < NITER; ++iter) {
    value = std::fma(value, alpha[iter], beta[iter]);
  }

  return value;
}

template <int NITER, int SX, int SY, int SZ, typename ViewType>
KOKKOS_INLINE_FUNCTION double ComputeUnifiedCellDirect(
    const ViewType &in, int b, int v, int z, int y, int x, const std::array<int, SX> &dx,
    const std::array<int, SY> &dy, const std::array<int, SZ> &dz,
    const std::array<double, NITER> &alpha, const std::array<double, NITER> &beta) {
  double value = in(b, v, z, y, x);

  for (int ix = 0; ix < SX; ++ix) {
    value += in(b, v, z, y, x + dx[ix]);
  }
  for (int iy = 0; iy < SY; ++iy) {
    value += in(b, v, z, y + dy[iy], x);
  }
  for (int iz = 0; iz < SZ; ++iz) {
    value += in(b, v, z + dz[iz], y, x);
  }

  for (int iter = 0; iter < NITER; ++iter) {
    value = std::fma(value, alpha[iter], beta[iter]);
  }

  return value;
}

template <int NITER>
KOKKOS_INLINE_FUNCTION std::array<double, NITER> MakeAlpha() {
  std::array<double, NITER> alpha{};
  for (int i = 0; i < NITER; ++i) {
    alpha[i] = 1.000000000000001 + 1.0e-15 * i;
  }
  return alpha;
}

template <int NITER>
KOKKOS_INLINE_FUNCTION std::array<double, NITER> MakeBeta() {
  std::array<double, NITER> beta{};
  for (int i = 0; i < NITER; ++i) {
    beta[i] = 3.0e-15 + 2.0e-15 * i;
  }
  return beta;
}

}  // namespace plb2
