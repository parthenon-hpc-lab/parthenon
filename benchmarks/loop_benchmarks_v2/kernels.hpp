#pragma once

#include <array>
#include <cmath>

#include <Kokkos_Core.hpp>

namespace plb2 {

template <int SX, int SY, int SZ>
struct UnifiedSpanAccess {
  const double *center = nullptr;
  std::array<const double *, SX> x_ptrs{};
  std::array<const double *, SY> y_ptrs{};
  std::array<const double *, SZ> z_ptrs{};
};

template <int SX, int SY, int SZ, typename ViewType>
KOKKOS_INLINE_FUNCTION UnifiedSpanAccess<SX, SY, SZ> BuildUnifiedCellHoistedPointers(
    const ViewType &in, int b, int v, int k, int j, int i, const std::array<int, 3> &strides,
    const std::array<int, SX> &dx, const std::array<int, SY> &dy,
    const std::array<int, SZ> &dz) {
  UnifiedSpanAccess<SX, SY, SZ> access;
  access.center = &in(b, v, k, j, i);
  for (int ix = 0; ix < SX; ++ix) {
    access.x_ptrs[ix] = access.center + dx[ix];
  }
  for (int iy = 0; iy < SY; ++iy) {
    access.y_ptrs[iy] = access.center + dy[iy] * strides[0];
  }
  for (int iz = 0; iz < SZ; ++iz) {
    access.z_ptrs[iz] = access.center + dz[iz] * strides[0] * strides[1];
  }
  return access;
}

template <int NITER, int SX, int SY, int SZ>
KOKKOS_INLINE_FUNCTION double ComputeUnifiedCellHoisted(
    const UnifiedSpanAccess<SX, SY, SZ> &access, int idx,
    const std::array<double, NITER> &alpha, const std::array<double, NITER> &beta) {
  double value = access.center[idx];

  for (int ix = 0; ix < SX; ++ix) {
    value += access.x_ptrs[ix][idx];
  }
  for (int iy = 0; iy < SY; ++iy) {
    value += access.y_ptrs[iy][idx];
  }
  for (int iz = 0; iz < SZ; ++iz) {
    value += access.z_ptrs[iz][idx];
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
