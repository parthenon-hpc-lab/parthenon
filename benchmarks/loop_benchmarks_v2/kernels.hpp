#pragma once

#include <array>
#include <cmath>

#include <Kokkos_Core.hpp>

namespace plb2 {

inline constexpr int kMaxNiter = 128;

KOKKOS_INLINE_FUNCTION double
ApplyKernelIterations(double value, const std::array<double, kMaxNiter> &alpha,
                      const std::array<double, kMaxNiter> &beta, int niter) {
  for (int iter = 0; iter < niter; ++iter) {
    value = value * alpha[iter] + beta[iter];
  }
  return value;
}

template <int SX, int SY, int SZ>
struct UnifiedSpanAccess {
  const double *center = nullptr;
  std::array<const double *, SX> x_ptrs{};
  std::array<const double *, SY> y_ptrs{};
  std::array<const double *, SZ> z_ptrs{};
};

template <int SX, int SY, int SZ, typename ViewType>
KOKKOS_INLINE_FUNCTION UnifiedSpanAccess<SX, SY, SZ> BuildUnifiedCellHoistedPointers(
    const ViewType &in, int b, int v, int k, int j, int i, const std::array<int, SX> &dx,
    const std::array<int, SY> &dy, const std::array<int, SZ> &dz) {
  UnifiedSpanAccess<SX, SY, SZ> access;
  access.center = &in(b, v, k, j, i);
  for (int ix = 0; ix < SX; ++ix) {
    access.x_ptrs[ix] = &in(b, v, k, j, i + dx[ix]);
  }
  for (int iy = 0; iy < SY; ++iy) {
    access.y_ptrs[iy] = &in(b, v, k, j + dy[iy], i);
  }
  for (int iz = 0; iz < SZ; ++iz) {
    access.z_ptrs[iz] = &in(b, v, k + dz[iz], j, i);
  }
  return access;
}

template <int SX, int SY, int SZ>
KOKKOS_INLINE_FUNCTION double
ComputeUnifiedCellHoisted(const UnifiedSpanAccess<SX, SY, SZ> &access, int idx,
                          const std::array<double, kMaxNiter> &alpha,
                          const std::array<double, kMaxNiter> &beta, int niter) {
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

  return ApplyKernelIterations(value, alpha, beta, niter);
}

template <int SX, int SY, int SZ, typename ViewType>
KOKKOS_INLINE_FUNCTION double
ComputeUnifiedCellDirect(const ViewType &in, int b, int v, int z, int y, int x,
                         const std::array<int, SX> &dx, const std::array<int, SY> &dy,
                         const std::array<int, SZ> &dz,
                         const std::array<double, kMaxNiter> &alpha,
                         const std::array<double, kMaxNiter> &beta, int niter) {
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

  return ApplyKernelIterations(value, alpha, beta, niter);
}

KOKKOS_INLINE_FUNCTION std::array<double, kMaxNiter> MakeAlpha() {
  std::array<double, kMaxNiter> alpha{};
  for (int i = 0; i < kMaxNiter; ++i) {
    alpha[i] = 1.000000000000001 + 1.0e-15 * i;
  }
  return alpha;
}

KOKKOS_INLINE_FUNCTION std::array<double, kMaxNiter> MakeBeta() {
  std::array<double, kMaxNiter> beta{};
  for (int i = 0; i < kMaxNiter; ++i) {
    beta[i] = 3.0e-15 + 2.0e-15 * i;
  }
  return beta;
}

} // namespace plb2
