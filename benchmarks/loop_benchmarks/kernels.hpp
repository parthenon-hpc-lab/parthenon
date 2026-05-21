#pragma once

#include <cstddef>

#include <Kokkos_Core.hpp>

namespace plb {

KOKKOS_INLINE_FUNCTION
double ComputeLightCell(double in, double aux) { return 1.125 * in - 0.375 * aux; }

KOKKOS_INLINE_FUNCTION
double ComputeFluxCell(double in, double fx_up, double fx_lo, double fy_up, double fy_lo,
                       double fz_up, double fz_lo) {
  return 0.875 * in + (fx_up - fx_lo) + (fy_up - fy_lo) + (fz_up - fz_lo);
}

KOKKOS_INLINE_FUNCTION
double ComputeStencilCell(double center, double im1, double ip1, double jm1, double jp1,
                          double km1, double kp1, double aux, double fx_up,
                          double fx_lo) {
  const double lap = -6.0 * center + im1 + ip1 + jm1 + jp1 + km1 + kp1;
  return 0.625 * center + 0.125 * lap + 0.25 * aux + 0.5 * (fx_up - fx_lo);
}

KOKKOS_INLINE_FUNCTION
double ComputeHeavyCell(double in, double aux, int heavy_iterations) {
  double x = in;
  double y = aux;
  double acc = 0.0;
  for (int iter = 0; iter < heavy_iterations; ++iter) {
    const double t0 = x * 1.001 + y * 0.125 + acc * 0.03125;
    const double t1 = y * 0.999 - x * 0.0625 + acc * 0.015625;
    acc = t0 * t1 + acc * 0.5;
    x = t0 + 0.1 * acc;
    y = t1 - 0.05 * acc;
  }
  return acc + x - y;
}

} // namespace plb
