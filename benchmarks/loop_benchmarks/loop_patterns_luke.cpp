#include "loop_patterns.hpp"

#include <algorithm>
#include <stdexcept>

#include <Kokkos_Core.hpp>

#include "decomposition.hpp"
#include "kernels.hpp"

namespace plb {
#if defined(__clang__) || defined(__GNUC__)
__attribute__((used, noinline))
#endif
void RunCpuHierarchicalLoop2(const Dataset &dataset, const RawMemoryIndexer &idxer) {
  const auto &shape = dataset.problem;
  auto data = dataset.data;

  constexpr int NITER{10};
  constexpr std::array<double, NITER> alpha = [] {
    std::array<double, NITER> a{};
    for (int k = 0; k < NITER; ++k) {
      a[k] = 1.000000000000001 + 1.0e-15 * k;
    }
    return a;
  }();
  constexpr std::array<double, NITER> beta = [] {
    std::array<double, NITER> b{};
    for (int k = 0; k < NITER; ++k) {
      b[k] = 3.0e-15 + 2.0e-15 * k;
    }
    return b;
  }();

  for (int b = 0; b < shape.blocks; ++b) {
    const int nvar = data.active_counts(b);
    for (int idx_out = 0; idx_out < idxer.GetNouter(); ++idx_out) {
      const auto [ks, js, is] = idxer.GetStartIndices(idx_out);
      const int ninner = idxer.GetNinnerRaw(idx_out);
      for (int v = 0; v < nvar; ++v) {
        const double *const __restrict__ in = &data.in(b, v, ks, js, is);
        const double *const __restrict__ in_ip1 = &data.in(b, v, ks, js, is + 1);
        const double *const __restrict__ in_im1 = &data.in(b, v, ks, js, is - 1);
        const double *const __restrict__ in_jp1 = &data.in(b, v, ks, js + 1, is);
        const double *const __restrict__ in_jm1 = &data.in(b, v, ks, js - 1, is);
        const double *const __restrict__ in_kp1 = &data.in(b, v, ks + 1, js, is);
        const double *const __restrict__ in_km1 = &data.in(b, v, ks - 1, js, is);
        double *const out = &data.out(b, v, ks, js, is);
#pragma omp simd
        for (int idx = 0; idx < ninner; ++idx) {
          double a = in[idx];
          a += in_ip1[idx] + in_im1[idx] + in_jp1[idx] + in_jm1[idx] + in_kp1[idx] +
               in_km1[idx];
#pragma unroll
          for (int r = 0; r < NITER; ++r) {
            a = std::fma(a, alpha[r], beta[r]);
          }
          out[idx] = a;
        }
      }
    }
  }
}

} // namespace plb
