#pragma once

#include <array>
#include <cmath>
#include <optional>

#include <Kokkos_Core.hpp>

#include "kernels.hpp"
#include "loop_abstraction.hpp"

namespace plb2 {

template <loop_abstraction::loop_tag LOOP_TAG, loop_abstraction::inner_tag INNER_TAG,
          int SX, int SY, int SZ, typename InputView, typename OutputView,
          typename CountsView>
inline void RunUnifiedKernelWithLoopAbstraction(
    const InputView &input, OutputView &output, const CountsView &active_counts, int nblocks,
    int nx, int ny, int nz, int nghost, const std::array<int, SX> &dx,
    const std::array<int, SY> &dy, const std::array<int, SZ> &dz,
    const std::array<double, kMaxNiter> &alpha, const std::array<double, kMaxNiter> &beta,
    int niter, std::optional<int> ninner = std::nullopt) {
  loop_abstraction::index_space_t<LOOP_TAG, INNER_TAG> idx_space(nblocks, nx, ny, nz, nghost,
                                                                  ninner);

  loop_abstraction::outer(idx_space,
                          KOKKOS_LAMBDA(const auto &idx_range, int b) {
                            const int nvars = active_counts(b);
                            for (int v = 0; v < nvars; ++v) {
                              auto in = idx_range.view(input, v);
                              auto out = idx_range.view(output, v);

                              std::array<decltype(in), SX> x_views{};
                              std::array<decltype(in), SY> y_views{};
                              std::array<decltype(in), SZ> z_views{};

                              for (int ix = 0; ix < SX; ++ix) {
                                x_views[ix] = idx_range.view(input, v, {0, 0, dx[ix]});
                              }
                              for (int iy = 0; iy < SY; ++iy) {
                                y_views[iy] = idx_range.view(input, v, {0, dy[iy], 0});
                              }
                              for (int iz = 0; iz < SZ; ++iz) {
                                z_views[iz] = idx_range.view(input, v, {dz[iz], 0, 0});
                              }

                              loop_abstraction::inner(
                                  idx_range, KOKKOS_LAMBDA(auto idx) {
                                    double value = in(idx);
                                    for (int ix = 0; ix < SX; ++ix) {
                                      value += x_views[ix](idx);
                                    }
                                    for (int iy = 0; iy < SY; ++iy) {
                                      value += y_views[iy](idx);
                                    }
                                    for (int iz = 0; iz < SZ; ++iz) {
                                      value += z_views[iz](idx);
                                    }
                                    out(idx) = ApplyKernelIterations(value, alpha, beta, niter);
                                  });
                            }
                          });
}

template <loop_abstraction::loop_tag LOOP_TAG, loop_abstraction::inner_tag INNER_TAG,
          int SX, int SY, int SZ, typename InputView, typename OutputView,
          typename CountsView>
inline void RunUnifiedKernelWithLoopAbstractionDirect(
    const InputView &input, OutputView &output, const CountsView &active_counts, int nblocks,
    int nx, int ny, int nz, int nghost, const std::array<int, SX> &dx,
    const std::array<int, SY> &dy, const std::array<int, SZ> &dz,
    const std::array<double, kMaxNiter> &alpha, const std::array<double, kMaxNiter> &beta,
    int niter, std::optional<int> ninner = std::nullopt) {
  static_assert(LOOP_TAG == loop_abstraction::loop_tag::boiv,
                "direct abstraction kernel is intended for boiv only");
  static_assert(INNER_TAG == loop_abstraction::inner_tag::logical,
                "direct abstraction kernel is intended for logical inner coordinates only");

  loop_abstraction::index_space_t<LOOP_TAG, INNER_TAG> idx_space(nblocks, nx, ny, nz, nghost,
                                                                  ninner);

  loop_abstraction::outer(idx_space,
                          KOKKOS_LAMBDA(const auto &idx_range, int b) {
                            const int nvars = active_counts(b);
                            for (int v = 0; v < nvars; ++v) {
                              loop_abstraction::inner(
                                  idx_range, KOKKOS_LAMBDA(auto idx) {
                                    double value = input(b, v, idx.k, idx.j, idx.i);
                                    for (int ix = 0; ix < SX; ++ix) {
                                      value += input(b, v, idx.k, idx.j, idx.i + dx[ix]);
                                    }
                                    for (int iy = 0; iy < SY; ++iy) {
                                      value += input(b, v, idx.k, idx.j + dy[iy], idx.i);
                                    }
                                    for (int iz = 0; iz < SZ; ++iz) {
                                      value += input(b, v, idx.k + dz[iz], idx.j, idx.i);
                                    }
                                    output(b, v, idx.k, idx.j, idx.i) =
                                        ApplyKernelIterations(value, alpha, beta, niter);
                                  });
                            }
                          });
}

}  // namespace plb2
