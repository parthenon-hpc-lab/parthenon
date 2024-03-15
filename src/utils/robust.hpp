
//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2024 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// Copied from https://github.com/lanl/phoebus
//========================================================================================
// © 2022. Triad National Security, LLC. All rights reserved.  This
// program was produced under U.S. Government contract
// 89233218CNA000001 for Los Alamos National Laboratory (LANL), which
// is operated by Triad National Security, LLC for the U.S.
// Department of Energy/National Nuclear Security Administration. All
// rights in the program are reserved by Triad National Security, LLC,
// and the U.S. Department of Energy/National Nuclear Security
// Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works,
// distribute copies to the public, perform publicly and display
// publicly, and to permit others to do so.

#ifndef UTILS_ROBUST_HPP_
#define UTILS_ROBUST_HPP_

#include <config.hpp>
#include <kokkos_abstraction.hpp>
#include <limits>

namespace parthenon {
namespace robust {

template <typename T = Real>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto LARGE() {
  return 0.1 * std::numeric_limits<T>::max();
}
template <typename T = Real>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto SMALL() {
  return 10 * std::numeric_limits<T>::min();
}
template <typename T = Real>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto EPS() {
  return 10 * std::numeric_limits<T>::epsilon();
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION auto make_positive(const T val) {
  return std::max(val, EPS<T>());
}

KOKKOS_FORCEINLINE_FUNCTION
Real make_bounded(const Real val, const Real vmin, const Real vmax) {
  return std::min(std::max(val, vmin + EPS()), vmax * (1.0 - EPS()));
}

template <typename T>
KOKKOS_INLINE_FUNCTION int sgn(const T &val) {
  return (T(0) <= val) - (val < T(0));
}
template <typename A, typename B>
KOKKOS_INLINE_FUNCTION auto ratio(const A &a, const B &b) {
  const B sgn = b >= 0 ? 1 : -1;
  return a / (b + sgn * SMALL<B>());
}
} // namespace robust
} // namespace parthenon
#endif // UTILS_ROBUST_HPP_