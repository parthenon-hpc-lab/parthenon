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

#ifndef BATCHED_LINEAR_ALGEBRA_EXECUTION_UTILS_HPP_
#define BATCHED_LINEAR_ALGEBRA_EXECUTION_UTILS_HPP_

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "kokkos_abstraction.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {
namespace batched_linear_algebra {

// Functors passed to the loop wrappers below are only ever invoked by the calling
// thread (serially or through the inner-loop Kokkos patterns), never launched as a
// kernel, so they should be written as plain [&] lambdas rather than KOKKOS_LAMBDA.

using serial_tm_t = int; // TODO(LFR): parthenon::team_mbr_t;

// The execution handles the wrappers below know how to dispatch on. Anything else
// would otherwise silently compile to a no-op.
template <class tm_t>
inline constexpr bool is_supported_tm_v =
    std::is_same_v<tm_t, serial_tm_t> || std::is_same_v<tm_t, parthenon::team_mbr_t>;

template <class tm_t>
KOKKOS_FORCEINLINE_FUNCTION constexpr void check_execution_handle() {
  static_assert(is_supported_tm_v<tm_t>,
                "Execution handle must be serial_tm_t or parthenon::team_mbr_t.");
}

KOKKOS_INLINE_FUNCTION
double safe_sqrt(const double a) { return std::sqrt(std::max(a, 0.)); }

template <typename T>
KOKKOS_INLINE_FUNCTION int sign_of(T val) {
  constexpr T zero{0};
  // Zero is counted as positive for Householder reflector stability
  return (zero <= val) - (val < zero);
}

template <class tm_t>
KOKKOS_FORCEINLINE_FUNCTION void barrier(tm_t tm) {
  check_execution_handle<tm_t>();
  if constexpr (std::is_same_v<tm_t, parthenon::team_mbr_t>) {
    tm.team_barrier();
  }
}

template <class tm_t>
KOKKOS_FORCEINLINE_FUNCTION int rank(tm_t tm) {
  check_execution_handle<tm_t>();
  if constexpr (std::is_same_v<tm_t, parthenon::team_mbr_t>) {
    return tm.team_rank();
  } else {
    return 0;
  }
}

template <class tm_t, class F>
KOKKOS_FORCEINLINE_FUNCTION void once_per_team(tm_t tm, const F &func) {
  check_execution_handle<tm_t>();
  if constexpr (std::is_same_v<tm_t, serial_tm_t>) {
    func();
  } else if constexpr (std::is_same_v<tm_t, parthenon::team_mbr_t>) {
    Kokkos::single(Kokkos::PerTeam(tm), func);
  }
}

template <class F>
KOKKOS_FORCEINLINE_FUNCTION void sequential_loop(const int il, const int iu,
                                                 const F &func) {
  for (int i = il; i <= iu; ++i)
    func(i);
}

template <class tm_t, class F>
KOKKOS_FORCEINLINE_FUNCTION void parallel_loop(tm_t tm, const int il, const int iu,
                                               const F &func) {
  check_execution_handle<tm_t>();
  if constexpr (std::is_same_v<tm_t, serial_tm_t>) {
    for (int i = il; i <= iu; ++i)
      func(i);
  } else if constexpr (std::is_same_v<tm_t, parthenon::team_mbr_t>) {
    parthenon::par_for_inner(tm, il, iu, func);
  }
}

template <class tm_t, class F>
KOKKOS_FORCEINLINE_FUNCTION void parallel_loop(tm_t tm, const int jl, const int ju,
                                               const int il, const int iu,
                                               const F &func) {
  check_execution_handle<tm_t>();
  if constexpr (std::is_same_v<tm_t, serial_tm_t>) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        func(j, i);
      }
    }
  } else if constexpr (std::is_same_v<tm_t, parthenon::team_mbr_t>) {
    parthenon::par_for_inner(tm, jl, ju, il, iu, func);
  }
}

template <class tm_t, class F>
KOKKOS_FORCEINLINE_FUNCTION void summation(tm_t tm, const int il, const int iu,
                                           const F &func, double &sum) {
  check_execution_handle<tm_t>();
  if constexpr (std::is_same_v<tm_t, serial_tm_t>) {
    for (int i = il; i <= iu; ++i)
      func(i, sum);
  } else if constexpr (std::is_same_v<tm_t, parthenon::team_mbr_t>) {
    parthenon::par_reduce_inner(parthenon::inner_loop_pattern_ttr_tag, tm, il, iu, func,
                                Kokkos::Sum<double>(sum));
  }
}

template <class tm_t, class F>
KOKKOS_FORCEINLINE_FUNCTION void find_maximum(tm_t tm, const int il, const int iu,
                                              const F &func, double &mx) {
  check_execution_handle<tm_t>();
  if constexpr (std::is_same_v<tm_t, serial_tm_t>) {
    for (int i = il; i <= iu; ++i)
      func(i, mx);
  } else if constexpr (std::is_same_v<tm_t, parthenon::team_mbr_t>) {
    parthenon::par_reduce_inner(parthenon::inner_loop_pattern_ttr_tag, tm, il, iu, func,
                                Kokkos::Max<double>(mx));
  }
}

template <class tm_t, class F>
KOKKOS_FORCEINLINE_FUNCTION void summation(tm_t tm, const int jl, const int ju,
                                           const int il, const int iu, const F &func,
                                           double *sum) {
  check_execution_handle<tm_t>();
  if constexpr (std::is_same_v<tm_t, serial_tm_t>) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        func(j, i, sum);
      }
    }
  } else if constexpr (std::is_same_v<tm_t, parthenon::team_mbr_t>) {
    PARTHENON_FAIL("2D summation is not yet implemented for team execution.");
  }
}

} // namespace batched_linear_algebra
} // namespace parthenon

#endif // BATCHED_LINEAR_ALGEBRA_EXECUTION_UTILS_HPP_
