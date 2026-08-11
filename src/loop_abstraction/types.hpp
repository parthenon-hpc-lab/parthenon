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
#ifndef LOOP_ABSTRACTION_TYPES_HPP_
#define LOOP_ABSTRACTION_TYPES_HPP_

// This file was made in part with generative AI.

// Foundational, dependency-free vocabulary for the loop abstraction: the loop and
// inner tag enums that select a loop shape at compile time, the backend enum, the
// small logical (Index3) and memory (MemoryOffset) offset types, and the
// call-signature traits used to distinguish body signatures.

#include <algorithm>
#include <array>
#include <concepts>
#include <cstdint>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "basic_types.hpp"
#include "interface/mesh_data.hpp"
#include "kokkos_types.hpp"
#include "mesh/mesh.hpp"
#include "utils/concepts_lite.hpp"
#include "utils/indexer.hpp"

namespace parthenon::loop_abstraction {

using device_team_member_t =
    typename Kokkos::TeamPolicy<parthenon::DevExecSpace>::member_type;

// Selects the loop-abstraction backend at compile time. raw is a plain host loop
// nest; kokkos dispatches through Kokkos parallel policies.
enum class loop_backend { raw, kokkos };

inline constexpr loop_backend default_loop_backend_v =
    std::is_same_v<parthenon::DevExecSpace, parthenon::HostExecSpace>
        ? loop_backend::raw
        : loop_backend::kokkos;

// Loop shape: position of the variable (v) loop relative to the outer (o) and inner
// (i) loops within the (block, k, j, i) hierarchy. See LOOP_ABSTRACTION_CONTRACTS.md.
enum class loop_tag { bvoi, bovi, boiv };

// How the innermost chunk is traversed: flat/coordinate logical-cell coverage, or a
// contiguous memory span (which may touch ghost cells). See the contract document.
enum class inner_tag { logical_flat, logical_coords, memory };

// A logical (k, j, i) offset or point. Supports the small affine arithmetic used to
// express stencil/halo shifts in logical index space.
struct Index3 {
  int k, j, i;
  KOKKOS_DEFAULTED_FUNCTION
  constexpr Index3() = default;

  KOKKOS_INLINE_FUNCTION
  constexpr Index3(int k_, int j_, int i_) : k(k_), j(j_), i(i_) {}

  KOKKOS_INLINE_FUNCTION
  explicit constexpr Index3(const std::tuple<int, int, int> &t)
      : k(std::get<0>(t)), j(std::get<1>(t)), i(std::get<2>(t)) {}
};

KOKKOS_INLINE_FUNCTION
constexpr Index3 operator+(Index3 a, Index3 b) {
  return {a.k + b.k, a.j + b.j, a.i + b.i};
}

KOKKOS_INLINE_FUNCTION
constexpr Index3 operator-(Index3 a, Index3 b) {
  return {a.k - b.k, a.j - b.j, a.i - b.i};
}

KOKKOS_INLINE_FUNCTION
constexpr Index3 operator-(Index3 a) { return {-a.k, -a.j, -a.i}; }

KOKKOS_INLINE_FUNCTION
constexpr Index3 operator*(int n, Index3 a) { return {n * a.k, n * a.j, n * a.i}; }

KOKKOS_INLINE_FUNCTION
constexpr Index3 operator*(Index3 a, int n) { return n * a; }

// A memory-space offset. Carries both the (dk, dj, di) components and the
// precomputed flat displacement; implicitly converts to the flat int for direct
// pointer/view indexing.
struct MemoryOffset {
  int dk = 0;
  int dj = 0;
  int di = 0;
  int flat = 0;

  KOKKOS_INLINE_FUNCTION constexpr operator int() const { return flat; }
};

KOKKOS_INLINE_FUNCTION
constexpr MemoryOffset operator+(MemoryOffset a, MemoryOffset b) {
  return {a.dk + b.dk, a.dj + b.dj, a.di + b.di, a.flat + b.flat};
}

KOKKOS_INLINE_FUNCTION
constexpr MemoryOffset operator-(MemoryOffset a, MemoryOffset b) {
  return {a.dk - b.dk, a.dj - b.dj, a.di - b.di, a.flat - b.flat};
}

KOKKOS_INLINE_FUNCTION
constexpr MemoryOffset operator-(MemoryOffset a) {
  return {-a.dk, -a.dj, -a.di, -a.flat};
}

KOKKOS_INLINE_FUNCTION
constexpr MemoryOffset operator*(int n, MemoryOffset a) {
  return {n * a.dk, n * a.dj, n * a.di, n * a.flat};
}

KOKKOS_INLINE_FUNCTION
constexpr MemoryOffset operator*(MemoryOffset a, int n) { return n * a; }

namespace impl {
// Threaded from outer_reduce into inner_reduce (see kokkos.hpp). member is null in the
// boiv (flat RangePolicy) case; otherwise it is the enclosing team. update points at the
// reduce's accumulator (per-team for bvoi/bovi, per-work-item for boiv). reducer is
// carried by value so inner_reduce reuses its join() without the caller restating it;
// Kokkos reducers are cheap, trivially copyable value types. The definition is
// backend-agnostic (templated on the reducer), but reductions themselves are Kokkos-only.
template <class Reducer>
struct ReduceHandle {
  using reducer_type = Reducer;
  using value_type = typename Reducer::value_type;
  const device_team_member_t *member = nullptr;
  value_type *update = nullptr;
  Reducer reducer;
};

// Traits that detect whether a body's operator() takes a single explicit `int`
// argument. Used to reject explicit-int bodies where they would lose halo offset
// coordinates (see the boiv/logical_flat static_asserts in the backends).
template <class>
struct ExplicitUnaryIntCall : std::false_type {};

template <class R, class C, class Arg>
struct ExplicitUnaryIntCall<R (C::*)(Arg)> : std::is_same<std::remove_cvref_t<Arg>, int> {
};

template <class R, class C, class Arg>
struct ExplicitUnaryIntCall<R (C::*)(Arg) const>
    : std::is_same<std::remove_cvref_t<Arg>, int> {};

template <class F, class = void>
struct HasExplicitUnaryIntCall : std::false_type {};

template <class F>
struct HasExplicitUnaryIntCall<
    F, std::void_t<decltype(&std::remove_reference_t<F>::operator())>>
    : ExplicitUnaryIntCall<decltype(&std::remove_reference_t<F>::operator())> {};

template <class F>
inline constexpr bool has_explicit_unary_int_call_v = HasExplicitUnaryIntCall<F>::value;

} // namespace impl

// Bundles the three types involved in a reduction, keyed on the Kokkos reducer (the type
// of the instance passed last to outer_reduce, e.g. Kokkos::Min<Real>). Declare one alias
// at the top of a reduction kernel and name the pieces from it -- this keeps the
// KOKKOS_LAMBDA parameter types readable, and they *must* be named rather than `auto`
// because an extended __host__ __device__ lambda cannot take an `auto` parameter under
// nvcc (the inner_reduce body lambdas are ordinary lambdas, so their `auto` is fine):
//   using reduce_t = Reduction<Kokkos::Min<Real>>;
//   reduce_t::value_t result = 0.0;
//   outer_reduce(idx_space,
//     KOKKOS_LAMBDA(const idx_space_t::idx_range_t &r, int b,
//                   const reduce_t::handle_t &handle) {
//       inner_reduce(r, handle, [&](auto idx, reduce_t::value_t &v) { ... });
//     },
//     reduce_t::reducer_t(result));
template <class Reducer>
struct Reduction {
  using reducer_t = Reducer;                    // the Kokkos reducer, e.g. Kokkos::Sum
  using value_t = typename Reducer::value_type; // the reduced value / host result type
  using handle_t = impl::ReduceHandle<Reducer>; // threaded into the outer_reduce body
};

} // namespace parthenon::loop_abstraction

#endif // LOOP_ABSTRACTION_TYPES_HPP_
