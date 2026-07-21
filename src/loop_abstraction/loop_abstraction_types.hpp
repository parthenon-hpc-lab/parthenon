//========================================================================================
// (C) (or copyright) 2024-2026. Triad National Security, LLC. All rights reserved.
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
#ifndef LOOP_ABSTRACTION_LOOP_ABSTRACTION_TYPES_HPP_
#define LOOP_ABSTRACTION_LOOP_ABSTRACTION_TYPES_HPP_

// This file was made in part with generative AI.

// Foundational, dependency-free vocabulary for the loop abstraction: the loop and
// inner tag enums that select a loop shape at compile time, the backend enum, the
// small logical (Index3) and memory (MemoryOffset) offset types, and the
// call-signature traits used to distinguish body signatures. Everything else in the
// abstraction builds on these types, so this header carries the external include
// block and has no intra-abstraction dependencies.

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

} // namespace parthenon::loop_abstraction

#endif // LOOP_ABSTRACTION_LOOP_ABSTRACTION_TYPES_HPP_
