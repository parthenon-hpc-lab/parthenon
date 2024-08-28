//========================================================================================
// (C) (or copyright) 2020-2024. Triad National Security, LLC. All rights reserved.
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

#ifndef LOOP_BOUNDS_HPP_
#define LOOP_BOUNDS_HPP_

#include <utility>

#include <Kokkos_Core.hpp>

#include "basic_types.hpp"
#include "utils/concepts_lite.hpp"
#include "utils/type_list.hpp"

namespace parthenon {

// Struct for translating between loop bounds given to par_dispatch into an array of
// IndexRanges
//
template <class... Bound_ts>
struct LoopBoundTranslator {
 private:
  using BoundTypes = TypeList<IndexRange>;
  // overloads for different launch bound types.
  template <typename... Bounds>
  KOKKOS_INLINE_FUNCTION void GetIndexRanges_impl(const int idx, const int s, const int e,
                                                  Bounds &&...bounds) {
    bound_arr[idx].s = s;
    bound_arr[idx].e = e;
    if constexpr (sizeof...(Bounds) > 0) {
      GetIndexRanges_impl(idx + 1, std::forward<Bounds>(bounds)...);
    }
  }
  template <typename... Bounds>
  KOKKOS_INLINE_FUNCTION void GetIndexRanges_impl(const int idx, const IndexRange ir,
                                                  Bounds &&...bounds) {
    bound_arr[idx] = ir;
    if constexpr (sizeof...(Bounds) > 0) {
      GetIndexRanges_impl(idx + 1, std::forward<Bounds>(bounds)...);
    }
  }

  using Bound_tl = TypeList<Bound_ts...>;

 public:
  template <typename Bound>
  static constexpr bool isBoundType() {
    using btype = base_type<Bound>;
    return std::is_same_v<IndexRange, btype> || std::is_integral_v<btype>;
  }

  template <typename... Bnds>
  static constexpr std::size_t GetNumBounds(TypeList<Bnds...>) {
    using TL = TypeList<Bnds...>;
    if constexpr (sizeof...(Bnds) == 0) {
      return 0;
    } else {
      using Bnd0 = typename TL::template type<0>;
      static_assert(isBoundType<Bnd0>(), "unrecognized launch bound in par_dispatch");
      if constexpr (std::is_same_v<base_type<Bnd0>, IndexRange>) {
        return 2 + GetNumBounds(typename TL::template continuous_sublist<1>());
      } else if constexpr (std::is_integral_v<base_type<Bnd0>>) {
        using Bnd1 = typename TL::template type<1>;
        static_assert(std::is_integral_v<base_type<Bnd1>>,
                      "integer launch bounds need to come in (start, end) pairs");
        return 2 + GetNumBounds(typename TL::template continuous_sublist<2>());
      }
    }
    // should never get here but makes older cuda compilers happy
    return 0;
  }
  static constexpr std::size_t Rank = GetNumBounds(Bound_tl()) / 2;
  Kokkos::Array<IndexRange, Rank> bound_arr;

  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<IndexRange, Rank> GetIndexRanges(Bound_ts &&...bounds) {
    GetIndexRanges_impl(0, std::forward<Bound_ts>(bounds)...);
    return bound_arr;
  }
};

template <class... Bound_ts>
struct LoopBoundTranslator<TypeList<Bound_ts...>>
    : public LoopBoundTranslator<Bound_ts...> {};

template <class F, class = void>
struct is_functor : std::false_type {};

template <class F>
struct is_functor<F, void_t<decltype(&F::operator())>> : std::true_type {};

template <class TL, int idx = 0>
constexpr int FirstFuncIdx() {
  if constexpr (idx == TL::n_types) {
    return TL::n_types;
  } else {
    using cur_type = base_type<typename TL::template type<idx>>;
    if constexpr (is_functor<cur_type>::value) return idx;
    if constexpr (std::is_function<std::remove_pointer<cur_type>>::value) return idx;
    return FirstFuncIdx<TL, idx + 1>();
  }
  // should never get here, but makes older cuda versions happy
  return TL::n_types;
}

template <size_t, typename>
struct FunctionSignature {};

template <size_t Rank, typename R, typename T, typename Arg0, typename... Args>
struct FunctionSignature<Rank, R (T::*)(Arg0, Args...) const> {
 private:
  using team_mbr_t = Kokkos::TeamPolicy<>::member_type;
  static constexpr bool team_mbr = std::is_same_v<team_mbr_t, base_type<Arg0>>;
  using TL = TypeList<Arg0, Args...>;

 public:
  using FArgs = typename TL::template continuous_sublist<Rank + team_mbr>;
};

template <size_t Rank, typename F>
using function_signature = FunctionSignature<Rank, decltype(&base_type<F>::operator())>;

} // namespace parthenon

#endif // LOOP_BOUNDS_HPP_
