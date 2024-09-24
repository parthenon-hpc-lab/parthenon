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

#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

#include "basic_types.hpp"
#include "utils/concepts_lite.hpp"
#include "utils/type_list.hpp"

namespace parthenon {

// struct that can be specialized to register new types that can be processed to obtain
// loop bounds in a par_for* loop
template <typename Bound, typename T = void>
struct ProcessLoopBound : std::false_type {
  template <typename Bnd0, typename... Bnds>
  static constexpr std::size_t GetNumBounds(TypeList<Bnd0, Bnds...>) {
    static_assert(always_false<Bound, Bnds...>, "Invalid loop bound type");
    return 0;
  }

  template <std::size_t N, typename... Bnds>
  KOKKOS_INLINE_FUNCTION static void
  GetIndexRanges(const int &idx, Kokkos::Array<IndexRange, N> &bound_arr, Bound &bound,
                 Bnds &&...bounds) {
    static_assert(always_false<Bound, Bnds...>, "Invalid loop bound type");
  }
};

namespace LoopBounds {
template <typename... Bnds>
constexpr std::size_t GetNumBounds(TypeList<Bnds...>) {
  if constexpr (sizeof...(Bnds) > 0) {
    using TL = TypeList<Bnds...>;
    using NextBound = ProcessLoopBound<base_type<typename TL::template type<0>>>;
    static_assert(NextBound::value, "unrecognized loop bound");
    return NextBound::GetNumBounds(TL());
  }
  return 0;
}

template <std::size_t N, typename... Bnds>
KOKKOS_INLINE_FUNCTION void GetIndexRanges(const int &idx,
                                           Kokkos::Array<IndexRange, N> &bound_arr,
                                           Bnds &&...bounds) {
  if constexpr (sizeof...(Bnds) > 0) {
    using TL = TypeList<Bnds...>;
    using NextBound = typename TypeList<Bnds...>::template type<0>;
    ProcessLoopBound<base_type<NextBound>>::GetIndexRanges(idx, bound_arr,
                                                           std::forward<Bnds>(bounds)...);
  }
}
} // namespace LoopBounds

template <typename Bound>
struct ProcessLoopBound<Bound, std::enable_if_t<std::is_integral_v<Bound>>>
    : std::true_type {

  template <typename Bnd0, typename Bnd1, typename... Bnds>
  static constexpr std::size_t GetNumBounds(TypeList<Bnd0, Bnd1, Bnds...>) {
    static_assert(std::is_integral_v<base_type<Bnd0>> &&
                      std::is_integral_v<base_type<Bnd1>>,
                  "Integer bounds must come in pairs");

    return 1 + LoopBounds::GetNumBounds(TypeList<Bnds...>());
  }

  template <std::size_t N, typename... Bnds>
  KOKKOS_INLINE_FUNCTION static void
  GetIndexRanges(const int &idx, Kokkos::Array<IndexRange, N> &bound_arr, const int &s,
                 const int &e, Bnds &&...bounds) {
    bound_arr[idx].s = s;
    bound_arr[idx].e = e;
    LoopBounds::GetIndexRanges(idx + 1, bound_arr, std::forward<Bnds>(bounds)...);
  }
};

template <>
struct ProcessLoopBound<IndexRange> : std::true_type {
  template <typename T>
  using isIdRng = std::is_same<T, IndexRange>;

  template <typename Bnd0, typename... Bnds>
  static constexpr std::size_t GetNumBounds(TypeList<Bnd0, Bnds...>) {
    static_assert(std::is_same_v<base_type<Bnd0>, IndexRange>,
                  "expected IndexRange loop bound");

    return 1 + LoopBounds::GetNumBounds(TypeList<Bnds...>());
  }

  template <std::size_t N, typename... Bnds>
  KOKKOS_INLINE_FUNCTION static void
  GetIndexRanges(const int &idx, Kokkos::Array<IndexRange, N> &bound_arr,
                 const IndexRange &idr, Bnds &&...bounds) {
    bound_arr[idx] = idr;
    LoopBounds::GetIndexRanges(idx + 1, bound_arr, std::forward<Bnds>(bounds)...);
  }
};

// Struct for translating between loop bounds given to par_dispatch into an array of
// IndexRanges
template <class... Bound_ts>
struct LoopBoundTranslator {
 public:
  // make sure all the Bound_ts... types are valid loop bounds and count the number of
  // bounds contained in each type
  static constexpr std::size_t Rank = LoopBounds::GetNumBounds(TypeList<Bound_ts...>());

  // process all of the loop bounds into an array of IndexRanges
  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<IndexRange, Rank> GetIndexRanges(Bound_ts &&...bounds) {
    Kokkos::Array<IndexRange, Rank> bound_arr;
    LoopBounds::GetIndexRanges(0, bound_arr, std::forward<Bound_ts>(bounds)...);
    return bound_arr;
  }
};

template <class... Bound_ts>
struct LoopBoundTranslator<TypeList<Bound_ts...>>
    : public LoopBoundTranslator<Bound_ts...> {};

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
