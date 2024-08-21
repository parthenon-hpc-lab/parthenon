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

#ifndef UTILS_TYPE_LIST_HPP_
#define UTILS_TYPE_LIST_HPP_

#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace parthenon {

// Convenience struct for holding a variadic pack of types
// and providing compile time indexing into that pack as
// well as the ability to get the index of a given type within
// the pack. Functions are available below for compile time
// concatenation of TypeLists
template <class... Args>
struct TypeList {
  using types = std::tuple<Args...>;

  static constexpr std::size_t n_types{sizeof...(Args)};

  template <std::size_t I>
  using type = typename std::tuple_element<I, types>::type;

  template <std::size_t... Idxs>
  using sublist = TypeList<type<Idxs>...>;

  template <class T, std::size_t I = 0>
  static constexpr std::size_t GetIdx() {
    static_assert(I < n_types, "Type is not present in TypeList.");
    if constexpr (std::is_same_v<T, type<I>>) {
      return I;
    } else {
      return GetIdx<T, I + 1>();
    }
  }

  template <class F>
  static void IterateTypes(F func) {
    (func(Args()), ...);
  }

  template <std::size_t Start, std::size_t End>
  static auto ContinuousSublistImpl() {
    return ContinuousSublistImpl<Start>(std::make_index_sequence<End - Start + 1>());
  }
  template <std::size_t Start, std::size_t... Is>
  static auto ContinuousSublistImpl(std::index_sequence<Is...>) {
    return sublist<(Start + Is)...>();
  }

  template <std::size_t Start, std::size_t End = sizeof...(Args) - 1>
  using continuous_sublist = decltype(ContinuousSublistImpl<Start, End>());
};

namespace impl {
template <class... Args>
auto ConcatenateTypeLists(TypeList<Args...>) {
  return TypeList<Args...>();
}

template <class... Args1, class... Args2, class... Args>
auto ConcatenateTypeLists(TypeList<Args1...>, TypeList<Args2...>, Args...) {
  return ConcatenateTypeLists(TypeList<Args1..., Args2...>(), Args()...);
}
} // namespace impl

template <class... TLs>
using concatenate_type_lists_t =
    decltype(impl::ConcatenateTypeLists(std::declval<TLs>()...));

// Relevant only for lists of variable types
template <class TL>
auto GetNames() {
  std::vector<std::string> names;
  TL::IterateTypes([&names](auto t) { names.push_back(decltype(t)::name()); });
  return names;
}

namespace impl {
template <class TL, int cidx>
static constexpr int FirstNonIntegralImpl() {
  if constexpr (cidx == TL::n_types) {
    return TL::n_types;
  } else {
    if constexpr (std::is_integral_v<typename std::remove_reference<
                      typename TL::template type<cidx>>::type>)
      return FirstNonIntegralImpl<TL, cidx + 1>();
    return cidx;
  }
}
} // namespace impl

template <class TL>
static constexpr int FirstNonIntegralIdx() {
  return impl::FirstNonIntegralImpl<TL, 0>();
}

template <class F, class = void>
struct is_functor : std::false_type {};

template <class F>
struct is_functor<F, void_t<decltype(&F::operator())>> : std::true_type {};

template <class TL, int idx = 0>
constexpr int FirstFuncIdx() {
  if constexpr (idx == TL::n_types) {
    return TL::n_types;
  } else {
    using cur_type = typename TL::template type<idx>;
    if constexpr (is_functor<cur_type>::value) return idx;
    if constexpr (std::is_function<std::remove_pointer<cur_type>>::value) return idx;
    return FirstFuncIdx<TL, idx + 1>();
  }
}

template <class Function>
struct FuncSignature;

template <class Functor>
struct FuncSignature : public FuncSignature<decltype(&Functor::operator())> {};

template <class R, class... Args>
struct FuncSignature<R(Args...)> {
  using type = R(Args...);
  using arg_types_tl = TypeList<Args...>;
  using ret_type = R;
};

template <class R, class T, class... Args>
struct FuncSignature<R (T::*)(Args...) const> {
  using type = R (T::*)(Args...);
  using arg_types_tl = TypeList<Args...>;
  using ret_type = R;
};

} // namespace parthenon

#endif // UTILS_TYPE_LIST_HPP_
