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

  template <std::size_t Start, std::size_t End = n_types - 1>
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

template <class T, std::size_t I, class... Ts>
static auto InsertTypeImpl(TypeList<Ts...>) {
  if constexpr (I == 0) {
    return TypeList<T, Ts...>();
  } else if constexpr (I == sizeof...(Ts)) {
    return TypeList<Ts..., T>();
  } else {
    using TL = TypeList<Ts...>;
    return ConcatenateTypeLists(typename TL::template continuous_sublist<0, I>(),
                                TypeList<T>(),
                                typename TL::template continuous_sublist<I + 1>());
  }
}
} // namespace impl

template <class... TLs>
using concatenate_type_lists_t =
    decltype(impl::ConcatenateTypeLists(std::declval<TLs>()...));

template <class T, class TL, std::size_t I = TL::n_types>
using insert_type_list_t = decltype(impl::InsertTypeImpl<T, I>(TL()));

// Relevant only for lists of variable types
template <class TL>
auto GetNames() {
  std::vector<std::string> names;
  TL::IterateTypes([&names](auto t) { names.push_back(decltype(t)::name()); });
  return names;
}

namespace impl {

template <size_t N, class T>
auto ListOfType() {
  if constexpr (N == 1) {
    return TypeList<T>();
  } else {
    return concatenate_type_lists_t<TypeList<T>, decltype(ListOfType<N - 1, T>())>();
  }
}

template <size_t, size_t, typename>
struct SequenceOfInt {};

template <size_t VAL, size_t... ones>
struct SequenceOfInt<0, VAL, std::integer_sequence<size_t, ones...>> {
  using value = typename std::integer_sequence<size_t, ones...>;
};

template <size_t N, size_t VAL, size_t... ones>
struct SequenceOfInt<N, VAL, std::integer_sequence<size_t, ones...>> {
  using value =
      typename SequenceOfInt<N - 1, VAL,
                             std::integer_sequence<size_t, VAL, ones...>>::value;
};

} // namespace impl

template <size_t N, class T>
using list_of_type_t = decltype(impl::ListOfType<N, T>());

template <size_t N, size_t VAL = 1>
using sequence_of_int_v =
    typename impl::SequenceOfInt<N - 1, VAL, std::integer_sequence<size_t, VAL>>::value;

} // namespace parthenon

#endif // UTILS_TYPE_LIST_HPP_
