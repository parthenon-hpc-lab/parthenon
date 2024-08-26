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
#include <variant>
#include <vector>

#include "basic_types.hpp"
#include "concepts_lite.hpp"

namespace parthenon {

// c++-20 has std:remove_cvref_t that does this same thing
template <typename T>
using base_type = typename std::remove_cv_t<typename std::remove_reference_t<T>>;

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

  template <class T, std::size_t I = 0>
  static constexpr bool IsIn() {
    if constexpr (I == n_types) {
      return false;
    } else {
      if constexpr (std::is_same_v<T, type<I>>) {
        return true;
      } else {
        return IsIn<T, I + 1>();
      }
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

template <size_t N, class T>
auto ListOfType() {
  if constexpr (N == 1) {
    return TypeList<T>();
  } else {
    return concatenate_type_lists_t<TypeList<T>, decltype(ListOfType<N - 1, T>())>();
  }
}
} // namespace impl

template <size_t N, class T>
using list_of_type_t = decltype(impl::ListOfType<N, T>());

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
    using cur_type = base_type<typename TL::template type<idx>>;
    if constexpr (is_functor<cur_type>::value) return idx;
    if constexpr (std::is_function<std::remove_pointer<cur_type>>::value) return idx;
    return FirstFuncIdx<TL, idx + 1>();
  }
}

// Recognized bound types
// additional types should be translated in BoundTranslator (kokkos_abstraction.hpp)
template <typename Bound>
constexpr bool isBoundType() {
  using BoundTypes = TypeList<IndexRange>;
  using btype = base_type<Bound>;
  return std::is_same_v<IndexRange, btype> || std::is_integral_v<btype>;
}

template <typename... Bnds>
constexpr std::size_t GetNumBounds(TypeList<Bnds...>) {
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
  using IndexND = typename TL::template continuous_sublist<0, Rank + team_mbr - 1>;
  using FArgs = typename TL::template continuous_sublist<Rank + team_mbr>;
};

template <size_t Rank, typename F>
using function_signature = FunctionSignature<Rank, decltype(&base_type<F>::operator())>;
} // namespace parthenon

#endif // UTILS_TYPE_LIST_HPP_
