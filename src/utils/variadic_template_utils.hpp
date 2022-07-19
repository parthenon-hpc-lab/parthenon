//========================================================================================
// (C) (or copyright) 2022. Triad National Security, LLC. All rights reserved.
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
#ifndef UTILS_VARIADIC_TEMPLATE_UTILS_HPP_
#define UTILS_VARIADIC_TEMPLATE_UTILS_HPP_
//! \file variadic_template_utils.hpp
//  \brief Some template metafunctions for working with type lists and template parameter 
//  lists 

#include <type_traits> 

namespace parthenon {

template <int... IN>
struct multiply;

template <>
struct multiply<> : std::integral_constant<std::size_t, 1> {};

template <int I0, int... IN>
struct multiply<I0, IN...> : std::integral_constant<int, I0 * multiply<IN...>::value> {};

// GetTypeIdx is taken from Stack Overflow 26169198, should cause compile time failure if
// type is not in list
template <typename T, typename... Ts>
struct GetTypeIdx;

template <typename T, typename... Ts>
struct GetTypeIdx<T, T, Ts...> : std::integral_constant<std::size_t, 0> {
  using type = void;
};

template <typename T, typename U, typename... Ts>
struct GetTypeIdx<T, U, Ts...>
    : std::integral_constant<std::size_t, 1 + GetTypeIdx<T, Ts...>::value> {
  using type = void;
};

template <class T, class... Ts>
struct IncludesType;

template <typename T>
struct IncludesType<T, T> : std::true_type {};

template <typename T, typename... Ts>
struct IncludesType<T, T, Ts...> : std::true_type {};

template <typename T, typename U>
struct IncludesType<T, U> : std::false_type {};

template <typename T, typename U, typename... Ts>
struct IncludesType<T, U, Ts...> : IncludesType<T, Ts...> {};

} // namespace parthenon

#endif // UTILS_VARIADIC_TEMPLATE_UTILS_HPP_
