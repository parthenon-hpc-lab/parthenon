//========================================================================================
// (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
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
#ifndef UTILS_CONCEPTS_LITE_HPP_
#define UTILS_CONCEPTS_LITE_HPP_

#include <type_traits>
#include <utility>

// This is a class template that is required for doing something like static_assert(false)
// in constexpr if blocks. Actually writing static_assert(false) will always cause a
// compilation error, even if it is an unchosen constexpr if block. This is fixed in C++23
// I think.
template <class...>
constexpr std::false_type always_false{};

// Include a useful type trait for checking if a type is a specialization of
// a template. Only works if all template arguments are types
template <class SPECIAL, template <class...> class TEMPL>
struct is_specialization_of : public std::false_type {};

template <template <class...> class TEMPL, class... TPARAMS>
struct is_specialization_of<TEMPL<TPARAMS...>, TEMPL> : public std::true_type {};

// This is a variadic template class that accepts any set of types
// and is always equal to void as long as the types are well formed.
// Although it seems simple, it is the basis of the SFINAE "void_t
// trick" from Walter Brown. Probably just easiest to google it for
// a better description than I can give, there are some nice talks
// by Walter Brown about it on YouTube.

template <class... Ts>
using void_t = void;

template <class F, class = void>
struct is_functor : std::false_type {};

template <class F>
struct is_functor<F, void_t<decltype(&F::operator())>> : std::true_type {};

template <typename T>
concept FundamentalCArray = std::is_fundamental_v<std::remove_extent_t<T>> &&
                            std::is_array_v<T> && (std::rank_v<T> == 1);

// Concept for a general container, not necessarily with
// contiguous data storage

template <typename T>
concept Container = requires(T x) {
  x.size();
  typename T::value_type;
};

template <typename T>
concept ContiguousContainer_arr = requires(T x) {
  x.size();
  x.data();
  typename T::value_type;
};

template <typename T>
concept ContiguousContainer_scalar = std::is_fundamental<T>::value && !Container<T>;

template <typename T>
concept ContiguousContainer = ContiguousContainer_scalar<T> || ContiguousContainer_arr<T>;

// Below are helper functions and types for treating both
// contiguous containers and single objects as contiguous
// containers. Note that this should fail for objects that
// are containers but not contiguous_containers, since there
// isn't a (easy) way to treat them as contiguous
struct contiguous_container {
  template <class T>
    requires(ContiguousContainer_arr<T>)
  static std::size_t size(const T &x) {
    return x.size();
  }

  template <class T>
    requires(ContiguousContainer_scalar<T>)
  static std::size_t size(const T &x) {
    return 1;
  }

  template <class T>
    requires(FundamentalCArray<T>)
  static std::size_t size(const T &) {
    return std::extent_v<T>;
  }

  template <class T>
    requires(ContiguousContainer_arr<T>)
  static typename T::value_type *data(T &x) {
    return x.data();
  }

  template <class T>
    requires(ContiguousContainer_scalar<T>)
  static T *data(T &x) {
    return &x;
  }

  template <class T>
    requires(FundamentalCArray<T>)
  static std::remove_extent_t<T> *data(T &x) {
    return x;
  }

  template <class T>
    requires(ContiguousContainer_arr<T>)
  static typename T::value_type value_type(const T &);

  template <class T>
    requires(ContiguousContainer_scalar<T>)
  static T value_type(T &);

  template <class T>
    requires(FundamentalCArray<T>)
  static std::remove_extent_t<T> value_type(const T &);
};

template <typename T>
concept Integral = std::is_integral<T>::value;

template <typename T>
concept IntegralOrEnum = std::is_integral<T>::value || std::is_enum<T>::value;

template <typename T>
concept Scalar = std::is_scalar<T>::value;

template <typename>
struct is_pair : std::false_type {};

template <typename T, typename U>
struct is_pair<std::pair<T, U>> : std::true_type {};

template <typename T>
concept IntegralOrEnumOrPair =
    std::is_integral<T>::value || std::is_enum<T>::value || is_pair<T>::value;

template <typename T>
concept KokkosView = ContiguousContainer<T> && requires {
  typename T::host_mirror_type;
  typename T::execution_space;
  typename T::memory_space;
  typename T::device_type;
  typename T::memory_traits;
  typename T::host_mirror_space;
};

//---------------------------------------------------------
// Templates for dealing with template packs
//---------------------------------------------------------

// Multiply a list of integer template parameters
template <int... IN>
struct multiply;

template <>
struct multiply<> : std::integral_constant<std::size_t, 1> {};

template <int I0, int... IN>
struct multiply<I0, IN...> : std::integral_constant<int, I0 * multiply<IN...>::value> {};

// GetTypeIdx is taken from Stack Overflow 26169198, should cause compile time failure if
// type is not in list
template <class T, class... Ts>
struct GetTypeIdx;

template <class T, class... Ts>
struct GetTypeIdx<T, T, Ts...> : std::integral_constant<std::size_t, 0> {};

template <class T, class U, class... Ts>
struct GetTypeIdx<T, U, Ts...>
    : std::integral_constant<std::size_t, 1 + GetTypeIdx<T, Ts...>::value> {};

// Check if the typelist Ts... includes the type T
template <class T, class... Ts>
struct IncludesType;

template <class T>
struct IncludesType<T, T> : std::true_type {};

template <class T, class... Ts>
struct IncludesType<T, T, Ts...> : std::true_type {};

template <class T, class U>
struct IncludesType<T, U> : std::false_type {};

template <class T, class U, class... Ts>
struct IncludesType<T, U, Ts...> : IncludesType<T, Ts...> {};

template <class T, class = void>
struct UnderlyingType {
  using type = T;
};

template <class T>
  requires(std::is_enum_v<T>)
struct UnderlyingType<T> : std::underlying_type<T> {};

#endif // UTILS_CONCEPTS_LITE_HPP_
