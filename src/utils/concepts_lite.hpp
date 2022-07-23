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

// These macros are just to make code more readable and self-explanatory,
// generally it is best to write template<..., REQUIRES(... && ...)> in the code
// but there are some instance where this causes issues. Switching to the construct
// template<..., class = ENABLEIF(... && ...)> sometimes fixes those problems.
#define REQUIRES(...) typename std::enable_if<(__VA_ARGS__), int>::type = 0
#define ENABLEIF(...) typename std::enable_if<(__VA_ARGS__), int>::type
using TYPE_OF_SUCCESSFUL_REQUIRES = int;

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

// implements is a template struct for checking if type T implements a particular
// concept, which here simply means that it conforms to some interface.
// (I think people call this concepts lite, since there are more
// powerful things that concepts can do in C++20 and the compiler error
// messages from this type of implementation can be kind of crazy). This cleaner
// interface for using the void_t trick is partly inspired by Stackoverflow 26513095

// General template that is accepted if the specialization below
// is not well formed, inherits from false type so that implements<...>()
// will return false. The default parameter for the second template argument
// means that when you write implements<T>, the compiler inteprets this as
// implements<T, void> and then looks to see if there is a specialization that
// matches this pattern

template <class T, class = void>
struct implements : std::false_type {};

// all_implements just checks if all types in a parameter pack implement 
// a given concept
template <class T, class = void>
struct all_implement : std::false_type {};

// Specialization of implements that is chosen if all of the template
// arguments to void_t are well formed, since in that case void_t = void
// and this specialization matches the defined template pattern. Note that
// pattern Concept(Ts...) is interpreted as a function taking types Ts...
// and returning type Concept, but such a function doesn't get used for anything
// we are doing. It is just a clean way of deducing multiple types from a single
// input type Concept(Ts...) to the base template.

template <class Concept, class... Ts>
struct implements<Concept(Ts...), void_t<decltype(std::declval<Concept>().requires_(
                                      std::declval<Ts>()...))>> : std::true_type {};

template <class Concept, class... Ts>
struct all_implement<Concept(Ts...), void_t<decltype(std::declval<Concept>().requires_(
                                      std::declval<Ts>()))...>> : std::true_type {};

//---------------------------
// Various concepts are implemented below. The general useage of a
// concept would be:
//
// template<class T, REQUIRES(implements<my_concept(T)>::value)>
// void foo(T& in){ implementation when T conforms to my_concept}
//
// template<class T, REQUIRES(!implements<my_concept(T)>::value)>
// void foo(T& in){ implementation when T doesn't conform to my_concept}
//
// Strangely, for some compilers, replacing implements<my_concept(T)>::value
// with implements<my_concept(T)>() causes the code not to compile, so
// we use value everywhere even though it is slightly more verbose.
//---------------------------

// This trying to use c-style arrays in the concepts pattern below seems not
// to work for reasons I don't understand
template <class T, class = void>
struct is_fundamental_c_array : std::false_type {};
template <class T, std::size_t N>
struct is_fundamental_c_array<T[N], void_t<ENABLEIF(std::is_fundamental<T>::value)>>
    : std::true_type {};

// Concept for a general container, not necessarily with
// contiguous data storage
struct container {
  // Every concept needs a requires_ method declaration, no
  // implementation of requires_ is necessary though.
  // requires_ should be well formed if the object T matches the
  // concept of the struct (in this case a contiguous container).
  // We just use void_t here since it is a variadic template that
  // can accept any number of types and we don't care about the
  // actual return type of requires_. Of course for it to be specialized,
  // all of the type parameters should be well formed. Including
  // the typename in front of the member type is important, since if
  // it is not included T::value_type will not be interpreted as a
  // type even if exists in T, void_t<...> will not be well formed
  // since it only accepts types, the definition of requires_ will
  // fail silently, implements<contiguous_container(T)> will always
  // inherit from false_type (even if T is a contiguous container),
  // and you will be left wondering what the hell is going on.
  template <class T>
  auto requires_(T &&x) -> void_t<decltype(x.size()), typename T::value_type>;
};

// Concept defining the interface of a container with continuous
// storage. Also defines helper functions for treating single
// objects as contiguous containers of size one
struct contiguous_container {
  template <class T>
  auto requires_(T &&x)
      -> void_t<decltype(x.size()), decltype(x.data()), typename T::value_type>;

  // Below are helper functions and types for treating both
  // contiguous containers and single objects as contiguous
  // containers. Note that this should fail for objects that
  // are containers but not contiguous_containers, since there
  // isn't a (easy) way to treat them as contiguous
  template <class T, REQUIRES(implements<contiguous_container(T)>::value)>
  static std::size_t size(const T &x) {
    return x.size();
  }

  template <class T,
            REQUIRES(!implements<container(T)>::value && std::is_fundamental<T>::value)>
  static std::size_t size(const T &x) {
    return 1;
  }

  template <class T, std::size_t N, REQUIRES(is_fundamental_c_array<T[N]>::value)>
  static std::size_t size(const T (&)[N]) {
    return N;
  }

  template <class T, REQUIRES(implements<contiguous_container(T)>::value)>
  static typename T::value_type *data(T &x) {
    return x.data();
  }

  template <class T,
            REQUIRES(!implements<container(T)>::value && std::is_fundamental<T>::value)>
  static T *data(T &x) {
    return &x;
  }

  template <class T, std::size_t N, REQUIRES(is_fundamental_c_array<T[N]>::value)>
  static T *data(T (&x)[N]) {
    return x;
  }

  template <class T, REQUIRES(implements<contiguous_container(T)>::value)>
  static typename T::value_type value_type(T &);

  template <class T,
            REQUIRES(!implements<container(T)>::value && std::is_fundamental<T>::value)>
  static T value_type(T &);

  template <class T, std::size_t N, REQUIRES(is_fundamental_c_array<T[N]>::value)>
  static T value_type(T (&)[N]);
};

struct integral {
  template <class T>
  auto requires_(T) -> void_t<ENABLEIF(std::is_integral<T>::value)>;
};

struct kokkos_view {
  template <class T>
  auto requires_(T x) -> void_t<ENABLEIF(implements<contiguous_container(T)>::value),
                                typename T::HostMirror, typename T::execution_space,
                                typename T::memory_space, typename T::device_type,
                                typename T::memory_traits, typename T::host_mirror_space>;
};

#endif // UTILS_CONCEPTS_LITE_HPP_
