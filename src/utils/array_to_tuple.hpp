//========================================================================================
// (C) (or copyright) 2023. Triad National Security, LLC. All rights reserved.
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
#ifndef UTILS_ARRAY_TO_TUPLE_HPP_
#define UTILS_ARRAY_TO_TUPLE_HPP_

#include <type_traits>

namespace parthenon {

template <class T, std::size_t... I>
auto ArrayToTuple(const T &arr_in, std::index_sequence<I...>) {
  return std::make_tuple((arr_in[I])...);
}

template <class T, std::size_t N>
auto ArrayToTuple(const std::array<T, N> &arr_in) {
  return ArrayToTuple(arr_in, std::make_index_sequence<N>());
}

template <class T, std::size_t... I>
auto ArrayToReverseTuple(const T &arr_in, std::index_sequence<I...>) {
  return std::make_tuple((arr_in[sizeof...(I) - I - 1])...);
}

template <class T, std::size_t N>
auto ArrayToReverseTuple(const std::array<T, N> &arr_in) {
  return ArrayToReverseTuple(arr_in, std::make_index_sequence<N>());
}

} // namespace parthenon

#endif // UTILS_ARRAY_TO_TUPLE_HPP_
