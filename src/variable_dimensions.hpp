//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
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
#ifndef VARIABLE_DIMENSIONS_HPP_
#define VARIABLE_DIMENSIONS_HPP_

#include <type_traits>

#include "basic_types.hpp"

#define MAX_VARIABLE_DIMENSION 7

namespace parthenon {

template <class T, int N>
struct multi_pointer : multi_pointer<std::add_pointer_t<T>, N - 1> {};

template <class T>
struct multi_pointer<T, 0> {
  using type = T;
};

template <class T, int N = MAX_VARIABLE_DIMENSION>
using multi_pointer_t = typename multi_pointer<T, N>::type;

} // namespace parthenon

#endif // VARIABLE_DIMENSIONS_HPP_