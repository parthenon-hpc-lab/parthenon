//========================================================================================
// (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
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

#ifndef UTILS_CLEANTYPES_HPP_
#define UTILS_CLEANTYPES_HPP_

namespace parthenon {
namespace cleantypes {

// From:
// https://stackoverflow.com/questions/9851594/standard-c11-way-to-remove-all-pointers-of-a-type
template <typename T>
struct remove_all_pointers {
  using type = T;
};

template <typename T>
struct remove_all_pointers<T *> {
  using type = typename remove_all_pointers<T>::type;
};

template <typename T>
struct remove_all_pointers<T *const> {
  using type = typename remove_all_pointers<T>::type;
};

template <typename T>
struct remove_all_pointers<T *volatile> {
  using type = typename remove_all_pointers<T>::type;
};

template <typename T>
struct remove_all_pointers<T *const volatile> {
  using type = typename remove_all_pointers<T>::type;
};

} // namespace cleantypes
} // namespace parthenon

#endif // UTILS_CLEANTYPES_HPP_
