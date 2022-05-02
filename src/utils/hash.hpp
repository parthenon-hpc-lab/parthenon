//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
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

#ifndef UTILS_HASH_HPP_
#define UTILS_HASH_HPP_

#include <functional>
#include <tuple>

namespace parthenon {
namespace impl {
template <class T>
std::size_t hash_combine(std::size_t lhs, const T &v) {
  std::size_t rhs = std::hash<T>()(v);
  // The boost hash combine function
  lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
  return lhs;
}

template <class Tup, std::size_t I = std::tuple_size<Tup>::value - 1>
struct TupHash {
  static std::size_t val(const Tup &tup, std::size_t seed = 0) {
    seed = TupHash<Tup, I - 1>::val(tup, seed);
    return hash_combine(seed, std::get<I>(tup));
  }
};

template <class Tup>
struct TupHash<Tup, 0> {
  static std::size_t val(const Tup &tup, std::size_t seed) {
    return hash_combine(seed, std::get<0>(tup));
  }
};
} // namespace impl

template <class T>
struct tuple_hash {
  using argument_type = T;
  using result_type = std::size_t;
  std::size_t operator()(const argument_type &tup) const {
    return parthenon::impl::TupHash<argument_type>::val(tup);
  }
};
} // namespace parthenon

#endif // UTILS_HASH_HPP_
