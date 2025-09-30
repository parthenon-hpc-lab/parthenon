//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2022 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2022-2024. Triad National Security, LLC. All rights reserved.
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
#include <memory>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

namespace parthenon {

// A hash struct that can be used as a template class in
// std::unordered_map, etc. to
// hash a tuple by hashing each of its elements then combining the
// hashes into a single hash using hash_combine. May or may not be
// optimal way of hashing, but it certainly works.
namespace impl {
inline void boost_hash_combine(std::size_t &lhs, std::size_t rhs) {
  lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
}

template <class... Ts>
inline std::size_t hash_combine(const Ts &...ts) {
  std::size_t h{0};
  (boost_hash_combine(h, std::hash<Ts>()(ts)), ...);
  return h;
}
} // namespace impl
} // namespace parthenon

template <class T, class _Alloc>
struct std::hash<std::vector<T, _Alloc>> {
  std::size_t operator()(const std::vector<T, _Alloc> &vec) const {
    std::size_t h{0};
    for (auto &&t : vec)
      parthenon::impl::boost_hash_combine(h, hash<T>()(t));
    return h;
  }
};

template <class T, int N>
struct std::hash<std::array<T, N>> {
  std::size_t operator()(const std::array<T, N> &arr) const {
    std::size_t h{0};
    for (auto &&t : arr)
      parthenon::impl::boost_hash_combine(h, hash<T>()(t));
    return h;
  }
};

template <class T>
struct std::hash<std::set<T>> {
  std::size_t operator()(const std::set<T> &vec) const {
    std::size_t h{0};
    for (auto &&t : vec)
      parthenon::impl::boost_hash_combine(h, hash<T>()(t));
    return h;
  }
};

template <class... Ts>
struct std::hash<std::tuple<Ts...>> {
  std::size_t operator()(const std::tuple<Ts...> &tup) const {
    return std::apply(parthenon::impl::hash_combine<Ts...>, tup);
  }
};

namespace parthenon {
template <class T>
struct WeakPtrHash {
  std::size_t operator()(const std::weak_ptr<T> &wp) const {
    if (auto sp = wp.lock()) {
      return std::hash<std::shared_ptr<T>>()(sp);
    }
    return 0;
  }
};

template <class T>
struct WeakPtrEqual {
  bool operator()(const std::weak_ptr<T> &lhs, const std::weak_ptr<T> &rhs) const {
    return !lhs.owner_before(rhs) && !rhs.owner_before(lhs);
  }
};

// This is just here for backward compatibility
template <class T>
struct tuple_hash {
  std::size_t operator()(const T &tup) const { return std::hash<T>()(tup); }
};
} // namespace parthenon

#endif // UTILS_HASH_HPP_
