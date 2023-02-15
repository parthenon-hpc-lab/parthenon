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

#ifndef UTILS_UNIQUE_ID_HPP_
#define UTILS_UNIQUE_ID_HPP_

#include <cstddef>

#include <unordered_map>

namespace parthenon {
template <typename T>
class UniqueIDGenerator {
 public:
  std::size_t operator()(const T &key) {
    if (uids_.count(key) > 0) {
      return uids_.at(key);
    }
    std::size_t uid = uids_.size();
    uids_.emplace(key, uid);
    return uid;
  }

 private:
  std::unordered_map<T, std::size_t> uids_;
};
} // namespace parthenon

#endif // UTILS_UNIQUE_ID_HPP_
