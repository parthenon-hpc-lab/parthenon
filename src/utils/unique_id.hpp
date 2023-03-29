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
using Uid_t = std::size_t; // change this to something else if needed
constexpr Uid_t INVALID_UID = 0;
template <typename T>
class UniqueIDGenerator {
 public:
  Uid_t operator()(const T &key) {
    if (uids_.count(key) > 0) {
      return uids_.at(key);
    }
    // Ensure that all uids > 0, 0 reserved
    // as an invalid id
    Uid_t uid = uids_.size() + 1;
    uids_.emplace(key, uid);
    return uid;
  }

 private:
  std::unordered_map<T, Uid_t> uids_;
};
} // namespace parthenon

#endif // UTILS_UNIQUE_ID_HPP_
