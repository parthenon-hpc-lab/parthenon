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
#ifndef INTERFACE_PARAMS_HPP_
#define INTERFACE_PARAMS_HPP_

#include <string>
#include <utility>

#include "utils/dict_anytype.hpp"

namespace parthenon {

/// Defines a class that can be used to hold parameters
/// of any kind
class Params {
 public:
  Params() {}

  // can't copy because we have a map of unique_ptr
  Params(const Params &p) = delete;

  /// Adds object based on type of the value
  ///
  /// Throws an error if the key is already in use
  template <typename T>
  void Add(const std::string &key, T value) {
    myParams_.Add(key, value);
  }

  void reset() { myParams_.reset(); }

  template <typename T, typename... Args>
  const T &Get(Args... args) {
    return myParams_.Get<T>(std::forward<Args>(args)...);
  }

  bool hasKey(const std::string &key) const { return myParams.hasKey(key); }

  // void Params::
  void list() { myParams_.list(); }

 private:
  DictAnyType myParams_;
} // namespace parthenon

#endif // INTERFACE_PARAMS_HPP_
