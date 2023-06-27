//========================================================================================
// (C) (or copyright) 2022. Triad National Security, LLC. All rights reserved.
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

#ifndef INTERFACE_PACKAGES_HPP_
#define INTERFACE_PACKAGES_HPP_

#include <memory>
#include <string>

#include "basic_types.hpp"

namespace parthenon {
class StateDescriptor; // forward declaration
class Packages_t {
 public:
  Packages_t() = default;
  void Add(const std::shared_ptr<StateDescriptor> &package);

  std::shared_ptr<StateDescriptor> const &Get(const std::string &name) {
    return packages_.at(name);
  }

  const Dictionary<std::shared_ptr<StateDescriptor>> &AllPackages() const {
    return packages_;
  }
  Dictionary<std::shared_ptr<StateDescriptor>> &AllPackages() { return packages_; }

 private:
  Dictionary<std::shared_ptr<StateDescriptor>> packages_;
};
} // namespace parthenon

#endif // INTERFACE_PACKAGES_HPP_
