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

  // Templated version for retrieving a package with a particular type
  // Allows subclassing 'StateDescriptor' to add user package types to list
  template<typename T>
  T* const &Get(const std::string &name) {
    return static_cast<T*>(packages_.at(name).get());
  }

  const Dictionary<std::shared_ptr<StateDescriptor>> &AllPackages() const {
    return packages_;
  }

  // Returns a sub-Dictionary containing just pointers to packages of type T.
  // Dictionary is a *new copy*, and members are bare pointers, not shared_ptr.
  template <typename T>
  const Dictionary<T*> AllPackagesOfType() const {
    Dictionary<T*> sub_dict;
    for (auto package : packages_) {
      if (T *cast_package = dynamic_cast<T*>(package.second.get())) {
        sub_dict[package.first] = cast_package;
      }
    }
    return sub_dict;
  }

  // Returns a list of pointers to packages of type T.
  // List contains bare pointers, not shared_ptr objects
  template <typename T>
  const std::vector<T*> ListPackagesOfType() const {
    std::vector<T*> sub_list;
    for (auto package : packages_) {
      if (T *cast_package = dynamic_cast<T*>(package.second.get())) {
        sub_list.push_back(cast_package);
      }
    }
    return sub_list;
  }

 private:
  Dictionary<std::shared_ptr<StateDescriptor>> packages_;
};
} // namespace parthenon

#endif // INTERFACE_PACKAGES_HPP_
