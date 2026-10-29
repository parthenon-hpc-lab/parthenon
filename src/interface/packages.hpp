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

#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "basic_types.hpp"
#include "utils/callback_order.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {
class StateDescriptor; // forward declaration
// Compiled once, in state_descriptor.cpp
extern template class CallbackOrder<StateDescriptor>;

class Packages_t {
 public:
  Packages_t() = default;
  void Add(const std::shared_ptr<StateDescriptor> &package);

  std::shared_ptr<StateDescriptor> const &Get(const std::string &name) const {
    return packages_.at(name);
  }

  // Retrieve a package pointer, cast to a given type T
  template <typename T>
  T *Get(const std::string &name) const {
    return static_cast<T *>(packages_.at(name).get());
  }

  const Dictionary<std::shared_ptr<StateDescriptor>> &AllPackages() const {
    return packages_;
  }
  Dictionary<std::shared_ptr<StateDescriptor>> &AllPackages() { return packages_; }

  // Note the non-const accessor inserts an empty dictionary if a name
  // is accessed but not present.
  Dictionary<std::shared_ptr<StateDescriptor>> &
  AllPackagesWithSubMeshData(const std::string &name) {
    return packages_with_submeshdata_[name];
  }
  const Dictionary<std::shared_ptr<StateDescriptor>> &
  AllPackagesWithSubMeshData(const std::string &name) const {
    PARTHENON_REQUIRE(packages_with_submeshdata_.count(name) > 0,
                      "At least one package must contain submeshdata with subname " +
                          name);
    return packages_with_submeshdata_.at(name);
  }
  auto &AllPackagesWithSubMeshData() { return packages_with_submeshdata_; }
  const auto &AllPackagesWithSubMeshData() const { return packages_with_submeshdata_; }

  // Returns a sub-Dictionary containing just pointers to packages of type T.
  // Dictionary is a *new copy*, and members are bare pointers, not shared_ptr.
  template <typename T>
  const Dictionary<T *> AllPackagesOfType() const {
    Dictionary<T *> sub_dict;
    for (auto package : packages_) {
      if (T *cast_package = dynamic_cast<T *>(package.second.get())) {
        sub_dict[package.first] = cast_package;
      }
    }
    return sub_dict;
  }

  // Returns a list of pointers to packages of type T.
  // List contains bare pointers, not shared_ptr objects
  template <typename T>
  const std::vector<T *> ListPackagesOfType() const {
    std::vector<T *> sub_list;
    for (auto package : packages_) {
      if (T *cast_package = dynamic_cast<T *>(package.second.get())) {
        sub_list.append(cast_package);
      }
    }
    return sub_list;
  }

  // Orders the packages' callbacks by the dependencies of their CallbackRegistrations,
  // see utils/callback_order.hpp. The methods below forward to it.
  CallbackOrder<StateDescriptor> CallbackOrders() const {
    return CallbackOrder<StateDescriptor>(packages_, "StateDescriptor");
  }

  template <CallbackRegistrationGetter<StateDescriptor> Getter>
  std::vector<std::string> BuildExecutionOrder(Getter getter,
                                               const std::string &callback_name) const {
    return CallbackOrders().BuildExecutionOrder(getter, callback_name);
  }

  template <CallbackRegistrationGetter<StateDescriptor> Getter>
  void AddTasksDAG(Getter getter, const std::function<void(StateDescriptor *)> &executor,
                   const std::string &callback_name) const {
    CallbackOrders().AddTasksDAG(getter, executor, callback_name);
  }

  template <CallbackRegistrationGetter<StateDescriptor> Getter>
  void AddTasksDAG(const std::vector<std::string> &order, Getter getter,
                   const std::function<void(StateDescriptor *)> &executor) const {
    CallbackOrders().AddTasksDAG(order, getter, executor);
  }

  template <CallbackRegistrationGetter<StateDescriptor> Getter>
  void WriteCallbackGraph(std::ostream &stream, Getter getter,
                          const std::string &callback_name) const {
    CallbackOrders().WriteCallbackGraph(stream, getter, callback_name);
  }

  template <auto... Members>
  std::vector<std::string> ExecutionOrder() const {
    return CallbackOrders().ExecutionOrder<Members...>();
  }

  // Registrations of a StateDescriptor callback member in execution order, e.g.
  //   for (const auto &fill_derived_mesh :
  //        packages.Callbacks<&StateDescriptor::FillDerivedMesh>())
  //     fill_derived_mesh(md);
  template <auto Member>
  auto Callbacks() const {
    return CallbackOrders().Callbacks<Member>();
  }

  template <auto... Members, typename F>
  void ForEachInOrder(F &&f) const {
    CallbackOrders().ForEachInOrder<Members...>(std::forward<F>(f));
  }

 private:
  Dictionary<std::shared_ptr<StateDescriptor>> packages_;
  // A map from category of meshdata subset requirements to state
  // descriptors containing subsets of that category. Makes it easy to
  // group things for, e.g., source terms.
  Dictionary<Dictionary<std::shared_ptr<StateDescriptor>>> packages_with_submeshdata_;
};
} // namespace parthenon

#endif // INTERFACE_PACKAGES_HPP_
