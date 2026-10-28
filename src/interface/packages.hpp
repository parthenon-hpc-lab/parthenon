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

#include <exception>
#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "basic_types.hpp"
#include "utils/callback_dag.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {
class StateDescriptor; // forward declaration
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

  // Compute the execution order of a callback across all packages, based on the
  // depends_on/required_by lists of each package's CallbackRegistration. getter maps a
  // StateDescriptor* to its CallbackRegistration. Only packages with the callback
  // registered are included, and dependencies on packages without it are ignored.
  // Throws on cycles.
  template <typename Getter>
  std::vector<std::string> BuildExecutionOrder(Getter getter,
                                               const std::string &callback_name) const {
    try {
      return BuildCallbackDAG(getter).TopologicalSort();
    } catch (const std::exception &e) {
      PARTHENON_THROW("Error building execution order for " + callback_name +
                      " callbacks: " + e.what());
    }
  }

  // Call executor on each package with the callback registered, in dependency order.
  template <typename Getter>
  void AddTasksDAG(Getter getter, const std::function<void(StateDescriptor *)> &executor,
                   const std::string &callback_name) const {
    AddTasksDAG(BuildExecutionOrder(getter, callback_name), getter, executor);
  }

  // As above, but with a precomputed order from BuildExecutionOrder.
  template <typename Getter>
  void AddTasksDAG(const std::vector<std::string> &order, Getter getter,
                   const std::function<void(StateDescriptor *)> &executor) const {
    for (const auto &name : order) {
      StateDescriptor *package = packages_.at(name).get();
      if (getter(package).IsRegistered()) executor(package);
    }
  }

  // Write the dependency graph of a callback in GraphViz DOT format.
  template <typename Getter>
  void WriteCallbackGraph(std::ostream &stream, Getter getter,
                          const std::string &callback_name) const {
    stream << "// Callback execution order for: " << callback_name << "\n";
    stream << BuildCallbackDAG(getter);
  }

  // Execution order of a StateDescriptor callback member, e.g.
  // ExecutionOrder<&StateDescriptor::FillDerivedMesh>(). With several members, e.g. the
  // Mesh and Block variants of a callback, the packages registering any of them are
  // ordered together using the dependencies of all of them.
  template <auto... Members>
  std::vector<std::string> ExecutionOrder() const {
    return BuildExecutionOrder(CallbackGetter<Members...>(), "StateDescriptor");
  }

  // The CallbackRegistrations of one StateDescriptor callback member across packages,
  // in execution order, skipping packages that don't register it. Iterating yields
  // const references to the registrations, which can be called directly.
  template <auto Member>
  class OrderedCallbacks {
   public:
    using Order = std::vector<StateDescriptor *>;

    class iterator {
     public:
      explicit iterator(std::vector<StateDescriptor *>::const_iterator it) : it_(it) {}
      const auto &operator*() const { return (*it_)->*Member; }
      iterator &operator++() {
        ++it_;
        return *this;
      }
      bool operator!=(const iterator &other) const { return it_ != other.it_; }

     private:
      std::vector<StateDescriptor *>::const_iterator it_;
    };

    explicit OrderedCallbacks(Order order) : order_(std::move(order)) {}
    iterator begin() const { return iterator(order_.begin()); }
    iterator end() const { return iterator(order_.end()); }
    std::size_t size() const { return order_.size(); }

   private:
    Order order_;
  };

  // Registrations of a StateDescriptor callback member in execution order, e.g.
  //   for (const auto &cb : packages.Callbacks<&StateDescriptor::FillDerivedMesh>())
  //     cb(md);
  // The order is rebuilt on every call.
  template <auto Member>
  OrderedCallbacks<Member> Callbacks() const {
    return OrderedCallbacks<Member>(OrderedPackages<Member>());
  }

  // Call f(StateDescriptor *) on each package registering any of the callback members,
  // in execution order. Useful to order several members together, e.g. the Mesh and
  // Block variants of a callback.
  template <auto... Members, typename F>
  void ForEachInOrder(F &&f) const {
    for (auto *package : OrderedPackages<Members...>()) {
      f(package);
    }
  }

 private:
  // Dependencies of whichever of several callback members a package registers
  struct CallbackDeps {
    bool registered = false;
    std::vector<std::string> depends_on;
    std::vector<std::string> required_by;
    bool IsRegistered() const { return registered; }
  };

  template <auto First, auto... Rest>
  static auto CallbackGetter() {
    if constexpr (sizeof...(Rest) == 0) {
      return [](StateDescriptor *sd) -> auto & { return sd->*First; };
    } else {
      return [](StateDescriptor *sd) {
        CallbackDeps deps;
        auto add = [&](const auto &reg) {
          if (!reg.IsRegistered()) return;
          deps.registered = true;
          deps.depends_on.insert(deps.depends_on.end(), reg.depends_on.begin(),
                                 reg.depends_on.end());
          deps.required_by.insert(deps.required_by.end(), reg.required_by.begin(),
                                  reg.required_by.end());
        };
        add(sd->*First);
        (add(sd->*Rest), ...);
        return deps;
      };
    }
  }

  template <auto... Members>
  std::vector<StateDescriptor *> OrderedPackages() const {
    std::vector<StateDescriptor *> order;
    for (const auto &name : ExecutionOrder<Members...>()) {
      order.push_back(packages_.at(name).get());
    }
    return order;
  }

  template <typename Getter>
  CallbackDAG BuildCallbackDAG(Getter getter) const {
    Dictionary<StateDescriptor *> registered;
    for (const auto &[name, package] : packages_) {
      if (getter(package.get()).IsRegistered()) registered[name] = package.get();
    }

    CallbackDAG dag;
    for (const auto &[name, package] : registered) {
      dag.AddNode(name);
      const auto &registration = getter(package);
      // DependsOn: this package runs after those packages
      for (const auto &dependency : registration.depends_on) {
        if (registered.count(dependency) > 0) dag.AddEdge(dependency, name);
      }
      // RequiredBy: this package runs before those packages
      for (const auto &dependent : registration.required_by) {
        if (registered.count(dependent) > 0) dag.AddEdge(name, dependent);
      }
    }
    return dag;
  }

  Dictionary<std::shared_ptr<StateDescriptor>> packages_;
  // A map from category of meshdata subset requirements to state
  // descriptors containing subsets of that category. Makes it easy to
  // group things for, e.g., source terms.
  Dictionary<Dictionary<std::shared_ptr<StateDescriptor>>> packages_with_submeshdata_;
};
} // namespace parthenon

#endif // INTERFACE_PACKAGES_HPP_
