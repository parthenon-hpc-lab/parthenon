//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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
#ifndef UTILS_CALLBACK_ORDER_HPP_
#define UTILS_CALLBACK_ORDER_HPP_
// This file was made in part with generative AI.

#include <concepts>
#include <cstddef>
#include <exception>
#include <functional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "utils/callback_dag.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

// A callable mapping a T * to its registration of a callback: something with
// IsRegistered() and the depends_on/required_by names, such as a CallbackRegistration.
template <typename Getter, typename T>
concept CallbackRegistrationGetter = requires(Getter getter, T *item) {
  { getter(item).IsRegistered() } -> std::convertible_to<bool>;
  { getter(item).depends_on } -> std::convertible_to<const std::vector<std::string> &>;
  { getter(item).required_by } -> std::convertible_to<const std::vector<std::string> &>;
}; // NOLINT(readability/braces)

// Orders the callbacks of a set of named T, e.g. Parthenon's StateDescriptors, by the
// dependencies of their CallbackRegistrations. A callback runs after the items named in
// its depends_on and before those in its required_by. Only items with the callback
// registered are ordered, and dependencies on items without it are ignored. Cycles
// throw. The order is built on each call.
template <typename T>
class CallbackOrder {
 public:
  // items maps names to (smart) pointers to T, e.g. a
  // std::map<std::string, std::shared_ptr<T>>. label names T in error messages.
  template <typename Map>
  explicit CallbackOrder(const Map &items, std::string label = "callback")
      : label_(std::move(label)) {
    for (const auto &[name, item] : items) {
      items_[name] = &*item;
    }
  }

  // Execution order of a callback, where getter maps a T * to its registration.
  template <CallbackRegistrationGetter<T> Getter>
  std::vector<std::string> BuildExecutionOrder(Getter getter,
                                               const std::string &callback_name) const {
    return BuildExecutionOrderImpl(Erase(getter), callback_name);
  }

  // Call executor on each item with the callback registered, in dependency order.
  template <CallbackRegistrationGetter<T> Getter>
  void AddTasksDAG(Getter getter, const std::function<void(T *)> &executor,
                   const std::string &callback_name) const {
    for (T *item : OrderedItems(Erase(getter), callback_name)) {
      executor(item);
    }
  }

  // As above, but with a precomputed order from BuildExecutionOrder.
  template <CallbackRegistrationGetter<T> Getter>
  void AddTasksDAG(const std::vector<std::string> &order, Getter getter,
                   const std::function<void(T *)> &executor) const {
    for (const auto &name : order) {
      T *item = items_.at(name);
      if (getter(item).IsRegistered()) executor(item);
    }
  }

  // Write the dependency graph of a callback in GraphViz DOT format.
  template <CallbackRegistrationGetter<T> Getter>
  void WriteCallbackGraph(std::ostream &stream, Getter getter,
                          const std::string &callback_name) const {
    stream << "// Callback execution order for: " << callback_name << "\n";
    stream << BuildDAG(Erase(getter));
  }

  // Execution order of a callback member of T, e.g.
  // ExecutionOrder<&StateDescriptor::FillDerivedMesh>(). With several members, e.g. the
  // Mesh and Block variants of a callback, the items registering any of them are
  // ordered together using the dependencies of all of them.
  template <auto... Members>
  std::vector<std::string> ExecutionOrder() const {
    return BuildExecutionOrderImpl(MemberDeps<Members...>, label_);
  }

  // The CallbackRegistrations of one callback member across items, in execution order,
  // skipping items that don't register it. Iterating yields const references to the
  // registrations, which can be called directly.
  template <auto Member>
  class OrderedCallbacks {
   public:
    class iterator {
     public:
      explicit iterator(typename std::vector<T *>::const_iterator it) : it_(it) {}
      const auto &operator*() const { return (*it_)->*Member; }
      iterator &operator++() {
        ++it_;
        return *this;
      }
      bool operator!=(const iterator &other) const { return it_ != other.it_; }

     private:
      typename std::vector<T *>::const_iterator it_;
    };

    explicit OrderedCallbacks(std::vector<T *> order) : order_(std::move(order)) {}
    iterator begin() const { return iterator(order_.begin()); }
    iterator end() const { return iterator(order_.end()); }
    std::size_t size() const { return order_.size(); }

   private:
    std::vector<T *> order_;
  };

  // Registrations of a callback member in execution order, e.g.
  //   for (const auto &fill_derived_mesh :
  //        order.Callbacks<&StateDescriptor::FillDerivedMesh>())
  //     fill_derived_mesh(md);
  template <auto Member>
  OrderedCallbacks<Member> Callbacks() const {
    return OrderedCallbacks<Member>(OrderedItems(MemberDeps<Member>, label_));
  }

  // Call f(T *) on each item registering any of the callback members, in execution
  // order. Useful to order several members together, e.g. the Mesh and Block variants
  // of a callback.
  template <auto... Members, typename F>
  void ForEachInOrder(F &&f) const {
    for (T *item : OrderedItems(MemberDeps<Members...>, label_)) {
      f(item);
    }
  }

 private:
  // Dependencies of an item's registration of a callback, or of whichever of several
  // callback members it registers
  struct Deps {
    bool registered = false;
    std::vector<std::string> depends_on;
    std::vector<std::string> required_by;

    template <typename Registration>
    void Add(const Registration &registration) {
      if (!registration.IsRegistered()) return;
      registered = true;
      depends_on.insert(depends_on.end(), registration.depends_on.begin(),
                        registration.depends_on.end());
      required_by.insert(required_by.end(), registration.required_by.begin(),
                         registration.required_by.end());
    }
  };
  using DepsGetter = std::function<Deps(T *)>;

  template <typename Getter>
  static DepsGetter Erase(Getter getter) {
    return [getter](T *item) {
      Deps deps;
      deps.Add(getter(item));
      return deps;
    };
  }

  template <auto... Members>
  static Deps MemberDeps(T *item) {
    Deps deps;
    (deps.Add(item->*Members), ...);
    return deps;
  }

  // Not templated on the getter, so these are compiled once per T (and only once for
  // types with an explicit instantiation, such as StateDescriptor)
  CallbackDAG BuildDAG(const DepsGetter &deps_of) const;
  std::vector<std::string>
  BuildExecutionOrderImpl(const DepsGetter &deps_of,
                          const std::string &callback_name) const;
  std::vector<T *> OrderedItems(const DepsGetter &deps_of,
                                const std::string &callback_name) const;

  std::unordered_map<std::string, T *> items_;
  std::string label_;
};

template <typename T>
CallbackDAG CallbackOrder<T>::BuildDAG(const DepsGetter &deps_of) const {
  std::unordered_map<std::string, Deps> registered;
  for (const auto &[name, item] : items_) {
    auto deps = deps_of(item);
    if (deps.registered) registered.emplace(name, std::move(deps));
  }

  CallbackDAG dag;
  for (const auto &[name, deps] : registered) {
    dag.AddNode(name);
    // depends_on: this item runs after those items
    for (const auto &dependency : deps.depends_on) {
      if (registered.count(dependency) > 0) dag.AddEdge(dependency, name);
    }
    // required_by: this item runs before those items
    for (const auto &dependent : deps.required_by) {
      if (registered.count(dependent) > 0) dag.AddEdge(name, dependent);
    }
  }
  return dag;
}

template <typename T>
std::vector<std::string>
CallbackOrder<T>::BuildExecutionOrderImpl(const DepsGetter &deps_of,
                                          const std::string &callback_name) const {
  try {
    return BuildDAG(deps_of).TopologicalSort();
  } catch (const std::exception &e) {
    PARTHENON_THROW("Error building execution order for " + callback_name +
                    " callbacks: " + e.what());
  }
}

template <typename T>
std::vector<T *> CallbackOrder<T>::OrderedItems(const DepsGetter &deps_of,
                                                const std::string &callback_name) const {
  std::vector<T *> order;
  for (const auto &name : BuildExecutionOrderImpl(deps_of, callback_name)) {
    order.push_back(items_.at(name));
  }
  return order;
}

} // namespace parthenon

#endif // UTILS_CALLBACK_ORDER_HPP_
