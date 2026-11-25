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
#ifndef UTILS_CALLBACKS_HPP_
#define UTILS_CALLBACKS_HPP_
// This file was made in part with generative AI.

#include <cstddef>
#include <exception>
#include <map>
#include <ostream>
#include <set>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "utils/error_checking.hpp"

namespace parthenon {

/// Wrapper for callbacks with dependency information.
///
/// This template wraps a callback function along with metadata about its
/// dependencies on other packages' callbacks. It supports expressing both
/// "run after" and "run before" relationships.
///
/// @tparam Func The function type (typically std::function<...>)
template <typename Func>
struct CallbackRegistration {
  using FunctionType = Func;
  /// The actual callback function
  Func callback = nullptr;

  /// Names of units whose callbacks must execute before this one ("run after")
  std::vector<std::string> depends_on;

  /// Names of units whose callbacks must execute after this one ("run before")
  std::vector<std::string> required_by;

  /// Assignment operator to allow direct function assignment.
  ///
  /// This enables syntax like: callback_registration = some_function;
  /// The function is registered without any dependencies.
  ///
  /// @param fn The callback function to assign
  /// @return Reference to this registration
  CallbackRegistration &operator=(Func fn) {
    callback = fn;
    depends_on.clear();
    required_by.clear();
    return *this;
  }

  /// Register a callback with optional dependencies.
  ///
  /// @param fn The callback function to register
  /// @param after Units that must execute before this callback
  /// @param before Units that must execute after this callback
  /// @return Reference to this registration for potential chaining
  CallbackRegistration &Register(Func fn, const std::vector<std::string> &after = {},
                                 const std::vector<std::string> &before = {}) {
    callback = fn;
    depends_on = after;
    required_by = before;
    return *this;
  }

  bool IsRegistered() const { return callback != nullptr; }

  /// Implicit conversion to bool for checking registration status.
  /// @return true if callback is registered
  explicit operator bool() const { return IsRegistered(); }

  /// Call the underlying callback function.
  ///
  /// Allows the registration to be used like a function object.
  /// @param args Arguments to forward to the callback
  /// @return Whatever the callback returns
  template <typename... Args>
  auto operator()(Args &&...args) const {
    return callback(std::forward<Args>(args)...);
  }

  /// Compare against nullptr, matching the std::function interface.
  /// @return true if no callback is registered
  friend bool operator==(const CallbackRegistration &reg, std::nullptr_t) {
    return !reg.IsRegistered();
  }
};

/// Directed Acyclic Graph (DAG) for managing callback execution order.
///
/// This class builds a dependency graph from callback specifications and computes
/// a valid execution order via topological sorting. It detects and reports cycles
class CallbackDAG {
 public:
  CallbackDAG() = default;

  void AddNode(const std::string &name);
  void AddEdge(const std::string &from, const std::string &to);
  std::vector<std::string> TopologicalSort() const;
  void WriteGraphViz(std::ostream &stream) const;

  /// Stream insertion operator for easy output.
  friend std::ostream &operator<<(std::ostream &stream, const CallbackDAG &dag);

 private:
  void ValidateAcyclic() const;
  void TopologicalSortDFS(const std::string &node, std::set<std::string> &visited,
                          std::set<std::string> &rec_stack,
                          std::vector<std::string> &result) const;
  bool DetectCycleDFS(const std::string &node, std::set<std::string> &visited,
                      std::set<std::string> &rec_stack,
                      std::vector<std::string> &path) const;

  std::set<std::string> nodes_;
  std::map<std::string, std::vector<std::string>> adjacency_list_;
};

template <typename>
struct is_callback_registration : std::false_type {};
template <typename Func>
struct is_callback_registration<CallbackRegistration<Func>> : std::true_type {};

// Member points to a CallbackRegistration member of T, e.g.
// &StateDescriptor::FillDerivedMesh
template <auto Member, typename T>
concept CallbackMemberOf = requires(T *item) {
  requires is_callback_registration<std::remove_cvref_t<decltype(item->*Member)>>::value;
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
    requires CallbackMemberOf<Member, T>
  OrderedCallbacks<Member> Callbacks() const {
    return OrderedCallbacks<Member>(OrderedItems(&MemberDeps<Member>));
  }

  // Write the dependency graph of a callback member in GraphViz DOT format.
  template <auto Member>
    requires CallbackMemberOf<Member, T>
  void WriteCallbackGraph(std::ostream &stream, const std::string &callback_name) const {
    stream << "// Callback execution order for: " << callback_name << "\n";
    stream << BuildDAG(&MemberDeps<Member>);
  }

 private:
  // Dependencies of an item's registration of a callback member
  struct Deps {
    bool registered = false;
    const std::vector<std::string> *depends_on = nullptr;
    const std::vector<std::string> *required_by = nullptr;
  };
  using DepsOf = Deps (*)(T *);

  template <auto Member>
  static Deps MemberDeps(T *item) {
    const auto &registration = item->*Member;
    return {registration.IsRegistered(), &registration.depends_on,
            &registration.required_by};
  }

  // Not templated on the member, so these are compiled once per T (and only once for
  // types with an explicit instantiation, such as StateDescriptor)
  CallbackDAG BuildDAG(DepsOf deps_of) const;
  std::vector<T *> OrderedItems(DepsOf deps_of) const;

  std::unordered_map<std::string, T *> items_;
  std::string label_;
};

template <typename T>
CallbackDAG CallbackOrder<T>::BuildDAG(DepsOf deps_of) const {
  std::unordered_map<std::string, Deps> registered;
  for (const auto &[name, item] : items_) {
    auto deps = deps_of(item);
    if (deps.registered) registered.emplace(name, deps);
  }

  CallbackDAG dag;
  for (const auto &[name, deps] : registered) {
    dag.AddNode(name);
    // depends_on: this item runs after those items
    for (const auto &dependency : *deps.depends_on) {
      if (registered.count(dependency) > 0) dag.AddEdge(dependency, name);
    }
    // required_by: this item runs before those items
    for (const auto &dependent : *deps.required_by) {
      if (registered.count(dependent) > 0) dag.AddEdge(name, dependent);
    }
  }
  return dag;
}

template <typename T>
std::vector<T *> CallbackOrder<T>::OrderedItems(DepsOf deps_of) const {
  std::vector<std::string> names;
  try {
    names = BuildDAG(deps_of).TopologicalSort();
  } catch (const std::exception &e) {
    PARTHENON_THROW("Error building execution order for " + label_ +
                    " callbacks: " + e.what());
  }
  std::vector<T *> order;
  for (const auto &name : names) {
    order.push_back(items_.at(name));
  }
  return order;
}

} // namespace parthenon

#endif // UTILS_CALLBACKS_HPP_
