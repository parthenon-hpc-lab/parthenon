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
#ifndef UTILS_CALLBACK_REGISTRATION_HPP_
#define UTILS_CALLBACK_REGISTRATION_HPP_
// This file was made in part with generative AI.

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

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

} // namespace parthenon

#endif // UTILS_CALLBACK_REGISTRATION_HPP_
