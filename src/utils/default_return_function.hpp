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

#ifndef UTILS_DEFAULT_RETURN_FUNCTION_HPP_
#define UTILS_DEFAULT_RETURN_FUNCTION_HPP_

#include <type_traits>

namespace parthenon {
template <class return_t, return_t default_ret, class... args_t>
class DefaultReturnFunction {
 public:
  DefaultReturnFunction() : func_(nullptr) {}
  explicit DefaultReturnFunction(std::nullptr_t) : func_(nullptr) {}

  template <class F, REQUIRES(std::is_invocable_v<std::decay_t<F>, args_t...>)>
  explicit DefaultReturnFunction(F &&f) {
    assign(std::forward<F>(f));
  }

  template <class F, REQUIRES(std::is_invocable_v<std::decay_t<F>, args_t...>)>
  DefaultReturnFunction &operator=(F &&f) {
    assign(std::forward<F>(f));
    return *this;
  }

  TaskStatus operator()(args_t... args) const { return func_(args...); }
  bool operator==(std::nullptr_t) const { return !func_; }
  bool operator!=(std::nullptr_t) const { return static_cast<bool>(func_); }
  explicit operator bool() const { return static_cast<bool>(func_); }

 private:
  std::function<return_t(args_t...)> func_;

  template <class F>
  void assign(F &&f) {
    if constexpr (std::is_same_v<std::invoke_result_t<std::decay_t<F>, args_t...>,
                                 return_t>) {
      func_ = f;
    } else {
      func_ = [f](args_t... args) {
        f(args...);
        return default_ret;
      };
    }
  }
};

} // namespace parthenon

#endif // UTILS_DEFAULT_RETURN_FUNCTION_HPP_
