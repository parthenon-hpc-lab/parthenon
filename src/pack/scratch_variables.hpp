//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2025 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2025. Triad National Security, LLC. All rights reserved.
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

#ifndef PACK_SCRATCH_VARIABLES_HPP_
#define PACK_SCRATCH_VARIABLES_HPP_

#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "basic_types.hpp"
#include "interface/metadata.hpp"
#include "pack/pack_utils.hpp"
#include "utils/type_list.hpp"

namespace parthenon {

inline std::string range_regex(unsigned a, unsigned b) {
  std::ostringstream pattern;
  pattern << "((" << std::to_string(a) << ")";
  for (int i = a + 1; i <= b; i++) {
    pattern << "|(" << std::to_string(i) << ")";
  }
  pattern << ")";
  return pattern.str();
}

template <TopologicalType TT, int... NCOMPS>
struct ScratchVariable : public parthenon::variable_names::base_t<true, NCOMPS...> {
  using base_t = parthenon::variable_names::base_t<true, NCOMPS...>;
  template <typename... Ts>
  KOKKOS_INLINE_FUNCTION ScratchVariable(Ts &&...args)
      : base_t(std::forward<Ts>(args)...) {}
  static constexpr TopologicalType type = TT;
  static constexpr int ncomps = sizeof...(NCOMPS);
  static constexpr int size = (NCOMPS * ... * (1));
  static constexpr std::array<int, ncomps> shape{NCOMPS...};
};

constexpr bool debug_scratch_variables() {
#ifdef PARTHENON_DEBUG_SCRATCH
  return true;
#else
  return false;
#endif
}

// All this macro nonsense is necessary to pass the var_name as a string
// to use in the name() method. C++-20 allows parsing strings as template
// parameters, in which case ScratchVariable can just template on a
// compile time string to use as the name.
#define SCRATCH_VARIABLE_IMPL(var_name, TT, ...)                                         \
  struct var_name : public ScratchVariable<TT, __VA_ARGS__> {                            \
    friend class StateDescriptor;                                                        \
    template <typename... Ts>                                                            \
    KOKKOS_INLINE_FUNCTION var_name(Ts &&...args)                                        \
        : ScratchVariable<TT, __VA_ARGS__>(std::forward<Ts>(args)...) {}                 \
    static std::string name() {                                                          \
      if constexpr (debug_scratch_variables()) {                                         \
        return std::string("scratch_") + std::string(#var_name);                         \
      } else {                                                                           \
        return "scratch_" + TopologicalTypeToString(type) + "_" + range_regex(lb, ub);   \
      }                                                                                  \
    }                                                                                    \
                                                                                         \
   protected:                                                                            \
    inline static int lb;                                                                \
    inline static int ub;                                                                \
    static int update_bounds(const int lower) {                                          \
      lb = lower;                                                                        \
      ub = lower + size - 1;                                                             \
      return ub + 1;                                                                     \
    }                                                                                    \
    static const auto GetVarNames() {                                                    \
      std::array<std::string, size> vars;                                                \
      auto base = "scratch_" + TopologicalTypeToString(TT) + "_";                        \
      for (int i = 0; i < size; i++) {                                                   \
        vars[i] = base + std::to_string(i + lb);                                         \
      }                                                                                  \
      return vars;                                                                       \
    }                                                                                    \
  };

#define SCRATCH_VARIABLE_IMPL2(var_name, TT) SCRATCH_VARIABLE_IMPL(var_name, TT, 1)
#define SCRATCH_VARIABLE_IMPL3(var_name, TT, t) SCRATCH_VARIABLE_IMPL(var_name, TT, t)
#define SCRATCH_VARIABLE_IMPL4(var_name, TT, t, u)                                       \
  SCRATCH_VARIABLE_IMPL(var_name, TT, t, u)
#define SCRATCH_VARIABLE_IMPL5(var_name, TT, t, u, v)                                    \
  SCRATCH_VARIABLE_IMPL(var_name, TT, t, u, v)
#define SCRATCH_EXPAND(x) x
#define SCRATCH_GET_IMPL(_1, _2, _3, _4, _5, macro, ...) macro

#define SCRATCH_VARIABLE(...)                                                            \
  SCRATCH_EXPAND(SCRATCH_GET_IMPL(__VA_ARGS__, SCRATCH_VARIABLE_IMPL5,                   \
                                  SCRATCH_VARIABLE_IMPL4, SCRATCH_VARIABLE_IMPL3,        \
                                  SCRATCH_VARIABLE_IMPL2)(__VA_ARGS__))

} // namespace parthenon
#endif // PACK_SCRATCH_VARIABLES_HPP_
