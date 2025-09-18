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
#include "interface/state_descriptor.hpp"
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
struct ScratchVariable {
  using base_t = parthenon::variable_names::base_t<true, NCOMPS...>;
  static constexpr TopologicalType type = TT;
  static constexpr int ncomps = sizeof...(NCOMPS);
  static constexpr int size = (NCOMPS * ...);
  static constexpr std::array<int, ncomps> shape{NCOMPS...};
};

#define SCRATCH_VARIABLE(var_name, TT, ...)                                              \
  struct var_name##_t : public ScratchVariable<TT, __VA_ARGS__> {                        \
    static std::string name() { return #var_name; }                                      \
  };

template <typename SV, int lower>
struct ScratchVariable_impl : public SV::base_t {
  using type = SV;
  static constexpr int lb = lower;
  static constexpr int ub = lower + SV::size - 1;
  static constexpr auto shape = SV::shape;

  template <class... Ts>
  KOKKOS_INLINE_FUNCTION ScratchVariable_impl(Ts &&...args)
      : SV::base_t(std::forward<Ts>(args)...) {}

  static std::string name() {
#ifdef PARTHENON_DEBUG_SCRATCH
    return "scratch_" + SV::Name();
#else
    return "scratch_" + TopologicalTypeToString(SV::type) + "_" + range_regex(lb, ub);
#endif
  }
};

namespace impl {
template <typename...>
struct SVList_impl {};

template <typename SV>
struct SVList_impl<SV> {
  using type = ScratchVariable_impl<SV, 0>;
  using value = TypeList<type>;
};

template <typename SV, typename... SVs>
struct SVList_impl<SV, SVs...> {
  using list = SVList_impl<SVs...>;
  using type = ScratchVariable_impl<SV, list::type::ub + 1>;
  using value = concatenate_type_lists_t<TypeList<type>, typename list::value>;
};
} // namespace impl

// Gives a tuv index into the common scratch data for a given TopologicalType
// by using an agreed upon pool of scratch_TT_# overrideable var names
// that way the total memory allocated across all packages is the maximum
// size of any single ScratchVariableList for a given TT, but allows
// for each package to index into the common space with their own
// unique types & sizes
template <typename V, typename... SVs>
struct ScratchVariableList {
  static constexpr TopologicalType TT = V::type;
  static constexpr int n_vars = V::size + (SVs::size + ... + 0);
  using TL = TypeList<V, SVs...>;
  using list = impl::SVList_impl<V, SVs...>;

  template <typename SV>
  using type = typename list::value::template type<TL::template GetIdx<SV>()>;

  static const auto GetVarNames() {
    std::array<std::string, n_vars> vars;
    auto base = "scratch_" + TopologicalTypeToString(TT) + "_";
    for (int i = 0; i < n_vars; i++) {
      vars[i] = base + std::to_string(i);
    }
    return vars;
  }
};

} // namespace parthenon
#endif // PACK_SCRATCH_VARIABLES_HPP_
