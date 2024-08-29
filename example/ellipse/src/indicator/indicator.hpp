//========================================================================================
// (C) (or copyright) 2024. Triad National Security, LLC. All rights reserved.
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

#ifndef _INDICATOR_INDICATOR_HPP_
#define _INDICATOR_INDICATOR_HPP_

#include <memory>
#include <parthenon/package.hpp>

namespace Indicator {
using namespace parthenon::package::prelude;

struct phi : public parthenon::variable_names::base_t<false> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION phi(Ts &&...args)
    : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}
  static std::string name() { return "phi"; }
};

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

} // namespace Indicator

#endif // _INDICATOR_INDICATOR_HPP_
