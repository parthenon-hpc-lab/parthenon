//========================================================================================
// (C) (or copyright) 2020-2024. Triad National Security, LLC. All rights reserved.
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
#ifndef INTERFACE_SWARM_VARIABLE_TYPES_HPP_
#define INTERFACE_SWARM_VARIABLE_TYPES_HPP_

#include <string>

#include "swarm_pack.hpp"

#define SWARM_VARIABLE(type, ns, varname)                                                \
  struct varname : public parthenon::swarm_variable_names::base_t<false, type> {         \
    template <class... Ts>                                                               \
    KOKKOS_INLINE_FUNCTION varname(Ts &&...args)                                         \
        : parthenon::swarm_variable_names::base_t<false,                                 \
                                                  type>(std::forward<Ts>(args)...) {}    \
    static std::string name() { return #ns "." #varname; }                               \
  }

namespace swarm_position {

SWARM_VARIABLE(parthenon::Real, swarm, x);
SWARM_VARIABLE(parthenon::Real, swarm, y);
SWARM_VARIABLE(parthenon::Real, swarm, z);

} // namespace swarm_position

#endif // INTERFACE_SWARM_VARIABLE_TYPES_HPP_
