//========================================================================================
// (C) (or copyright) 2020-2026. Triad National Security, LLC. All rights reserved.
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
#ifndef PACK_DEFAULT_NAMES_HPP_
#define PACK_DEFAULT_NAMES_HPP_

// This file was made in part with generative AI.

#include <cstdint>
#include <string>
#include <utility>

#include "pack/sparse_pack/sparse_pack.hpp"
#include "pack/swarm_pack/swarm_pack.hpp"

#define PAR_VAR(ns, varname)                                                             \
  struct varname : public parthenon::variable_names::base_t<false> {                     \
    template <class... Ts>                                                               \
    KOKKOS_INLINE_FUNCTION varname(Ts &&...args)                                         \
        : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}         \
    static std::string name() { return #ns "." #varname; }                               \
  }

#define PAR_SWARMVAR(type, ns, varname)                                                  \
  struct varname : public parthenon::swarm_variable_names::base_t<type> {                \
    template <class... Ts>                                                               \
    KOKKOS_INLINE_FUNCTION varname(Ts &&...args)                                         \
        : parthenon::swarm_variable_names::base_t<type>(std::forward<Ts>(args)...) {}    \
    static std::string name() { return #ns "." #varname; }                               \
  }

namespace swarm_position {
PAR_SWARMVAR(std::uint64_t, swarm, id);
PAR_SWARMVAR(parthenon::Real, swarm, x1);
PAR_SWARMVAR(parthenon::Real, swarm, x2);
PAR_SWARMVAR(parthenon::Real, swarm, x3);
} // namespace swarm_position

#endif // PACK_DEFAULT_NAMES_HPP_
