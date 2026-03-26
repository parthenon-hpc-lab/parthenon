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
#ifndef PACK_SWARM_PACK_SWARM_PACK_TYPES_HPP_
#define PACK_SWARM_PACK_SWARM_PACK_TYPES_HPP_

// This file was made in part with generative AI

#include <cstdint>

#include "basic_types.hpp"
#include "utils/type_list.hpp"

namespace parthenon {

#define PARTHENON_SWARM_PACK_TYPES(X)                                                    \
  X(Real)                                                                                \
  X(int)                                                                                 \
  X(std::uint64_t)

using SwarmPackTypes = TypeList<int, Real, std::uint64_t>;

template <typename TYPE>
class SwarmPackCache;

template <typename TypeList>
struct SwarmPackCacheTuple;

template <typename... Ts>
struct SwarmPackCacheTuple<TypeList<Ts...>> {
  using type = std::tuple<SwarmPackCache<Ts>...>;
};

using SwarmPackCaches = typename SwarmPackCacheTuple<SwarmPackTypes>::type;

} // namespace parthenon

#endif // PACK_SWARM_PACK_SWARM_PACK_TYPES_HPP_
