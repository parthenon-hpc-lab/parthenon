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
#ifndef INTERFACE_MAKE_SWARM_PACK_DESCRIPTOR_HPP_
#define INTERFACE_MAKE_SWARM_PACK_DESCRIPTOR_HPP_

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <regex>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "interface/mesh_data.hpp"
#include "interface/meshblock_data.hpp"
#include "interface/metadata.hpp"
#include "interface/state_descriptor.hpp"
#include "interface/swarm_pack.hpp"
#include "interface/variable.hpp"
#include "mesh/mesh.hpp"

namespace parthenon {

template <typename TYPE>
inline auto MakeSwarmPackDescriptor(const std::string &swarm_name,
                                    const std::vector<std::string> &vars) {
  impl::SwarmPackDescriptor<TYPE> base_desc(swarm_name, vars);
  return typename SwarmPack<TYPE>::Descriptor(base_desc);
}

template <class... Ts>
inline auto MakeSwarmPackDescriptor(const std::string &swarm_name) {
  static_assert(sizeof...(Ts) > 0, "Must have at least one variable type for type pack");
  using TYPE = typename GetDataType<Ts...>::value;

  std::vector<std::string> vars{Ts::name()...};

  return typename SwarmPack<TYPE, Ts...>::Descriptor(
      static_cast<impl::SwarmPackDescriptor<TYPE>>(
          MakeSwarmPackDescriptor<TYPE>(swarm_name, vars)));
}

} // namespace parthenon

#endif // INTERFACE_MAKE_SWARM_PACK_DESCRIPTOR_HPP_
