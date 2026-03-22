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

// This file was made in part with generative AI

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "pack/swarm_pack/swarm_pack_descriptor.hpp"

namespace parthenon {
namespace impl {

template <typename TYPE>
SwarmPackDescriptor<TYPE>::SwarmPackDescriptor(const std::string &swarm_name,
                                               const std::vector<std::string> &vars)
    : swarm_name(swarm_name), vars(vars), identifier(GetIdentifier()) {}

template <typename TYPE>
bool SwarmPackDescriptor<TYPE>::IncludeVariable(
    int vidx, const std::shared_ptr<ParticleVariable<TYPE>> &pv) const {
  return vars[vidx] == pv->label();
}

template <typename TYPE>
std::string SwarmPackDescriptor<TYPE>::GetIdentifier() const {
  std::string ident("");
  for (const auto &var : vars)
    ident += var;
  ident += "|swarm_name:";
  ident += swarm_name;
  return ident;
}

template struct SwarmPackDescriptor<Real>;
template struct SwarmPackDescriptor<int>;
template struct SwarmPackDescriptor<std::uint64_t>;

} // namespace impl
} // namespace parthenon
