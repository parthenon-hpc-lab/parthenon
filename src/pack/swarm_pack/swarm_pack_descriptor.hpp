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
#ifndef PACK_SWARM_PACK_SWARM_PACK_DESCRIPTOR_HPP_
#define PACK_SWARM_PACK_SWARM_PACK_DESCRIPTOR_HPP_

// This file was made in part with generative AI

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "interface/variable.hpp"

namespace parthenon {

namespace impl {
template <typename TYPE>
struct SwarmPackDescriptor {
  void Print() const;

  // default constructor needed for certain use cases
  SwarmPackDescriptor() : swarm_name(""), vars({}), identifier("") {}

  SwarmPackDescriptor(const std::string &swarm_name, const std::vector<std::string> &vars)
      : swarm_name(swarm_name), vars(vars), identifier(GetIdentifier()) {}

  // Determining if variable pv should be included in SwarmPack
  bool IncludeVariable(int vidx,
                       const std::shared_ptr<ParticleVariable<TYPE>> &pv) const {
    if (vars[vidx] == pv->label()) return true;
    return false;
  }

  const std::string swarm_name;
  const std::vector<std::string> vars;
  const std::string identifier;

 private:
  std::string GetIdentifier() const {
    std::string ident("");
    for (const auto &var : vars)
      ident += var;
    ident += "|swarm_name:";
    ident += swarm_name;
    return ident;
  }
};
} // namespace impl

} // namespace parthenon

#endif // PACK_SWARM_PACK_SWARM_PACK_DESCRIPTOR_HPP_
