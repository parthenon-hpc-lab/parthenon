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
#include <string>
#include <vector>

#include "pack/swarm_pack/make_swarm_pack_descriptor.hpp"

namespace parthenon {

template <typename TYPE>
typename SwarmPack<TYPE>::Descriptor
MakeSwarmPackDescriptor(const std::string &swarm_name,
                        const std::vector<std::string> &vars) {
  impl::SwarmPackDescriptor<TYPE> base_desc(swarm_name, vars);
  return typename SwarmPack<TYPE>::Descriptor(base_desc);
}

template SwarmPack<Real>::Descriptor
MakeSwarmPackDescriptor<Real>(const std::string &, const std::vector<std::string> &);
template SwarmPack<int>::Descriptor
MakeSwarmPackDescriptor<int>(const std::string &, const std::vector<std::string> &);
template SwarmPack<std::uint64_t>::Descriptor
MakeSwarmPackDescriptor<std::uint64_t>(const std::string &,
                                       const std::vector<std::string> &);

} // namespace parthenon
