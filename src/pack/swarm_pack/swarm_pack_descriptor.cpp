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

// This file was made in part with generative AI

#include <cstdio>
#include <string>

#include "pack/swarm_pack/swarm_pack_descriptor.hpp"
#include "pack/swarm_pack/swarm_pack_types.hpp"

namespace parthenon {
namespace impl {

template <typename TYPE>
void SwarmPackDescriptor<TYPE>::Print() const {
  printf("--------------------\n");
  printf("%s\n", identifier.c_str());
  printf("--------------------\n");
}

#define INSTANTIATE_PRINT(TYPE) template void SwarmPackDescriptor<TYPE>::Print() const;
PARTHENON_SWARM_PACK_TYPES(INSTANTIATE_PRINT)
#undef INSTANTIATE_PRINT

} // namespace impl
} // namespace parthenon
