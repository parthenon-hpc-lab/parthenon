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
#ifndef EXAMPLE_BOUNDARY_EXCHANGE_BOUNDARY_EXCHANGE_HPP_
#define EXAMPLE_BOUNDARY_EXCHANGE_BOUNDARY_EXCHANGE_HPP_

// Standard Includes
#include <memory>
#include <string>
#include <utility>
#include <vector>

// Parthenon Includes
#include <interface/state_descriptor.hpp>
#include <parthenon/package.hpp>

namespace boundary_exchange {
using namespace parthenon::package::prelude;
using parthenon::Packages_t;

struct neighbor_info : public parthenon::variable_names::base_t<false, 4> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION neighbor_info(Ts &&...args)
      : parthenon::variable_names::base_t<false, 4>(std::forward<Ts>(args)...) {}
  static std::string name() { return "neighbor_info"; }
};

TaskStatus SetBlockValues(MeshData<Real> *rc);
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

} // namespace boundary_exchange

#endif // EXAMPLE_BOUNDARY_EXCHANGE_BOUNDARY_EXCHANGE_HPP_
