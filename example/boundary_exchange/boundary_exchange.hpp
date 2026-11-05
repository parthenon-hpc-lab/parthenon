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

// Fields used to check BoundaryType::faces_only_* against BoundaryType::any
struct linear_field : public parthenon::variable_names::base_t<false> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION linear_field(Ts &&...args)
      : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}
  static std::string name() { return "linear_field"; }
};

struct neighbor_info_ref : public parthenon::variable_names::base_t<false, 8> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION neighbor_info_ref(Ts &&...args)
      : parthenon::variable_names::base_t<false, 8>(std::forward<Ts>(args)...) {}
  static std::string name() { return "neighbor_info_ref"; }
};

struct linear_field_ref : public parthenon::variable_names::base_t<false> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION linear_field_ref(Ts &&...args)
      : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}
  static std::string name() { return "linear_field_ref"; }
};

TaskStatus SetBlockValues(MeshData<Real> *rc);
TaskStatus CopyToReference(MeshData<Real> *md);
TaskStatus CompareFaceGhosts(MeshData<Real> *md);
// Returns the number of mismatches between the send and receive sides of faces-only
// communication over all ranks
int CheckFacesOnlyConsistency(parthenon::Mesh *pmesh);
// Prints a summary of the faces-only check and returns true if it passed
bool ReportFacesOnlyCheck(parthenon::Mesh *pmesh);
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

} // namespace boundary_exchange

#endif // EXAMPLE_BOUNDARY_EXCHANGE_BOUNDARY_EXCHANGE_HPP_
