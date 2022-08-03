//========================================================================================
// (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
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
#ifndef EXAMPLE_SPARSE_ADVECTION_SPARSE_ADVECTION_PACKAGE_HPP_
#define EXAMPLE_SPARSE_ADVECTION_SPARSE_ADVECTION_PACKAGE_HPP_

#include <memory>

#include <parthenon/package.hpp>

namespace sparse_advection_package {

using namespace parthenon::package::prelude;

static constexpr int NUM_FIELDS = 4;
using RealArr_t = Kokkos::Array<Real, NUM_FIELDS>;

struct sparse_vt : public parthenon::variable_names::base_t<true> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION sparse_vt(Ts &&...args)
      : parthenon::variable_names::base_t<true>(std::forward<Ts>(args)...) {}
  static std::string name() { return "sparse_.*"; }
};

struct vx_vt : public parthenon::variable_names::base_t<true> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION vx_vt(Ts &&...args)
      : parthenon::variable_names::base_t<true>(std::forward<Ts>(args)...) {}
  static std::string name() { return "vx_.*"; }
};

struct vy_vt : public parthenon::variable_names::base_t<true> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION vy_vt(Ts &&...args)
      : parthenon::variable_names::base_t<true>(std::forward<Ts>(args)...) {}
  static std::string name() { return "vy_.*"; }
};

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

AmrTag CheckRefinement(MeshBlockData<Real> *rc);

Real EstimateTimestepBlock(MeshBlockData<Real> *rc);

TaskStatus CalculateFluxes(std::shared_ptr<MeshData<Real>> &rc);
} // namespace sparse_advection_package

#endif // EXAMPLE_SPARSE_ADVECTION_SPARSE_ADVECTION_PACKAGE_HPP_
