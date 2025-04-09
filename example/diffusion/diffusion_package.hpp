//========================================================================================
// (C) (or copyright) 2023-2025. Triad National Security, LLC. All rights reserved.
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
#ifndef EXAMPLE_DIFFUSION_DIFFUSION_PACKAGE_HPP_
#define EXAMPLE_DIFFUSION_DIFFUSION_PACKAGE_HPP_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>

#define VARIABLE(ns, varname)                                                            \
  struct varname : public parthenon::variable_names::base_t<false> {                     \
    template <class... Ts>                                                               \
    KOKKOS_INLINE_FUNCTION varname(Ts &&...args)                                         \
        : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}         \
    static std::string name() { return #ns "." #varname; }                               \
  }

namespace diffusion_package {
using namespace parthenon::package::prelude;

VARIABLE(diffusion, D);
VARIABLE(diffusion, u);

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
TaskStatus SetRHS(std::shared_ptr<MeshData<Real>> md,
                  std::shared_ptr<MeshData<Real>> md_rhs);
parthenon::TaskStatus SetDiffusionCoefficient(std::shared_ptr<MeshData<Real>> md,
                                              const Real dt);
Real EstimateTimestep(MeshData<Real> *md);

} // namespace diffusion_package

#endif // EXAMPLE_DIFFUSION_DIFFUSION_PACKAGE_HPP_
