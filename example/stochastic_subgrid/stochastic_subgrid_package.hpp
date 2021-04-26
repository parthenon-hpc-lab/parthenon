//========================================================================================
// (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
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
#ifndef EXAMPLE_STOCHASTIC_SUBGRID_STOCHASTIC_SUBGRID_PACKAGE_HPP_
#define EXAMPLE_STOCHASTIC_SUBGRID_STOCHASTIC_SUBGRID_PACKAGE_HPP_

#include <memory>

#include <interface/state_descriptor.hpp>
#include <parthenon/package.hpp>

namespace stochastic_subgrid_package {
using namespace parthenon::package::prelude;
using parthenon::Packages_t;

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
AmrTag CheckRefinement(MeshBlockData<Real> *rc);
TaskStatus ComputeNumIter(std::shared_ptr<MeshData<Real>> &md, Packages_t &packages);
void DoLotsOfWork(MeshBlockData<Real> *rc);
Real EstimateTimestepBlock(MeshBlockData<Real> *rc);
TaskStatus CalculateFluxes(std::shared_ptr<MeshBlockData<Real>> &rc);

} // namespace stochastic_subgrid_package

#endif // EXAMPLE_STOCHASTIC_SUBGRID_STOCHASTIC_SUBGRID_PACKAGE_HPP_
