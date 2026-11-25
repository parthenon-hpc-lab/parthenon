//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2026 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
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

#ifndef EXAMPLE_LUKEWARM_TRACERS_LUKEWARM_TRACERS_HPP_
#define EXAMPLE_LUKEWARM_TRACERS_LUKEWARM_TRACERS_HPP_

// This file was made in part with generative AI.

#include <memory>

#include "Kokkos_Random.hpp"

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;
using namespace parthenon;

// The tracers of example/particle_tracers, carried by lukewarms instead of swarms
namespace lukewarm_tracers {

class ParticleDriver : public EvolutionDriver {
 public:
  using Integrator_t = parthenon::LowStorageIntegrator;
  using IntegratorPtr_t = std::unique_ptr<Integrator_t>;
  ParticleDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm)
      : EvolutionDriver(pin, app_in, pm),
        integrator(std::make_unique<Integrator_t>(pin)) {}
  TaskListStatus Step();
  TaskCollection StepTasks();

 protected:
  IntegratorPtr_t integrator;
};

Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin);
void ProblemGenerator(Mesh *pm, ParameterInput *pin, MeshData<Real> *md);

namespace particles_package {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
Real EstimateTimestepMesh(MeshData<Real> *rc);
void SourceTracers(Mesh *pm, ParameterInput *pin, MeshData<Real> *md);

} // namespace particles_package

namespace advection_package {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
Real EstimateTimestepMesh(MeshData<Real> *rc);

} // namespace advection_package

} // namespace lukewarm_tracers

#endif // EXAMPLE_LUKEWARM_TRACERS_LUKEWARM_TRACERS_HPP_
