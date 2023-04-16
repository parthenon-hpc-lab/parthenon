//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2021 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
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
#ifndef EXAMPLE_PARTICLE_LEAPFROG_PARTICLE_LEAPFROG_HPP_
#define EXAMPLE_PARTICLE_LEAPFROG_PARTICLE_LEAPFROG_HPP_

#include <memory>

#include "Kokkos_Random.hpp"

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;
using namespace parthenon;

namespace particles_leapfrog {

class ParticleDriver : public EvolutionDriver {
 public:
  ParticleDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm)
      : EvolutionDriver(pin, app_in, pm), integrator(pin) {}
  TaskCollection MakeParticlesUpdateTaskCollection() const;
  TaskCollection MakeFinalizationTaskCollection() const;
  TaskListStatus Step();

 private:
  LowStorageIntegrator integrator;
};

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin);
Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin);

namespace Particles {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
AmrTag CheckRefinement(MeshBlockData<Real> *rc);
Real EstimateTimestepBlock(MeshBlockData<Real> *rc);

} // namespace Particles

} // namespace particles_leapfrog

#endif // EXAMPLE_PARTICLE_LEAPFROG_PARTICLE_LEAPFROG_HPP_
