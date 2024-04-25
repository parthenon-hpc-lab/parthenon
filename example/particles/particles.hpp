//========================================================================================
// (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
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
#ifndef EXAMPLE_PARTICLES_PARTICLES_HPP_
#define EXAMPLE_PARTICLES_PARTICLES_HPP_

#include <memory>

#include "Kokkos_Random.hpp"

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;
using namespace parthenon;

namespace particles_example {

// Duplicates of ParticleBoundIX1Outflow and ParticleBoundOX1Outflow to illustrate custom
// particle boundary conditions
class ParticleBoundIX1User : public ParticleBound {
 public:
  KOKKOS_INLINE_FUNCTION void Apply(const int n, Real &x, Real &y, Real &z,
                                    const SwarmDeviceContext &swarm_d) const override {
    if (x < swarm_d.x_min_global_) {
      swarm_d.MarkParticleForRemoval(n);
    }
  }
};

class ParticleBoundOX1User : public ParticleBound {
 public:
  KOKKOS_INLINE_FUNCTION void Apply(const int n, Real &x, Real &y, Real &z,
                                    const SwarmDeviceContext &swarm_d) const override {
    if (x > swarm_d.x_max_global_) {
      swarm_d.MarkParticleForRemoval(n);
    }
  }
};

typedef Kokkos::Random_XorShift64_Pool<> RNGPool;

class ParticleDriver : public EvolutionDriver {
 public:
  ParticleDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm)
      : EvolutionDriver(pin, app_in, pm), integrator(pin) {}
  TaskCollection MakeParticlesCreationTaskCollection() const;
  TaskCollection MakeParticlesUpdateTaskCollection() const;
  TaskCollection MakeFinalizationTaskCollection() const;
  TaskListStatus Step();

 private:
  LowStorageIntegrator integrator;
};

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin);
Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin);

std::unique_ptr<ParticleBound, DeviceDeleter<parthenon::DevMemSpace>> SetSwarmIX1UserBC();

std::unique_ptr<ParticleBound, DeviceDeleter<parthenon::DevMemSpace>> SetSwarmOX1UserBC();

namespace Particles {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
AmrTag CheckRefinement(MeshBlockData<Real> *rc);
Real EstimateTimestepBlock(MeshBlockData<Real> *rc);

} // namespace Particles

} // namespace particles_example

#endif // EXAMPLE_PARTICLES_PARTICLES_HPP_
