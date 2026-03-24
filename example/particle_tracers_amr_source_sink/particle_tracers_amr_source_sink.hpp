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

// This file was made in part with generative AI

#ifndef EXAMPLE_PARTICLE_TRACERS_AMR_SOURCE_SINK_PARTICLE_TRACERS_AMR_SOURCE_SINK_HPP_
#define EXAMPLE_PARTICLE_TRACERS_AMR_SOURCE_SINK_PARTICLE_TRACERS_AMR_SOURCE_SINK_HPP_

#include <memory>

#include "Kokkos_Random.hpp"

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;
using namespace parthenon;

namespace particle_tracers_amr_source_sink {

struct advected {
  static std::string name() { return "advected"; }
  static bool regex() { return false; }
  static constexpr int idx = 0;
};

struct tracer_deposition {
  static std::string name() { return "tracer_deposition"; }
  static bool regex() { return false; }
  static constexpr int idx = 0;
};

class ParticleDriver : public MultiStageDriver {
 public:
  ParticleDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm)
      : MultiStageDriver(pin, app_in, pm) {}
  TaskCollection MakeTaskCollection(BlockList_t &blocks, int stage);
};

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin);
Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin);

namespace particles_package {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
AmrTag CheckRefinement(MeshBlockData<Real> *rc);
void FinalInitialization(Mesh *pm, ParameterInput *pin, MeshData<Real> *md);

} // namespace particles_package

namespace advection_package {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

} // namespace advection_package

} // namespace particle_tracers_amr_source_sink

#endif // EXAMPLE_PARTICLE_TRACERS_AMR_SOURCE_SINK_PARTICLE_TRACERS_AMR_SOURCE_SINK_HPP_
