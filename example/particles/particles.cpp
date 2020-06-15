//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
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

#include "particles.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "bvals/boundary_conditions.hpp"
#include "bvals/bvals.hpp"
#include "driver/multistage.hpp"
#include "interface/params.hpp"
#include "interface/state_descriptor.hpp"
#include "mesh/mesh.hpp"
#include "parthenon_manager.hpp"
#include "reconstruct/reconstruction.hpp"
#include "refinement/refinement.hpp"

using parthenon::BlockStageNamesIntegratorTask;
using parthenon::BlockStageNamesIntegratorTaskFunc;
using parthenon::BlockTask;
using parthenon::CellVariable;
using parthenon::Integrator;
using parthenon::Metadata;
using parthenon::Params;
using parthenon::ParArrayND;
using parthenon::ParthenonManager;

// *************************************************//
// redefine some weakly linked parthenon functions *//
// *************************************************//

namespace parthenon {

Packages_t ParthenonManager::ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;
  packages["Particles"] = particles_example::Particles::Initialize(pin.get());
  return packages;
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  auto pkg = packages["Particles"];

  SwarmContainer &sc = real_containers.GetSwarmContainer();
  Swarm &s = sc.Get("my particles");

  // Here we demonstrate the different ways to add particles

  // Add the number of empty particles requested in parameter file
  const int &num_particles_to_add = pkg->Param<int>("num_particles");
  std::vector<int> empty_particle_indices = s.AddEmptyParticles(num_particles_to_add);

  // WARNING do not get these references before resizing the swarm -- you'll get
  // segfaults
  auto &x = s.GetReal("x").Get();
  auto &y = s.GetReal("y").Get();
  auto &z = s.GetReal("z").Get();
  auto &vx = s.GetReal("vx").Get();
  auto &vy = s.GetReal("vy").Get();
  auto &vz = s.GetReal("vz").Get();

  for (int n : empty_particle_indices) {
    x(n) = 1.e-1*n;
    y(n) = 1.e-2*n;
    z(n) = 1.e-3*n;
    vx(n) = 0.1;
    vy(n) = 1.e-5;
    vz(n) = 1.e-4*n;
  }

  // Add 2 uniformly spaced particles
  //auto uniform_particle_indices = s.AddUniformParticles(2);
  //for (auto n : empty_particle_indices) {
  //  vx(n) = 0.1;
  //  vy(n) = 0.;
  //  vz(n) = 0.;
  //}
}

} // namespace parthenon

// *************************************************//
// define the "physics" package Particles, which   *//
// includes defining various functions that control*//
// how parthenon functions and any tasks needed to *//
// implement the "physics"                         *//
// *************************************************//

namespace particles_example {
namespace Particles {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("Particles");

  int num_particles = pin->GetOrAddInteger("Particles", "num_particles", 100);
  pkg->AddParam<>("num_particles", num_particles);
  Real particle_speed = pin->GetOrAddReal("Particles", "particle_speed", 1.0);
  pkg->AddParam<>("particle_speed", particle_speed);

  std::string swarm_name = "my particles";
  Metadata swarm_metadata;
  pkg->AddSwarm(swarm_name, swarm_metadata);
  Metadata real_swarmvalue_metadata({Metadata::Real});
  pkg->AddSwarmValue("weight", swarm_name, real_swarmvalue_metadata);
  pkg->AddSwarmValue("vx", swarm_name, real_swarmvalue_metadata);
  pkg->AddSwarmValue("vy", swarm_name, real_swarmvalue_metadata);
  pkg->AddSwarmValue("vz", swarm_name, real_swarmvalue_metadata);

  pkg->EstimateTimestep = EstimateTimestep;

  return pkg;
}

AmrTag CheckRefinement(Container<Real> &rc) {
  return AmrTag::same;
}

Real EstimateTimestep(Container<Real> &rc) {
  return 0.5;
}

TaskStatus SetTimestepTask(Container<Real> &rc) {
  MeshBlock *pmb = rc.pmy_block;
  pmb->SetBlockTimestep(parthenon::Update::EstimateTimestep(rc));
  return TaskStatus::complete;
}

//}

} // namespace Particles

// *************************************************//
// define the application driver. in this case,    *//
// that just means defining the MakeTaskList       *//
// function.                                       *//
// *************************************************//
// first some helper tasks
TaskStatus UpdateContainer(MeshBlock *pmb, int stage,
                           std::vector<std::string> &stage_name, Integrator *integrator) {
  // const Real beta = stage_wghts[stage-1].beta;
  const Real beta = integrator->beta[stage - 1];
  const Real dt = integrator->dt;
  Container<Real> &base = pmb->real_containers.Get();
  Container<Real> &cin = pmb->real_containers.Get(stage_name[stage - 1]);
  Container<Real> &cout = pmb->real_containers.Get(stage_name[stage]);
  Container<Real> &dudt = pmb->real_containers.Get("dUdt");
  parthenon::Update::AverageContainers(cin, base, beta);
  parthenon::Update::UpdateContainer(cin, dudt, beta * dt, cout);
  return TaskStatus::complete;
}

TaskStatus UpdateSwarm(MeshBlock *pmb, int stage,
                       std::vector<std::string> &stage_name,
                       Integrator *integrator) {
  Swarm &swarm = pmb->real_containers.GetSwarmContainer().Get("my particles");
  parthenon::Update::TransportSwarm(swarm, swarm, integrator->dt);
  return TaskStatus::complete;
}

TaskStatus MyContainerTask(Container<Real> container) {
  return TaskStatus::complete;
}

// See the advection.hpp declaration for a description of how this function gets called.
TaskList ParticleDriver::MakeTaskList(MeshBlock *pmb, int stage) {
  TaskList tl;

  TaskID none(0);
  // first make other useful containers
  if (stage == 1) {
    Container<Real> &container = pmb->real_containers.Get();
    pmb->real_containers.Add("my container", container);
    SwarmContainer &base = pmb->real_containers.GetSwarmContainer();
  }

  SwarmContainer sc = pmb->real_containers.GetSwarmContainer();

  Swarm &swarm = sc.Get("my particles");

  auto update_swarm = tl.AddTask<SwarmTask>(UpdateSwarm, none, pmb, stage,
                                            stage_name, integrator);

  Container<Real> container = pmb->real_containers.Get("my container");

  auto update_container = tl.AddTask<ContainerTask>(MyContainerTask, none, container);

  // estimate next time step
  if (stage == integrator->nstages) {
    // Do nothing
    //AddContainerTask();
    //pmb->SetBlockTimestep(0.25);
  }
  return tl;
}

} // namespace particles_example
