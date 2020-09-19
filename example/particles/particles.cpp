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

// *************************************************//
// redefine some internal parthenon functions      *//
// *************************************************//
namespace particles_example {

Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;
  packages["particles_package"] = particles_example::Particles::Initialize(pin.get());
  return packages;
}

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  /*auto &rc = pmb->real_containers.Get();
  auto pkg = pmb->packages["particles_package"];

  auto &sc = pmb->real_containers.GetSwarmContainer();
  auto &s = sc->Get("my particles");

  // Add the number of empty particles requested in parameter file
  const int &num_particles_to_add = pkg->Param<int>("num_particles");
  auto new_particle_mask = s->AddEmptyParticles(num_particles_to_add);

  // WARNING do not get these references before resizing the swarm. Otherwise,
  // you'll get segfaults
  auto &x = s->GetReal("x").Get();
  auto &y = s->GetReal("y").Get();
  auto &z = s->GetReal("z").Get();
  auto &vx = s->GetReal("vx").Get();
  auto &vy = s->GetReal("vy").Get();
  auto &vz = s->GetReal("vz").Get();
  auto &weight = s->GetReal("weight").Get();
  auto &mask = s->GetMask().Get();

  const Real &v = pkg->Param<Real>("particle_speed");

  printf("Problem generator!");

  pmb->par_for("particles_package::ProblemGenerator", 0, s->get_max_active_index(),
    KOKKOS_LAMBDA(const int n) {
      if (new_particle_mask(n)) {
        x(n) = 1.e-1*n;
        y(n) = 1.e-2*n;
        z(n) = 1.e-3*n;
        vx(n) = v;
        vy(n) = 0.;
        vz(n) = 0.;
        weight(n) = 1.0;
      }
    });*/
}

// *************************************************//
// define the "physics" package particles_package, *//
// which includes defining various functions that  *//
// control how parthenon functions and any tasks   *//
// needed to implement the "physics"               *//
// *************************************************//

namespace Particles {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("particles_package");

  int num_particles = pin->GetOrAddInteger("Particles", "num_particles", 100);
  pkg->AddParam<>("num_particles", num_particles);
  Real particle_speed = pin->GetOrAddReal("Particles", "particle_speed", 1.0);
  pkg->AddParam<>("particle_speed", particle_speed);
  Real const_dt = pin->GetOrAddReal("Particles", "const_dt", 1.0);
  pkg->AddParam<>("const_dt", const_dt);

  // Initialize random number generator pool
  int rng_seed = pin->GetInteger("Particles", "rng_seed");
  pkg->AddParam<>("rng_seed", rng_seed);
  RNGPool rng_pool(rng_seed);
  pkg->AddParam<>("rng_pool", rng_pool);

  std::string swarm_name = "my particles";
  Metadata swarm_metadata;
  pkg->AddSwarm(swarm_name, swarm_metadata);
  Metadata real_swarmvalue_metadata({Metadata::Real});
  pkg->AddSwarmValue("vx", swarm_name, real_swarmvalue_metadata);
  pkg->AddSwarmValue("vy", swarm_name, real_swarmvalue_metadata);
  pkg->AddSwarmValue("vz", swarm_name, real_swarmvalue_metadata);
  pkg->AddSwarmValue("weight", swarm_name, real_swarmvalue_metadata);

  std::string field_name = "particle_deposition";
  Metadata m({Metadata::Cell, Metadata::Independent});
  pkg->AddField(field_name, m);

  pkg->EstimateTimestep = EstimateTimestep;

  return pkg;
}

AmrTag CheckRefinement(Container<Real> &rc) {
  return AmrTag::same;
}

Real EstimateTimestep(std::shared_ptr<Container<Real>> &rc) {
  auto pmb = rc->pmy_block;
  auto pkg = pmb->packages["particles_package"];
  const Real &dt = pkg->Param<Real>("const_dt");
  return dt;
}

TaskStatus SetTimestepTask(std::shared_ptr<Container<Real>> &rc) {
  MeshBlock *pmb = rc->pmy_block;
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
/*TaskStatus UpdateContainer(MeshBlock *pmb, int stage,
                           std::vector<std::string> &stage_name, Integrator *integrator) {
  // const Real beta = stage_wghts[stage-1].beta;
  const Real beta = integrator->beta[stage - 1];
  const Real dt = integrator->dt;
  auto &base = pmb->real_containers.Get();
  auto &cin = pmb->real_containers.Get(stage_name[stage - 1]);
  auto &cout = pmb->real_containers.Get(stage_name[stage]);
  auto &dudt = pmb->real_containers.Get("dUdt");
  parthenon::Update::AverageContainers(cin, base, beta);
  parthenon::Update::UpdateContainer(cin, dudt, beta * dt, cout);
  return TaskStatus::complete;
}*/

TaskStatus DestroySomeParticles(MeshBlock *pmb, int stage,
  std::vector<std::string> &stage_name, Integrator *integrator) {
  auto pkg = pmb->packages["particles_package"];
  auto swarm = pmb->real_containers.GetSwarmContainer()->Get("my particles");
  auto rng_pool = pkg->Param<RNGPool>("rng_pool");

  auto &mask = swarm->GetMask().Get();
  auto &marked_for_removal = swarm->GetMarkedForRemoval().Get();

  pmb->par_for("DestroySomeParticles", 0, swarm->get_max_active_index(),
    KOKKOS_LAMBDA(const int n) {
      if (mask(n)) {
        auto rng_gen = rng_pool.get_state();
        // Randomly remove 10% of particles
        if (rng_gen.drand() > 0.9) {
          printf("Removing particle %i!\n", n);
          marked_for_removal(n) = true;
        }
        rng_pool.free_state(rng_gen);
      }
    });

  printf("removing marked particles after destruction!\n");
  swarm->RemoveMarkedParticles();

  return TaskStatus::complete;
}

TaskStatus DepositParticles(MeshBlock *pmb, int stage,
  std::vector<std::string> &stage_name, Integrator *integrator) {
  auto swarm = pmb->real_containers.GetSwarmContainer()->Get("my particles");

  // Meshblock geometry
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  Real dx_i = pmb->coords.dx1f(pmb->cellbounds.is(IndexDomain::interior));
  Real dx_j = pmb->coords.dx2f(pmb->cellbounds.js(IndexDomain::interior));
  Real dx_k = pmb->coords.dx3f(pmb->cellbounds.ks(IndexDomain::interior));
  Real minx_i = pmb->coords.x1v(ib.s);
  Real minx_j = pmb->coords.x1v(jb.s);
  Real minx_k = pmb->coords.x1v(kb.s);

  auto &x = swarm->GetReal("x").Get();
  auto &y = swarm->GetReal("y").Get();
  auto &z = swarm->GetReal("z").Get();
  auto &weight = swarm->GetReal("weight").Get();
  auto &mask = swarm->GetMask().Get();

  auto &particle_dep = pmb->real_containers.Get()->Get("particle_deposition").data;
  // Reset particle count
  pmb->par_for("ZeroParticleDep", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
    KOKKOS_LAMBDA(const int k, const int j, const int i) {
      particle_dep(k, j, i) = 0.;
    });

  pmb->par_for("DepositParticles", 0, swarm->get_max_active_index(),
    KOKKOS_LAMBDA(const int n) {
      if (mask(n)) {
        int i = static_cast<int>((x(n) - minx_i)/dx_i) + ib.s;
        int j = static_cast<int>((y(n) - minx_j)/dx_j) + jb.s;
        int k = static_cast<int>((z(n) - minx_k)/dx_k) + kb.s;

        Kokkos::atomic_add(&particle_dep(k,j,i), weight(n));
      }
    });

  return TaskStatus::complete;
}


TaskStatus CreateSomeParticles(MeshBlock *pmb, int stage,
  std::vector<std::string> &stage_name, Integrator *integrator) {

  auto pkg = pmb->packages["particles_package"];
  auto swarm = pmb->real_containers.GetSwarmContainer()->Get("my particles");
  auto rng_pool = pkg->Param<RNGPool>("rng_pool");
  auto num_particles = pkg->Param<int>("num_particles");
  auto v = pkg->Param<Real>("particle_speed");

  auto new_particles_mask = swarm->AddEmptyParticles(num_particles);

  // Meshblock geometry
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  int nx_i = pmb->cellbounds.ncellsi(IndexDomain::interior);
  int nx_j = pmb->cellbounds.ncellsj(IndexDomain::interior);
  int nx_k = pmb->cellbounds.ncellsk(IndexDomain::interior);
  Real dx_i = pmb->coords.dx1f(pmb->cellbounds.is(IndexDomain::interior));
  Real dx_j = pmb->coords.dx2f(pmb->cellbounds.js(IndexDomain::interior));
  Real dx_k = pmb->coords.dx3f(pmb->cellbounds.ks(IndexDomain::interior));
  Real minx_i = pmb->coords.x1v(ib.s);
  Real minx_j = pmb->coords.x1v(jb.s);
  Real minx_k = pmb->coords.x1v(kb.s);

  auto &x = swarm->GetReal("x").Get();
  auto &y = swarm->GetReal("y").Get();
  auto &z = swarm->GetReal("z").Get();
  auto &vx = swarm->GetReal("vx").Get();
  auto &vy = swarm->GetReal("vy").Get();
  auto &vz = swarm->GetReal("vz").Get();
  auto &weight = swarm->GetReal("weight").Get();

  pmb->par_for("CreateSomeParticles", 0, swarm->get_max_active_index(),
    KOKKOS_LAMBDA(const int n) {
      if (new_particles_mask(n)) {
        printf("Creating particle %i!\n", n);
        auto rng_gen = rng_pool.get_state();

        // Randomly sample in space in this meshblock
        x(n) = minx_i + nx_i*dx_i*rng_gen.drand();
        y(n) = minx_j + nx_j*dx_j*rng_gen.drand();
        z(n) = minx_k + nx_k*dx_k*rng_gen.drand();

        // Randomly sample direction on the unit sphere, fixing speed
        Real theta = acos(2. * rng_gen.drand() - 1.);
        Real phi = 2. * M_PI * rng_gen.drand();
        vx(n) = v*sin(theta)*cos(phi);
        vy(n) = v*sin(theta)*sin(phi);
        vz(n) = v*cos(theta);

        weight(n) = 1.0;

        rng_pool.free_state(rng_gen);
      }
    });

  return TaskStatus::complete;
}

TaskStatus UpdateSwarm(MeshBlock *pmb, int stage,
                       std::vector<std::string> &stage_name,
                       Integrator *integrator) {
  auto swarm = pmb->real_containers.GetSwarmContainer()->Get("my particles");
  //parthenon::Update::TransportSwarm(swarm, swarm, integrator->dt);
  return TaskStatus::complete;
}

/*TaskStatus RemoveSecondParticle(MeshBlock *pmb, int stage,
 std::vector<std::string> &stage_name, Integrator *integrator) {

  auto swarm = pmb->real_containers.GetSwarmContainer()->Get("my particles");

  auto &mask = swarm->GetMask().Get();
  auto &marked_for_removal = swarm->GetMarkedForRemoval().Get();

  pmb->par_for("RemoveSecondParticle", 0, swarm->get_max_active_index(),
    KOKKOS_LAMBDA(const int n) {
      if (mask(n) && n == 1) {
        marked_for_removal(n) = true;
      }
    });

  swarm->RemoveMarkedParticles();

  return TaskStatus::complete;
}*/

TaskStatus Defrag(MeshBlock *pmb, int stage,
  std::vector<std::string> &stage_name, Integrator *integrator) {

  auto s = pmb->real_containers.GetSwarmContainer()->Get("my particles");
  // Don't need to do this every timestep
  s->Defrag();

  printf("num active: %i max index: %i\n", s->get_num_active(),
    s->get_max_active_index());

  return TaskStatus::complete;
}

/*TaskStatus AddTwoParticles(MeshBlock *pmb, int stage,
  std::vector<std::string> &stage_name, Integrator *integrator) {

  auto s = pmb->real_containers.GetSwarmContainer()->Get("my particles");
  auto pkg = pmb->packages["particles_package"];

  auto new_particle_mask = s->AddEmptyParticles(2);

  auto &x = s->GetReal("x").Get();
  auto &y = s->GetReal("y").Get();
  auto &z = s->GetReal("z").Get();
  auto &vx = s->GetReal("vx").Get();
  auto &vy = s->GetReal("vy").Get();
  auto &vz = s->GetReal("vz").Get();
  auto &weight = s->GetReal("weight").Get();

  const Real &v = pkg->Param<Real>("particle_speed");

  pmb->par_for("particles_package::AddTwoParticles", 0, s->get_max_active_index(),
    KOKKOS_LAMBDA(const int n) {
      if (new_particle_mask(n)) {
        x(n) = 1.e-1*n;
        y(n) = 1.e-2*n;
        z(n) = 1.e-3*n;
        vx(n) = v;
        vy(n) = 0.;//1.e-5;
        vz(n) = 0.;//1.e-4*n;
        weight(n) = 1.0;
      }
  });

  return TaskStatus::complete;
}

TaskStatus MyContainerTask(std::shared_ptr<Container<Real>> container) {
  return TaskStatus::complete;
}*/

// See the advection_driver.hpp declaration for a description of how this function gets called.
TaskList ParticleDriver::MakeTaskList(MeshBlock *pmb, int stage) {
  TaskList tl;

  TaskID none(0);
  // first make other useful containers
  if (stage == 1) {
    auto container = pmb->real_containers.Get();
    pmb->real_containers.Add("my container", container);
    auto base = pmb->real_containers.GetSwarmContainer();
  }

  auto sc = pmb->real_containers.GetSwarmContainer();

  auto swarm = sc->Get("my particles");

  printf("update swarm\n");
  auto update_swarm = tl.AddTask(UpdateSwarm, none, pmb, stage,
                                            stage_name, integrator);

  printf("destroy some particles\n");
  auto destroy_some_particles = tl.AddTask(DestroySomeParticles, update_swarm, pmb, stage,
    stage_name, integrator);

  printf("create some particles\n");
  auto create_some_particles = tl.AddTask(CreateSomeParticles, destroy_some_particles, pmb, stage, stage_name, integrator);

  printf("deposit particles\n");
  auto deposit_particles = tl.AddTask(DepositParticles, create_some_particles, pmb, stage, stage_name, integrator);

  //auto remove_second_particle = tl.AddTask(RemoveSecondParticle, update_swarm, pmb, stage,
  //                                         stage_name, integrator);

  //auto add_two_particles = tl.AddTask(AddTwoParticles, remove_second_particle, pmb, stage,
  //                                    stage_name, integrator);

  printf("defrag\n");
  auto defrag = tl.AddTask(Defrag, deposit_particles, pmb, stage,
                           stage_name, integrator);

  //auto container = pmb->real_containers.Get("my container");

  //auto update_container = tl.AddTask(MyContainerTask, none, container);

  return tl;
}

} // namespace particles_example
