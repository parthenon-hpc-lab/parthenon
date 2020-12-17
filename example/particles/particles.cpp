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
  // Don't do anything for now
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
  pkg->AddSwarmValue("t", swarm_name, real_swarmvalue_metadata);
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

AmrTag CheckRefinement(Container<Real> &rc) { return AmrTag::same; }

Real EstimateTimestep(std::shared_ptr<Container<Real>> &rc) {
  auto pmb = rc->GetBlockPointer();
  auto pkg = pmb->packages["particles_package"];
  const Real &dt = pkg->Param<Real>("const_dt");
  return dt;
}

TaskStatus SetTimestepTask(std::shared_ptr<Container<Real>> &rc) {
  auto pmb = rc->GetBlockPointer();
  pmb->SetBlockTimestep(parthenon::Update::EstimateTimestep(rc));
  return TaskStatus::complete;
}

} // namespace Particles

// *************************************************//
// define the application driver. in this case,    *//
// that just means defining the MakeTaskList       *//
// function.                                       *//
// *************************************************//
// first some helper tasks

TaskStatus DestroySomeParticles(MeshBlock *pmb) {
  auto pkg = pmb->packages["particles_package"];
  auto swarm = pmb->real_containers.GetSwarmContainer()->Get("my particles");
  auto rng_pool = pkg->Param<RNGPool>("rng_pool");

  // TODO(BRR) short-circuiting this function!
  return TaskStatus::complete;

  // The swarm mask is managed internally and should always be treated as constant. This
  // may be enforced later.
  auto swarm_d = swarm->GetDeviceContext();

  // Randomly mark 10% of particles each timestep for removal
  pmb->par_for(
      "DestroySomeParticles", 0, swarm->get_max_active_index(),
      KOKKOS_LAMBDA(const int n) {
        if (swarm_d.IsActive(n)) {
          auto rng_gen = rng_pool.get_state();
          if (rng_gen.drand() > 0.9) {
            swarm_d.MarkParticleForRemoval(n);
          }
          rng_pool.free_state(rng_gen);
        }
      });

  // Remove marked particles
  swarm->RemoveMarkedParticles();

  return TaskStatus::complete;
}

TaskStatus DepositParticles(MeshBlock *pmb) {
  fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD);
  auto swarm = pmb->real_containers.GetSwarmContainer()->Get("my particles");

  // Meshblock geometry
  const IndexRange &ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange &jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange &kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  const Real &dx_i = pmb->coords.dx1f(pmb->cellbounds.is(IndexDomain::interior));
  const Real &dx_j = pmb->coords.dx2f(pmb->cellbounds.js(IndexDomain::interior));
  const Real &dx_k = pmb->coords.dx3f(pmb->cellbounds.ks(IndexDomain::interior));
  const Real &minx_i = pmb->coords.x1f(ib.s);
  const Real &minx_j = pmb->coords.x2f(jb.s);
  const Real &minx_k = pmb->coords.x3f(kb.s);

  const auto &x = swarm->GetReal("x").Get();
  const auto &y = swarm->GetReal("y").Get();
  const auto &z = swarm->GetReal("z").Get();
  const auto &weight = swarm->GetReal("weight").Get();
  auto swarm_d = swarm->GetDeviceContext();

  auto &particle_dep = pmb->real_containers.Get()->Get("particle_deposition").data;
  // Reset particle count
  pmb->par_for(
      "ZeroParticleDep", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        particle_dep(k, j, i) = 0.;
      });

  pmb->par_for(
      "DepositParticles", 0, swarm->get_max_active_index(), KOKKOS_LAMBDA(const int n) {
        if (swarm_d.IsActive(n)) {
          int i = static_cast<int>((x(n) - minx_i) / dx_i) + ib.s;
          int j = static_cast<int>((y(n) - minx_j) / dx_j) + jb.s;
          int k = static_cast<int>((z(n) - minx_k) / dx_k) + kb.s;

          if (i >= ib.s && i <= ib.e && j >= jb.s && j <= jb.e && k >= kb.s &&
              k <= kb.e) {
            Kokkos::atomic_add(&particle_dep(k, j, i), weight(n));
          }
        }
      });

  return TaskStatus::complete;
}

TaskStatus CreateSomeParticles(MeshBlock *pmb, double t0) {
  printf("[%i] CreateSomeParticles\n", Globals::my_rank);
  auto pkg = pmb->packages["particles_package"];
  auto swarm = pmb->real_containers.GetSwarmContainer()->Get("my particles");
  auto rng_pool = pkg->Param<RNGPool>("rng_pool");
  auto num_particles = pkg->Param<int>("num_particles");
  auto v = pkg->Param<Real>("particle_speed");

  printf("num_particles: %i\n", num_particles);

  const auto new_particles_mask = swarm->AddEmptyParticles(num_particles);

  // Meshblock geometry
  const IndexRange &ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange &jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange &kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  const int &nx_i = pmb->cellbounds.ncellsi(IndexDomain::interior);
  const int &nx_j = pmb->cellbounds.ncellsj(IndexDomain::interior);
  const int &nx_k = pmb->cellbounds.ncellsk(IndexDomain::interior);
  const Real &dx_i = pmb->coords.dx1f(pmb->cellbounds.is(IndexDomain::interior));
  const Real &dx_j = pmb->coords.dx2f(pmb->cellbounds.js(IndexDomain::interior));
  const Real &dx_k = pmb->coords.dx3f(pmb->cellbounds.ks(IndexDomain::interior));
  const Real &minx_i = pmb->coords.x1f(ib.s);
  const Real &minx_j = pmb->coords.x2f(jb.s);
  const Real &minx_k = pmb->coords.x3f(kb.s);

  auto &t = swarm->GetReal("t").Get();
  auto &x = swarm->GetReal("x").Get();
  auto &y = swarm->GetReal("y").Get();
  auto &z = swarm->GetReal("z").Get();
  auto &vx = swarm->GetReal("vx").Get();
  auto &vy = swarm->GetReal("vy").Get();
  auto &vz = swarm->GetReal("vz").Get();
  auto &weight = swarm->GetReal("weight").Get();

  pmb->par_for(
      "CreateSomeParticles", 0, swarm->get_max_active_index(),
      KOKKOS_LAMBDA(const int n) {
        if (new_particles_mask(n)) {
          auto rng_gen = rng_pool.get_state();

          // Randomly sample in space in this meshblock
          x(n) = minx_i + nx_i * dx_i * rng_gen.drand();
          y(n) = minx_j + nx_j * dx_j * rng_gen.drand();
          z(n) = minx_k + nx_k * dx_k * rng_gen.drand();

          // Randomly sample direction on the unit sphere, fixing speed
          Real theta = acos(2. * rng_gen.drand() - 1.);
          Real phi = 2. * M_PI * rng_gen.drand();
          vx(n) = v * sin(theta) * cos(phi);
          vy(n) = v * sin(theta) * sin(phi);
          vz(n) = v * cos(theta);

          t(n) = t0;

          weight(n) = 1.0;

          rng_pool.free_state(rng_gen);
        }
      });

  return TaskStatus::complete;
}

TaskStatus TransportParticles(MeshBlock *pmb, double t0, Integrator *integrator) {
  printf("[%i] TransportParticles\n", Globals::my_rank);
  auto swarm = pmb->real_containers.GetSwarmContainer()->Get("my particles");

  int max_active_index = swarm->get_max_active_index();

  Real dt = integrator->dt;

  auto &t = swarm->GetReal("t").Get();
  auto &x = swarm->GetReal("x").Get();
  auto &y = swarm->GetReal("y").Get();
  auto &z = swarm->GetReal("z").Get();
  const auto &vx = swarm->GetReal("vx").Get();
  const auto &vy = swarm->GetReal("vy").Get();
  const auto &vz = swarm->GetReal("vz").Get();

  const Real &dx_i = pmb->coords.dx1f(pmb->cellbounds.is(IndexDomain::interior));
  const Real &dx_j = pmb->coords.dx2f(pmb->cellbounds.js(IndexDomain::interior));
  const Real &dx_k = pmb->coords.dx3f(pmb->cellbounds.ks(IndexDomain::interior));
  const Real &dx_push = std::min<Real>(dx_i, std::min<Real>(dx_j, dx_k));

  const IndexRange &ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange &jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange &kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  const Real &x_min = pmb->coords.x1f(ib.s);
  const Real &y_min = pmb->coords.x2f(jb.s);
  const Real &z_min = pmb->coords.x3f(kb.s);
  const Real &x_max = pmb->coords.x1f(ib.e + 1);
  const Real &y_max = pmb->coords.x2f(jb.e + 1);
  const Real &z_max = pmb->coords.x3f(kb.e + 1);

  //const Real x_min_global = pmb->pmy_mesh->mesh_size.x1min;

  auto swarm_d = swarm->GetDeviceContext();

  //ParArrayND<Real> t("time", max_active_index + 1);

  // Simple particle push: push particles half a zone width until they have
  // traveled one integrator timestep's worth of time
  pmb->par_for(
      "TransportParticles", 0, max_active_index, KOKKOS_LAMBDA(const int n) {
        printf("[%i] is on current: %i\n", n, swarm_d.IsOnCurrentMeshBlock(n));
        if (swarm_d.IsActive(n) && swarm_d.IsOnCurrentMeshBlock(n)) {
          Real v = sqrt(vx(n) * vx(n) + vy(n) * vy(n) + vz(n) * vz(n));
          while (t(n) < t0 + dt) {
            Real dt_cell = dx_push / v;
            Real dt_end = t0 + dt - t(n);
            Real dt_push = std::min<Real>(dt_cell, dt_end);

            x(n) += vx(n) * dt_push;
            y(n) += vy(n) * dt_push;
            z(n) += vz(n) * dt_push;
            t(n) += dt_push;

            // Apply physical boundaries before indicating communication?

            int neighborBlockIndex = swarm_d.GetNeighborBlockIndex(n, x(n), y(n), z(n));

            if (neighborBlockIndex != -1) {
              printf("[%i] particle (%e %e %e) -> %i\n",
                Globals::my_rank, x(n), y(n), z(n), neighborBlockIndex);
              // Particle no longer on this block
              break;
            }

            // If outside of meshblock, get neighbor index

            // Periodic boundaries (handled by MPI)
            /*if (x(n) < x_min) {
              x(n) = x_max - (x_min - x(n));
            }
            if (x(n) > x_max) {
              x(n) = x_min + (x(n) - x_max);
            }
            if (y(n) < y_min) {
              y(n) = y_max - (y_min - y(n));
            }
            if (y(n) > y_max) {
              y(n) = y_min + (y(n) - y_max);
            }
            if (z(n) < z_min) {
              z(n) = z_max - (z_min - z(n));
            }
            if (z(n) > z_max) {
              z(n) = z_min + (z(n) - z_max);
            }*/
          }
        }
      });

  // Only mark as finished if mpiStatus is complete
  //if (swarm->mpiStatus) {
  //  return TaskStatus::complete;
  //} else {
  //  return TaskStatus::incomplete;
  //}
  return TaskStatus::complete;
}

TaskStatus Defrag(MeshBlock *pmb) {
  auto s = pmb->real_containers.GetSwarmContainer()->Get("my particles");

  // Only do this if list is getting too sparse. This criterion (whether there
  // are *any* gaps in the list) is very aggressive
  if (s->get_num_active() <= s->get_max_active_index()) {
    s->Defrag();
  }

  return TaskStatus::complete;
}

// See the advection_driver.hpp declaration for a description of how this function gets
// called.
TaskList ParticleDriver::MakeTaskList(MeshBlock *pmb, int stage) {
  TaskList tl;

  TaskID none(0);

  double t0 = tm.time;

  auto sc = pmb->real_containers.GetSwarmContainer();

  auto swarm = sc->Get("my particles");

  auto start_comm = tl.AddTask(none, &SwarmContainer::StartCommunication, sc.get(),
    BoundaryCommSubset::all);

  auto create_some_particles =
      tl.AddTask(none, CreateSomeParticles, pmb, t0);

  auto transport_particles = tl.AddTask(create_some_particles, TransportParticles, pmb, t0, integrator);
  auto send = tl.AddTask(create_some_particles, &SwarmContainer::Send, sc.get(), BoundaryCommSubset::all);
  auto receive = tl.AddTask(create_some_particles, &SwarmContainer::Receive, sc.get(), BoundaryCommSubset::all);

  auto finalize_comm = tl.AddTask(create_some_particles, &SwarmContainer::FinishCommunication,
    sc.get(), BoundaryCommSubset::all);

  auto deposit_particles = tl.AddTask(finalize_comm, DepositParticles, pmb);

  auto defrag = tl.AddTask(deposit_particles, Defrag, pmb);

/*
  auto transport_particles = tl.AddTask(create_some_particles, TransportParticles, pmb, t0, integrator);

  auto destroy_some_particles =
      tl.AddTask(transport_particles, DestroySomeParticles, pmb);

  auto silly_update = tl.AddTask(destroy_some_particles, &SwarmContainer::SillyUpdate,
  sc.get());

  auto send = tl.AddTask(destroy_some_particles, &SwarmContainer::Send, sc.get(), BoundaryCommSubset::all);
  auto receive = tl.AddTask(destroy_some_particles, &SwarmContainer::Receive, sc.get(), BoundaryCommSubset::all);

  auto finalize_comm = tl.AddTask(destroy_some_particles, &SwarmContainer::FinishCommunication,
    sc.get(), BoundaryCommSubset::all);

  auto deposit_particles = tl.AddTask(finalize_comm, DepositParticles, pmb);

  auto defrag = tl.AddTask(deposit_particles, Defrag, pmb);
*/
  return tl;
}

} // namespace particles_example
