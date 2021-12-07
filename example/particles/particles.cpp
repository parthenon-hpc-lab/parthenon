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
  packages.Add(particles_example::Particles::Initialize(pin.get()));
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

  Real destroy_particles_frac =
      pin->GetOrAddReal("Particles", "destroy_particles_frac", 0.0);
  pkg->AddParam<>("destroy_particles_frac", destroy_particles_frac);
  PARTHENON_REQUIRE(
      destroy_particles_frac >= 0. && destroy_particles_frac <= 1.,
      "Fraction of particles to destroy each timestep must be between 0 and 1");

  bool orbiting_particles =
      pin->GetOrAddBoolean("Particles", "orbiting_particles", false);
  pkg->AddParam<>("orbiting_particles", orbiting_particles);

  // Initialize random number generator pool
  int rng_seed = pin->GetInteger("Particles", "rng_seed");
  pkg->AddParam<>("rng_seed", rng_seed);
  RNGPool rng_pool(rng_seed);
  pkg->AddParam<>("rng_pool", rng_pool);

  std::string swarm_name = "my particles";
  Metadata swarm_metadata({Metadata::Provides, Metadata::None});
  pkg->AddSwarm(swarm_name, swarm_metadata);
  Metadata real_swarmvalue_metadata({Metadata::Real});
  pkg->AddSwarmValue("t", swarm_name, real_swarmvalue_metadata);
  pkg->AddSwarmValue("vx", swarm_name, real_swarmvalue_metadata);
  pkg->AddSwarmValue("vy", swarm_name, real_swarmvalue_metadata);
  pkg->AddSwarmValue("vz", swarm_name, real_swarmvalue_metadata);
  pkg->AddSwarmValue("weight", swarm_name, real_swarmvalue_metadata);

  std::string field_name = "particle_deposition";
  Metadata m({Metadata::Cell, Metadata::Independent, Metadata::WithFluxes});
  pkg->AddField(field_name, m);

  pkg->EstimateTimestepBlock = EstimateTimestepBlock;

  return pkg;
}

AmrTag CheckRefinement(MeshBlockData<Real> *rc) { return AmrTag::same; }

Real EstimateTimestepBlock(MeshBlockData<Real> *rc) {
  auto pmb = rc->GetBlockPointer();
  auto pkg = pmb->packages.Get("particles_package");
  const Real &dt = pkg->Param<Real>("const_dt");
  return dt;
}

} // namespace Particles

// *************************************************//
// define the application driver. in this case,    *//
// that just means defining the MakeTaskList       *//
// function.                                       *//
// *************************************************//
// first some helper tasks

TaskStatus DestroySomeParticles(MeshBlock *pmb) {
  Kokkos::Profiling::pushRegion("Task_Particles_DestroySomeParticles");

  auto pkg = pmb->packages.Get("particles_package");
  auto swarm = pmb->swarm_data.Get()->Get("my particles");
  auto rng_pool = pkg->Param<RNGPool>("rng_pool");
  const auto destroy_particles_frac = pkg->Param<Real>("destroy_particles_frac");

  // The swarm mask is managed internally and should always be treated as constant. This
  // may be enforced later.
  auto swarm_d = swarm->GetDeviceContext();

  // Randomly mark some fraction of particles each timestep for removal
  pmb->par_for(
      "DestroySomeParticles", 0, swarm->GetMaxActiveIndex(), KOKKOS_LAMBDA(const int n) {
        if (swarm_d.IsActive(n)) {
          auto rng_gen = rng_pool.get_state();
          if (rng_gen.drand() > 1.0 - destroy_particles_frac) {
            swarm_d.MarkParticleForRemoval(n);
          }
          rng_pool.free_state(rng_gen);
        }
      });

  // Remove marked particles
  swarm->RemoveMarkedParticles();

  Kokkos::Profiling::popRegion(); // Task_Particles_DestroySomeParticles
  return TaskStatus::complete;
}

TaskStatus DepositParticles(MeshBlock *pmb) {
  Kokkos::Profiling::pushRegion("Task_Particles_DepositParticles");

  auto swarm = pmb->swarm_data.Get()->Get("my particles");

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

  const auto &x = swarm->Get<Real>("x").Get();
  const auto &y = swarm->Get<Real>("y").Get();
  const auto &z = swarm->Get<Real>("z").Get();
  const auto &weight = swarm->Get<Real>("weight").Get();
  auto swarm_d = swarm->GetDeviceContext();

  auto &particle_dep = pmb->meshblock_data.Get()->Get("particle_deposition").data;
  // Reset particle count
  pmb->par_for(
      "ZeroParticleDep", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        particle_dep(k, j, i) = 0.;
      });

  const int ndim = pmb->pmy_mesh->ndim;

  pmb->par_for(
      "DepositParticles", 0, swarm->GetMaxActiveIndex(), KOKKOS_LAMBDA(const int n) {
        if (swarm_d.IsActive(n)) {
          int i = static_cast<int>(std::floor((x(n) - minx_i) / dx_i) + ib.s);
          int j = 0;
          if (ndim > 1) {
            j = static_cast<int>(std::floor((y(n) - minx_j) / dx_j) + jb.s);
          }
          int k = 0;
          if (ndim > 2) {
            k = static_cast<int>(std::floor((z(n) - minx_k) / dx_k) + kb.s);
          }

          if (i >= ib.s && i <= ib.e && j >= jb.s && j <= jb.e && k >= kb.s &&
              k <= kb.e) {
            Kokkos::atomic_add(&particle_dep(k, j, i), weight(n));
          }
        }
      });

  Kokkos::Profiling::popRegion(); // Task_Particles_DepositParticles
  return TaskStatus::complete;
}

TaskStatus CreateSomeParticles(MeshBlock *pmb, const double t0) {
  Kokkos::Profiling::pushRegion("Task_Particles_CreateSomeParticles");

  auto pkg = pmb->packages.Get("particles_package");
  auto swarm = pmb->swarm_data.Get()->Get("my particles");
  auto rng_pool = pkg->Param<RNGPool>("rng_pool");
  auto num_particles = pkg->Param<int>("num_particles");
  auto v = pkg->Param<Real>("particle_speed");
  const auto orbiting_particles = pkg->Param<bool>("orbiting_particles");

  ParArrayND<int> new_indices;
  const auto new_particles_mask = swarm->AddEmptyParticles(num_particles, new_indices);

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

  auto &t = swarm->Get<Real>("t").Get();
  auto &x = swarm->Get<Real>("x").Get();
  auto &y = swarm->Get<Real>("y").Get();
  auto &z = swarm->Get<Real>("z").Get();
  auto &vx = swarm->Get<Real>("vx").Get();
  auto &vy = swarm->Get<Real>("vy").Get();
  auto &vz = swarm->Get<Real>("vz").Get();
  auto &weight = swarm->Get<Real>("weight").Get();

  auto swarm_d = swarm->GetDeviceContext();

  if (orbiting_particles) {
    pmb->par_for(
        "CreateSomeOrbitingParticles", 0, swarm->GetMaxActiveIndex(),
        KOKKOS_LAMBDA(const int n) {
          if (new_particles_mask(n)) {
            auto rng_gen = rng_pool.get_state();

            // Randomly sample in space in this meshblock while staying within 0.5 of
            // origin
            Real r;
            do {
              x(n) = minx_i + nx_i * dx_i * rng_gen.drand();
              y(n) = minx_j + nx_j * dx_j * rng_gen.drand();
              z(n) = minx_k + nx_k * dx_k * rng_gen.drand();
              r = sqrt(x(n) * x(n) + y(n) * y(n) + z(n) * z(n));
            } while (r > 0.5);

            // Randomly sample direction perpendicular to origin
            Real theta = acos(2. * rng_gen.drand() - 1.);
            Real phi = 2. * M_PI * rng_gen.drand();
            vx(n) = sin(theta) * cos(phi);
            vy(n) = sin(theta) * sin(phi);
            vz(n) = cos(theta);
            // Project v onto plane normal to sphere
            Real vdN = vx(n) * x(n) + vy(n) * y(n) + vz(n) * z(n);
            Real NdN = r * r;
            vx(n) = vx(n) - vdN / NdN * x(n);
            vy(n) = vy(n) - vdN / NdN * y(n);
            vz(n) = vz(n) - vdN / NdN * z(n);

            // Normalize
            Real v_tmp = sqrt(vx(n) * vx(n) + vy(n) * vy(n) + vz(n) * vz(n));
            vx(n) *= v / v_tmp;
            vy(n) *= v / v_tmp;
            vz(n) *= v / v_tmp;

            // Create particles at the beginning of the timestep
            t(n) = t0;

            weight(n) = 1.0;

            rng_pool.free_state(rng_gen);
          }
        });
  } else {
    pmb->par_for(
        "CreateSomeParticles", 0, swarm->GetMaxActiveIndex(), KOKKOS_LAMBDA(const int n) {
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

            // Create particles at the beginning of the timestep
            t(n) = t0;

            weight(n) = 1.0;

            rng_pool.free_state(rng_gen);
          }
        });
  }

  Kokkos::Profiling::popRegion(); // Task_Particles_CreateSomeParticles
  return TaskStatus::complete;
}

TaskStatus TransportParticles(MeshBlock *pmb, const StagedIntegrator *integrator,
                              const double t0) {
  Kokkos::Profiling::pushRegion("Task_Particles_TransportParticles");

  auto swarm = pmb->swarm_data.Get()->Get("my particles");
  auto pkg = pmb->packages.Get("particles_package");
  const auto orbiting_particles = pkg->Param<bool>("orbiting_particles");

  int max_active_index = swarm->GetMaxActiveIndex();

  Real dt = integrator->dt;

  auto &t = swarm->Get<Real>("t").Get();
  auto &x = swarm->Get<Real>("x").Get();
  auto &y = swarm->Get<Real>("y").Get();
  auto &z = swarm->Get<Real>("z").Get();
  auto &vx = swarm->Get<Real>("vx").Get();
  auto &vy = swarm->Get<Real>("vy").Get();
  auto &vz = swarm->Get<Real>("vz").Get();

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

  auto swarm_d = swarm->GetDeviceContext();

  // Simple particle push: push particles half the smallest zone width until they have
  // traveled one integrator timestep's worth of time. Particles orbit the origin.
  if (orbiting_particles) {
    // Particles orbit the origin
    pmb->par_for(
        "TransportOrbitingParticles", 0, max_active_index, KOKKOS_LAMBDA(const int n) {
          if (swarm_d.IsActive(n)) {
            Real v = sqrt(vx(n) * vx(n) + vy(n) * vy(n) + vz(n) * vz(n));
            while (t(n) < t0 + dt) {
              Real dt_cell = dx_push / v;
              Real dt_end = t0 + dt - t(n);
              Real dt_push = std::min<Real>(dt_cell, dt_end);

              Real r = sqrt(x(n) * x(n) + y(n) * y(n) + z(n) * z(n));

              x(n) += vx(n) * dt_push;
              y(n) += vy(n) * dt_push;
              z(n) += vz(n) * dt_push;
              t(n) += dt_push;

              // Force point back onto spherical shell
              Real r_tmp = sqrt(x(n) * x(n) + y(n) * y(n) + z(n) * z(n));
              x(n) *= r / r_tmp;
              y(n) *= r / r_tmp;
              z(n) *= r / r_tmp;

              // Project v onto plane normal to sphere
              Real vdN = vx(n) * x(n) + vy(n) * y(n) + vz(n) * z(n);
              Real NdN = r * r;
              vx(n) = vx(n) - vdN / NdN * x(n);
              vy(n) = vy(n) - vdN / NdN * y(n);
              vz(n) = vz(n) - vdN / NdN * z(n);

              // Normalize
              Real v_tmp = sqrt(vx(n) * vx(n) + vy(n) * vy(n) + vz(n) * vz(n));
              vx(n) *= v / v_tmp;
              vy(n) *= v / v_tmp;
              vz(n) *= v / v_tmp;

              bool on_current_mesh_block = true;
              // This call is required to trigger internal boundary condition machinery
              swarm_d.GetNeighborBlockIndex(n, x(n), y(n), z(n), on_current_mesh_block);

              if (!on_current_mesh_block) {
                // Particle no longer on this block
                break;
              }
            }
          }
        });
  } else {
    // Particles move in straight lines
    pmb->par_for(
        "TransportParticles", 0, max_active_index, KOKKOS_LAMBDA(const int n) {
          if (swarm_d.IsActive(n)) {
            Real v = sqrt(vx(n) * vx(n) + vy(n) * vy(n) + vz(n) * vz(n));
            while (t(n) < t0 + dt) {
              Real dt_cell = dx_push / v;
              Real dt_end = t0 + dt - t(n);
              Real dt_push = std::min<Real>(dt_cell, dt_end);

              x(n) += vx(n) * dt_push;
              y(n) += vy(n) * dt_push;
              z(n) += vz(n) * dt_push;
              t(n) += dt_push;

              bool on_current_mesh_block = true;
              // This call is required to trigger internal boundary condition machinery
              swarm_d.GetNeighborBlockIndex(n, x(n), y(n), z(n), on_current_mesh_block);

              if (!on_current_mesh_block) {
                // Particle no longer on this block. Can still be communicated to a
                // neighbor block or have boundary conditions applied so transport
                // can continue.
                break;
              }
            }
          }
        });
  }

  Kokkos::Profiling::popRegion(); // Task_Particles_TransportParticles
  return TaskStatus::complete;
}

// Custom step function to allow for looping over MPI-related tasks until complete
TaskListStatus ParticleDriver::Step() {
  TaskListStatus status;
  integrator.dt = tm.dt;

  BlockList_t &blocks = pmesh->block_list;
  auto num_task_lists_executed_independently = blocks.size();

  // Create all the particles that will be created during the step
  status = MakeParticlesCreationTaskCollection().Execute();

  // Loop over repeated MPI calls until every particle is finished. This logic is
  // required because long-distance particle pushes can lead to a large, unpredictable
  // number of MPI sends and receives.
  bool particles_update_done = false;
  while (!particles_update_done) {
    status = MakeParticlesUpdateTaskCollection().Execute();

    particles_update_done = true;
    for (auto &block : blocks) {
      // TODO(BRR) Despite this "my particles"-specific call, this function feels like it
      // should be generalized
      auto swarm = block->swarm_data.Get()->Get("my particles");
      if (!swarm->finished_transport) {
        particles_update_done = false;
      }
    }
  }

  // Use a more traditional task list for predictable post-MPI evaluations.
  status = MakeFinalizationTaskCollection().Execute();

  return status;
}

// TODO(BRR) This should really be in parthenon/src... but it can't just live in Swarm
// because of the loop over blocks
TaskStatus StopCommunicationMesh(const BlockList_t &blocks) {
  Kokkos::Profiling::pushRegion("Task_Particles_StopCommunicationMesh");

  int num_sent_local = 0;
  for (auto &block : blocks) {
    auto sc = block->swarm_data.Get();
    auto swarm = sc->Get("my particles");
    swarm->finished_transport = false;
    num_sent_local += swarm->num_particles_sent_;
  }

  int num_sent_global = num_sent_local; // potentially overwritten by following Allreduce
#ifdef MPI_PARALLEL
  for (auto &block : blocks) {
    auto swarm = block->swarm_data.Get()->Get("my particles");
    for (int n = 0; n < block->pbval->nneighbor; n++) {
      NeighborBlock &nb = block->pbval->neighbor[n];
      // TODO(BRR) May want logic like this if we have non-blocking TaskRegions
      // if (nb.snb.rank != Globals::my_rank) {
      //  if (swarm->vbswarm->bd_var_.flag[nb.bufid] != BoundaryStatus::completed) {
      //    printf("[%i] Neighbor %i not complete!\n", Globals::my_rank, n);
      //    //return TaskStatus::incomplete;
      //  }
      //}

      // TODO(BRR) May want to move this logic into a per-cycle initialization call
      if (swarm->vbswarm->bd_var_.flag[nb.bufid] == BoundaryStatus::completed) {
        swarm->vbswarm->bd_var_.req_send[nb.bufid] = MPI_REQUEST_NULL;
      }
    }
  }

  MPI_Allreduce(&num_sent_local, &num_sent_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif // MPI_PARALLEL

  if (num_sent_global == 0) {
    for (auto &block : blocks) {
      auto &pmb = block;
      auto sc = pmb->swarm_data.Get();
      auto swarm = sc->Get("my particles");
      swarm->finished_transport = true;
    }
  }

  // Reset boundary statuses
  for (auto &block : blocks) {
    auto &pmb = block;
    auto sc = pmb->swarm_data.Get();
    auto swarm = sc->Get("my particles");
    for (int n = 0; n < swarm->vbswarm->bd_var_.nbmax; n++) {
      auto &nb = pmb->pbval->neighbor[n];
      swarm->vbswarm->bd_var_.flag[nb.bufid] = BoundaryStatus::waiting;
    }
  }

  Kokkos::Profiling::popRegion(); // Task_Particles_StopCommunicationMesh
  return TaskStatus::complete;
}

TaskCollection ParticleDriver::MakeParticlesCreationTaskCollection() const {
  TaskCollection tc;
  TaskID none(0);
  const double t0 = tm.time;
  const BlockList_t &blocks = pmesh->block_list;

  auto num_task_lists_executed_independently = blocks.size();
  TaskRegion &async_region0 = tc.AddRegion(num_task_lists_executed_independently);
  for (int i = 0; i < blocks.size(); i++) {
    auto &pmb = blocks[i];
    auto &tl = async_region0[i];
    auto create_some_particles = tl.AddTask(none, CreateSomeParticles, pmb.get(), t0);
  }

  return tc;
}

TaskCollection ParticleDriver::MakeParticlesUpdateTaskCollection() const {
  TaskCollection tc;
  TaskID none(0);
  const double t0 = tm.time;
  const BlockList_t &blocks = pmesh->block_list;

  auto num_task_lists_executed_independently = blocks.size();

  TaskRegion &async_region0 = tc.AddRegion(num_task_lists_executed_independently);
  for (int i = 0; i < blocks.size(); i++) {
    auto &pmb = blocks[i];

    auto &sc = pmb->swarm_data.Get();

    auto &tl = async_region0[i];

    auto transport_particles =
        tl.AddTask(none, TransportParticles, pmb.get(), &integrator, t0);

    auto send = tl.AddTask(transport_particles, &SwarmContainer::Send, sc.get(),
                           BoundaryCommSubset::all);
    auto receive =
        tl.AddTask(send, &SwarmContainer::Receive, sc.get(), BoundaryCommSubset::all);
  }

  TaskRegion &sync_region0 = tc.AddRegion(1);
  {
    auto &tl = sync_region0[0];
    auto stop_comm = tl.AddTask(none, StopCommunicationMesh, blocks);
  }

  return tc;
}

TaskCollection ParticleDriver::MakeFinalizationTaskCollection() const {
  TaskCollection tc;
  TaskID none(0);
  BlockList_t &blocks = pmesh->block_list;

  auto num_task_lists_executed_independently = blocks.size();
  TaskRegion &async_region1 = tc.AddRegion(num_task_lists_executed_independently);

  for (int i = 0; i < blocks.size(); i++) {
    auto &pmb = blocks[i];
    auto &sc = pmb->swarm_data.Get();
    auto &sc1 = pmb->meshblock_data.Get();
    auto &tl = async_region1[i];

    auto destroy_some_particles = tl.AddTask(none, DestroySomeParticles, pmb.get());

    auto sort_particles = tl.AddTask(destroy_some_particles,
                                     &SwarmContainer::SortParticlesByCell, sc.get());

    auto deposit_particles = tl.AddTask(sort_particles, DepositParticles, pmb.get());

    // Defragment if swarm memory pool occupancy is 90%
    auto defrag = tl.AddTask(deposit_particles, &SwarmContainer::Defrag, sc.get(), 0.9);

    // estimate next time step
    auto new_dt = tl.AddTask(
        defrag, parthenon::Update::EstimateTimestep<MeshBlockData<Real>>, sc1.get());
  }

  return tc;
}

} // namespace particles_example
