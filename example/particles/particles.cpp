//========================================================================================
// (C) (or copyright) 2020-2024. Triad National Security, LLC. All rights reserved.
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

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "particles.hpp"

// *************************************************//
// redefine some internal parthenon functions      *//
// *************************************************//
namespace particles_example {

using namespace parthenon;
using namespace parthenon::BoundaryFunction;

Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;
  packages.Add(particles_example::Particles::Initialize(pin.get()));
  return packages;
}

enum class DepositionMethod { per_particle, per_cell };

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

  std::string deposition_method =
      pin->GetOrAddString("Particles", "deposition_method", "per_particle");
  if (deposition_method == "per_particle") {
    pkg->AddParam<>("deposition_method", DepositionMethod::per_particle);
  } else if (deposition_method == "per_cell") {
    pkg->AddParam<>("deposition_method", DepositionMethod::per_cell);
  } else {
    PARTHENON_THROW("deposition method not recognized");
  }

  bool orbiting_particles =
      pin->GetOrAddBoolean("Particles", "orbiting_particles", false);
  pkg->AddParam<>("orbiting_particles", orbiting_particles);

  // Initialize random number generator pool
  int rng_seed = pin->GetInteger("Particles", "rng_seed");
  pkg->AddParam<>("rng_seed", rng_seed);
  RNGPool rng_pool(rng_seed);
  pkg->AddParam<>("rng_pool", rng_pool);

  std::string swarm_name = "my_particles";
  Metadata swarm_metadata({Metadata::Provides, Metadata::None});
  pkg->AddSwarm(swarm_name, swarm_metadata);
  Metadata real_swarmvalue_metadata({Metadata::Real});
  pkg->AddSwarmValue("t", swarm_name, real_swarmvalue_metadata);
  Metadata real_vec_swarmvalue_metadata({Metadata::Real}, std::vector<int>{3});
  pkg->AddSwarmValue("v", swarm_name, real_vec_swarmvalue_metadata);
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
  PARTHENON_INSTRUMENT

  auto pkg = pmb->packages.Get("particles_package");
  auto swarm = pmb->meshblock_data.Get()->GetSwarmData()->Get("my_particles");
  auto rng_pool = pkg->Param<RNGPool>("rng_pool");
  const auto destroy_particles_frac = pkg->Param<Real>("destroy_particles_frac");

  // The swarm mask is managed internally and should always be treated as constant. This
  // may be enforced later.
  auto swarm_d = swarm->GetDeviceContext();

  // Randomly mark some fraction of particles each timestep for removal
  pmb->par_for(
      PARTHENON_AUTO_LABEL, 0, swarm->GetMaxActiveIndex(), KOKKOS_LAMBDA(const int n) {
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

  return TaskStatus::complete;
}

TaskStatus SortParticlesIfUsingPerCellDeposition(MeshBlock *pmb) {
  auto pkg = pmb->packages.Get("particles_package");
  const auto deposition_method = pkg->Param<DepositionMethod>("deposition_method");
  if (deposition_method == DepositionMethod::per_cell) {
    auto swarm = pmb->meshblock_data.Get()->GetSwarmData()->Get("my_particles");
    swarm->SortParticlesByCell();
  }

  return TaskStatus::complete;
}

TaskStatus DepositParticles(MeshBlock *pmb) {
  PARTHENON_INSTRUMENT

  auto swarm = pmb->meshblock_data.Get()->GetSwarmData()->Get("my_particles");

  auto pkg = pmb->packages.Get("particles_package");
  const auto deposition_method = pkg->Param<DepositionMethod>("deposition_method");

  // Meshblock geometry
  const IndexRange &ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange &jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange &kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  const Real &dx_i = pmb->coords.Dxf<1>(pmb->cellbounds.is(IndexDomain::interior));
  const Real &dx_j = pmb->coords.Dxf<2>(pmb->cellbounds.js(IndexDomain::interior));
  const Real &dx_k = pmb->coords.Dxf<3>(pmb->cellbounds.ks(IndexDomain::interior));
  const Real &minx_i = pmb->coords.Xf<1>(ib.s);
  const Real &minx_j = pmb->coords.Xf<2>(jb.s);
  const Real &minx_k = pmb->coords.Xf<3>(kb.s);

  const auto &x = swarm->Get<Real>(swarm_position::x::name()).Get();
  const auto &y = swarm->Get<Real>(swarm_position::y::name()).Get();
  const auto &z = swarm->Get<Real>(swarm_position::z::name()).Get();
  const auto &weight = swarm->Get<Real>("weight").Get();
  auto swarm_d = swarm->GetDeviceContext();

  auto &particle_dep = pmb->meshblock_data.Get()->Get("particle_deposition").data;
  const int ndim = pmb->pmy_mesh->ndim;

  if (deposition_method == DepositionMethod::per_particle) {
    // Reset particle count
    pmb->par_for(
        PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          particle_dep(k, j, i) = 0.;
        });

    pmb->par_for(
        PARTHENON_AUTO_LABEL, 0, swarm->GetMaxActiveIndex(), KOKKOS_LAMBDA(const int n) {
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
  } else if (deposition_method == DepositionMethod::per_cell) {
    pmb->par_for(
        PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          particle_dep(k, j, i) = 0.;
          for (int n = 0; n < swarm_d.GetParticleCountPerCell(k, j, i); n++) {
            const int idx = swarm_d.GetFullIndex(k, j, i, n);
            particle_dep(k, j, i) += weight(idx);
          }
        });
  }

  return TaskStatus::complete;
}

TaskStatus CreateSomeParticles(MeshBlock *pmb, const double t0) {
  PARTHENON_INSTRUMENT

  auto pkg = pmb->packages.Get("particles_package");
  auto swarm = pmb->meshblock_data.Get()->GetSwarmData()->Get("my_particles");
  auto rng_pool = pkg->Param<RNGPool>("rng_pool");
  auto num_particles = pkg->Param<int>("num_particles");
  auto vel = pkg->Param<Real>("particle_speed");
  const auto orbiting_particles = pkg->Param<bool>("orbiting_particles");

  // Create new particles and get accessor
  auto newParticlesContext = swarm->AddEmptyParticles(num_particles);

  // Meshblock geometry
  const IndexRange &ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange &jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange &kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  const int &nx_i = pmb->cellbounds.ncellsi(IndexDomain::interior);
  const int &nx_j = pmb->cellbounds.ncellsj(IndexDomain::interior);
  const int &nx_k = pmb->cellbounds.ncellsk(IndexDomain::interior);
  const Real &dx_i = pmb->coords.Dxf<1>(pmb->cellbounds.is(IndexDomain::interior));
  const Real &dx_j = pmb->coords.Dxf<2>(pmb->cellbounds.js(IndexDomain::interior));
  const Real &dx_k = pmb->coords.Dxf<3>(pmb->cellbounds.ks(IndexDomain::interior));
  const Real &minx_i = pmb->coords.Xf<1>(ib.s);
  const Real &minx_j = pmb->coords.Xf<2>(jb.s);
  const Real &minx_k = pmb->coords.Xf<3>(kb.s);

  auto &t = swarm->Get<Real>("t").Get();
  auto &x = swarm->Get<Real>(swarm_position::x::name()).Get();
  auto &y = swarm->Get<Real>(swarm_position::y::name()).Get();
  auto &z = swarm->Get<Real>(swarm_position::z::name()).Get();
  auto &v = swarm->Get<Real>("v").Get();
  auto &weight = swarm->Get<Real>("weight").Get();

  auto swarm_d = swarm->GetDeviceContext();

  if (orbiting_particles) {
    pmb->par_for(
        PARTHENON_AUTO_LABEL, 0, newParticlesContext.GetNewParticlesMaxIndex(),
        KOKKOS_LAMBDA(const int new_n) {
          const int n = newParticlesContext.GetNewParticleIndex(new_n);
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
          v(0, n) = sin(theta) * cos(phi);
          v(1, n) = sin(theta) * sin(phi);
          v(2, n) = cos(theta);
          // Project v onto plane normal to sphere
          Real vdN = v(0, n) * x(n) + v(1, n) * y(n) + v(2, n) * z(n);
          Real NdN = r * r;
          v(0, n) = v(0, n) - vdN / NdN * x(n);
          v(1, n) = v(1, n) - vdN / NdN * y(n);
          v(2, n) = v(2, n) - vdN / NdN * z(n);

          // Normalize
          Real v_tmp = sqrt(v(0, n) * v(0, n) + v(1, n) * v(1, n) + v(2, n) * v(2, n));
          PARTHENON_DEBUG_REQUIRE(v_tmp > 0., "Speed must be > 0!");
          for (int ii = 0; ii < 3; ii++) {
            v(ii, n) *= vel / v_tmp;
          }

          // Create particles at the beginning of the timestep
          t(n) = t0;

          weight(n) = 1.0;

          rng_pool.free_state(rng_gen);
        });
  } else {
    pmb->par_for(
        PARTHENON_AUTO_LABEL, 0, newParticlesContext.GetNewParticlesMaxIndex(),
        KOKKOS_LAMBDA(const int new_n) {
          const int n = newParticlesContext.GetNewParticleIndex(new_n);
          auto rng_gen = rng_pool.get_state();

          // Randomly sample in space in this meshblock
          x(n) = minx_i + nx_i * dx_i * rng_gen.drand();
          y(n) = minx_j + nx_j * dx_j * rng_gen.drand();
          z(n) = minx_k + nx_k * dx_k * rng_gen.drand();

          // Randomly sample direction on the unit sphere, fixing speed
          Real theta = acos(2. * rng_gen.drand() - 1.);
          Real phi = 2. * M_PI * rng_gen.drand();
          v(0, n) = vel * sin(theta) * cos(phi);
          v(1, n) = vel * sin(theta) * sin(phi);
          v(2, n) = vel * cos(theta);

          // Create particles at the beginning of the timestep
          t(n) = t0;

          weight(n) = 1.0;

          rng_pool.free_state(rng_gen);
        });
  }

  return TaskStatus::complete;
}

TaskStatus TransportParticles(MeshBlock *pmb, const double t0, const double dt) {
  PARTHENON_INSTRUMENT

  auto swarm = pmb->meshblock_data.Get()->GetSwarmData()->Get("my_particles");
  auto pkg = pmb->packages.Get("particles_package");
  const auto orbiting_particles = pkg->Param<bool>("orbiting_particles");

  int max_active_index = swarm->GetMaxActiveIndex();

  auto &t = swarm->Get<Real>("t").Get();
  auto &x = swarm->Get<Real>(swarm_position::x::name()).Get();
  auto &y = swarm->Get<Real>(swarm_position::y::name()).Get();
  auto &z = swarm->Get<Real>(swarm_position::z::name()).Get();
  auto &v = swarm->Get<Real>("v").Get();

  const Real &dx_i = pmb->coords.Dxf<1>(pmb->cellbounds.is(IndexDomain::interior));
  const Real &dx_j = pmb->coords.Dxf<2>(pmb->cellbounds.js(IndexDomain::interior));
  const Real &dx_k = pmb->coords.Dxf<3>(pmb->cellbounds.ks(IndexDomain::interior));
  const Real &dx_push = std::min<Real>(dx_i, std::min<Real>(dx_j, dx_k));

  const IndexRange &ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange &jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange &kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  const Real &x_min = pmb->coords.Xf<1>(ib.s);
  const Real &y_min = pmb->coords.Xf<2>(jb.s);
  const Real &z_min = pmb->coords.Xf<3>(kb.s);
  const Real &x_max = pmb->coords.Xf<1>(ib.e + 1);
  const Real &y_max = pmb->coords.Xf<2>(jb.e + 1);
  const Real &z_max = pmb->coords.Xf<3>(kb.e + 1);

  auto swarm_d = swarm->GetDeviceContext();

  // Simple particle push: push particles half the smallest zone width until they have
  // traveled one integrator timestep's worth of time. Particles orbit the origin.
  if (orbiting_particles) {
    // Particles orbit the origin
    pmb->par_for(
        PARTHENON_AUTO_LABEL, 0, max_active_index, KOKKOS_LAMBDA(const int n) {
          if (swarm_d.IsActive(n)) {
            Real vel = sqrt(v(0, n) * v(0, n) + v(1, n) * v(1, n) + v(2, n) * v(2, n));
            PARTHENON_DEBUG_REQUIRE(vel > 0., "Speed must be > 0!");
            while (t(n) < t0 + dt) {
              Real dt_cell = dx_push / vel;
              Real dt_end = t0 + dt - t(n);
              Real dt_push = std::min<Real>(dt_cell, dt_end);

              Real r = sqrt(x(n) * x(n) + y(n) * y(n) + z(n) * z(n));

              x(n) += v(0, n) * dt_push;
              y(n) += v(1, n) * dt_push;
              z(n) += v(2, n) * dt_push;
              t(n) += dt_push;

              // Force point back onto spherical shell
              Real r_tmp = sqrt(x(n) * x(n) + y(n) * y(n) + z(n) * z(n));
              PARTHENON_DEBUG_REQUIRE(r_tmp > 0., "r_tmp must be > 0 for division!");
              x(n) *= r / r_tmp;
              y(n) *= r / r_tmp;
              z(n) *= r / r_tmp;

              // Project v onto plane normal to sphere
              Real vdN = v(0, n) * x(n) + v(1, n) * y(n) + v(2, n) * z(n);
              Real NdN = r * r;
              PARTHENON_DEBUG_REQUIRE(NdN > 0., "NdN must be > 0 for division!");
              v(0, n) = v(0, n) - vdN / NdN * x(n);
              v(1, n) = v(1, n) - vdN / NdN * y(n);
              v(2, n) = v(2, n) - vdN / NdN * z(n);

              // Normalize
              Real v_tmp =
                  sqrt(v(0, n) * v(0, n) + v(1, n) * v(1, n) + v(2, n) * v(2, n));
              PARTHENON_DEBUG_REQUIRE(v_tmp > 0., "v_tmp must be > 0 for division!");
              for (int ii = 0; ii < 3; ii++) {
                v(ii, n) *= vel / v_tmp;
              }

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
        PARTHENON_AUTO_LABEL, 0, max_active_index, KOKKOS_LAMBDA(const int n) {
          if (swarm_d.IsActive(n)) {
            Real vel = sqrt(v(0, n) * v(0, n) + v(1, n) * v(1, n) + v(2, n) * v(2, n));
            PARTHENON_DEBUG_REQUIRE(vel > 0., "vel must be > 0 for division!");
            while (t(n) < t0 + dt) {
              Real dt_cell = dx_push / vel;
              Real dt_end = t0 + dt - t(n);
              Real dt_push = std::min<Real>(dt_cell, dt_end);

              x(n) += v(0, n) * dt_push;
              y(n) += v(1, n) * dt_push;
              z(n) += v(2, n) * dt_push;
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

  return TaskStatus::complete;
}

// Custom step function to allow for looping over MPI-related tasks until complete
TaskListStatus ParticleDriver::Step() {
  TaskListStatus status;

  PARTHENON_REQUIRE(integrator.nstages == 1,
                    "Only first order time integration supported!");

  BlockList_t &blocks = pmesh->block_list;
  auto num_task_lists_executed_independently = blocks.size();

  // Create all the particles that will be created during the step
  status = MakeParticlesCreationTaskCollection().Execute();
  PARTHENON_REQUIRE(status == TaskListStatus::complete,
                    "ParticlesCreation task list failed!");

  // Transport particles iteratively until all particles reach final time
  status = IterativeTransport();
  // status = MakeParticlesTransportTaskCollection().Execute();
  PARTHENON_REQUIRE(status == TaskListStatus::complete,
                    "IterativeTransport task list failed!");

  // Use a more traditional task list for predictable post-MPI evaluations.
  status = MakeFinalizationTaskCollection().Execute();
  PARTHENON_REQUIRE(status == TaskListStatus::complete, "Finalization task list failed!");

  return status;
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

TaskStatus CountNumSent(const BlockList_t &blocks, const double tf_, bool *done) {
  int num_unfinished = 0;
  for (auto &block : blocks) {
    auto sc = block->meshblock_data.Get()->GetSwarmData();
    auto swarm = sc->Get("my_particles");
    int max_active_index = swarm->GetMaxActiveIndex();

    auto &t = swarm->Get<Real>("t").Get();

    auto swarm_d = swarm->GetDeviceContext();

    const auto &tf = tf_;

    parthenon::par_reduce(
        PARTHENON_AUTO_LABEL, 0, max_active_index,
        KOKKOS_LAMBDA(const int n, int &num_unfinished) {
          if (swarm_d.IsActive(n)) {
            if (t(n) < tf) {
              num_unfinished++;
            }
          }
        },
        Kokkos::Sum<int>(num_unfinished));
  }

#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &num_unfinished, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif // MPI_PARALLEL

  if (num_unfinished > 0) {
    *done = false;
  } else {
    *done = true;
  }

  return TaskStatus::complete;
}

TaskCollection ParticleDriver::IterativeTransportTaskCollection(bool *done) const {
  TaskCollection tc;
  TaskID none(0);
  const BlockList_t &blocks = pmesh->block_list;
  const int nblocks = blocks.size();
  const double t0 = tm.time;
  const double dt = tm.dt;

  TaskRegion &async_region = tc.AddRegion(nblocks);
  for (int i = 0; i < nblocks; i++) {
    auto &pmb = blocks[i];
    auto &sc = pmb->meshblock_data.Get()->GetSwarmData();
    auto &tl = async_region[i];

    auto transport = tl.AddTask(none, TransportParticles, pmb.get(), t0, dt);
    auto reset_comms =
        tl.AddTask(transport, &SwarmContainer::ResetCommunication, sc.get());
    auto send =
        tl.AddTask(reset_comms, &SwarmContainer::Send, sc.get(), BoundaryCommSubset::all);
    auto receive =
        tl.AddTask(send, &SwarmContainer::Receive, sc.get(), BoundaryCommSubset::all);
  }

  TaskRegion &sync_region = tc.AddRegion(1);
  {
    auto &tl = sync_region[0];
    auto check_completion = tl.AddTask(none, CountNumSent, blocks, t0 + dt, done);
  }

  return tc;
}

// TODO(BRR) to be replaced by iterative tasklist machinery
TaskListStatus ParticleDriver::IterativeTransport() const {
  TaskListStatus status;
  bool transport_done = false;
  int n_transport_iter = 0;
  int n_transport_iter_max = 1000;
  while (!transport_done) {
    status = IterativeTransportTaskCollection(&transport_done).Execute();

    n_transport_iter++;
    PARTHENON_REQUIRE(n_transport_iter < n_transport_iter_max,
                      "Too many transport iterations!");
  }

  return status;
}

TaskCollection ParticleDriver::MakeFinalizationTaskCollection() const {
  TaskCollection tc;
  TaskID none(0);
  BlockList_t &blocks = pmesh->block_list;

  auto num_task_lists_executed_independently = blocks.size();
  TaskRegion &async_region1 = tc.AddRegion(num_task_lists_executed_independently);

  for (int i = 0; i < blocks.size(); i++) {
    auto &pmb = blocks[i];
    auto &sc = pmb->meshblock_data.Get()->GetSwarmData();
    auto &sc1 = pmb->meshblock_data.Get();
    auto &tl = async_region1[i];

    auto destroy_some_particles = tl.AddTask(none, DestroySomeParticles, pmb.get());

    auto sort_particles = tl.AddTask(destroy_some_particles,
                                     SortParticlesIfUsingPerCellDeposition, pmb.get());

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
