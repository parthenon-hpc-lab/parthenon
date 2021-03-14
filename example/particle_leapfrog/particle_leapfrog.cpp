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

#include "particle_leapfrog.hpp"
#include "basic_types.hpp"
#include "globals.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
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

// *************************************************//
// define the "physics" package particles_package, *//
// which includes defining various functions that  *//
// control how parthenon functions and any tasks   *//
// needed to implement the "physics"               *//
// *************************************************//

namespace Particles {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("particles_package");

  Real cfl = pin->GetOrAddReal("Particles", "cfl", 0.3);
  pkg->AddParam<>("cfl", cfl);

  auto write_particle_log =
      pin->GetOrAddBoolean("Particles", "write_particle_log", false);
  pkg->AddParam<>("write_particle_log", write_particle_log);

  std::string swarm_name = "my particles";
  Metadata swarm_metadata;
  pkg->AddSwarm(swarm_name, swarm_metadata);
  Metadata real_swarmvalue_metadata({Metadata::Real});
  pkg->AddSwarmValue("t", swarm_name, real_swarmvalue_metadata);
  pkg->AddSwarmValue("id", swarm_name, Metadata({Metadata::Integer}));
  pkg->AddSwarmValue("vx", swarm_name, real_swarmvalue_metadata);
  pkg->AddSwarmValue("vy", swarm_name, real_swarmvalue_metadata);
  pkg->AddSwarmValue("vz", swarm_name, real_swarmvalue_metadata);
  pkg->AddSwarmValue("weight", swarm_name, real_swarmvalue_metadata);

  pkg->EstimateTimestepBlock = EstimateTimestepBlock;

  return pkg;
}

AmrTag CheckRefinement(MeshBlockData<Real> *rc) { return AmrTag::same; }

Real EstimateTimestepBlock(MeshBlockData<Real> *rc) {
  auto pmb = rc->GetBlockPointer();
  auto swarm = pmb->swarm_data.Get()->Get("my particles");
  auto pkg = pmb->packages.Get("particles_package");
  const auto &cfl = pkg->Param<Real>("cfl");

  int max_active_index = swarm->GetMaxActiveIndex();

  const auto &vx = swarm->Get<Real>("vx").Get();
  const auto &vy = swarm->Get<Real>("vy").Get();
  const auto &vz = swarm->Get<Real>("vz").Get();

  // Assumes a grid with constant dx, dy, dz within a block
  const Real &dx_i = pmb->coords.dx1f(0);
  const Real &dx_j = pmb->coords.dx2f(0);
  const Real &dx_k = pmb->coords.dx3f(0);
  const Real &dx_push = std::min<Real>(dx_i, std::min<Real>(dx_j, dx_k));

  auto swarm_d = swarm->GetDeviceContext();

  Real min_dt;
  pmb->par_reduce(
      "particle_leapfrog:EstimateTimestep", 0, max_active_index,
      KOKKOS_LAMBDA(const int n, Real &lmin_dt) {
        if (swarm_d.IsActive(n)) {
          Real v = sqrt(vx(n) * vx(n) + vy(n) * vy(n) + vz(n) * vz(n));
          lmin_dt = std::min(lmin_dt, dx_push / v);
        }
      },
      Kokkos::Min<Real>(min_dt));

  return cfl * min_dt;
}

} // namespace Particles

// *************************************************//
// define the application driver. in this case,    *//
// that just means defining the MakeTaskList       *//
// function.                                       *//
// *************************************************//

TaskStatus WriteParticleLog(MeshBlock *pmb) {
  auto pkg = pmb->packages.Get("particles_package");

  const auto write_particle_log = pkg->Param<bool>("write_particle_log");
  if (!write_particle_log) {
    return TaskStatus::complete;
  }

  auto swarm = pmb->swarm_data.Get()->Get("my particles");

  const auto &id = swarm->Get<int>("id").Get().GetHostMirrorAndCopy();
  const auto &x = swarm->Get<Real>("x").Get().GetHostMirrorAndCopy();
  const auto &y = swarm->Get<Real>("y").Get().GetHostMirrorAndCopy();
  const auto &z = swarm->Get<Real>("z").Get().GetHostMirrorAndCopy();
  const auto &vx = swarm->Get<Real>("vx").Get().GetHostMirrorAndCopy();
  const auto &vy = swarm->Get<Real>("vy").Get().GetHostMirrorAndCopy();
  const auto &vz = swarm->Get<Real>("vz").Get().GetHostMirrorAndCopy();

  std::stringstream buffer;
  for (auto n = 0; n < x.GetSize(); n++) {
    buffer << Globals::my_rank << " , " << id(n) << " , " << x(n) << " , " << y(n)
           << " , " << z(n) << " , " << vx(n) << " , " << vy(n) << " , " << vz(n)
           << std::endl;
  }

  std::cout << buffer.str();

  return TaskStatus::complete;
}

// initial particle position: x,y,z,vx,vy,vz
constexpr int num_test_particles = 5;
constexpr int num_particles_max = 1024; // temp limit to ensure unique ids, needs fix
const std::array<std::array<Real, 6>, num_test_particles> particles_ic = {{
    {0.1, 0.2, 0.3, 1.0, 0.0, 0.0},  // along x direction
    {0.5, -0.1, 0.3, 1.0, 1.0, 0.0}, // along y direction
    {-0.1, 0.3, 0.2, 1.0, 0.0, 1.0}, // along z direction
    {0.12, 0.2, -0.3, std::sqrt(3.0), std::sqrt(3.0), std::sqrt(3.0)}, // along diagnonal
    {0.3, 0.0, 0.0, 1.0, 0.0, 0.0},                                    // orbiting
}};

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  Real t0 = 0.0;
  auto pkg = pmb->packages.Get("particles_package");
  auto swarm = pmb->swarm_data.Get()->Get("my particles");
  auto num_particles = num_test_particles;

  ParArrayND<int> new_indices;
  const auto new_particles_mask = swarm->AddEmptyParticles(num_particles, new_indices);

  auto &id = swarm->Get<int>("id").Get();
  auto &x = swarm->Get<Real>("x").Get();
  auto &y = swarm->Get<Real>("y").Get();
  auto &z = swarm->Get<Real>("z").Get();
  auto &vx = swarm->Get<Real>("vx").Get();
  auto &vy = swarm->Get<Real>("vy").Get();
  auto &vz = swarm->Get<Real>("vz").Get();

  auto swarm_d = swarm->GetDeviceContext();
  const auto &ic = particles_ic;
  const auto &id_offset = num_particles_max;
  const auto &my_rank = Globals::my_rank;
  // This hardcoded implementation should only used in PGEN and not during runtime
  // addition of particles as indices need to be taken into account.
  // TODO(pgrete) need MPI support, i.e., only deposit particles that are on this
  // meshblock!
  pmb->par_for(
      "CreateParticles", 0, num_particles - 1, KOKKOS_LAMBDA(const int n) {
        id(n) = id_offset * my_rank + n; // global unique id
        y(n) = ic.at(n).at(1);
        z(n) = ic.at(n).at(2);
        vx(n) = ic.at(n).at(3);
        vy(n) = ic.at(n).at(4);
        vz(n) = ic.at(n).at(5);
      });
}

TaskStatus TransportParticles(MeshBlock *pmb, const StagedIntegrator *integrator,
                              const double t0) {
  auto swarm = pmb->swarm_data.Get()->Get("my particles");
  auto pkg = pmb->packages.Get("particles_package");
  return TaskStatus::complete;

  int max_active_index = swarm->GetMaxActiveIndex();

  Real dt = integrator->dt;

  auto &t = swarm->Get<Real>("t").Get();
  auto &x = swarm->Get<Real>("x").Get();
  auto &y = swarm->Get<Real>("y").Get();
  auto &z = swarm->Get<Real>("z").Get();
  const auto &vx = swarm->Get<Real>("vx").Get();
  const auto &vy = swarm->Get<Real>("vy").Get();
  const auto &vz = swarm->Get<Real>("vz").Get();

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
            swarm_d.GetNeighborBlockIndex(n, x(n), y(n), z(n), on_current_mesh_block);

            if (!on_current_mesh_block) {
              // Particle no longer on this block
              break;
            }
          }
        }
      });

  return TaskStatus::complete;
}

TaskStatus Defrag(MeshBlock *pmb) {
  auto s = pmb->swarm_data.Get()->Get("my particles");

  // Only do this if list is getting too sparse. This criterion (whether there
  // are *any* gaps in the list) is very aggressive
  if (s->GetNumActive() <= s->GetMaxActiveIndex()) {
    s->Defrag();
  }

  return TaskStatus::complete;
}

// Custom step function to allow for looping over MPI-related tasks until complete
TaskListStatus ParticleDriver::Step() {
  TaskListStatus status;
  integrator.dt = tm.dt;

  BlockList_t &blocks = pmesh->block_list;
  auto num_task_lists_executed_independently = blocks.size();

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
  int num_sent_local = 0;
  for (auto &block : blocks) {
    auto &pmb = block;
    auto sc = pmb->swarm_data.Get();
    auto swarm = sc->Get("my particles");
    swarm->finished_transport = false;
    num_sent_local += swarm->num_particles_sent_;
  }

  // Boundary transfers on same MPI proc are blocking
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

  int num_sent_global = 0;
  MPI_Allreduce(&num_sent_local, &num_sent_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if (num_sent_global == 0) {
    for (auto &block : blocks) {
      auto &pmb = block;
      auto sc = pmb->swarm_data.Get();
      auto swarm = sc->Get("my particles");
      swarm->finished_transport = true;

      // TODO(BRR) should this really be done at an initialization step for each cycle?
      for (int n = 0; n < swarm->vbswarm->bd_var_.nbmax; n++) {
        auto &nb = pmb->pbval->neighbor[n];
        swarm->vbswarm->bd_var_.flag[nb.bufid] = BoundaryStatus::waiting;
      }
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
    // auto create_some_particles = tl.AddTask(none, CreateSomeParticles, pmb.get(), t0);
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

    auto sc = pmb->swarm_data.Get();

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
    auto &tl = async_region1[i];

    auto defrag = tl.AddTask(none, Defrag, pmb.get());
  }

  TaskRegion &sync_region = tc.AddRegion(1);
  {
    for (int i = 0; i < blocks.size(); i++) {
      auto &pmb = blocks[i];
      auto write_particle_log = sync_region[0].AddTask(none, WriteParticleLog, pmb.get());
    }
  }

  return tc;
}

} // namespace particles_example
