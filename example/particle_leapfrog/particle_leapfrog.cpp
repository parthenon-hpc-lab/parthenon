//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2021-2022 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
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

#include "particle_leapfrog.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "basic_types.hpp"
#include "config.hpp"
#include "globals.hpp"
#include "interface/update.hpp"
#include "kokkos_abstraction.hpp"

// *************************************************//
// redefine some internal parthenon functions      *//
// *************************************************//
namespace particles_leapfrog {

Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;
  packages.Add(Particles::Initialize(pin.get()));
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

  auto write_particle_log_nth_cycle =
      pin->GetOrAddInteger("Particles", "write_particle_log_nth_cycle", 0);
  pkg->AddParam<>("write_particle_log_nth_cycle", write_particle_log_nth_cycle);

  std::string swarm_name = "my_particles";
  Metadata swarm_metadata({Metadata::Provides, Metadata::None, Metadata::Independent});
  pkg->AddSwarm(swarm_name, swarm_metadata);
  pkg->AddSwarmValue("id", swarm_name, Metadata({Metadata::Integer}));
  Metadata vreal_swarmvalue_metadata({Metadata::Real, Metadata::Vector},
                                     std::vector<int>{3});
  pkg->AddSwarmValue("v", swarm_name, vreal_swarmvalue_metadata);
  Metadata vvreal_swarmvalue_metadata({Metadata::Real}, std::vector<int>{3, 3});
  pkg->AddSwarmValue("vv", swarm_name, vvreal_swarmvalue_metadata);

  pkg->EstimateTimestepBlock = EstimateTimestepBlock;

  return pkg;
}

Real EstimateTimestepBlock(MeshBlockData<Real> *rc) {
  auto pmb = rc->GetBlockPointer();
  auto swarm = pmb->swarm_data.Get()->Get("my_particles");
  auto pkg = pmb->packages.Get("particles_package");
  const auto &cfl = pkg->Param<Real>("cfl");

  int max_active_index = swarm->GetMaxActiveIndex();

  const auto &v = swarm->Get<Real>("v").Get();

  // Assumes a grid with constant dx, dy, dz within a block
  const Real &dx_i = pmb->coords.Dxf<1>(0);
  const Real &dx_j = pmb->coords.Dxf<2>(0);
  const Real &dx_k = pmb->coords.Dxf<3>(0);
  const Real &dx_push = std::min<Real>(dx_i, std::min<Real>(dx_j, dx_k));

  auto swarm_d = swarm->GetDeviceContext();

  Real min_dt;
  pmb->par_reduce(
      "particle_leapfrog:EstimateTimestep", 0, max_active_index,
      KOKKOS_LAMBDA(const int n, Real &lmin_dt) {
        if (swarm_d.IsActive(n)) {
          const Real vel =
              sqrt(v(0, n) * v(0, n) + v(1, n) * v(1, n) + v(2, n) * v(2, n));
          if (vel != 0.0) {
            lmin_dt = std::min(lmin_dt, dx_push / vel);
          }
        }
      },
      Kokkos::Min<Real>(min_dt));

  return cfl * min_dt;
}

} // namespace Particles

// initial particle position: x,y,z,vx,vy,vz
constexpr int num_test_particles = 14;
const Kokkos::Array<Kokkos::Array<Real, 6>, num_test_particles> particles_ic = {{
    {-0.1, 0.2, 0.3, 1.0, 0.0, 0.0},   // along x direction
    {0.4, -0.1, 0.3, 0.0, 1.0, 0.0},   // along y direction
    {-0.1, 0.3, 0.2, 0.0, 0.0, 0.5},   // along z direction
    {0.0, 0.0, 0.0, -1.0, 0.0, 0.0},   // along -x direction
    {0.0, 0.0, 0.0, 0.0, -1.0, 0.0},   // along -y direction
    {0.0, 0.0, 0.0, 0.0, 0.0, -1.0},   // along -z direction
    {0.0, 0.0, 0.0, 1.0, 1.0, 1.0},    // along xyz diagonal
    {0.0, 0.0, 0.0, -1.0, 1.0, 1.0},   // along -xyz diagonal
    {0.0, 0.0, 0.0, 1.0, -1.0, 1.0},   // along x-yz diagonal
    {0.0, 0.0, 0.0, 1.0, 1.0, -1.0},   // along xy-z diagonal
    {0.0, 0.0, 0.0, -1.0, -1.0, 1.0},  // along -x-yz diagonal
    {0.0, 0.0, 0.0, 1.0, -1.0, -1.0},  // along x-y-z diagonal
    {0.0, 0.0, 0.0, -1.0, 1.0, -1.0},  // along -xy-z diagonal
    {0.0, 0.0, 0.0, -1.0, -1.0, -1.0}, // along -x-y-z diagonal
}};

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  auto pkg = pmb->packages.Get("particles_package");
  auto swarm = pmb->swarm_data.Get()->Get("my_particles");

  const IndexRange &ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange &jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange &kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  const Real &x_min = pmb->coords.Xf<1>(ib.s);
  const Real &y_min = pmb->coords.Xf<2>(jb.s);
  const Real &z_min = pmb->coords.Xf<3>(kb.s);
  const Real &x_max = pmb->coords.Xf<1>(ib.e + 1);
  const Real &y_max = pmb->coords.Xf<2>(jb.e + 1);
  const Real &z_max = pmb->coords.Xf<3>(kb.e + 1);

  const auto &ic = particles_ic;

  // determine which particles belong to this block
  size_t num_particles_this_block = 0;
  auto ids_this_block =
      ParArray1D<int>("indices of particles in test", num_test_particles);

  auto ids_this_block_h =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), ids_this_block);

  for (auto n = 0; n < num_test_particles; n++) {
    const Real &x_ = ic[n][0];
    const Real &y_ = ic[n][1];
    const Real &z_ = ic[n][2];

    if ((x_ >= x_min) && (x_ < x_max) && (y_ >= y_min) && (y_ < y_max) && (z_ >= z_min) &&
        (z_ < z_max)) {
      ids_this_block_h(num_particles_this_block) = n;
      num_particles_this_block++;
    }
  }

  Kokkos::deep_copy(pmb->exec_space, ids_this_block, ids_this_block_h);

  ParArrayND<int> new_indices;
  const auto new_particles_mask =
      swarm->AddEmptyParticles(num_particles_this_block, new_indices);

  auto &id = swarm->Get<int>("id").Get();
  auto &x = swarm->Get<Real>("x").Get();
  auto &y = swarm->Get<Real>("y").Get();
  auto &z = swarm->Get<Real>("z").Get();
  auto &v = swarm->Get<Real>("v").Get();
  auto &vv = swarm->Get<Real>("vv").Get();

  // This hardcoded implementation should only used in PGEN and not during runtime
  // addition of particles as indices need to be taken into account.
  pmb->par_for(
      "CreateParticles", 0, num_particles_this_block - 1, KOKKOS_LAMBDA(const int n) {
        const auto &m = ids_this_block(n);

        id(n) = m; // global unique id
        x(n) = ic[m][0];
        y(n) = ic[m][1];
        z(n) = ic[m][2];
        v(0, n) = ic[m][3];
        v(1, n) = ic[m][4];
        v(2, n) = ic[m][5];
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) {
            vv(i, j, n) = v(i, n) * v(j, n);
          }
        }
      });
}

TaskStatus TransportParticles(MeshBlock *pmb, const StagedIntegrator *integrator) {
  auto swarm = pmb->swarm_data.Get()->Get("my_particles");
  auto pkg = pmb->packages.Get("particles_package");

  int max_active_index = swarm->GetMaxActiveIndex();

  Real dt = integrator->dt;

  auto &id = swarm->Get<int>("id").Get();
  auto &x = swarm->Get<Real>("x").Get();
  auto &y = swarm->Get<Real>("y").Get();
  auto &z = swarm->Get<Real>("z").Get();
  auto &v = swarm->Get<Real>("v").Get();

  auto swarm_d = swarm->GetDeviceContext();
  // keep particles on existing trajectory for now
  const Real ax = 0.0;
  const Real ay = 0.0;
  const Real az = 0.0;
  pmb->par_for(
      "Leapfrog", 0, max_active_index, KOKKOS_LAMBDA(const int n) {
        if (swarm_d.IsActive(n)) {
          // drift
          x(n) += v(0, n) * 0.5 * dt;
          y(n) += v(1, n) * 0.5 * dt;
          z(n) += v(2, n) * 0.5 * dt;

          // kick
          v(0, n) += ax * dt;
          v(1, n) += ay * dt;
          v(2, n) += az * dt;

          // drift
          x(n) += v(0, n) * 0.5 * dt;
          y(n) += v(1, n) * 0.5 * dt;
          z(n) += v(2, n) * 0.5 * dt;

          bool on_current_mesh_block = true;
          swarm_d.GetNeighborBlockIndex(n, x(n), y(n), z(n), on_current_mesh_block);
        }
      });

  return TaskStatus::complete;
}

// Custom step function to allow for looping over MPI-related tasks until complete
TaskListStatus ParticleDriver::Step() {
  TaskListStatus status;
  integrator.dt = tm.dt;

  BlockList_t &blocks = pmesh->block_list;
  auto num_task_lists_executed_independently = blocks.size();

  status = MakeParticlesUpdateTaskCollection().Execute();

  // Use a more traditional task list for predictable post-MPI evaluations.
  status = MakeFinalizationTaskCollection().Execute();

  return status;
}

TaskCollection ParticleDriver::MakeParticlesUpdateTaskCollection() const {
  TaskCollection tc;
  TaskID none(0);
  const BlockList_t &blocks = pmesh->block_list;

  auto num_task_lists_executed_independently = blocks.size();

  TaskRegion &sync_region0 = tc.AddRegion(1);
  {
    for (int i = 0; i < blocks.size(); i++) {
      auto &tl = sync_region0[0];
      auto &pmb = blocks[i];
      auto &sc = pmb->swarm_data.Get();
      auto reset_comms = tl.AddTask(none, &SwarmContainer::ResetCommunication, sc.get());
    }
  }

  TaskRegion &async_region0 = tc.AddRegion(num_task_lists_executed_independently);
  for (int i = 0; i < blocks.size(); i++) {
    auto &pmb = blocks[i];

    auto &sc = pmb->swarm_data.Get();

    auto &tl = async_region0[i];

    auto transport_particles =
        tl.AddTask(none, TransportParticles, pmb.get(), &integrator);

    auto send = tl.AddTask(transport_particles, &SwarmContainer::Send, sc.get(),
                           BoundaryCommSubset::all);
    auto receive =
        tl.AddTask(send, &SwarmContainer::Receive, sc.get(), BoundaryCommSubset::all);
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
    auto &tl = async_region1[i];

    // Defragment if swarm memory pool occupancy is 90%
    auto defrag = tl.AddTask(none, &SwarmContainer::Defrag, sc.get(), 0.9);

    auto new_dt =
        tl.AddTask(defrag, parthenon::Update::EstimateTimestep<MeshBlockData<Real>>,
                   pmb->meshblock_data.Get().get());
  }

  return tc;
}

} // namespace particles_leapfrog
