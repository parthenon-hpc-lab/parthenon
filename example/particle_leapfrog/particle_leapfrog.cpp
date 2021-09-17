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

  std::string swarm_name = "my particles";
  Metadata swarm_metadata({Metadata::Provides, Metadata::None});
  pkg->AddSwarm(swarm_name, swarm_metadata);
  Metadata real_swarmvalue_metadata({Metadata::Real});
  pkg->AddSwarmValue("id", swarm_name, Metadata({Metadata::Integer}));
  pkg->AddSwarmValue("vx", swarm_name, real_swarmvalue_metadata);
  pkg->AddSwarmValue("vy", swarm_name, real_swarmvalue_metadata);
  pkg->AddSwarmValue("vz", swarm_name, real_swarmvalue_metadata);

  pkg->EstimateTimestepBlock = EstimateTimestepBlock;

  return pkg;
}

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
          if (v != 0.0) {
            lmin_dt = std::min(lmin_dt, dx_push / v);
          }
        }
      },
      Kokkos::Min<Real>(min_dt));

  return cfl * min_dt;
}

} // namespace Particles

// Simple log function to ease regression tests and debugging until there's a proper
// particles output method. Do NOT use in practice.
TaskStatus WriteParticleLog(BlockList_t &blocks, int ncycle) {
  auto pkg = blocks[0]->packages.Get("particles_package");
  const auto write_particle_log_nth_cycle =
      pkg->Param<int>("write_particle_log_nth_cycle");
  if ((write_particle_log_nth_cycle < 1) ||
      (ncycle % write_particle_log_nth_cycle != 0)) {
    return TaskStatus::complete;
  }

  // Step 1: Gather number of particles on this rank
  int num_particles_this_rank = 0;

  for (int i = 0; i < blocks.size(); i++) {
    auto &pmb = blocks[i];

    auto swarm = pmb->swarm_data.Get()->Get("my particles");
    const auto &is_active = swarm->GetMask().Get().GetHostMirrorAndCopy();
    for (auto n = 0; n < is_active.GetSize(); n++) {
      if (is_active(n)) {
        num_particles_this_rank += 1;
      }
    }
  }

  // Step 2a: Gather actual data locally
  constexpr int num_fields = 8; // block->gid, id, x, y, z, vx, vy, vz
  Kokkos::View<Real *, LayoutWrapper, HostMemSpace> particle_output_this_rank(
      "particle_output_this_rank", num_fields * num_particles_this_rank);

  int offset = 0;
  for (int i = 0; i < blocks.size(); i++) {
    auto &pmb = blocks[i];
    auto swarm = pmb->swarm_data.Get()->Get("my particles");

    const auto &id = swarm->Get<int>("id").Get().GetHostMirrorAndCopy();
    const auto &x = swarm->Get<Real>("x").Get().GetHostMirrorAndCopy();
    const auto &y = swarm->Get<Real>("y").Get().GetHostMirrorAndCopy();
    const auto &z = swarm->Get<Real>("z").Get().GetHostMirrorAndCopy();
    const auto &vx = swarm->Get<Real>("vx").Get().GetHostMirrorAndCopy();
    const auto &vy = swarm->Get<Real>("vy").Get().GetHostMirrorAndCopy();
    const auto &vz = swarm->Get<Real>("vz").Get().GetHostMirrorAndCopy();

    const auto &is_active = swarm->GetMask().Get().GetHostMirrorAndCopy();
    for (auto n = 0; n < is_active.GetSize(); n++) {
      if (is_active(n)) {
        particle_output_this_rank(offset++) = static_cast<Real>(pmb->gid);
        particle_output_this_rank(offset++) = static_cast<Real>(id(n));
        particle_output_this_rank(offset++) = x(n);
        particle_output_this_rank(offset++) = y(n);
        particle_output_this_rank(offset++) = z(n);
        particle_output_this_rank(offset++) = vx(n);
        particle_output_this_rank(offset++) = vy(n);
        particle_output_this_rank(offset++) = vz(n);
      }
    }
  }

  // Step 2b. Gather data on root process
  Kokkos::View<Real *, LayoutWrapper, HostMemSpace> particle_output_all_ranks;
  std::vector<int> num_particles_all_ranks(Globals::nranks);
#ifdef MPI_PARALLEL
  if (Globals::my_rank == 0) {
    PARTHENON_MPI_CHECK(MPI_Gather(&num_particles_this_rank, 1, MPI_INT,
                                   &num_particles_all_ranks.front(), 1, MPI_INT, 0,
                                   MPI_COMM_WORLD));

    int num_particles_total = 0;
    std::vector<int> displacements(Globals::nranks);
    std::vector<int> recvsizes(Globals::nranks);
    for (int i = 0; i < Globals::nranks; i++) {
      num_particles_total += num_particles_all_ranks.at(i);
      recvsizes.at(i) = num_particles_all_ranks.at(i) * num_fields;
      if (i > 0) {
        displacements.at(i) = displacements.at(i - 1) + recvsizes.at(i - 1);
      }
    }
    particle_output_all_ranks = Kokkos::View<Real *, LayoutWrapper, HostMemSpace>(
        "particle_output_all_ranks", num_fields * num_particles_total);

    PARTHENON_MPI_CHECK(MPI_Gatherv(
        particle_output_this_rank.data(), particle_output_this_rank.size(),
        MPI_PARTHENON_REAL, particle_output_all_ranks.data(), &recvsizes.front(),
        &displacements.front(), MPI_PARTHENON_REAL, 0, MPI_COMM_WORLD));

  } else {
    PARTHENON_MPI_CHECK(MPI_Gather(&num_particles_this_rank, 1, MPI_INT, nullptr, 1,
                                   MPI_INT, 0, MPI_COMM_WORLD));
    PARTHENON_MPI_CHECK(MPI_Gatherv(particle_output_this_rank.data(),
                                    particle_output_this_rank.size(), MPI_PARTHENON_REAL,
                                    nullptr, nullptr, nullptr, MPI_PARTHENON_REAL, 0,
                                    MPI_COMM_WORLD));
  }

#else
  particle_output_all_ranks = Kokkos::View<Real *, LayoutWrapper, HostMemSpace>(
      "particle_output_all_ranks", num_fields * num_particles_this_rank);
  Kokkos::deep_copy(particle_output_all_ranks, particle_output_this_rank);
  num_particles_all_ranks.at(0) = num_particles_this_rank;
#endif

  // Step 3: Root process write data
  if (Globals::my_rank == 0) {
    std::stringstream buffer;
    auto open_mode = std::ios_base::app; // default append
    // write header
    if (ncycle == 0) {
      buffer << "ncycle , rank , block gid , particles id , x , y , z , vx , vy , vz"
             << std::endl;
      open_mode = std::ios_base::out; // start writing clean file
    }
    // set precision for float fields
    buffer << std::fixed << std::setprecision(10);
    int offset = 0;
    for (auto rank = 0; rank < Globals::nranks; rank++) {
      for (auto p = 0; p < num_particles_all_ranks.at(rank); p++) {
        buffer << ncycle << " , " << rank << " , "
               << static_cast<int>(particle_output_all_ranks(offset)) // block id
               << " , "
               << static_cast<int>(particle_output_all_ranks(offset + 1)); // particle id
        offset += 2;
        for (auto j = 2; j < num_fields; j++) {
          buffer << " , " << particle_output_all_ranks(offset++);
        }
        buffer << std::endl;
      }
    }

    std::ofstream outfile("particles.csv", open_mode);
    if (outfile.is_open()) {
      outfile << buffer.str();
    } else {
      PARTHENON_THROW("Unable to open particles output file");
    }
  }

  return TaskStatus::complete;
}

// initial particle position: x,y,z,vx,vy,vz
constexpr int num_test_particles = 4;
const std::array<std::array<Real, 6>, num_test_particles> particles_ic = {{
    {-0.1, 0.2, 0.3, 1.0, 0.0, 0.0},  // along x direction
    {0.4, -0.1, 0.3, 0.0, 1.0, 0.0},  // along y direction
    {-0.1, 0.3, 0.2, 0.0, 0.0, 0.5},  // along z direction
    {0.12, 0.2, -0.3, 1.0, 1.0, 1.0}, // along diagonal
}};

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  auto pkg = pmb->packages.Get("particles_package");
  auto swarm = pmb->swarm_data.Get()->Get("my particles");

  const IndexRange &ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange &jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange &kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  const Real &x_min = pmb->coords.x1f(ib.s);
  const Real &y_min = pmb->coords.x2f(jb.s);
  const Real &z_min = pmb->coords.x3f(kb.s);
  const Real &x_max = pmb->coords.x1f(ib.e + 1);
  const Real &y_max = pmb->coords.x2f(jb.e + 1);
  const Real &z_max = pmb->coords.x3f(kb.e + 1);

  const auto &ic = particles_ic;

  // determine which particles belong to this block
  size_t num_particles_this_block = 0;
  auto ids_this_block =
      ParArray1D<int>("indices of particles in test", num_test_particles);

  auto ids_this_block_h =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), ids_this_block);

  for (auto n = 0; n < num_test_particles; n++) {
    const Real &x_ = ic.at(n).at(0);
    const Real &y_ = ic.at(n).at(1);
    const Real &z_ = ic.at(n).at(2);

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
  auto &vx = swarm->Get<Real>("vx").Get();
  auto &vy = swarm->Get<Real>("vy").Get();
  auto &vz = swarm->Get<Real>("vz").Get();

  auto swarm_d = swarm->GetDeviceContext();
  // This hardcoded implementation should only used in PGEN and not during runtime
  // addition of particles as indices need to be taken into account.
  pmb->par_for(
      "CreateParticles", 0, num_particles_this_block - 1, KOKKOS_LAMBDA(const int n) {
        const auto &m = ids_this_block(n);

        id(n) = m; // global unique id
        x(n) = ic.at(m).at(0);
        y(n) = ic.at(m).at(1);
        z(n) = ic.at(m).at(2);
        vx(n) = ic.at(m).at(3);
        vy(n) = ic.at(m).at(4);
        vz(n) = ic.at(m).at(5);
      });
}

TaskStatus TransportParticles(MeshBlock *pmb, const StagedIntegrator *integrator) {
  auto swarm = pmb->swarm_data.Get()->Get("my particles");
  auto pkg = pmb->packages.Get("particles_package");

  int max_active_index = swarm->GetMaxActiveIndex();

  Real dt = integrator->dt;

  auto &id = swarm->Get<int>("id").Get();
  auto &x = swarm->Get<Real>("x").Get();
  auto &y = swarm->Get<Real>("y").Get();
  auto &z = swarm->Get<Real>("z").Get();
  auto &vx = swarm->Get<Real>("vx").Get();
  auto &vy = swarm->Get<Real>("vy").Get();
  auto &vz = swarm->Get<Real>("vz").Get();

  auto swarm_d = swarm->GetDeviceContext();
  // keep particles on existing trajectory for now
  const Real ax = 0.0;
  const Real ay = 0.0;
  const Real az = 0.0;
  pmb->par_for(
      "Leapfrog", 0, max_active_index, KOKKOS_LAMBDA(const int n) {
        if (swarm_d.IsActive(n)) {
          Real v = sqrt(vx(n) * vx(n) + vy(n) * vy(n) + vz(n) * vz(n));

          // drift
          x(n) += vx(n) * 0.5 * dt;
          y(n) += vy(n) * 0.5 * dt;
          z(n) += vz(n) * 0.5 * dt;

          // kick
          vx(n) += ax * dt;
          vy(n) += ay * dt;
          vz(n) += az * dt;

          // drift
          x(n) += vx(n) * 0.5 * dt;
          y(n) += vy(n) * 0.5 * dt;
          z(n) += vz(n) * 0.5 * dt;
          printf("[%i] xyz = %e %e %e\n", n, x(n), y(n), z(n));

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

  // Directly add single region with single task
  tc.AddRegion(1)[0].AddTask(none, WriteParticleLog, blocks, tm.ncycle);

  return tc;
}

} // namespace particles_leapfrog
