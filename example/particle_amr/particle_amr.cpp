//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2025 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2025. Triad National Security, LLC. All rights reserved.
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

#include "particle_amr.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "basic_types.hpp"
#include "bvals/comms/bvals_in_one.hpp"
#include "config.hpp"
#include "globals.hpp"
#include "interface/metadata.hpp"
#include "interface/update.hpp"
#include "kokkos_abstraction.hpp"
#include "pack/swarm_default_names.hpp"
#include "prolong_restrict/prolong_restrict.hpp"

using namespace parthenon::driver::prelude;
using namespace parthenon::Update;

typedef Kokkos::Random_XorShift64_Pool<> RNGPool;

namespace particle_amr_example {

// Add multiple packages, one for the advected background and one for the particles.
Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;
  packages.Add(advection_package::Initialize(pin.get()));
  packages.Add(particles_package::Initialize(pin.get()));
  return packages;
}

// Create separate packages for background field and particles

namespace advection_package {

// *************************************************//
// define the advection package, including         *//
// initialization and update functions.            *//
// *************************************************//

Real EstimateTimestepBlock(MeshBlockData<Real> *mbd) {
  auto pmb = mbd->GetBlockPointer();
  auto pkg = pmb->packages.Get("advection_package");
  const auto &cfl = pkg->Param<Real>("cfl");

  const auto &vx = pkg->Param<Real>("vx");
  const auto &vy = pkg->Param<Real>("vy");
  const auto &vz = pkg->Param<Real>("vz");

  // Assumes a grid with constant dx, dy, dz within a block
  const Real &dx_i = pmb->coords.Dxc<1>(0);
  const Real &dx_j = pmb->coords.Dxc<2>(0);
  const Real &dx_k = pmb->coords.Dxc<3>(0);

  Real min_dt = dx_i / std::abs(vx + TINY_NUMBER);
  min_dt = std::min(min_dt, dx_j / std::abs(vy + TINY_NUMBER));
  min_dt = std::min(min_dt, dx_k / std::abs(vz + TINY_NUMBER));

  return cfl * min_dt;
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("advection_package");

  Real vx = pin->GetOrAddReal("Background", "vx", 0.5);
  pkg->AddParam<>("vx", vx);
  Real vy = pin->GetOrAddReal("Background", "vy", 0.0);
  pkg->AddParam<>("vy", vy);
  Real vz = pin->GetOrAddReal("Background", "vz", 0.0);
  pkg->AddParam<>("vz", vz);

  Real cfl = pin->GetOrAddReal("Background", "cfl", 0.3);
  pkg->AddParam<>("cfl", cfl);

  // Add advected field
  std::string field_name = "advected";
  Metadata mfield(
      {Metadata::Cell, Metadata::Independent, Metadata::FillGhost, Metadata::WithFluxes});
  pkg->AddField(field_name, mfield);

  // Add field to track particle count per cell (for visualization)
  field_name = "particle_count";
  Metadata m_count({Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
  pkg->AddField(field_name, m_count);

  pkg->EstimateTimestepBlock = EstimateTimestepBlock;

  return pkg;
}

} // namespace advection_package

namespace particles_package {

// *************************************************//
// define the particles package, including         *//
// initialization and update functions, plus       *//
// AMR refinement criteria based on particle count *//
// *************************************************//

Real EstimateTimestepBlock(MeshBlockData<Real> *mbd) {
  auto pmb = mbd->GetBlockPointer();
  auto pkg = pmb->packages.Get("advection_package");

  const auto &vx = pkg->Param<Real>("vx");
  const auto &vy = pkg->Param<Real>("vy");
  const auto &vz = pkg->Param<Real>("vz");

  // Assumes a grid with constant dx, dy, dz within a block
  const Real &dx_i = pmb->coords.Dxc<1>(0);
  const Real &dx_j = pmb->coords.Dxc<2>(0);
  const Real &dx_k = pmb->coords.Dxc<3>(0);

  Real min_dt = dx_i / std::abs(vx + TINY_NUMBER);
  min_dt = std::min(min_dt, dx_j / std::abs(vy + TINY_NUMBER));
  min_dt = std::min(min_dt, dx_k / std::abs(vz + TINY_NUMBER));

  // No CFL number for particles
  return min_dt;
}

AmrTag CheckRefinement(MeshBlockData<Real> *rc) {
  auto pmb = rc->GetBlockPointer();
  auto pkg = pmb->packages.Get("particles_package");

  const int refine_above = pkg->Param<int>("refine_above");
  const int derefine_below = pkg->Param<int>("derefine_below");

  // Check if swarm exists
  auto swarm_container = pmb->meshblock_data.Get()->GetSwarmData();
  if (swarm_container == nullptr || !swarm_container->Contains("particles")) {
    return AmrTag::derefine;
  }

  auto swarm = swarm_container->Get("particles");
  int num_active = swarm->GetNumActive();

  if (num_active > refine_above) {
    return AmrTag::refine;
  } else if (num_active < derefine_below) {
    return AmrTag::derefine;
  }

  return AmrTag::same;
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("particles_package");

  int num_particles = pin->GetOrAddReal("Particles", "num_particles", 1000);
  pkg->AddParam<>("num_particles", num_particles);

  // AMR thresholds based on particle count
  int refine_above = pin->GetOrAddInteger("Particles", "refine_above", 100);
  pkg->AddParam<>("refine_above", refine_above);
  int derefine_below = pin->GetOrAddInteger("Particles", "derefine_below", 10);
  pkg->AddParam<>("derefine_below", derefine_below);

  // Add swarm of particles
  std::string swarm_name = "particles";
  Metadata swarm_metadata({Metadata::Provides, Metadata::None});
  pkg->AddSwarm(swarm_name, swarm_metadata);
  Metadata real_swarmvalue_metadata({Metadata::Real});

  pkg->EstimateTimestepBlock = EstimateTimestepBlock;
  pkg->CheckRefinementBlock = CheckRefinement;

  return pkg;
}

} // namespace particles_package

TaskStatus AdvectParticles(MeshBlock *pmb, const StagedIntegrator *integrator) {
  auto swarm_container = pmb->meshblock_data.Get()->GetSwarmData();
  if (swarm_container == nullptr || !swarm_container->Contains("particles")) {
    return TaskStatus::complete;
  }

  auto swarm = swarm_container->Get("particles");
  auto adv_pkg = pmb->packages.Get("advection_package");

  int max_active_index = swarm->GetMaxActiveIndex();

  Real dt = integrator->dt;

  auto &x = swarm->Get<Real>(swarm_position::x::name()).Get();
  auto &y = swarm->Get<Real>(swarm_position::y::name()).Get();
  auto &z = swarm->Get<Real>(swarm_position::z::name()).Get();

  const auto &vx = adv_pkg->Param<Real>("vx");
  const auto &vy = adv_pkg->Param<Real>("vy");
  const auto &vz = adv_pkg->Param<Real>("vz");

  auto swarm_d = swarm->GetDeviceContext();
  pmb->par_for(
      PARTHENON_AUTO_LABEL, 0, max_active_index, KOKKOS_LAMBDA(const int n) {
        if (swarm_d.IsActive(n)) {
          x(n) += vx * dt;
          y(n) += vy * dt;
          z(n) += vz * dt;
        }
      });

  return TaskStatus::complete;
}

TaskStatus CountParticles(MeshBlock *pmb) {
  auto swarm_container = pmb->meshblock_data.Get()->GetSwarmData();
  if (swarm_container == nullptr || !swarm_container->Contains("particles")) {
    return TaskStatus::complete;
  }

  auto swarm = swarm_container->Get("particles");

  // Meshblock geometry
  const IndexRange &ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange &jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange &kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  const Real &dx_i = pmb->coords.Dxc<1>(0);
  const Real &dx_j = pmb->coords.Dxf<2>(0);
  const Real &dx_k = pmb->coords.Dxf<3>(0);
  const Real &minx_i = pmb->coords.Xf<1>(ib.s);
  const Real &minx_j = pmb->coords.Xf<2>(jb.s);
  const Real &minx_k = pmb->coords.Xf<3>(kb.s);

  const auto &x = swarm->Get<Real>(swarm_position::x::name()).Get();
  const auto &y = swarm->Get<Real>(swarm_position::y::name()).Get();
  const auto &z = swarm->Get<Real>(swarm_position::z::name()).Get();
  auto swarm_d = swarm->GetDeviceContext();

  auto &particle_count = pmb->meshblock_data.Get()->Get("particle_count").data;
  // Reset particle count
  pmb->par_for(
      PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) { particle_count(k, j, i) = 0.; });

  const int ndim = pmb->pmy_mesh->ndim;

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
            Kokkos::atomic_add(&particle_count(k, j, i), 1.0);
          }
        }
      });

  return TaskStatus::complete;
}

TaskStatus CalculateFluxes(MeshBlockData<Real> *mbd) {
  auto pmb = mbd->GetBlockPointer();
  auto adv_pkg = pmb->packages.Get("advection_package");
  const auto &vx = adv_pkg->Param<Real>("vx");
  const auto &vy = adv_pkg->Param<Real>("vy");
  const auto &vz = adv_pkg->Param<Real>("vz");

  const auto ndim = pmb->pmy_mesh->ndim;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto advected = mbd->Get("advected").data;

  parthenon::ParArray4D<Real> x1flux = mbd->Get("bnd_flux::advected").data.Get(0, 0, 0);
  // Spatially first order upwind method
  pmb->par_for(
      PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e + 1,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        // X1
        if (vx > 0.) {
          x1flux(0, k, j, i) = advected(k, j, i - 1) * vx;
        } else {
          x1flux(0, k, j, i) = advected(k, j, i) * vx;
        }
      });

  if (ndim > 1) {
    parthenon::ParArray4D<Real> x2flux = mbd->Get("bnd_flux::advected").data.Get(1, 0, 0);
    pmb->par_for(
        PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e + 1, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          // X2
          if (vy > 0.) {
            x2flux(0, k, j, i) = advected(k, j - 1, i) * vy;
          } else {
            x2flux(0, k, j, i) = advected(k, j, i) * vy;
          }
        });
  }

  if (ndim > 2) {
    parthenon::ParArray4D<Real> x3flux = mbd->Get("bnd_flux::advected").data.Get(2, 0, 0);
    pmb->par_for(
        PARTHENON_AUTO_LABEL, kb.s, kb.e + 1, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          // X3
          if (vz > 0.) {
            x3flux(0, k, j, i) = advected(k - 1, j, i) * vz;
          } else {
            x3flux(0, k, j, i) = advected(k, j, i) * vz;
          }
        });
  }

  return TaskStatus::complete;
}

// *************************************************//
// define the application driver. in this case,    *//
// that just means defining the MakeTaskList       *//
// function.                                       *//
// *************************************************//

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  auto &adv_pkg = pmb->packages.Get("advection_package");
  auto &pt_pkg = pmb->packages.Get("particles_package");
  auto &mbd = pmb->meshblock_data.Get();
  auto &advected = mbd->Get("advected").data;

  // Get swarm container (already exists in MeshBlockData)
  auto swarm_container = mbd->GetSwarmData();

  std::shared_ptr<Swarm> swarm;
  if (!swarm_container->Contains("particles")) {
    Metadata swarm_metadata({Metadata::Provides, Metadata::None});
    swarm = std::make_shared<Swarm>("particles", swarm_metadata);
    swarm_container->Add(swarm);
    // Note: SetBlockPointer is not needed here as the swarm will get its block pointer
    // during initialization by the swarm framework

    // Add position and id variables
    Metadata m_real({Metadata::Real});
    swarm->Add("x", m_real);
    swarm->Add("y", m_real);
    swarm->Add("z", m_real);

    Metadata m_int({Metadata::Integer});
    swarm->Add("id", m_int);
  } else {
    swarm = swarm_container->Get("particles");
  }

  const int num_particles = pt_pkg->Param<int>("num_particles");
  auto rng_pool =
      RNGPool(pmb->gid); // Seed is meshblock gid for consistency across MPI decomposition

  const int ndim = pmb->pmy_mesh->ndim;

  const IndexRange &ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange &jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange &kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  auto coords = pmb->coords;

  const Real advected_mean = 1.0;
  const Real advected_amp = 0.5;

  const Real &x_min = pmb->coords.Xf<1>(ib.s);
  const Real &y_min = pmb->coords.Xf<2>(jb.s);
  const Real &z_min = pmb->coords.Xf<3>(kb.s);
  const Real &x_max = pmb->coords.Xf<1>(ib.e + 1);
  const Real &y_max = pmb->coords.Xf<2>(jb.e + 1);
  const Real &z_max = pmb->coords.Xf<3>(kb.e + 1);

  const auto mesh_size = pmb->pmy_mesh->mesh_size;
  const Real x_min_mesh = mesh_size.xmin(X1DIR);
  const Real y_min_mesh = mesh_size.xmin(X2DIR);
  const Real z_min_mesh = mesh_size.xmin(X3DIR);
  const Real x_max_mesh = mesh_size.xmax(X1DIR);
  const Real y_max_mesh = mesh_size.xmax(X2DIR);
  const Real z_max_mesh = mesh_size.xmax(X3DIR);

  const Real kwave = 2. * M_PI / (x_max_mesh - x_min_mesh);

  // Initialize advected field
  pmb->par_for(
      PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        advected(k, j, i) = advected_mean + advected_amp * sin(kwave * coords.Xc<1>(i));
      });

  // Calculate fraction of total particles on this meshblock
  Real volume_meshblock = (x_max - x_min) * (y_max - y_min) * (z_max - z_min);
  Real volume_mesh = (x_max_mesh - x_min_mesh) * (y_max_mesh - y_min_mesh) *
                     (z_max_mesh - z_min_mesh);

  int num_particles_meshblock =
      std::round(num_particles * volume_meshblock / volume_mesh);

  auto new_particles_context = swarm->AddEmptyParticles(num_particles_meshblock);

  auto &x = swarm->Get<Real>(swarm_position::x::name()).Get();
  auto &y = swarm->Get<Real>(swarm_position::y::name()).Get();
  auto &z = swarm->Get<Real>(swarm_position::z::name()).Get();
  auto &id = swarm->Get<int>("id").Get();

  auto swarm_d = swarm->GetDeviceContext();
  int gid = pmb->gid;
  pmb->par_for(
      PARTHENON_AUTO_LABEL, 0, new_particles_context.GetNewParticlesMaxIndex(),
      KOKKOS_LAMBDA(const int new_n) {
        const int n = new_particles_context.GetNewParticleIndex(new_n);
        auto rng_gen = rng_pool.get_state();

        // Uniformly distribute particles in this meshblock
        x(n) = x_min + rng_gen.drand() * (x_max - x_min);
        y(n) = y_min + rng_gen.drand() * (y_max - y_min);
        z(n) = z_min + rng_gen.drand() * (z_max - z_min);

        // Create unique ID combining gid and particle index
        id(n) = gid * 1000000 + n;

        rng_pool.free_state(rng_gen);
      });
}

TaskCollection ParticleDriver::MakeTaskCollection(BlockList_t &blocks, int stage) {
  TaskCollection tc;
  TaskID none(0);
  auto pmesh = blocks[0]->pmy_mesh;

  const Real beta = integrator->beta[stage - 1];
  const Real dt = integrator->dt;
  const auto &stage_name = integrator->stage_name;
  const int nstages = integrator->nstages;

  const auto nblocks = blocks.size();
  TaskRegion &async_region0 = tc.AddRegion(nblocks);

  // Staged advection update of advected field
  for (int n = 0; n < nblocks; n++) {
    auto &pmb = blocks[n];
    auto &tl = async_region0[n];

    auto &base = pmb->meshblock_data.Get();
    if (stage == 1) {
      pmb->meshblock_data.Add("dUdt", base);
      for (int m = 1; m < nstages; m++) {
        pmb->meshblock_data.Add(stage_name[m], base);
      }
    }

    auto &sc0 = pmb->meshblock_data.Get(stage_name[stage - 1]);
    auto advect_flux = tl.AddTask(none, particle_amr_example::CalculateFluxes, sc0.get());
  }

  auto partitions = pmesh->GetDefaultBlockPartitions();
  const int num_partitions = partitions.size();
  TaskRegion &single_tasklist_per_pack_region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = single_tasklist_per_pack_region[i];
    auto &mbase = pmesh->mesh_data.Add("base", partitions[i]);
    auto &mc0 = pmesh->mesh_data.Add(stage_name[stage - 1], mbase);
    auto &mc1 = pmesh->mesh_data.Add(stage_name[stage], mbase);
    auto &mdudt = pmesh->mesh_data.Add("dUdt", mbase);

    const auto any = parthenon::BoundaryType::any;

    tl.AddTask(none, parthenon::StartReceiveBoundBufs<any>, mc1);
    tl.AddTask(none, parthenon::StartReceiveFluxCorrections, mc0);

    auto send_flx = tl.AddTask(none, parthenon::LoadAndSendFluxCorrections, mc0);
    auto recv_flx = tl.AddTask(none, parthenon::ReceiveFluxCorrections, mc0);
    auto set_flx = tl.AddTask(recv_flx, parthenon::SetFluxCorrections, mc0);

    // compute the divergence of fluxes of conserved variables
    auto flux_div =
        tl.AddTask(set_flx, FluxDivergence<MeshData<Real>>, mc0.get(), mdudt.get());

    auto avg_data = tl.AddTask(flux_div, AverageIndependentData<MeshData<Real>>,
                               mc0.get(), mbase.get(), beta);
    // apply du/dt to all independent fields in the container
    auto update = tl.AddTask(avg_data, UpdateIndependentData<MeshData<Real>>, mc0.get(),
                             mdudt.get(), beta * dt, mc1.get());

    // do boundary exchange
    parthenon::AddBoundaryExchangeTasks(update, tl, mc1, pmesh->multilevel);
  }

  TaskRegion &async_region1 = tc.AddRegion(nblocks);
  for (int n = 0; n < nblocks; n++) {
    auto &pmb = blocks[n];
    auto &tl = async_region1[n];
    auto &sc1 = pmb->meshblock_data.Get(stage_name[stage]);

    auto set_bc = tl.AddTask(none, parthenon::ApplyBoundaryConditions, sc1);

    auto advect_particles = tl.AddTask(set_bc, AdvectParticles, pmb.get(), integrator.get());
    auto count_particles = tl.AddTask(advect_particles, CountParticles, pmb.get());

    if (stage == integrator->nstages) {
      auto new_dt = tl.AddTask(
          count_particles, parthenon::Update::EstimateTimestep<MeshBlockData<Real>>, sc1.get());
    }
  }

  // Particle boundary communication
  if (stage == integrator->nstages) {
    TaskRegion &sync_region0 = tc.AddRegion(1);
    {
      for (int i = 0; i < blocks.size(); i++) {
        auto &tl = sync_region0[0];
        auto &pmb = blocks[i];
        auto &sc = pmb->meshblock_data.Get()->GetSwarmData();
        auto reset_comms =
            tl.AddTask(none, &SwarmContainer::ResetCommunication, sc.get());
      }
    }

    TaskRegion &async_region2 = tc.AddRegion(nblocks);
    for (int n = 0; n < nblocks; n++) {
      auto &tl = async_region2[n];
      auto &pmb = blocks[n];
      auto &sc = pmb->meshblock_data.Get()->GetSwarmData();

      auto send = tl.AddTask(none, &SwarmContainer::Send, sc.get(),
                             BoundaryCommSubset::all);

      auto receive =
          tl.AddTask(send, &SwarmContainer::Receive, sc.get(), BoundaryCommSubset::all);

      // Defragment if swarm memory pool occupancy is 90%
      auto defrag = tl.AddTask(receive, &SwarmContainer::Defrag, sc.get(), 0.9);
    }
  }

  return tc;
}

} // namespace particle_amr_example
