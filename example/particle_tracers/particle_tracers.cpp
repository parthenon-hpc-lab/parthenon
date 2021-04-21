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

#include "particle_tracers.hpp"
#include "Kokkos_CopyViews.hpp"
#include "Kokkos_HostSpace.hpp"
#include "Kokkos_Random.hpp"
#include "basic_types.hpp"
#include "config.hpp"
#include "globals.hpp"
#include "interface/metadata.hpp"
#include "interface/update.hpp"
#include "kokkos_abstraction.hpp"
#include "parthenon/driver.hpp"
#include "refinement/refinement.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#define SMALL (1.e-10)

using namespace parthenon::driver::prelude;
using namespace parthenon::Update;

typedef Kokkos::Random_XorShift64_Pool<> RNGPool;

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

  Real vmatx = pin->GetOrAddReal("Material", "vx", 1.0);
  pkg->AddParam<>("vmatx", vmatx);
  Real vmaty = pin->GetOrAddReal("Material", "vy", 0.0);
  pkg->AddParam<>("vmaty", vmaty);
  Real vmatz = pin->GetOrAddReal("Material", "vz", 0.0);
  pkg->AddParam<>("vmatz", vmatz);

  Real cfl = pin->GetOrAddReal("Material", "cfl", 0.3);
  pkg->AddParam<>("cfl", cfl);

  auto write_particle_log =
      pin->GetOrAddBoolean("Particles", "write_particle_log", false);
  pkg->AddParam<>("write_particle_log", write_particle_log);

  int num_tracers = pin->GetOrAddReal("Tracers", "num_tracers", 100);
  pkg->AddParam<>("num_tracers", num_tracers);

  // Initialize random number generator pool
  int rng_seed = pin->GetOrAddInteger("Tracers", "rng_seed", 1273);
  pkg->AddParam<>("rng_seed", rng_seed);
  RNGPool rng_pool(rng_seed);
  pkg->AddParam<>("rng_pool", rng_pool);

  std::string swarm_name = "tracers";
  Metadata swarm_metadata;
  pkg->AddSwarm(swarm_name, swarm_metadata);
  Metadata real_swarmvalue_metadata({Metadata::Real});
  pkg->AddSwarmValue("id", swarm_name, Metadata({Metadata::Integer}));

  std::string field_name = "density";
  Metadata mfield({Metadata::Cell, Metadata::Independent, Metadata::FillGhost});
  pkg->AddField(field_name, mfield);

  pkg->EstimateTimestepBlock = EstimateTimestepBlock;

  return pkg;
}

AmrTag CheckRefinement(MeshBlockData<Real> *rc) { return AmrTag::same; }

Real EstimateTimestepBlock(MeshBlockData<Real> *rc) {
  auto pmb = rc->GetBlockPointer();
  //  auto swarm = pmb->swarm_data.Get()->Get("tracers");
  auto pkg = pmb->packages.Get("particles_package");
  const auto &cfl = pkg->Param<Real>("cfl");

  // int max_active_index = swarm->GetMaxActiveIndex();

  const auto &vmatx = pkg->Param<Real>("vmatx");
  const auto &vmaty = pkg->Param<Real>("vmaty");
  const auto &vmatz = pkg->Param<Real>("vmatz");

  // Assumes a grid with constant dx, dy, dz within a block
  const Real &dx_i = pmb->coords.dx1f(0);
  const Real &dx_j = pmb->coords.dx2f(0);
  const Real &dx_k = pmb->coords.dx3f(0);

  Real min_dt = dx_i / std::abs(vmatx + SMALL);
  min_dt = std::min(min_dt, dx_j / std::abs(vmaty + SMALL));
  min_dt = std::min(min_dt, dx_k / std::abs(vmatz + SMALL));

  return cfl * min_dt;
}

} // namespace Particles

// *************************************************//
// define the application driver. in this case,    *//
// that just means defining the MakeTaskList       *//
// function.                                       *//
// *************************************************//

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  auto &pkg = pmb->packages.Get("particles_package");
  auto &rc = pmb->meshblock_data.Get();
  auto &density = rc->Get("density").data;
  auto &swarm = pmb->swarm_data.Get()->Get("tracers");
  const auto num_tracers = pkg->Param<int>("num_tracers");
  auto rng_pool = pkg->Param<RNGPool>("rng_pool");

  const IndexRange &ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange &jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange &kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  auto coords = pmb->coords;

  const Real density_mean = 1.0;
  const Real density_amp = 0.5;
  PARTHENON_REQUIRE(density_mean > density_amp, "Cannot have negative densities!");

  auto density_h = density.GetHostMirror();
  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e; j++) {
      for (int i = ib.s; i <= ib.e; i++) {
        density_h(k, j, i) = density_mean + density_amp * sin(2. * M_PI * coords.x1v(i));
      }
    }
  }
  density.DeepCopy(density_h);

  const Real &x_min = pmb->coords.x1f(ib.s);
  const Real &y_min = pmb->coords.x2f(jb.s);
  const Real &z_min = pmb->coords.x3f(kb.s);
  const Real &x_max = pmb->coords.x1f(ib.e + 1);
  const Real &y_max = pmb->coords.x2f(jb.e + 1);
  const Real &z_max = pmb->coords.x3f(kb.e + 1);

  const auto mesh_size = pmb->pmy_mesh->mesh_size;
  const Real x_min_mesh = mesh_size.x1min;
  const Real y_min_mesh = mesh_size.x2min;
  const Real z_min_mesh = mesh_size.x3min;
  const Real x_max_mesh = mesh_size.x1max;
  const Real y_max_mesh = mesh_size.x2max;
  const Real z_max_mesh = mesh_size.x3max;

  // Calculate fraction of total tracer particles on this meshblock
  Real mass_meshblock = density_mean*(x_max - x_min) - density_amp/(2.*M_PI)*(cos(2.*M_PI*x_max) - cos(2.*M_PI*x_min));
  mass_meshblock *= (y_max - y_min)*(z_max - z_min);
  printf("mass_meshblock: %e\n", mass_meshblock);
  Real mass_mesh = density_mean*(x_max_mesh - x_min_mesh);
  mass_mesh -= density_amp/(2.*M_PI)*(cos(2.*M_PI*x_max_mesh) - cos(2.*M_PI*x_min_mesh));
  mass_mesh *= (y_max_mesh - y_min_mesh)*(z_max_mesh - z_min_mesh);
  printf("mess_mesh: %e\n", mass_mesh);

  int num_tracers_meshblock = std::round(num_tracers*mass_meshblock/mass_mesh);
  printf("num_tracers_meshblock: %i\n", num_tracers_meshblock);

  ParArrayND<int> new_indices;
  const auto new_particles_mask =
      swarm->AddEmptyParticles(num_tracers_meshblock, new_indices);

  auto &x = swarm->Get<Real>("x").Get();
  auto &y = swarm->Get<Real>("y").Get();
  auto &z = swarm->Get<Real>("z").Get();

  auto swarm_d = swarm->GetDeviceContext();
  const auto &my_rank = Globals::my_rank;
  const auto &gid = pmb->gid;
  // This hardcoded implementation should only used in PGEN and not during runtime
  // addition of particles as indices need to be taken into account.
  pmb->par_for(
      "CreateParticles", 0, num_tracers_meshblock - 1, KOKKOS_LAMBDA(const int n) {

        auto rng_gen = rng_pool.get_state();

        x(n) = x_min + rng_gen.drand()*(x_max - x_min);
        y(n) = y_min + rng_gen.drand()*(y_max - y_min);
        z(n) = z_min + rng_gen.drand()*(z_max - z_min);

        rng_pool.free_state(rng_gen);
      });
}

TaskStatus AdvectTracers(MeshBlock *pmb, const StagedIntegrator *integrator) {
    auto swarm = pmb->swarm_data.Get()->Get("tracers");
    auto pkg = pmb->packages.Get("particles_package");

    int max_active_index = swarm->GetMaxActiveIndex();

    Real dt = integrator->dt;

    auto &x = swarm->Get<Real>("x").Get();
    auto &y = swarm->Get<Real>("y").Get();
    auto &z = swarm->Get<Real>("z").Get();

    const auto &vmatx = pkg->Param<Real>("vmatx");

    auto swarm_d = swarm->GetDeviceContext();
    pmb->par_for("Tracer advection", 0, max_active_index, KOKKOS_LAMBDA(const int n) {
          if (swarm_d.IsActive(n)) {

            x(n) += vmatx * dt;

            bool on_current_mesh_block = true;
            swarm_d.GetNeighborBlockIndex(n, x(n), y(n), z(n), on_current_mesh_block);
          }
      });

  return TaskStatus::complete;
}

// Mark all MPI requests as NULL / initialize boundary flags.
// TODO(BRR) Should this be a Swarm method?
TaskStatus InitializeCommunicationMesh(const BlockList_t &blocks) {
  /*  // Boundary transfers on same MPI proc are blocking
    for (auto &block : blocks) {
      auto swarm = block->swarm_data.Get()->Get("tracers");
      for (int n = 0; n < block->pbval->nneighbor; n++) {
        NeighborBlock &nb = block->pbval->neighbor[n];
        swarm->vbswarm->bd_var_.req_send[nb.bufid] = MPI_REQUEST_NULL;
      }
    }

    for (auto &block : blocks) {
      auto &pmb = block;
      auto sc = pmb->swarm_data.Get();
      auto swarm = sc->Get("tracers");

      for (int n = 0; n < swarm->vbswarm->bd_var_.nbmax; n++) {
        auto &nb = pmb->pbval->neighbor[n];
        swarm->vbswarm->bd_var_.flag[nb.bufid] = BoundaryStatus::waiting;
      }
    }

    // Reset boundary statuses
    for (auto &block : blocks) {
      auto &pmb = block;
      auto sc = pmb->swarm_data.Get();
      auto swarm = sc->Get("tracers");
      for (int n = 0; n < swarm->vbswarm->bd_var_.nbmax; n++) {
        auto &nb = pmb->pbval->neighbor[n];
        swarm->vbswarm->bd_var_.flag[nb.bufid] = BoundaryStatus::waiting;
      }
    }
  */
  return TaskStatus::complete;
}

TaskStatus Defrag(MeshBlock *pmb) {
  /*auto s = pmb->swarm_data.Get()->Get("tracers");

  // Only do this if list is getting too sparse. This criterion (whether there
  // are *any* gaps in the list) is very aggressive
  if (s->GetNumActive() <= s->GetMaxActiveIndex()) {
    s->Defrag();
  }*/

  return TaskStatus::complete;
}

TaskStatus CalculateFluxes(MeshBlockData<Real> *rc) {
  auto pmb = rc->GetBlockPointer();
  auto pkg = pmb->packages.Get("particles_package");

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto density = rc->Get("density").data;
  auto x1flux = rc->Get("density").flux[X1DIR].Get<4>();

  const auto &vmatx = pkg->Param<Real>("vmatx");
  const auto &vmaty = pkg->Param<Real>("vmaty");
  const auto &vmatz = pkg->Param<Real>("vmatz");

  PARTHENON_REQUIRE(pmb->pmy_mesh->ndim == 1, "2D and 3D fluxes not supported!");

  // Upwind method
  pmb->par_for(
      "CalculateFluxesX1", kb.s, kb.e, jb.s, jb.e, ib.s - 1, ib.e + 1,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        x1flux(0, k, j, i) = density(k, j, i - 1) * vmatx;
      });

  return TaskStatus::complete;
}

TaskStatus DepositParticles(MeshBlock *pmb) {
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
            Kokkos::atomic_add(&particle_dep(k, j, i), 1.0);
          }
        }
      });

  return TaskStatus::complete;
}

TaskCollection ParticleDriver::MakeTaskCollection(BlockList_t &blocks, int stage) {
  TaskCollection tc;
  TaskID none(0);

  const Real beta = integrator->beta[stage - 1];
  const Real dt = integrator->dt;
  const auto &stage_name = integrator->stage_name;
  const int nstages = integrator->nstages;

  const auto nblocks = blocks.size();
  TaskRegion &async_region0 = tc.AddRegion(nblocks);

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
    auto &dudt = pmb->meshblock_data.Get("dUdt");
    auto &sc1 = pmb->meshblock_data.Get(stage_name[stage]);

    auto start_recv = tl.AddTask(none, &MeshBlockData<Real>::StartReceiving, sc1.get(),
                                 BoundaryCommSubset::all);

    auto advect_flux = tl.AddTask(none, particles_example::CalculateFluxes, sc0.get());

    auto send_flux =
        tl.AddTask(advect_flux, &MeshBlockData<Real>::SendFluxCorrection, sc0.get());

    auto recv_flux =
        tl.AddTask(advect_flux, &MeshBlockData<Real>::ReceiveFluxCorrection, sc0.get());

    auto flux_div =
        tl.AddTask(recv_flux, FluxDivergence<MeshBlockData<Real>>, sc0.get(), dudt.get());

    auto avg_data = tl.AddTask(flux_div, AverageIndependentData<MeshBlockData<Real>>,
                               sc0.get(), base.get(), beta);

    auto update = tl.AddTask(avg_data, UpdateIndependentData<MeshBlockData<Real>>,
                             sc0.get(), dudt.get(), beta * dt, sc1.get());

    auto send = tl.AddTask(update, &MeshBlockData<Real>::SendBoundaryBuffers, sc1.get());

    auto recv = tl.AddTask(send, &MeshBlockData<Real>::ReceiveBoundaryBuffers, sc1.get());

    auto fill_from_bufs =
        tl.AddTask(recv, &MeshBlockData<Real>::SetBoundaries, sc1.get());

    auto clear_comm_flags =
        tl.AddTask(fill_from_bufs, &MeshBlockData<Real>::ClearBoundary, sc1.get(),
                   BoundaryCommSubset::all);

    auto prolongBound = tl.AddTask(fill_from_bufs, parthenon::ProlongateBoundaries, sc1);

    auto set_bc = tl.AddTask(prolongBound, parthenon::ApplyBoundaryConditions, sc1);

    if (stage == integrator->nstages) {
      auto new_dt = tl.AddTask(
          set_bc, parthenon::Update::EstimateTimestep<MeshBlockData<Real>>, sc1.get());

      // First-order operator split tracer particle update

      // Initialize comms

      auto tracerAdvect = tl.AddTask(set_bc, AdvectTracers, pmb.get(), integrator.get());

      // Send

      // Receive

      // Deposit

      // Defrag
    }
  }

  return tc;
}

} // namespace particles_example
