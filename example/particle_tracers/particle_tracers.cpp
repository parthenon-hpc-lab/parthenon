//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2021-2024 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2021-2026. Triad National Security, LLC. All rights reserved.
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

// C++ includes
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// Parthenon includes
#include "basic_types.hpp"
#include "bvals/comms/bvals_in_one.hpp"
#include "config.hpp"
#include "globals.hpp"
#include "interface/metadata.hpp"
#include "interface/update.hpp"
#include "kokkos_abstraction.hpp"
#include "pack/default_names.hpp"
#include "prolong_restrict/prolong_restrict.hpp"

#include "particle_tracers.hpp"

using namespace parthenon::driver::prelude;
using namespace parthenon::Update;

typedef Kokkos::Random_XorShift64_Pool<> RNGPool;

namespace tracers_example {

// ****************************************************//
// Define variables/names to be used by this example. *//
// ****************************************************//

static const char swarm_name[] = "tracers";

namespace field {
PAR_VAR(field, advected);
PAR_VAR(field, deposition);
} // namespace field

// ****************************************************//
// Add multiple packages, one for the advected        *//
// background and one for the tracer particles.       *//
// ****************************************************//

Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;
  packages.Add(advection_package::Initialize(pin.get()));
  packages.Add(particles_package::Initialize(pin.get()));
  return packages;
}

namespace advection_package {

// ****************************************************//
// Define the advection package, including            *//
// timestep and package initialization functions.     *//
// ****************************************************//

Real EstimateTimestepMesh(MeshData<Real> *md) {
  auto pm = md->GetParentPointer();
  auto adv_pkg = pm->packages.Get("advection_package");

  const auto &cfl = adv_pkg->Param<Real>("cfl");
  const auto &vx = adv_pkg->Param<Real>("vx");
  const auto &vy = adv_pkg->Param<Real>("vy");
  const auto &vz = adv_pkg->Param<Real>("vz");

  // Assumes a grid with constant dx, dy, dz within a block
  std::array<Real, 3> dx;
  dx.fill(std::numeric_limits<Real>::max());
  for (auto &pmb : pm->block_list) {
    const auto &reg = pmb->block_size;
    dx[0] = std::min(dx[0], pmb->coords.Dxc<X1DIR>(0));
    dx[1] = std::min(dx[1], pmb->coords.Dxc<X2DIR>(0));
    dx[2] = std::min(dx[2], pmb->coords.Dxc<X3DIR>(0));
  }

  Real min_dt = dx[0] / std::abs(vx + TINY_NUMBER);
  min_dt = std::min(min_dt, dx[1] / std::abs(vy + TINY_NUMBER));
  min_dt = std::min(min_dt, dx[2] / std::abs(vz + TINY_NUMBER));

  return cfl * min_dt;
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto adv_pkg = std::make_shared<StateDescriptor>("advection_package");

  adv_pkg->AddParam("vx", pin->GetOrAddReal("Background", "vx", 1.0));
  adv_pkg->AddParam("vy", pin->GetOrAddReal("Background", "vy", 0.0));
  adv_pkg->AddParam("vz", pin->GetOrAddReal("Background", "vz", 0.0));
  adv_pkg->AddParam("cfl", pin->GetOrAddReal("Background", "cfl", 0.3));

  const Real advected_mean = 1.0;
  const Real advected_amp = 0.5;
  adv_pkg->AddParam("advected_mean", advected_mean);
  adv_pkg->AddParam("advected_amp", advected_amp);
  PARTHENON_REQUIRE(advected_mean > advected_amp,
                    "Advected field must be everywere positive!");

  // Add advected field
  Metadata madv(
      {Metadata::Cell, Metadata::Independent, Metadata::FillGhost, Metadata::WithFluxes});
  adv_pkg->AddField<field::advected>(madv);

  // Add deposition field
  Metadata mdep({Metadata::Cell, Metadata::Derived});
  adv_pkg->AddField<field::deposition>(mdep);

  // Assign package timestep hook
  adv_pkg->EstimateTimestepMesh = EstimateTimestepMesh;

  return adv_pkg;
}

} // namespace advection_package

namespace particles_package {

// *************************************************//
// Define the tracer particles package, including  *//
// timestep and initialization functions.          *//
// *************************************************//

// NOTE(@pdmullen): The below tracers timestep function is currently redundant with the
// advection timestep, however, its inclusion demonstrates having two packages with votes
// towards the global timestep.
Real EstimateTimestepMesh(MeshData<Real> *md) {
  auto pm = md->GetParentPointer();
  auto adv_pkg = pm->packages.Get("advection_package");

  const auto &cfl = adv_pkg->Param<Real>("cfl");
  const auto &vx = adv_pkg->Param<Real>("vx");
  const auto &vy = adv_pkg->Param<Real>("vy");
  const auto &vz = adv_pkg->Param<Real>("vz");

  // Assumes a grid with constant dx, dy, dz within a block
  std::array<Real, 3> dx;
  dx.fill(std::numeric_limits<Real>::max());
  for (auto &pmb : pm->block_list) {
    const auto &reg = pmb->block_size;
    dx[0] = std::min(dx[0], pmb->coords.Dxc<X1DIR>(0));
    dx[1] = std::min(dx[1], pmb->coords.Dxc<X2DIR>(0));
    dx[2] = std::min(dx[2], pmb->coords.Dxc<X3DIR>(0));
  }

  Real min_dt = dx[0] / std::abs(vx + TINY_NUMBER);
  min_dt = std::min(min_dt, dx[1] / std::abs(vy + TINY_NUMBER));
  min_dt = std::min(min_dt, dx[2] / std::abs(vz + TINY_NUMBER));

  // No CFL number for particles
  return min_dt;
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto tr_pkg = std::make_shared<StateDescriptor>("particles_package");

  tr_pkg->AddParam("num_tracers", pin->GetOrAddInteger("Tracers", "num_tracers", 100));

  // `NoPersistentParticleIds` is just passed to test this aspect in the regression tests.
  // For typical tracers, persistent ids might be important.
  Metadata m({Metadata::Provides, Metadata::None, Metadata::NoPersistentParticleIds});
  tr_pkg->AddSwarm(swarm_name, m);

  // Assign package timestep hook
  tr_pkg->EstimateTimestepMesh = EstimateTimestepMesh;

  // Assign package final initialization hook
  tr_pkg->FinalInitializationBlock = SourceTracers;

  return tr_pkg;
}

void SourceTracers(MeshBlock *pmb, ParameterInput *pin) {
  auto &mbd = pmb->meshblock_data.Get();
  auto &adv_pkg = pmb->packages.Get("advection_package");
  auto &tr_pkg = pmb->packages.Get("particles_package");

  // Advection package params
  const Real &advected_mean = adv_pkg->Param<Real>("advected_mean");
  const Real &advected_amp = adv_pkg->Param<Real>("advected_amp");

  // Tracer package params and swarm
  auto &swarm = mbd->GetSwarmData()->Get(swarm_name);
  const auto num_tracers = tr_pkg->Param<int>("num_tracers");

  // RNG seed is meshblock gid for consistency across MPI decomposition
  auto rng_pool = RNGPool(pmb->gid);

  // Indexing
  const IndexRange &ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange &jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange &kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // Block physical size
  auto coords = pmb->coords;
  const Real &x_min = coords.Xf<1>(ib.s);
  const Real &y_min = coords.Xf<2>(jb.s);
  const Real &z_min = coords.Xf<3>(kb.s);
  const Real &x_max = coords.Xf<1>(ib.e + 1);
  const Real &y_max = coords.Xf<2>(jb.e + 1);
  const Real &z_max = coords.Xf<3>(kb.e + 1);

  // Mesh physical size
  const auto mesh_size = pmb->pmy_mesh->mesh_size;
  const Real x_min_mesh = mesh_size.xmin(X1DIR);
  const Real y_min_mesh = mesh_size.xmin(X2DIR);
  const Real z_min_mesh = mesh_size.xmin(X3DIR);
  const Real x_max_mesh = mesh_size.xmax(X1DIR);
  const Real y_max_mesh = mesh_size.xmax(X2DIR);
  const Real z_max_mesh = mesh_size.xmax(X3DIR);
  const Real kwave = 2. * M_PI / (x_max_mesh - x_min_mesh);

  // Calculate frac of total tracers on this MeshBlock by integrating the advected profile
  // over the Mesh and this MeshBlock. Tracer number follows number = advected*volume.
  const Real nmesh = (advected_mean * (x_max_mesh - x_min_mesh) -
                      advected_amp / kwave *
                          (std::cos(kwave * x_max_mesh) - std::cos(kwave * x_min_mesh))) *
                     (y_max_mesh - y_min_mesh) * (z_max_mesh - z_min_mesh);
  const Real nmeshblock =
      (advected_mean * (x_max - x_min) -
       advected_amp / kwave * (std::cos(kwave * x_max) - std::cos(kwave * x_min))) *
      (y_max - y_min) * (z_max - z_min);

  // Add these particles
  int num_tracers_meshblock = std::round(num_tracers * nmeshblock / nmesh);
  auto new_particles_context = swarm->AddEmptyParticles(num_tracers_meshblock);

  // Create pack
  static auto desc =
      MakeSwarmPackDescriptor<swarm_position::x, swarm_position::y, swarm_position::z>(
          swarm_name);
  auto pack = desc.GetPack(mbd.get());

  // Initialize tracers on this block
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "InitializeTracers", DevExecSpace(), 0,
      new_particles_context.GetNewParticlesMaxIndex(), KOKKOS_LAMBDA(const int new_n) {
        const int n = new_particles_context.GetNewParticleIndex(new_n);
        auto rng_gen = rng_pool.get_state();

        // Extract particle position
        Real &xx = pack(0, swarm_position::x(), n);
        Real &yy = pack(0, swarm_position::y(), n);
        Real &zz = pack(0, swarm_position::z(), n);

        // Rejection sample the x position
        Real val;
        do {
          xx = x_min + rng_gen.drand() * (x_max - x_min);
          val = advected_mean + advected_amp * std::sin(2. * M_PI * xx);
        } while (val < rng_gen.drand() * (advected_mean + advected_amp));

        yy = y_min + rng_gen.drand() * (y_max - y_min);
        zz = z_min + rng_gen.drand() * (z_max - z_min);

        rng_pool.free_state(rng_gen);
      });
}

} // namespace particles_package

// *************************************************//
// Now we define the tasks.  We here exmplify      *//
// tasks operating on MeshData<Real> registers,    *//
// such that all blocks within the MeshData        *//
// partition are updated.                          *//
// *************************************************//

TaskStatus AdvectTracers(MeshData<Real> *md, const Real dt) {
  auto pm = md->GetParentPointer();

  // Advection params
  auto adv_pkg = pm->packages.Get("advection_package");
  const auto &vx = adv_pkg->Param<Real>("vx");
  const auto &vy = adv_pkg->Param<Real>("vy");
  const auto &vz = adv_pkg->Param<Real>("vz");

  // Create pack
  static auto desc =
      MakeSwarmPackDescriptor<swarm_position::x, swarm_position::y, swarm_position::z>(
          swarm_name);
  auto pack = desc.GetPack(md);

  // Advect tracers
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "AdvectTracers", DevExecSpace(), 0, pack.GetMaxFlatIndex(),
      KOKKOS_LAMBDA(const int idx) {
        auto [b, n] = pack.GetBlockParticleIndices(idx);
        const auto &swarm_d = pack.GetContext(b);
        if (swarm_d.IsActive(n)) {
          pack(b, swarm_position::x(), n) += vx * dt;
          pack(b, swarm_position::y(), n) += vy * dt;
          pack(b, swarm_position::z(), n) += vz * dt;
        }
      });

  return TaskStatus::complete;
}

TaskStatus DepositTracers(MeshData<Real> *md) {
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  // Indexing
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);

  // Create packs
  static auto desc = MakePackDescriptor<field::deposition>(resolved_pkgs.get());
  static auto pdesc =
      MakeSwarmPackDescriptor<swarm_position::x, swarm_position::y, swarm_position::z>(
          swarm_name);
  auto vmesh = desc.GetPack(md);
  auto vpart = pdesc.GetPack(md);

  // First zero the deposition field
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Zero", parthenon::DevExecSpace(), 0, md->NumBlocks() - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        vmesh(b, field::deposition(), k, j, i) = 0.0;
      });

  // Now atomically add to depositon field
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "DepositTracers", DevExecSpace(), 0, vpart.GetMaxFlatIndex(),
      KOKKOS_LAMBDA(const int idx) {
        auto [b, n] = vpart.GetBlockParticleIndices(idx);
        const auto &swarm_d = vpart.GetContext(b);
        if (swarm_d.IsActive(n)) {
          int ip, jp, kp;
          const Real &xx = vpart(b, swarm_position::x(), n);
          const Real &yy = vpart(b, swarm_position::y(), n);
          const Real &zz = vpart(b, swarm_position::z(), n);
          swarm_d.Xtoijk(xx, yy, zz, ip, jp, kp);

          // For testing in this example we make sure the indices are correct; these
          // could be demoted to Debug-only calls
          const bool inside_x = ip >= ib.s && ip <= ib.e;
          const bool inside_y = jp >= jb.s && jp <= jb.e;
          const bool inside_z = kp >= kb.s && kp <= kb.e;
          if (inside_x && inside_y && inside_z) {
            Kokkos::atomic_add(&vmesh(b, field::deposition(), kp, jp, ip), 1.0);
          } else {
            PARTHENON_FAIL("Tracer outside of active domain during deposition.");
          }
        }
      });

  return TaskStatus::complete;
}

TaskStatus CalculateFluxes(MeshData<Real> *md) {
  auto pm = md->GetParentPointer();
  auto &resolved_pkgs = pm->resolved_packages;

  // Advection package params
  auto adv_pkg = pm->packages.Get("advection_package");
  const Real &vx = adv_pkg->Param<Real>("vx");
  const Real &vy = adv_pkg->Param<Real>("vy");
  const Real &vz = adv_pkg->Param<Real>("vz");

  // Indexing and dimensionality
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);
  const auto ndim = pm->ndim;

  // Create pack
  static auto desc = parthenon::MakePackDescriptor<field::advected>(
      resolved_pkgs.get(), {}, {parthenon::PDOpt::WithFluxes});
  auto pack = desc.GetPack(md);

  // X1-Flux
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "X1-Flux", parthenon::DevExecSpace(), 0, md->NumBlocks() - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e + 1,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        pack.flux(b, X1DIR, field::advected(), k, j, i) =
            vx * ((vx > 0) ? pack(b, field::advected(), k, j, i - 1)
                           : pack(b, field::advected(), k, j, i));
      });

  if (ndim > 1) {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "X2-Flux", parthenon::DevExecSpace(), 0,
        md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e + 1, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          pack.flux(b, X2DIR, field::advected(), k, j, i) =
              vy * ((vy > 0) ? pack(b, field::advected(), k, j - 1, i)
                             : pack(b, field::advected(), k, j, i));
        });
  }

  if (ndim > 2) {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "X3-Flux", parthenon::DevExecSpace(), 0,
        md->NumBlocks() - 1, kb.s, kb.e + 1, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          pack.flux(b, X3DIR, field::advected(), k, j, i) =
              vz * ((vz > 0) ? pack(b, field::advected(), k - 1, j, i)
                             : pack(b, field::advected(), k, j, i));
        });
  }

  return TaskStatus::complete;
}

// ********************************************************//
// Define the application driver. in this case, that just *//
// that just means defining the Step and StepTasks.       *//
// ********************************************************//

TaskListStatus ParticleDriver::Step() { return StepTasks().Execute(); }

TaskCollection ParticleDriver::StepTasks() {
  using namespace parthenon::Update;
  TaskCollection tc;

  // MeshData Registers
  auto &base = pmesh->mesh_data.Get();
  auto &u0 = pmesh->mesh_data.AddShallow("u0", base);
  auto &u1 = pmesh->mesh_data.Add("u1", u0);

  TaskID none(0);
  const auto any = parthenon::BoundaryType::any;
  const int num_partitions = pmesh->DefaultNumPartitions();

  // For multi-stage integrator logic, deep copy u0 --> u1
  auto &init_region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = init_region[i];
    auto &u0 = pmesh->mesh_data.GetOrAdd("u0", i);
    auto &u1 = pmesh->mesh_data.GetOrAdd("u1", i);
    tl.AddTask(none, CopyData<std::vector<MetadataFlag>, MeshData<Real>>,
               std::vector<MetadataFlag>({Metadata::Independent}), u0.get(), u1.get());
  }

  // Execute multi-stage integrator logic for advection
  integrator->dt = tm.dt;
  for (int stage = 1; stage <= integrator->nstages; stage++) {
    const Real g0 = integrator->gam0[stage - 1];
    const Real g1 = integrator->gam1[stage - 1];
    const Real bdt = integrator->beta[stage - 1] * integrator->dt;

    TaskRegion &tr = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
      auto &tl = tr[i];
      auto &u0 = pmesh->mesh_data.GetOrAdd("u0", i);
      auto &u1 = pmesh->mesh_data.GetOrAdd("u1", i);
      auto start_recv = tl.AddTask(none, parthenon::StartReceiveBoundBufs<any>, u0);
      auto start_flx_recv = tl.AddTask(none, parthenon::StartReceiveFluxCorrections, u0);
      auto calc_flx = tl.AddTask(none, CalculateFluxes, u0.get());
      auto send_flx = tl.AddTask(calc_flx, parthenon::LoadAndSendFluxCorrections, u0);
      auto recv_flx = tl.AddTask(none, parthenon::ReceiveFluxCorrections, u0);
      auto set_flx = tl.AddTask(calc_flx | recv_flx, parthenon::SetFluxCorrections, u0);
      auto update =
          tl.AddTask(calc_flx | set_flx, UpdateWithFluxDivergence<MeshData<Real>>,
                     u0.get(), u1.get(), g0, g1, bdt);
      auto bcs = parthenon::AddBoundaryExchangeTasks(update, tl, u0, pmesh->multilevel);
    }
  }

  // Compute post-advection timestep
  auto &dt_region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = dt_region[i];
    auto &base = pmesh->mesh_data.GetOrAdd("base", i);
    auto new_dt = tl.AddTask(none, EstimateTimestep<MeshData<Real>>, base.get());
  }

  // Operator split tracer push
  integrator->dt = tm.dt;
  TaskRegion &tracer_region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = tracer_region[i];
    auto &base = pmesh->mesh_data.GetOrAdd("base", i);
    auto reset = tl.AddTask(none, parthenon::ResetSwarmsCommunicationMesh, base);
    auto advect = tl.AddTask(reset, AdvectTracers, base.get(), integrator->dt);
    auto send = tl.AddTask(advect, parthenon::SendSwarmsMesh, base);
    auto receive = tl.AddTask(advect | send, parthenon::ReceiveSwarmsMesh, base);
    auto deposit = tl.AddTask(receive, DepositTracers, base.get());
    auto defrag = tl.AddTask(deposit, parthenon::DefragSwarmsMesh, base, 0.9);
  }

  return tc;
}

// *************************************************//
// Define the ProblemGenerator. Initializing the,  *//
// advected field.  Recall that initial particle   *//
// sourcing is handled in FinalInitialization      *//
// owned by the particles package.                 */
// *************************************************//

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  auto &mbd = pmb->meshblock_data.Get();
  auto &resolved_pkgs = pmb->resolved_packages;
  PARTHENON_REQUIRE(pmb->pmy_mesh->ndim <= 2,
                    "Tracer particles example only supports <= 2D!");

  // Advection package params
  auto &adv_pkg = pmb->packages.Get("advection_package");
  const Real &advected_mean = adv_pkg->Param<Real>("advected_mean");
  const Real &advected_amp = adv_pkg->Param<Real>("advected_amp");

  // Indexing
  const IndexRange &ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange &jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange &kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // Mesh physical size
  const auto mesh_size = pmb->pmy_mesh->mesh_size;
  const Real x_min_mesh = mesh_size.xmin(X1DIR);
  const Real x_max_mesh = mesh_size.xmax(X1DIR);
  const Real kwave = 2.0 * M_PI / (x_max_mesh - x_min_mesh);

  // Create pack
  static auto desc = parthenon::MakePackDescriptor<field::advected>(resolved_pkgs.get());
  auto pack = desc.GetPack(mbd.get());
  auto coords = pmb->coords;

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ProblemGenerator", parthenon::DevExecSpace(), kb.s, kb.e,
      jb.s, jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x1v = coords.Xc<1>(i);
        pack(0, field::advected(), k, j, i) =
            advected_mean + advected_amp * std::sin(kwave * x1v);
      });
}

} // namespace tracers_example
