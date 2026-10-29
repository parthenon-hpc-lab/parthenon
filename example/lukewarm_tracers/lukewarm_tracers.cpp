//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2026 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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

// This file was made in part with generative AI.

// The tracers of example/particle_tracers, carried by lukewarms instead of swarms.
// Lukewarm particles are owned by a MeshData rather than by its blocks, so a tracer that
// crosses from one block into another of the same MeshData simply keeps going, and only
// tracers that leave their MeshData are exchanged.

// C++ includes
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

// Parthenon includes
#include "amr_criteria/refinement_package.hpp"
#include "basic_types.hpp"
#include "bvals/comms/bvals_in_one.hpp"
#include "globals.hpp"
#include "interface/metadata.hpp"
#include "interface/update.hpp"
#include "kokkos_abstraction.hpp"
#include "lukewarm/lukewarm_pack.hpp"
#include "lukewarm/lukewarm_relocate.hpp"
#include "outputs/outputs.hpp"
#include "pack/default_names.hpp"

#include "lukewarm_tracers.hpp"

using namespace parthenon::driver::prelude;
using namespace parthenon::Update;

typedef Kokkos::Random_XorShift64_Pool<> RNGPool;

namespace lukewarm_tracers {

// ****************************************************//
// Define variables/names to be used by this example. *//
// ****************************************************//

static const char lukewarm_name[] = "tracers";

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
    dx[0] = std::min(dx[0], pmb->coords.Dxc<X1DIR>(0));
    dx[1] = std::min(dx[1], pmb->coords.Dxc<X2DIR>(0));
    dx[2] = std::min(dx[2], pmb->coords.Dxc<X3DIR>(0));
  }

  Real min_dt = dx[0] / std::abs(vx + TINY_NUMBER);
  min_dt = std::min(min_dt, static_cast<Real>(dx[1] / std::abs(vy + TINY_NUMBER)));
  min_dt = std::min(min_dt, static_cast<Real>(dx[2] / std::abs(vz + TINY_NUMBER)));

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

  const auto &vx = adv_pkg->Param<Real>("vx");
  const auto &vy = adv_pkg->Param<Real>("vy");
  const auto &vz = adv_pkg->Param<Real>("vz");

  // Assumes a grid with constant dx, dy, dz within a block
  std::array<Real, 3> dx;
  dx.fill(std::numeric_limits<Real>::max());
  for (auto &pmb : pm->block_list) {
    dx[0] = std::min(dx[0], pmb->coords.Dxc<X1DIR>(0));
    dx[1] = std::min(dx[1], pmb->coords.Dxc<X2DIR>(0));
    dx[2] = std::min(dx[2], pmb->coords.Dxc<X3DIR>(0));
  }

  Real min_dt = dx[0] / std::abs(vx + TINY_NUMBER);
  min_dt = std::min(min_dt, static_cast<Real>(dx[1] / std::abs(vy + TINY_NUMBER)));
  min_dt = std::min(min_dt, static_cast<Real>(dx[2] / std::abs(vz + TINY_NUMBER)));

  // No CFL number for particles
  return min_dt;
}

// History outputs get the MeshData of every block on this rank. Unless that is also
// the only partition, it does not own lukewarms, so count the tracers of the MeshData
// that do.
Real NumTracers(MeshData<Real> *md) {
  auto pm = md->GetParentPointer();
  int num_tracers = 0;
  for (auto &partition : pm->GetDefaultBlockPartitions()) {
    auto &base = pm->mesh_data.Add("base", partition);
    num_tracers += base->GetLukewarm(lukewarm_name).NumParticles();
  }
  return num_tracers;
}

// Every tracer deposits one, so this matches NumTracers() once tracers have been
// deposited
Real TotalDeposition(MeshData<Real> *md) {
  auto pm = md->GetParentPointer();
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);

  static auto desc = MakePackDescriptor<field::deposition>(pm->resolved_packages.get());
  auto pack = desc.GetPack(md);

  Real total = 0.0;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "TotalDeposition", DevExecSpace(), 0,
      pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &sum) {
        sum += pack(b, field::deposition(), k, j, i);
      },
      Kokkos::Sum<Real>(total));
  return total;
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto tr_pkg = std::make_shared<StateDescriptor>("particles_package");

  tr_pkg->AddParam("num_tracers", pin->GetOrAddInteger("Tracers", "num_tracers", 100));

  // Every lukewarm carries positions and persistent ids
  Metadata m({Metadata::Provides});
  tr_pkg->AddLukewarm(lukewarm_name, m);

  // Assign package timestep hook
  tr_pkg->EstimateTimestepMesh = EstimateTimestepMesh;

  // Tracers are sourced once the initialization hierarchy has converged, on each
  // MeshData that owns them
  tr_pkg->PostInitializationMesh = SourceTracers;

  parthenon::HstVar_list hst_vars = {};
  hst_vars.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                                    NumTracers, "num_tracers"));
  hst_vars.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                                    TotalDeposition, "total_deposition"));
  tr_pkg->AddParam<>(parthenon::hist_param_key, hst_vars);

  return tr_pkg;
}

void SourceTracers(Mesh *pm, ParameterInput *pin, MeshData<Real> *md) {
  auto &adv_pkg = pm->packages.Get("advection_package");
  auto &tr_pkg = pm->packages.Get("particles_package");

  // Advection package params
  const Real &advected_mean = adv_pkg->Param<Real>("advected_mean");
  const Real &advected_amp = adv_pkg->Param<Real>("advected_amp");

  // Tracer package params and lukewarm
  auto &tracers = md->GetLukewarm(lukewarm_name);
  const auto num_tracers = tr_pkg->Param<int>("num_tracers");

  // Mesh physical size
  const auto mesh_size = pm->mesh_size;
  const Real x_min_mesh = mesh_size.xmin(X1DIR);
  const Real y_min_mesh = mesh_size.xmin(X2DIR);
  const Real z_min_mesh = mesh_size.xmin(X3DIR);
  const Real x_max_mesh = mesh_size.xmax(X1DIR);
  const Real y_max_mesh = mesh_size.xmax(X2DIR);
  const Real z_max_mesh = mesh_size.xmax(X3DIR);
  const Real kwave = 2. * M_PI / (x_max_mesh - x_min_mesh);
  const Real nmesh = (advected_mean * (x_max_mesh - x_min_mesh) -
                      advected_amp / kwave *
                          (std::cos(kwave * x_max_mesh) - std::cos(kwave * x_min_mesh))) *
                     (y_max_mesh - y_min_mesh) * (z_max_mesh - z_min_mesh);

  const int nblocks = md->NumBlocks();
  const auto geometry = md->GetLukewarmGeometry();

  // Calculate frac of total tracers on each MeshBlock by integrating the advected
  // profile over the Mesh and the MeshBlock. Tracer number follows number =
  // advected*volume.
  ParArray1DRaw<int> counts("num_tracers", nblocks);
  parthenon::par_for(
      PARTHENON_AUTO_LABEL, 0, nblocks - 1, KOKKOS_LAMBDA(const int b) {
        const Real x_min = geometry.Xmin(b, X1DIR);
        const Real x_max = geometry.Xmax(b, X1DIR);
        const Real nmeshblock =
            (advected_mean * (x_max - x_min) -
             advected_amp / kwave * (std::cos(kwave * x_max) - std::cos(kwave * x_min))) *
            (geometry.Xmax(b, X2DIR) - geometry.Xmin(b, X2DIR)) *
            (geometry.Xmax(b, X3DIR) - geometry.Xmin(b, X3DIR));
        counts(b) = Kokkos::round(num_tracers * nmeshblock / nmesh);
      });

  // Add these particles, with block b's at [offsets(b), offsets(b + 1)). A pack is a
  // snapshot, so get it after creating them.
  const auto offsets = tracers.Create(counts);
  static auto desc = MakeLukewarmPackDescriptor<lukewarm::x, lukewarm::y, lukewarm::z>(
      md, lukewarm_name);
  auto pack = desc.GetPack(md);

  // Seeded by the id of this MeshData, so that each MeshData draws its own tracers
  auto rng_pool = RNGPool(md->GetLukewarmContainer().GetId());

  // Initialize the tracers of each block
  parthenon::par_for_outer(
      PARTHENON_AUTO_LABEL, 0, 0, 0, nblocks - 1,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b) {
        const Real x_min = geometry.Xmin(b, X1DIR);
        const Real y_min = geometry.Xmin(b, X2DIR);
        const Real z_min = geometry.Xmin(b, X3DIR);
        const Real x_max = geometry.Xmax(b, X1DIR);
        const Real y_max = geometry.Xmax(b, X2DIR);
        const Real z_max = geometry.Xmax(b, X3DIR);

        parthenon::par_for_inner(
            member, offsets(b), offsets(b + 1) - 1, [&](const int n) {
              auto rng_gen = rng_pool.get_state();

              // Extract particle position
              Real &xx = pack(lukewarm::x(), n);
              Real &yy = pack(lukewarm::y(), n);
              Real &zz = pack(lukewarm::z(), n);

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
  static auto desc = MakeLukewarmPackDescriptor<lukewarm::x, lukewarm::y, lukewarm::z>(
      md, lukewarm_name);
  auto pack = desc.GetPack(md);

  // Advect tracers. Storage is dense, so every particle is active.
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "AdvectTracers", DevExecSpace(), 0, pack.GetNParticles() - 1,
      KOKKOS_LAMBDA(const int n) {
        pack(lukewarm::x(), n) += vx * dt;
        pack(lukewarm::y(), n) += vy * dt;
        pack(lukewarm::z(), n) += vz * dt;
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
  static auto pdesc = MakeLukewarmPackDescriptor<lukewarm::x, lukewarm::y, lukewarm::z>(
      md, lukewarm_name);
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
      DEFAULT_LOOP_PATTERN, "DepositTracers", DevExecSpace(), 0,
      vpart.GetNParticles() - 1, KOKKOS_LAMBDA(const int n) {
        // For testing in this example we make sure every tracer was delivered to this
        // MeshData; GetBlockIndex(n) would skip the check
        const auto loc = vpart.Locate(n);
        if (loc.InMeshData()) {
          const int b = loc.Block();
          const auto [kp, jp, ip] = vpart.GetCellIndices(b, n);
          Kokkos::atomic_add(&vmesh(b, field::deposition(), kp, jp, ip), 1.0);
        } else {
          PARTHENON_FAIL("Tracer outside of its MeshData during deposition.");
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

  // Operator split tracer push. Tracers that leave their MeshData are exchanged with its
  // neighbors, on this rank or another, as fields exchange their boundaries.
  integrator->dt = tm.dt;
  TaskRegion &tracer_region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = tracer_region[i];
    auto &base = pmesh->mesh_data.GetOrAdd("base", i);
    auto advect = tl.AddTask(none, AdvectTracers, base.get(), integrator->dt);
    auto exchange =
        parthenon::AddLukewarmExchangeTasks(advect, tl, base.get(), lukewarm_name);
    auto deposit = tl.AddTask(exchange, DepositTracers, base.get());
  }

  if (pmesh->adaptive) {
    auto &amr_region = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; ++i) {
      auto &tl = amr_region[i];
      auto &base = pmesh->mesh_data.GetOrAdd("base", i);
      tl.AddTask(none, parthenon::Refinement::Tag<MeshData<Real>>, base.get());
    }
  }

  return tc;
}

// *************************************************//
// Define the ProblemGenerator, initializing the   *//
// advected field of each MeshData. Tracers are    *//
// sourced later, in PostInitializationMesh.       *//
// *************************************************//

void ProblemGenerator(Mesh *pm, ParameterInput *pin, MeshData<Real> *md) {
  auto &resolved_pkgs = pm->resolved_packages;
  PARTHENON_REQUIRE(pm->ndim <= 2, "Tracer particles example only supports <= 2D!");

  // Advection package params
  auto &adv_pkg = pm->packages.Get("advection_package");
  const Real &advected_mean = adv_pkg->Param<Real>("advected_mean");
  const Real &advected_amp = adv_pkg->Param<Real>("advected_amp");

  // Indexing
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);

  // Mesh physical size
  const auto mesh_size = pm->mesh_size;
  const Real x_min_mesh = mesh_size.xmin(X1DIR);
  const Real x_max_mesh = mesh_size.xmax(X1DIR);
  const Real kwave = 2.0 * M_PI / (x_max_mesh - x_min_mesh);

  // Create pack
  static auto desc = parthenon::MakePackDescriptor<field::advected>(resolved_pkgs.get());
  auto pack = desc.GetPack(md);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ProblemGenerator", parthenon::DevExecSpace(), 0,
      pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const Real x1v = pack.GetCoordinates(b).Xc<1>(i);
        pack(b, field::advected(), k, j, i) =
            advected_mean + advected_amp * std::sin(kwave * x1v);
      });
}

} // namespace lukewarm_tracers
