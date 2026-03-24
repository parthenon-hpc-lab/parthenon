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

// This file was made in part with generative AI

#include "particle_tracers_amr_source_sink.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "amr_criteria/refinement_package.hpp"
#include "basic_types.hpp"
#include "bvals/comms/bvals_in_one.hpp"
#include "config.hpp"
#include "globals.hpp"
#include "interface/metadata.hpp"
#include "interface/update.hpp"
#include "kokkos_abstraction.hpp"
#include "pack/swarm_pack/make_swarm_pack_descriptor.hpp"
#include "pack/swarm_pack/swarm_default_names.hpp"
#include "utils/robust.hpp"

using namespace parthenon::driver::prelude;
using namespace parthenon::Update;

typedef Kokkos::Random_XorShift64_Pool<> RNGPool;
constexpr Real kParticlePlacementOmega = 1.0 - 4.0e8 * parthenon::robust::EPS();

namespace particle_tracers_amr_source_sink {

namespace {

struct DrivingGeometry {
  Real band1_x;
  Real band1_y;
  Real band2_x;
  Real band2_y;
  Real blob_x;
  Real blob_y;
  Real source_x;
  Real sink_y;
};

//----------------------------------------------------------------------------------------
//! Wrap a coordinate into the periodic domain interval.
KOKKOS_INLINE_FUNCTION
Real WrapPeriodic(const Real x, const Real xmin, const Real xmax) {
  const Real width = xmax - xmin;
  Real out = x;
  while (out < xmin)
    out += width;
  while (out >= xmax)
    out -= width;
  return out;
}

//! Return the shortest periodic distance between two coordinates.
KOKKOS_INLINE_FUNCTION
Real PeriodicDistance(const Real x, const Real x0, const Real xmin, const Real xmax) {
  const Real width = xmax - xmin;
  const Real dx = std::abs(x - x0);
  return std::min(dx, width - dx);
}

//! Return the signed shortest periodic displacement from x0 to x.
KOKKOS_INLINE_FUNCTION
Real PeriodicDisplacement(const Real x, const Real x0, const Real xmin, const Real xmax) {
  const Real width = xmax - xmin;
  Real dx = x - x0;
  while (dx > 0.5 * width)
    dx -= width;
  while (dx < -0.5 * width)
    dx += width;
  return dx;
}

//! Compute the overlap length between two 1D intervals.
KOKKOS_INLINE_FUNCTION
Real IntervalOverlap(const Real xmin_a, const Real xmax_a, const Real xmin_b,
                     const Real xmax_b) {
  return std::max(0.0, std::min(xmax_a, xmax_b) - std::max(xmin_a, xmin_b));
}

//----------------------------------------------------------------------------------------
//! Seed the one-time initial tracer packet across all blocks in a MeshData partition.
void SeedInitialTracerPacket(MeshData<Real> *md) {
  auto pkg = md->GetMeshPointer()->packages.Get("particles_package");
  const auto num_tracers = pkg->Param<int>("num_tracers");
  const auto packet_width_x = pkg->Param<Real>("packet_width_x");
  const auto packet_width_y = pkg->Param<Real>("packet_width_y");
  const auto placement_omega = pkg->Param<Real>("particle_placement_omega");
  auto rng_pool = pkg->Param<RNGPool>("rng_pool");

  const int ndim = md->GetMeshPointer()->ndim;
  PARTHENON_REQUIRE(ndim <= 2, "Tracer particles example only supports <= 2D!");

  const auto mesh_size = md->GetMeshPointer()->mesh_size;
  const Real x_min_mesh = mesh_size.xmin(X1DIR);
  const Real y_min_mesh = mesh_size.xmin(X2DIR);
  const Real x_max_mesh = mesh_size.xmax(X1DIR);
  const Real y_max_mesh = mesh_size.xmax(X2DIR);

  // The initial tracer packet is deliberately tall in y so particles repeatedly meet
  // different fine/coarse interfaces as the hostile AMR mask sweeps diagonally.
  const Real packet_xmin = x_min_mesh + 0.12 * (x_max_mesh - x_min_mesh);
  const Real packet_xmax = packet_xmin + packet_width_x * (x_max_mesh - x_min_mesh);
  const Real packet_yctr = 0.5 * (y_min_mesh + y_max_mesh);
  const Real packet_ymin = packet_yctr - 0.5 * packet_width_y * (y_max_mesh - y_min_mesh);
  const Real packet_ymax = packet_yctr + 0.5 * packet_width_y * (y_max_mesh - y_min_mesh);
  const Real packet_area =
      (packet_xmax - packet_xmin) * (ndim > 1 ? (packet_ymax - packet_ymin) : 1.0);

  const int nblocks = md->NumBlocks();
  ParArray1D<int> num_new_particles("initial tracer counts", nblocks);
  auto num_new_particles_h = Kokkos::create_mirror_view(num_new_particles);
  ParArray1D<Real> overlap_xmins("initial overlap xmins", nblocks);
  ParArray1D<Real> overlap_xmaxs("initial overlap xmaxs", nblocks);
  ParArray1D<Real> overlap_ymins("initial overlap ymins", nblocks);
  ParArray1D<Real> overlap_ys("initial overlap ys", nblocks);
  ParArray1D<Real> block_zmins("initial block zmins", nblocks);
  ParArray1D<Real> block_zspans("initial block zspans", nblocks);
  ParArray1D<std::uint64_t> id_bases("initial id bases", nblocks);
  auto overlap_xmins_h = Kokkos::create_mirror_view(overlap_xmins);
  auto overlap_xmaxs_h = Kokkos::create_mirror_view(overlap_xmaxs);
  auto overlap_ymins_h = Kokkos::create_mirror_view(overlap_ymins);
  auto overlap_ys_h = Kokkos::create_mirror_view(overlap_ys);
  auto block_zmins_h = Kokkos::create_mirror_view(block_zmins);
  auto block_zspans_h = Kokkos::create_mirror_view(block_zspans);
  auto id_bases_h = Kokkos::create_mirror_view(id_bases);

  for (int b = 0; b < nblocks; ++b) {
    auto pmb = md->GetBlockData(b)->GetBlockPointer();
    const auto block_size = md->GetMeshPointer()->GetBlockSize(pmb->loc);
    const Real x_min = block_size.xmin(X1DIR);
    const Real y_min = block_size.xmin(X2DIR);
    const Real z_min = block_size.xmin(X3DIR);
    const Real x_max = block_size.xmax(X1DIR);
    const Real y_max = block_size.xmax(X2DIR);
    const Real z_max = block_size.xmax(X3DIR);

    const Real overlap_x =
        std::max(0.0, std::min(x_max, packet_xmax) - std::max(x_min, packet_xmin));
    const Real overlap_y =
        ndim > 1
            ? std::max(0.0, std::min(y_max, packet_ymax) - std::max(y_min, packet_ymin))
            : 1.0;
    num_new_particles_h(b) =
        packet_area > 0.0 ? std::round(num_tracers * overlap_x * overlap_y / packet_area)
                          : 0;
    overlap_xmins_h(b) = std::max(x_min, packet_xmin);
    overlap_xmaxs_h(b) = std::min(x_max, packet_xmax);
    overlap_ymins_h(b) = ndim > 1 ? std::max(y_min, packet_ymin) : 0.0;
    overlap_ys_h(b) = overlap_y;
    block_zmins_h(b) = z_min;
    block_zspans_h(b) = z_max - z_min;
    id_bases_h(b) = static_cast<std::uint64_t>(pmb->gid) * 1000000ULL;
  }

  Kokkos::deep_copy(num_new_particles, num_new_particles_h);
  Kokkos::deep_copy(overlap_xmins, overlap_xmins_h);
  Kokkos::deep_copy(overlap_xmaxs, overlap_xmaxs_h);
  Kokkos::deep_copy(overlap_ymins, overlap_ymins_h);
  Kokkos::deep_copy(overlap_ys, overlap_ys_h);
  Kokkos::deep_copy(block_zmins, block_zmins_h);
  Kokkos::deep_copy(block_zspans, block_zspans_h);
  Kokkos::deep_copy(id_bases, id_bases_h);

  auto new_particles = parthenon::AddEmptyParticles(md, "tracers", num_new_particles);
  static auto pos_desc =
      MakeSwarmPackDescriptor<swarm_position::x, swarm_position::y, swarm_position::z>(
          "tracers");
  auto pos_pack = pos_desc.GetPack(md);
  static auto cohort_desc = MakeSwarmPackDescriptor<Real>("tracers", {"cohort"});
  static auto cohort_map = cohort_desc.GetMap();
  auto cohort_pack = cohort_desc.GetPack(md);
  const int cohort_idx = cohort_map["cohort"];
  static auto id_desc = MakeSwarmPackDescriptor<swarm_position::id>("tracers");
  auto id_pack = id_desc.GetPack(md);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), 0,
      new_particles.GetMaxFlatIndex(), KOKKOS_LAMBDA(const int idx) {
        auto [b, new_n] = new_particles.GetBlockParticleIndices(idx);
        const int n = new_particles.GetNewParticleIndex(b, new_n);
        auto rng_gen = rng_pool.get_state();

        pos_pack(b, swarm_position::x(), n) =
            overlap_xmins(b) +
            placement_omega * rng_gen.drand() * (overlap_xmaxs(b) - overlap_xmins(b));
        pos_pack(b, swarm_position::y(), n) =
            ndim > 1
                ? (overlap_ymins(b) + placement_omega * rng_gen.drand() * overlap_ys(b))
                : 0.0;
        pos_pack(b, swarm_position::z(), n) =
            block_zmins(b) + placement_omega * rng_gen.drand() * block_zspans(b);
        id_pack(b, swarm_position::id(), n) =
            id_bases(b) + static_cast<std::uint64_t>(new_n);
        cohort_pack(b, cohort_idx, n) = -1.0;

        rng_pool.free_state(rng_gen);
      });
}

//----------------------------------------------------------------------------------------
//! Compute the moving AMR/source/sink geometry for the current simulation time.
DrivingGeometry ComputeDrivingGeometry(const Real time, const Real tlim,
                                       const RegionSize &mesh_size) {
  const Real xmin = mesh_size.xmin(X1DIR);
  const Real xmax = mesh_size.xmax(X1DIR);
  const Real ymin = mesh_size.xmin(X2DIR);
  const Real ymax = mesh_size.xmax(X2DIR);
  const Real width = xmax - xmin;
  const Real height = ymax - ymin;
  const Real phase = time / std::max(tlim, TINY_NUMBER);

  DrivingGeometry geom;
  // Two oblique refinement bands sweep across the domain with different frequencies.
  // Their motion is intentionally unrelated to the particle velocity so particles are
  // repeatedly overtaken by refinement and left behind by derefinement.
  geom.band1_x = WrapPeriodic(xmin + 0.10 * width + 1.05 * phase * width, xmin, xmax);
  geom.band1_y = ymin + 0.52 * height + 0.24 * height * std::sin(2.0 * M_PI * phase);
  geom.band2_x = WrapPeriodic(xmin + 0.88 * width - 0.80 * phase * width, xmin, xmax);
  geom.band2_y = ymin + 0.48 * height + 0.28 * height * std::cos(3.0 * M_PI * phase);

  // A compact blob adds local refine/derefine churn where the band structure overlaps.
  geom.blob_x = WrapPeriodic(
      xmin + 0.50 * width + 0.18 * width * std::cos(4.0 * M_PI * phase), xmin, xmax);
  geom.blob_y = ymin + 0.50 * height + 0.18 * height * std::sin(4.0 * M_PI * phase);

  // Source and sink regions move independently of both the particle packet and the AMR
  // mask. That makes the source/sink mode a stricter lifecycle test than the fixed
  // population case because particles can be born and die while the mesh is changing for
  // unrelated reasons.
  geom.source_x = WrapPeriodic(xmin + 0.18 * width + 0.67 * phase * width, xmin, xmax);
  geom.sink_y = WrapPeriodic(ymin + 0.74 * height - 0.58 * phase * height, ymin, ymax);

  return geom;
}

//----------------------------------------------------------------------------------------
//! Measure normal distance to a periodic refinement band centerline.
Real DistanceToPeriodicBand(const Real x, const Real y, const Real x0, const Real y0,
                            const Real nx, const Real ny, const RegionSize &mesh_size) {
  const Real dx =
      PeriodicDisplacement(x, x0, mesh_size.xmin(X1DIR), mesh_size.xmax(X1DIR));
  const Real dy =
      PeriodicDisplacement(y, y0, mesh_size.xmin(X2DIR), mesh_size.xmax(X2DIR));
  return std::abs(dx * nx + dy * ny);
}

//! Update package parameters that describe the moving AMR/source/sink geometry.
void UpdateDrivingGeometry(const SimTime &tm, MeshData<Real> *md) {
  if (md->NumBlocks() == 0) return;
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("particles_package");
  // The tracer source/sink tasks run during the current step, but the state written after
  // that step is labeled with the post-step time/cycle. Use the step-end geometry here so
  // the source strip shown in outputs lines up with particles born during that step.
  const auto geom =
      ComputeDrivingGeometry(tm.time + tm.dt, tm.tlim, pmb->pmy_mesh->mesh_size);

  *pkg->MutableParam<Real>("band1_x") = geom.band1_x;
  *pkg->MutableParam<Real>("band1_y") = geom.band1_y;
  *pkg->MutableParam<Real>("band2_x") = geom.band2_x;
  *pkg->MutableParam<Real>("band2_y") = geom.band2_y;
  *pkg->MutableParam<Real>("blob_x") = geom.blob_x;
  *pkg->MutableParam<Real>("blob_y") = geom.blob_y;
  *pkg->MutableParam<Real>("source_x") = geom.source_x;
  *pkg->MutableParam<Real>("sink_y") = geom.sink_y;
  *pkg->MutableParam<int>("current_cycle") = tm.ncycle + 1;
}

} // namespace

//----------------------------------------------------------------------------------------
// Add multiple packages, one for the advected background and one for the tracer
// particles.
//! Register the background-field and particle packages for this problem.
Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;
  packages.Add(advection_package::Initialize(pin.get()));
  packages.Add(particles_package::Initialize(pin.get()));
  return packages;
}

// Create separate packages for background field and tracer particles

namespace advection_package {

//----------------------------------------------------------------------------------------
// Background field package

//! Estimate a stable mesh timestep for the advected background field.
Real EstimateTimestep(MeshData<Real> *md) {
  auto pkg = md->GetMeshPointer()->packages.Get("advection_package");
  const auto &cfl = pkg->Param<Real>("cfl");
  const auto &vx = pkg->Param<Real>("vx");
  const auto &vy = pkg->Param<Real>("vy");
  const auto &vz = pkg->Param<Real>("vz");

  auto desc = parthenon::MakePackDescriptor<advected>(md);
  auto pack = desc.GetPack(md);

  const IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  const IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  const IndexRange kb = md->GetBoundsK(IndexDomain::interior);

  Real min_dt = std::numeric_limits<Real>::max();
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, PARTHENON_AUTO_LABEL, DevExecSpace(), 0,
      pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lmin_dt) {
        const auto &coords = pack.GetCoordinates(b);
        lmin_dt = std::min(
            lmin_dt, parthenon::robust::ratio(coords.Dxc<X1DIR>(k, j, i), std::abs(vx)));
        lmin_dt = std::min(
            lmin_dt, parthenon::robust::ratio(coords.Dxc<X2DIR>(k, j, i), std::abs(vy)));
        lmin_dt = std::min(
            lmin_dt, parthenon::robust::ratio(coords.Dxc<X3DIR>(k, j, i), std::abs(vz)));
      },
      Kokkos::Min<Real>(min_dt));

  return cfl * min_dt;
}

//! Initialize the background-field package and enroll its variables/parameters.
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("advection_package");

  pkg->AddParam<>("vx", pin->GetOrAddReal("Background", "vx", 1.0));
  pkg->AddParam<>("vy", pin->GetOrAddReal("Background", "vy", 0.0));
  pkg->AddParam<>("vz", pin->GetOrAddReal("Background", "vz", 0.0));
  pkg->AddParam<>("cfl", pin->GetOrAddReal("Background", "cfl", 0.3));

  // Add advected field
  std::string field_name = "advected";
  Metadata mfield(
      {Metadata::Cell, Metadata::Independent, Metadata::FillGhost, Metadata::WithFluxes});
  pkg->AddField(field_name, mfield);

  // Add field in which to deposit tracer densities
  field_name = "tracer_deposition";
  pkg->AddField(field_name, mfield);

  pkg->EstimateTimestepMesh = EstimateTimestep;

  return pkg;
}

} // namespace advection_package

namespace particles_package {

//----------------------------------------------------------------------------------------
// Particle package

//! Initialize the particle package and enroll swarm state plus AMR controls.
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("particles_package");

  const auto stress_mode = pin->GetOrAddString("Tracers", "stress_mode", "fixed");
  const bool enable_source_sink = (stress_mode == "source_sink");
  PARTHENON_REQUIRE(stress_mode == "fixed" || stress_mode == "source_sink",
                    "Tracers/stress_mode must be either 'fixed' or 'source_sink'.");
  pkg->AddParam<>("stress_mode", stress_mode);
  pkg->AddParam<>("enable_source_sink", enable_source_sink);

  pkg->AddParam<>("num_tracers", pin->GetOrAddInteger("Tracers", "num_tracers", 100));
  pkg->AddParam<>("packet_width_x", pin->GetOrAddReal("Tracers", "packet_width_x", 0.10));
  pkg->AddParam<>("packet_width_y", pin->GetOrAddReal("Tracers", "packet_width_y", 0.70));
  pkg->AddParam<>("refine_band_halfwidth",
                  pin->GetOrAddReal("Tracers", "refine_band_halfwidth", 0.08));
  pkg->AddParam<>("derefine_band_halfwidth",
                  pin->GetOrAddReal("Tracers", "derefine_band_halfwidth", 0.14));
  pkg->AddParam<>("refine_blob_radius",
                  pin->GetOrAddReal("Tracers", "refine_blob_radius", 0.12));
  pkg->AddParam<>("derefine_blob_radius",
                  pin->GetOrAddReal("Tracers", "derefine_blob_radius", 0.20));
  pkg->AddParam<>("source_particles_per_cycle",
                  pin->GetOrAddInteger("Tracers", "source_particles_per_cycle", 0));
  pkg->AddParam<>("source_strip_width",
                  pin->GetOrAddReal("Tracers", "source_strip_width", 0.08));
  pkg->AddParam<>("sink_strip_height",
                  pin->GetOrAddReal("Tracers", "sink_strip_height", 0.10));
  pkg->AddParam<>(
      "particle_placement_omega",
      pin->GetOrAddReal("Tracers", "particle_placement_omega", kParticlePlacementOmega));
  const Real xmin = pin->GetReal("parthenon/mesh", "x1min");
  const Real xmax = pin->GetReal("parthenon/mesh", "x1max");
  const Real ymin = pin->GetReal("parthenon/mesh", "x2min");
  const Real ymax = pin->GetReal("parthenon/mesh", "x2max");
  const Real width = xmax - xmin;
  const Real height = ymax - ymin;
  pkg->AddParam<>("band1_x", xmin + 0.10 * width, true);
  pkg->AddParam<>("band1_y", ymin + 0.52 * height, true);
  pkg->AddParam<>("band2_x", xmin + 0.88 * width, true);
  pkg->AddParam<>("band2_y", ymin + 0.48 * height, true);
  pkg->AddParam<>("blob_x", xmin + 0.50 * width, true);
  pkg->AddParam<>("blob_y", ymin + 0.50 * height, true);
  pkg->AddParam<>("source_x", xmin + 0.18 * width, true);
  pkg->AddParam<>("sink_y", ymin + 0.74 * height, true);
  pkg->AddParam<>("current_cycle", 0, true);
  pkg->AddParam<>("rng_seed", pin->GetOrAddInteger("Tracers", "rng_seed", 314159));
  pkg->AddParam<>("source_id_cycle_stride", static_cast<std::uint64_t>(10000000000ULL));
  pkg->AddParam<>("source_id_gid_stride", static_cast<std::uint64_t>(100000ULL));
  RNGPool rng_pool(pkg->Param<int>("rng_seed"));
  pkg->AddParam<>("rng_pool", rng_pool);

  // Add swarm of tracer particles
  std::string swarm_name = "tracers";
  Metadata swarm_metadata({Metadata::Provides});
  pkg->AddSwarm(swarm_name, swarm_metadata);
  Metadata real_swarmvalue_metadata({Metadata::Real});
  pkg->AddSwarmValue("cohort", swarm_name, real_swarmvalue_metadata);

  pkg->CheckRefinementBlock = CheckRefinement;
  pkg->PreStepDiagnosticsMesh = UpdateDrivingGeometry;
  pkg->PostStepDiagnosticsMesh = UpdateDrivingGeometry;
  pkg->FinalInitializationMesh = FinalInitialization;

  return pkg;
}

//! Evaluate the prescribed moving AMR mask for the current block.
AmrTag CheckRefinement(MeshBlockData<Real> *mbd) {
  auto pmb = mbd->GetBlockPointer();
  auto pkg = pmb->packages.Get("particles_package");
  auto pmesh = pmb->pmy_mesh;
  const auto &mesh_size = pmesh->mesh_size;

  const Real xmin = mesh_size.xmin(X1DIR);
  const Real xmax = mesh_size.xmax(X1DIR);
  const Real ymin = mesh_size.xmin(X2DIR);
  const Real ymax = mesh_size.xmax(X2DIR);

  const Real xc = 0.5 * (pmb->block_size.xmin(X1DIR) + pmb->block_size.xmax(X1DIR));
  const Real yc = 0.5 * (pmb->block_size.xmin(X2DIR) + pmb->block_size.xmax(X2DIR));

  const Real refine_band_halfwidth = pkg->Param<Real>("refine_band_halfwidth");
  const Real derefine_band_halfwidth = pkg->Param<Real>("derefine_band_halfwidth");
  const Real refine_blob_radius = pkg->Param<Real>("refine_blob_radius");
  const Real derefine_blob_radius = pkg->Param<Real>("derefine_blob_radius");

  const Real band1_x = pkg->Param<Real>("band1_x");
  const Real band1_y = pkg->Param<Real>("band1_y");
  const Real band2_x = pkg->Param<Real>("band2_x");
  const Real band2_y = pkg->Param<Real>("band2_y");
  const Real blob_x = pkg->Param<Real>("blob_x");
  const Real blob_y = pkg->Param<Real>("blob_y");

  // The AMR driver deliberately uses geometry that is independent of particle motion.
  // The two bands and one blob create refinement from multiple directions so ownership
  // changes are forced even when particles simply coast ballistically.
  const Real band1_dist =
      DistanceToPeriodicBand(xc, yc, band1_x, band1_y, 0.83, 0.55, pmesh->mesh_size);
  const Real band2_dist =
      DistanceToPeriodicBand(xc, yc, band2_x, band2_y, -0.57, 0.82, pmesh->mesh_size);
  const Real blob_dx = PeriodicDistance(xc, blob_x, xmin, xmax);
  const Real blob_dy = pmesh->ndim > 1 ? PeriodicDistance(yc, blob_y, ymin, ymax) : 0.0;
  const Real blob_dist = std::sqrt(blob_dx * blob_dx + blob_dy * blob_dy);

  const bool refine = (band1_dist < refine_band_halfwidth) ||
                      (band2_dist < refine_band_halfwidth) ||
                      (blob_dist < refine_blob_radius);
  const bool derefine = (band1_dist > derefine_band_halfwidth) &&
                        (band2_dist > derefine_band_halfwidth) &&
                        (blob_dist > derefine_blob_radius);
  if (refine) return AmrTag::refine;
  if (derefine) return AmrTag::derefine;
  return AmrTag::same;
}

//----------------------------------------------------------------------------------------
//! Seed the initial tracer population once the startup AMR mesh has converged.
void FinalInitialization(Mesh *, ParameterInput *, MeshData<Real> *md) {
  SeedInitialTracerPacket(md);
}

} // namespace particles_package

//----------------------------------------------------------------------------------------
//! Remove particles that have entered the moving sink strip.
TaskStatus DestroySinkParticles(MeshData<Real> *md) {
  auto pkg = md->GetMeshPointer()->packages.Get("particles_package");
  if (!pkg->Param<bool>("enable_source_sink")) return TaskStatus::complete;

  const auto &mesh_size = md->GetMeshPointer()->mesh_size;
  const Real sink_y = pkg->Param<Real>("sink_y");
  const Real sink_half_height = 0.5 * pkg->Param<Real>("sink_strip_height") *
                                (mesh_size.xmax(X2DIR) - mesh_size.xmin(X2DIR));
  static auto desc = MakeSwarmPackDescriptor<swarm_position::y>("tracers");
  auto pack = desc.GetPack(md);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), 0,
      pack.GetMaxFlatIndex(), KOKKOS_LAMBDA(const int idx) {
        auto [b, n] = pack.GetBlockParticleIndices(idx);
        const auto &swarm_d = pack.GetContext(b);
        if (swarm_d.IsActive(n)) {
          const Real dy =
              PeriodicDisplacement(pack(b, swarm_position::y(), n), sink_y,
                                   mesh_size.xmin(X2DIR), mesh_size.xmax(X2DIR));
          if (std::abs(dy) <= sink_half_height) {
            swarm_d.MarkParticleForRemoval(n);
          }
        }
      });
  parthenon::RemoveMarkedParticles(md, "tracers");

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! Source new particles into the moving strip using mesh-wide coordinated allocation.
TaskStatus SourceStripParticles(MeshData<Real> *md) {
  auto pkg = md->GetMeshPointer()->packages.Get("particles_package");
  if (!pkg->Param<bool>("enable_source_sink")) return TaskStatus::complete;

  auto rng_pool = pkg->Param<RNGPool>("rng_pool");
  const auto source_particles_per_cycle = pkg->Param<int>("source_particles_per_cycle");
  if (source_particles_per_cycle <= 0) return TaskStatus::complete;

  const auto &mesh_size = md->GetMeshPointer()->mesh_size;
  const Real xmin_mesh = mesh_size.xmin(X1DIR);
  const Real xmax_mesh = mesh_size.xmax(X1DIR);
  const Real ymin_mesh = mesh_size.xmin(X2DIR);
  const Real ymax_mesh = mesh_size.xmax(X2DIR);
  const Real mesh_width = xmax_mesh - xmin_mesh;
  const Real mesh_height = ymax_mesh - ymin_mesh;

  const Real source_center = pkg->Param<Real>("source_x");
  const Real source_half_width =
      0.5 * pkg->Param<Real>("source_strip_width") * mesh_width;
  const Real source_area = 2.0 * source_half_width * mesh_height;
  const int current_cycle = pkg->Param<int>("current_cycle");
  const std::uint64_t id_cycle_stride =
      pkg->Param<std::uint64_t>("source_id_cycle_stride");
  const std::uint64_t id_gid_stride = pkg->Param<std::uint64_t>("source_id_gid_stride");
  const Real placement_omega = pkg->Param<Real>("particle_placement_omega");

  const int nblocks = md->NumBlocks();
  ParArray1D<int> num_new_particles("num new particles", nblocks);
  auto num_new_particles_h = Kokkos::create_mirror_view(num_new_particles);
  ParArray1D<Real> source_overlap_xmin("source overlap xmin", nblocks);
  ParArray1D<Real> source_overlap_xmax("source overlap xmax", nblocks);
  ParArray1D<Real> block_ymins("block ymins", nblocks);
  ParArray1D<Real> overlap_ys("overlap ys", nblocks);
  ParArray1D<Real> block_zmins("block zmins", nblocks);
  ParArray1D<Real> block_zspans("block z spans", nblocks);
  ParArray1D<std::uint64_t> id_bases("source id bases", nblocks);
  auto source_overlap_xmin_h = Kokkos::create_mirror_view(source_overlap_xmin);
  auto source_overlap_xmax_h = Kokkos::create_mirror_view(source_overlap_xmax);
  auto block_ymins_h = Kokkos::create_mirror_view(block_ymins);
  auto overlap_ys_h = Kokkos::create_mirror_view(overlap_ys);
  auto block_zmins_h = Kokkos::create_mirror_view(block_zmins);
  auto block_zspans_h = Kokkos::create_mirror_view(block_zspans);
  auto id_bases_h = Kokkos::create_mirror_view(id_bases);

  for (int b = 0; b < nblocks; ++b) {
    auto pmb = md->GetBlockData(b)->GetBlockPointer();
    auto swarm = md->GetSwarmData(b)->Get("tracers");
    const Real block_xmin = pmb->block_size.xmin(X1DIR);
    const Real block_xmax = pmb->block_size.xmax(X1DIR);
    const Real block_ymin = pmb->block_size.xmin(X2DIR);
    const Real block_ymax = pmb->block_size.xmax(X2DIR);
    const Real block_zmin = pmb->block_size.xmin(X3DIR);
    const Real block_zmax = pmb->block_size.xmax(X3DIR);

    Real overlap_x = 0.0;
    Real overlap_xmin = block_xmin;
    Real overlap_xmax = block_xmin;
    for (const int shift : {-1, 0, 1}) {
      const Real center = source_center + shift * mesh_width;
      const Real overlap_lo = std::max(block_xmin, center - source_half_width);
      const Real overlap_hi = std::min(block_xmax, center + source_half_width);
      const Real overlap = std::max(0.0, overlap_hi - overlap_lo);
      if (overlap > overlap_x) {
        overlap_x = overlap;
        overlap_xmin = overlap_lo;
        overlap_xmax = overlap_hi;
      }
    }
    const Real overlap_y = IntervalOverlap(block_ymin, block_ymax, ymin_mesh, ymax_mesh);
    const Real overlap_area = overlap_x * overlap_y;
    const int num_new_particles =
        source_area > 0.0 ? static_cast<int>(std::round(source_particles_per_cycle *
                                                        overlap_area / source_area))
                          : 0;

    source_overlap_xmin_h(b) = overlap_xmin;
    source_overlap_xmax_h(b) = overlap_xmax;
    block_ymins_h(b) = block_ymin;
    overlap_ys_h(b) = overlap_y;
    block_zmins_h(b) = block_zmin;
    block_zspans_h(b) = block_zmax - block_zmin;
    num_new_particles_h(b) = num_new_particles;
    id_bases_h(b) = static_cast<std::uint64_t>(current_cycle) * id_cycle_stride +
                    static_cast<std::uint64_t>(pmb->gid) * id_gid_stride;
  }

  Kokkos::deep_copy(num_new_particles, num_new_particles_h);
  Kokkos::deep_copy(source_overlap_xmin, source_overlap_xmin_h);
  Kokkos::deep_copy(source_overlap_xmax, source_overlap_xmax_h);
  Kokkos::deep_copy(block_ymins, block_ymins_h);
  Kokkos::deep_copy(overlap_ys, overlap_ys_h);
  Kokkos::deep_copy(block_zmins, block_zmins_h);
  Kokkos::deep_copy(block_zspans, block_zspans_h);
  Kokkos::deep_copy(id_bases, id_bases_h);
  auto new_particles = parthenon::AddEmptyParticles(md, "tracers", num_new_particles);

  static auto pos_desc =
      MakeSwarmPackDescriptor<swarm_position::x, swarm_position::y, swarm_position::z>(
          "tracers");
  auto pos_pack = pos_desc.GetPack(md);
  static auto cohort_desc = MakeSwarmPackDescriptor<Real>("tracers", {"cohort"});
  static auto cohort_map = cohort_desc.GetMap();
  auto cohort_pack = cohort_desc.GetPack(md);
  const int cohort_idx = cohort_map["cohort"];
  static auto id_desc = MakeSwarmPackDescriptor<swarm_position::id>("tracers");
  auto id_pack = id_desc.GetPack(md);

  // Each block only sources particles into the geometric intersection between the moving
  // source strip and that block. That keeps newly created particles on the block that
  // owns them until the regular swarm communication step has a chance to migrate them.
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), 0,
      new_particles.GetMaxFlatIndex(), KOKKOS_LAMBDA(const int idx) {
        auto [b, new_n] = new_particles.GetBlockParticleIndices(idx);
        const int n = new_particles.GetNewParticleIndex(b, new_n);
        auto rng_gen = rng_pool.get_state();
        pos_pack(b, swarm_position::x(), n) =
            source_overlap_xmin(b) +
            placement_omega * rng_gen.drand() *
                (source_overlap_xmax(b) - source_overlap_xmin(b));
        pos_pack(b, swarm_position::y(), n) =
            block_ymins(b) + placement_omega * rng_gen.drand() * overlap_ys(b);
        pos_pack(b, swarm_position::z(), n) =
            block_zmins(b) + placement_omega * rng_gen.drand() * block_zspans(b);
        id_pack(b, swarm_position::id(), n) =
            id_bases(b) + static_cast<std::uint64_t>(new_n);
        cohort_pack(b, cohort_idx, n) = static_cast<Real>(current_cycle);
        rng_pool.free_state(rng_gen);
      });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! Advect all active tracer particles with the prescribed constant velocity.
TaskStatus AdvectTracers(MeshData<Real> *md, const StagedIntegrator *integrator) {
  auto adv_pkg = md->GetMeshPointer()->packages.Get("advection_package");
  static auto desc =
      MakeSwarmPackDescriptor<swarm_position::x, swarm_position::y, swarm_position::z>(
          "tracers");
  auto pack = desc.GetPack(md);

  Real dt = integrator->dt;

  const auto &vx = adv_pkg->Param<Real>("vx");
  const auto &vy = adv_pkg->Param<Real>("vy");
  const auto &vz = adv_pkg->Param<Real>("vz");

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), 0,
      pack.GetMaxFlatIndex(), KOKKOS_LAMBDA(const int idx) {
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

//----------------------------------------------------------------------------------------
//! Deposit tracer counts back onto the mesh for visualization and diagnostics.
TaskStatus DepositTracers(MeshData<Real> *md) {
  static auto desc =
      MakeSwarmPackDescriptor<swarm_position::x, swarm_position::y, swarm_position::z>(
          "tracers");
  auto pack = desc.GetPack(md);
  auto dep_desc = parthenon::MakePackDescriptor<tracer_deposition>(md);
  auto dep_pack = dep_desc.GetPack(md);
  const IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  const IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  const IndexRange kb = md->GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), 0,
      dep_pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        dep_pack(b, tracer_deposition(), k, j, i) = 0.0;
      });

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), 0,
      pack.GetMaxFlatIndex(), KOKKOS_LAMBDA(const int idx) {
        auto [b, n] = pack.GetBlockParticleIndices(idx);
        const auto &swarm_d = pack.GetContext(b);
        if (swarm_d.IsActive(n)) {
          // Use the swarm runtime's cell-mapping convention rather than open-coding a
          // second floor-based rule here. That keeps deposition consistent with exact
          // face handling used by sorting/transport and avoids reintroducing upper-face
          // edge cases in this example.
          int i, j, k;
          swarm_d.Xtoijk(pack(b, swarm_position::x(), n), pack(b, swarm_position::y(), n),
                         pack(b, swarm_position::z(), n), i, j, k);
          Kokkos::atomic_add(&dep_pack(b, tracer_deposition(), k, j, i), 1.0);
        }
      });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! Compute first-order upwind fluxes across every block in a MeshData partition.
TaskStatus CalculateFluxes(MeshData<Real> *md) {
  PARTHENON_INSTRUMENT

  auto adv_pkg = md->GetMeshPointer()->packages.Get("advection_package");
  const auto &vx = adv_pkg->Param<Real>("vx");
  const auto &vy = adv_pkg->Param<Real>("vy");
  const auto &vz = adv_pkg->Param<Real>("vz");

  const auto ndim = md->GetMeshPointer()->ndim;
  const IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  const IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  const IndexRange kb = md->GetBoundsK(IndexDomain::interior);

  static auto advected_desc =
      parthenon::MakePackDescriptor<advected>(md, {}, {parthenon::PDOpt::WithFluxes});
  auto advected_pack = advected_desc.GetPack(md);

  // Spatially first order upwind method
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), 0,
      advected_pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e + 1,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        advected_pack.flux(b, X1DIR, advected(), k, j, i) =
            (vx > 0.0 ? advected_pack(b, advected(), k, j, i - 1)
                      : advected_pack(b, advected(), k, j, i)) *
            vx;
      });

  if (ndim > 1) {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), 0,
        advected_pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e + 1, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          advected_pack.flux(b, X2DIR, advected(), k, j, i) =
              (vy > 0.0 ? advected_pack(b, advected(), k, j - 1, i)
                        : advected_pack(b, advected(), k, j, i)) *
              vy;
        });
  }

  if (ndim > 2) {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), 0,
        advected_pack.GetNBlocks() - 1, kb.s, kb.e + 1, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          advected_pack.flux(b, X3DIR, advected(), k, j, i) =
              (vz > 0.0 ? advected_pack(b, advected(), k - 1, j, i)
                        : advected_pack(b, advected(), k, j, i)) *
              vz;
        });
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! Initialize the background advected field on a mesh block.
void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  auto &mbd = pmb->meshblock_data.Get();
  static auto advected_desc = parthenon::MakePackDescriptor<advected>(mbd.get());
  auto advected_pack = advected_desc.GetPack(mbd.get());

  const int ndim = pmb->pmy_mesh->ndim;
  PARTHENON_REQUIRE(ndim <= 2, "Tracer particles example only supports <= 2D!");

  const IndexRange &ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange &jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange &kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  auto coords = pmb->coords;

  const Real advected_mean = 1.0;
  const Real advected_amp = 0.5;
  PARTHENON_REQUIRE(advected_mean > advected_amp, "Cannot have negative densities!");

  const auto mesh_size = pmb->pmy_mesh->mesh_size;
  const Real x_min_mesh = mesh_size.xmin(X1DIR);
  const Real x_max_mesh = mesh_size.xmax(X1DIR);

  const Real kwave = 2. * M_PI / (x_max_mesh - x_min_mesh);

  pmb->par_for(
      PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        advected_pack(0, advected(), k, j, i) =
            advected_mean + advected_amp * std::sin(kwave * coords.Xc<1>(i));
      });
}

//! Build the coupled field and particle task graph for one integrator stage.
TaskCollection ParticleDriver::MakeTaskCollection(BlockList_t &blocks, int stage) {
  TaskCollection tc;
  TaskID none(0);
  const auto any = parthenon::BoundaryType::any;

  const Real beta = integrator->beta[stage - 1];
  const Real dt = integrator->dt;
  const auto &stage_name = integrator->stage_name;
  const int nstages = integrator->nstages;

  auto partitions = pmesh->GetDefaultBlockPartitions();
  const int num_partitions = partitions.size();
  // note that task within this region that contains one tasklist per pack
  // could still be executed in parallel
  TaskRegion &tr = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = tr[i];
    auto &mbase = pmesh->mesh_data.Add("base", partitions[i]);
    auto &mc0 = pmesh->mesh_data.Add(stage_name[stage - 1], mbase);
    auto &mc1 = pmesh->mesh_data.Add(stage_name[stage], mbase);
    auto &mdudt = pmesh->mesh_data.Add("dUdt", mbase);

    auto start_recv = tl.AddTask(none, parthenon::StartReceiveBoundBufs<any>, mc1);
    auto start_flx_recv = tl.AddTask(none, parthenon::StartReceiveFluxCorrections, mc0);
    auto calc_flx = tl.AddTask(none,
                               static_cast<TaskStatus (*)(MeshData<Real> *)>(
                                   particle_tracers_amr_source_sink::CalculateFluxes),
                               mc0.get());

    auto send_flx = tl.AddTask(calc_flx, parthenon::LoadAndSendFluxCorrections, mc0);
    auto recv_flx = tl.AddTask(none, parthenon::ReceiveFluxCorrections, mc0);
    auto set_flx = tl.AddTask(calc_flx | recv_flx, parthenon::SetFluxCorrections, mc0);

    // compute the divergence of fluxes of conserved variables
    auto flux_div =
        tl.AddTask(set_flx, FluxDivergence<MeshData<Real>>, mc0.get(), mdudt.get());

    auto avg_data = tl.AddTask(flux_div, AverageIndependentData<MeshData<Real>>,
                               mc0.get(), mbase.get(), beta);
    // apply du/dt to all independent fields in the container
    auto update = tl.AddTask(avg_data, UpdateIndependentData<MeshData<Real>>, mc0.get(),
                             mdudt.get(), beta * dt, mc1.get());

    // do boundary exchange
    auto set_bc = parthenon::AddBoundaryExchangeTasks(update, tl, mc1, pmesh->multilevel);
  }

  // First-order operator split tracer particle update
  if (stage == integrator->nstages) {
    TaskRegion &pre_particle_region = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; ++i) {
      auto &tl = pre_particle_region[i];
      auto &mc1 = pmesh->mesh_data.Add(stage_name[stage], partitions[i]);
      auto new_dt = tl.AddTask(none, parthenon::Update::EstimateTimestep<MeshData<Real>>,
                               mc1.get());
      if (pmesh->adaptive) {
        auto &base = pmesh->mesh_data.Add("base", partitions[i]);
        auto refine =
            tl.AddTask(new_dt, parthenon::Refinement::Tag<MeshData<Real>>, base.get());
      }
    }

    TaskRegion &tracer_particles = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; ++i) {
      auto &tl = tracer_particles[i];
      auto &base = pmesh->mesh_data.Add("base", partitions[i]);
      auto reset = tl.AddTask(none, parthenon::ResetSwarmCommunication, base);
      auto advect = tl.AddTask(reset, particle_tracers_amr_source_sink::AdvectTracers,
                               base.get(), integrator.get());
      auto send =
          tl.AddTask(advect, parthenon::SendSwarms, base, BoundaryCommSubset::all);
      auto receive =
          tl.AddTask(send, parthenon::ReceiveSwarms, base, BoundaryCommSubset::all);
      auto sink = tl.AddTask(
          receive, particle_tracers_amr_source_sink::DestroySinkParticles, base.get());
      auto source = tl.AddTask(
          sink, particle_tracers_amr_source_sink::SourceStripParticles, base.get());
      auto deposit = tl.AddTask(source, particle_tracers_amr_source_sink::DepositTracers,
                                base.get());
      // Defragment if swarm memory pool occupancy is 90%
      auto defrag = tl.AddTask(deposit, parthenon::DefragSwarms, base, 0.9);
    }
  }

  return tc;
}

} // namespace particle_tracers_amr_source_sink
