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

KOKKOS_INLINE_FUNCTION
Real PeriodicDistance(const Real x, const Real x0, const Real xmin, const Real xmax) {
  const Real width = xmax - xmin;
  const Real dx = std::abs(x - x0);
  return std::min(dx, width - dx);
}

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

KOKKOS_INLINE_FUNCTION
Real IntervalOverlap(const Real xmin_a, const Real xmax_a, const Real xmin_b,
                     const Real xmax_b) {
  return std::max(0.0, std::min(xmax_a, xmax_b) - std::max(xmin_a, xmin_b));
}

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

Real DistanceToPeriodicBand(const Real x, const Real y, const Real x0, const Real y0,
                            const Real nx, const Real ny, const RegionSize &mesh_size) {
  const Real dx =
      PeriodicDisplacement(x, x0, mesh_size.xmin(X1DIR), mesh_size.xmax(X1DIR));
  const Real dy =
      PeriodicDisplacement(y, y0, mesh_size.xmin(X2DIR), mesh_size.xmax(X2DIR));
  return std::abs(dx * nx + dy * ny);
}

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

// Add multiple packages, one for the advected background and one for the tracer
// particles.
Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;
  packages.Add(advection_package::Initialize(pin.get()));
  packages.Add(particles_package::Initialize(pin.get()));
  return packages;
}

// Create separate packages for background field and tracer particles

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

  Real vx = pin->GetOrAddReal("Background", "vx", 1.0);
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

  // Add field in which to deposit tracer densities
  field_name = "tracer_deposition";
  pkg->AddField(field_name, mfield);

  pkg->EstimateTimestepBlock = EstimateTimestepBlock;

  return pkg;
}

} // namespace advection_package

namespace particles_package {

// *************************************************//
// define the tracer particles package, including  *//
// initialization and update functions.            *//
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

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("particles_package");

  const auto stress_mode = pin->GetOrAddString("Tracers", "stress_mode", "fixed");
  const bool enable_source_sink = (stress_mode == "source_sink");
  PARTHENON_REQUIRE(stress_mode == "fixed" || stress_mode == "source_sink",
                    "Tracers/stress_mode must be either 'fixed' or 'source_sink'.");
  pkg->AddParam<>("stress_mode", stress_mode);
  pkg->AddParam<>("enable_source_sink", enable_source_sink);

  int num_tracers = pin->GetOrAddInteger("Tracers", "num_tracers", 100);
  pkg->AddParam<>("num_tracers", num_tracers);
  const Real packet_width_x = pin->GetOrAddReal("Tracers", "packet_width_x", 0.10);
  pkg->AddParam<>("packet_width_x", packet_width_x);
  const Real packet_width_y = pin->GetOrAddReal("Tracers", "packet_width_y", 0.70);
  pkg->AddParam<>("packet_width_y", packet_width_y);
  const Real refine_band_halfwidth =
      pin->GetOrAddReal("Tracers", "refine_band_halfwidth", 0.08);
  pkg->AddParam<>("refine_band_halfwidth", refine_band_halfwidth);
  const Real derefine_band_halfwidth =
      pin->GetOrAddReal("Tracers", "derefine_band_halfwidth", 0.14);
  pkg->AddParam<>("derefine_band_halfwidth", derefine_band_halfwidth);
  const Real refine_blob_radius =
      pin->GetOrAddReal("Tracers", "refine_blob_radius", 0.12);
  pkg->AddParam<>("refine_blob_radius", refine_blob_radius);
  const Real derefine_blob_radius =
      pin->GetOrAddReal("Tracers", "derefine_blob_radius", 0.20);
  pkg->AddParam<>("derefine_blob_radius", derefine_blob_radius);
  const int source_particles_per_cycle =
      pin->GetOrAddInteger("Tracers", "source_particles_per_cycle", 0);
  pkg->AddParam<>("source_particles_per_cycle", source_particles_per_cycle);
  const Real source_strip_width =
      pin->GetOrAddReal("Tracers", "source_strip_width", 0.08);
  pkg->AddParam<>("source_strip_width", source_strip_width);
  const Real sink_strip_height = pin->GetOrAddReal("Tracers", "sink_strip_height", 0.10);
  pkg->AddParam<>("sink_strip_height", sink_strip_height);
  const Real placement_omega =
      pin->GetOrAddReal("Tracers", "particle_placement_omega", kParticlePlacementOmega);
  pkg->AddParam<>("particle_placement_omega", placement_omega);
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
  pkg->EstimateTimestepBlock = EstimateTimestepBlock;
  pkg->PreStepDiagnosticsMesh = UpdateDrivingGeometry;
  pkg->PostStepDiagnosticsMesh = UpdateDrivingGeometry;

  return pkg;
}

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

} // namespace particles_package

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
  for (int b = 0; b < md->NumBlocks(); ++b) {
    md->GetSwarmData(b)->Get("tracers")->RemoveMarkedParticles();
  }

  return TaskStatus::complete;
}

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
  ParArray1D<NewParticlesContext> new_contexts("new particle contexts", nblocks);
  auto new_contexts_h = Kokkos::create_mirror_view(new_contexts);
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
    id_bases_h(b) = static_cast<std::uint64_t>(current_cycle) * id_cycle_stride +
                    static_cast<std::uint64_t>(pmb->gid) * id_gid_stride;
    new_contexts_h(b) = swarm->AddEmptyParticles(num_new_particles);
  }

  Kokkos::deep_copy(new_contexts, new_contexts_h);
  Kokkos::deep_copy(source_overlap_xmin, source_overlap_xmin_h);
  Kokkos::deep_copy(source_overlap_xmax, source_overlap_xmax_h);
  Kokkos::deep_copy(block_ymins, block_ymins_h);
  Kokkos::deep_copy(overlap_ys, overlap_ys_h);
  Kokkos::deep_copy(block_zmins, block_zmins_h);
  Kokkos::deep_copy(block_zspans, block_zspans_h);
  Kokkos::deep_copy(id_bases, id_bases_h);

  static const std::vector<std::string> real_vars{swarm_position::x::name(),
                                                  swarm_position::y::name(),
                                                  swarm_position::z::name(), "cohort"};
  static auto real_desc = MakeSwarmPackDescriptor<Real>("tracers", real_vars);
  static auto real_map = real_desc.GetMap();
  auto real_pack = real_desc.GetPack(md);
  const int x_idx = real_map[swarm_position::x::name()];
  const int y_idx = real_map[swarm_position::y::name()];
  const int z_idx = real_map[swarm_position::z::name()];
  const int cohort_idx = real_map["cohort"];
  static auto id_desc = MakeSwarmPackDescriptor<swarm_position::id>("tracers");
  auto id_pack = id_desc.GetPack(md);

  // Each block only sources particles into the geometric intersection between the moving
  // source strip and that block. That keeps newly created particles on the block that
  // owns them until the regular swarm communication step has a chance to migrate them.
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), 0, nblocks - 1,
      KOKKOS_LAMBDA(const int b) {
        for (int new_n = 0; new_n <= new_contexts(b).GetNewParticlesMaxIndex(); ++new_n) {
          const int n = new_contexts(b).GetNewParticleIndex(new_n);
          auto rng_gen = rng_pool.get_state();
          real_pack(b, x_idx, n) = source_overlap_xmin(b) +
                                   placement_omega * rng_gen.drand() *
                                       (source_overlap_xmax(b) - source_overlap_xmin(b));
          real_pack(b, y_idx, n) =
              block_ymins(b) + placement_omega * rng_gen.drand() * overlap_ys(b);
          real_pack(b, z_idx, n) =
              block_zmins(b) + placement_omega * rng_gen.drand() * block_zspans(b);
          id_pack(b, swarm_position::id(), n) =
              id_bases(b) + static_cast<std::uint64_t>(new_n);
          real_pack(b, cohort_idx, n) = static_cast<Real>(current_cycle);
          rng_pool.free_state(rng_gen);
        }
      });

  return TaskStatus::complete;
}

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

TaskStatus DepositTracers(MeshData<Real> *md) {
  static auto desc =
      MakeSwarmPackDescriptor<swarm_position::x, swarm_position::y, swarm_position::z>(
          "tracers");
  auto pack = desc.GetPack(md);
  auto dep_desc =
      parthenon::MakePackDescriptor(md, std::vector<std::string>{"tracer_deposition"});
  auto dep_pack = dep_desc.GetPack(md);
  auto dep_map = dep_desc.GetMap();
  const PackIdx dep_idx(dep_map["tracer_deposition"]);

  for (int b = 0; b < md->NumBlocks(); ++b) {
    auto pmb = md->GetBlockData(b)->GetBlockPointer();
    const IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    const IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    const IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
    pmb->par_for(
        PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          dep_pack(b, dep_idx, k, j, i) = 0.0;
        });
  }

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
          Kokkos::atomic_add(&dep_pack(b, dep_idx, k, j, i), 1.0);
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
  auto &tr_pkg = pmb->packages.Get("particles_package");
  auto &mbd = pmb->meshblock_data.Get();
  auto &advected = mbd->Get("advected").data;
  auto &swarm = pmb->meshblock_data.Get()->GetSwarmData()->Get("tracers");
  const auto num_tracers = tr_pkg->Param<int>("num_tracers");
  const auto packet_width_x = tr_pkg->Param<Real>("packet_width_x");
  const auto packet_width_y = tr_pkg->Param<Real>("packet_width_y");
  auto rng_pool =
      RNGPool(pmb->gid); // Seed is meshblock gid for consistency across MPI decomposition

  const int ndim = pmb->pmy_mesh->ndim;
  PARTHENON_REQUIRE(ndim <= 2, "Tracer particles example only supports <= 2D!");

  const IndexRange &ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange &jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange &kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  auto coords = pmb->coords;

  const Real advected_mean = 1.0;
  const Real advected_amp = 0.5;
  PARTHENON_REQUIRE(advected_mean > advected_amp, "Cannot have negative densities!");

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

  pmb->par_for(
      PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        advected(k, j, i) = advected_mean + advected_amp * sin(kwave * coords.Xc<1>(i));
      });

  // The initial tracer packet is deliberately tall in y so particles repeatedly meet
  // different fine/coarse interfaces as the hostile AMR mask sweeps diagonally.
  const Real packet_xmin = x_min_mesh + 0.12 * (x_max_mesh - x_min_mesh);
  const Real packet_xmax = packet_xmin + packet_width_x * (x_max_mesh - x_min_mesh);
  const Real packet_yctr = 0.5 * (y_min_mesh + y_max_mesh);
  const Real packet_ymin = packet_yctr - 0.5 * packet_width_y * (y_max_mesh - y_min_mesh);
  const Real packet_ymax = packet_yctr + 0.5 * packet_width_y * (y_max_mesh - y_min_mesh);

  const Real overlap_x =
      std::max(0.0, std::min(x_max, packet_xmax) - std::max(x_min, packet_xmin));
  const Real overlap_y =
      ndim > 1
          ? std::max(0.0, std::min(y_max, packet_ymax) - std::max(y_min, packet_ymin))
          : 1.0;
  const Real packet_area =
      (packet_xmax - packet_xmin) * (ndim > 1 ? (packet_ymax - packet_ymin) : 1.0);
  int num_tracers_meshblock =
      packet_area > 0.0 ? std::round(num_tracers * overlap_x * overlap_y / packet_area)
                        : 0;

  auto new_particles_context = swarm->AddEmptyParticles(num_tracers_meshblock);

  static const std::vector<std::string> real_vars{swarm_position::x::name(),
                                                  swarm_position::y::name(),
                                                  swarm_position::z::name(), "cohort"};
  static auto real_desc = MakeSwarmPackDescriptor<Real>("tracers", real_vars);
  static auto real_map = real_desc.GetMap();
  auto real_pack = real_desc.GetPack(pmb->meshblock_data.Get().get());
  const int x_idx = real_map[swarm_position::x::name()];
  const int y_idx = real_map[swarm_position::y::name()];
  const int z_idx = real_map[swarm_position::z::name()];
  const int cohort_idx = real_map["cohort"];
  static auto id_desc = MakeSwarmPackDescriptor<swarm_position::id>("tracers");
  auto id_pack = id_desc.GetPack(pmb->meshblock_data.Get().get());
  const Real placement_omega = tr_pkg->Param<Real>("particle_placement_omega");

  pmb->par_for(
      PARTHENON_AUTO_LABEL, 0, new_particles_context.GetNewParticlesMaxIndex(),
      KOKKOS_LAMBDA(const int new_n) {
        const int n = new_particles_context.GetNewParticleIndex(new_n);
        auto rng_gen = rng_pool.get_state();

        real_pack(0, x_idx, n) =
            std::max(x_min, packet_xmin) + placement_omega * rng_gen.drand() * overlap_x;
        real_pack(0, y_idx, n) = ndim > 1
                                     ? (std::max(y_min, packet_ymin) +
                                        placement_omega * rng_gen.drand() * overlap_y)
                                     : 0.0;
        real_pack(0, z_idx, n) =
            z_min + placement_omega * rng_gen.drand() * (z_max - z_min);
        id_pack(0, swarm_position::id(), n) =
            static_cast<std::uint64_t>(pmb->gid) * 1000000ULL +
            static_cast<std::uint64_t>(new_n);
        // Reserve negative cohort values for the seed population so movie/debug views can
        // distinguish startup particles from particles sourced later during the run.
        real_pack(0, cohort_idx, n) = -1.0;

        rng_pool.free_state(rng_gen);
      });
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
    auto &dudt = pmb->meshblock_data.Get("dUdt");
    auto &sc1 = pmb->meshblock_data.Get(stage_name[stage]);

    auto advect_flux =
        tl.AddTask(none, particle_tracers_amr_source_sink::CalculateFluxes, sc0.get());
  }

  auto partitions = pmesh->GetDefaultBlockPartitions();
  const int num_partitions = partitions.size();
  // note that task within this region that contains one tasklist per pack
  // could still be executed in parallel
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

    if (stage == integrator->nstages) {
      auto new_dt = tl.AddTask(
          set_bc, parthenon::Update::EstimateTimestep<MeshBlockData<Real>>, sc1.get());
    }
  }

  // First-order operator split tracer particle update

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

    TaskRegion &tracer_transport = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; ++i) {
      auto &tl = tracer_transport[i];
      auto &base = pmesh->mesh_data.Add("base", partitions[i]);
      tl.AddTask(none, particle_tracers_amr_source_sink::AdvectTracers, base.get(),
                 integrator.get());
    }

    TaskRegion &async_region2 = tc.AddRegion(nblocks);
    for (int n = 0; n < nblocks; n++) {
      auto &tl = async_region2[n];
      auto &pmb = blocks[n];
      auto &sc = pmb->meshblock_data.Get()->GetSwarmData();
      auto send =
          tl.AddTask(none, &SwarmContainer::Send, sc.get(), BoundaryCommSubset::all);
      tl.AddTask(send, &SwarmContainer::Receive, sc.get(), BoundaryCommSubset::all);
    }

    TaskRegion &tracer_post_comm = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; ++i) {
      auto &tl = tracer_post_comm[i];
      auto &base = pmesh->mesh_data.Add("base", partitions[i]);
      auto sink = tl.AddTask(none, particle_tracers_amr_source_sink::DestroySinkParticles,
                             base.get());
      auto source = tl.AddTask(
          sink, particle_tracers_amr_source_sink::SourceStripParticles, base.get());
      tl.AddTask(source, particle_tracers_amr_source_sink::DepositTracers, base.get());
    }

    TaskRegion &async_region3 = tc.AddRegion(nblocks);
    for (int n = 0; n < nblocks; n++) {
      auto &tl = async_region3[n];
      auto &pmb = blocks[n];
      auto &sc = pmb->meshblock_data.Get()->GetSwarmData();
      // Defragment if swarm memory pool occupancy is 90%
      auto defrag = tl.AddTask(none, &SwarmContainer::Defrag, sc.get(), 0.9);
      if (pmesh->adaptive) {
        tl.AddTask(defrag, parthenon::Refinement::Tag<MeshBlockData<Real>>,
                   pmb->meshblock_data.Get().get());
      }
    }
  }

  return tc;
}

} // namespace particle_tracers_amr_source_sink
