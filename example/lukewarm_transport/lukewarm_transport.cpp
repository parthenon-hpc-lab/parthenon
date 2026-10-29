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

// An iterative tracer push. Each tracer carries its own time and is pushed until it
// reaches the end of the step, or until it enters a block of another MeshData, where it
// drops out to be exchanged. Its new MeshData pushes it the rest of the way in the next
// iteration, and iterations go on until no tracer anywhere has time left.
//
// Exchanged tracers are appended to the end of their new MeshData, and every tracer that
// stayed behind has reached the end of the step, so each iteration after the first only
// has to look at the tracers GetLastReceived() reports.

// C++ includes
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

// Third party includes
#include <Kokkos_Random.hpp>

// Parthenon includes
#include "basic_types.hpp"
#include "globals.hpp"
#include "interface/metadata.hpp"
#include "interface/update.hpp"
#include "kokkos_abstraction.hpp"
#include "lukewarm/lukewarm_pack.hpp"
#include "lukewarm/lukewarm_relocate.hpp"
#include "outputs/outputs.hpp"
#include "pack/default_names.hpp"
#include "utils/instrument.hpp"

#include "lukewarm_transport.hpp"

typedef Kokkos::Random_XorShift64_Pool<> RNGPool;

namespace lukewarm_transport {

Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;
  packages.Add(tracers_package::Initialize(pin.get()));
  return packages;
}

namespace tracers_package {

// The smallest cell width along any active direction of the blocks of md
Real MinCellWidth(MeshData<Real> *md) {
  const int ndim = md->GetParentPointer()->ndim;
  Real dx = std::numeric_limits<Real>::max();
  for (int b = 0; b < md->NumBlocks(); ++b) {
    const auto &coords = md->GetBlockData(b)->GetBlockPointer()->coords;
    dx = std::min(dx, coords.Dxc<X1DIR>(0));
    if (ndim > 1) dx = std::min(dx, coords.Dxc<X2DIR>(0));
    if (ndim > 2) dx = std::min(dx, coords.Dxc<X3DIR>(0));
  }
  return dx;
}

// Tracers may move several blocks in a step, which is what the iterative push is for
Real EstimateTimestepMesh(MeshData<Real> *md) {
  auto &pkg = md->GetParentPointer()->packages.Get("tracers_package");
  const Real speed = pkg->Param<Real>("speed");
  const Real cells_per_step = pkg->Param<Real>("cells_per_step");
  return cells_per_step * MinCellWidth(md) / speed;
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

// Iterations the push took in the last step, the same on every rank
Real NumIterations(MeshData<Real> *md) {
  auto &pkg = md->GetParentPointer()->packages.Get("tracers_package");
  return pkg->Param<int>("num_iterations");
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("tracers_package");

  pkg->AddParam("num_tracers_per_block",
                pin->GetOrAddInteger("Tracers", "num_tracers_per_block", 100));
  pkg->AddParam("speed", pin->GetOrAddReal("Tracers", "speed", 1.0));
  pkg->AddParam("cells_per_step", pin->GetOrAddReal("Tracers", "cells_per_step", 20.0));
  pkg->AddParam("max_iterations",
                pin->GetOrAddInteger("Tracers", "max_iterations", 1000));
  pkg->AddParam("start_time", pin->GetOrAddReal("parthenon/time", "start_time", 0.0));
  pkg->AddParam("num_iterations", 0, Params::Mutability::Mutable);

  // Every lukewarm carries positions and persistent ids
  pkg->AddLukewarm(lukewarm_name, Metadata({Metadata::Provides}));
  pkg->AddLukewarmField<tracer::time>(lukewarm_name, Metadata({Metadata::Real}));
  pkg->AddLukewarmField<tracer::velocity>(
      lukewarm_name, Metadata({Metadata::Real}, std::vector<int>{3}));

  pkg->AddField<field::deposition>(Metadata({Metadata::Cell, Metadata::Derived}));

  pkg->EstimateTimestepMesh = EstimateTimestepMesh;

  // Tracers are sourced once the mesh is initialized, on each MeshData that owns them
  pkg->PostInitializationMesh = SourceTracers;

  parthenon::HstVar_list hst_vars = {};
  hst_vars.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                                    NumTracers, "num_tracers"));
  hst_vars.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                                    TotalDeposition, "total_deposition"));
  hst_vars.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::max,
                                                    NumIterations, "num_iterations"));
  pkg->AddParam<>(parthenon::hist_param_key, hst_vars);

  return pkg;
}

void SourceTracers(Mesh *pm, ParameterInput *pin, MeshData<Real> *md) {
  auto &pkg = pm->packages.Get("tracers_package");
  const int num_tracers_per_block = pkg->Param<int>("num_tracers_per_block");
  const Real speed = pkg->Param<Real>("speed");
  const Real start_time = pkg->Param<Real>("start_time");
  const int ndim = pm->ndim;
  auto &tracers = md->GetLukewarm(lukewarm_name);
  const int nblocks = md->NumBlocks();

  // Every block gets the same number of tracers, with block b's at
  // [offsets(b), offsets(b + 1))
  ParArray1DRaw<int> counts("num_tracers", nblocks);
  Kokkos::deep_copy(counts, num_tracers_per_block);
  const auto offsets = tracers.Create(counts);

  // A pack is a snapshot, so get it after creating the tracers
  static auto desc =
      MakeLukewarmPackDescriptor<lukewarm::x, lukewarm::y, lukewarm::z, tracer::time,
                                 tracer::velocity>(md, lukewarm_name);
  auto pack = desc.GetPack(md);
  const auto geometry = md->GetLukewarmGeometry();

  // Seeded by the id of this MeshData, so that each MeshData draws its own tracers
  auto rng_pool = RNGPool(md->GetLukewarmContainer().GetId());

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

              pack(lukewarm::x(), n) = x_min + rng_gen.drand() * (x_max - x_min);
              pack(lukewarm::y(), n) = y_min + rng_gen.drand() * (y_max - y_min);
              pack(lukewarm::z(), n) = z_min + rng_gen.drand() * (z_max - z_min);
              pack(tracer::time(), n) = start_time;

              // A direction drawn uniformly over the active dimensions
              Real v[3] = {0.0, 0.0, 0.0};
              if (ndim == 1) {
                v[0] = rng_gen.drand() < 0.5 ? -1.0 : 1.0;
              } else {
                const Real phi = 2.0 * M_PI * rng_gen.drand();
                const Real mu = ndim == 3 ? 2.0 * rng_gen.drand() - 1.0 : 0.0;
                const Real sin_theta = std::sqrt(1.0 - mu * mu);
                v[0] = sin_theta * std::cos(phi);
                v[1] = sin_theta * std::sin(phi);
                v[2] = mu;
              }
              for (int d = 0; d < 3; ++d) {
                pack(tracer::velocity(d), n) = speed * v[d];
              }

              rng_pool.free_state(rng_gen);
            });
      });
}

} // namespace tracers_package

TaskStatus PushTracers(MeshData<Real> *md, const Real t_start, const Real dt,
                       int *iteration) {
  PARTHENON_INSTRUMENT
  auto &pkg = md->GetParentPointer()->packages.Get("tracers_package");

  // The first iteration pushes every tracer. Every tracer that stays on this MeshData is
  // pushed to the end of the step, so later iterations only push the tracers the last
  // exchange delivered.
  auto &tracers = md->GetLukewarm(lukewarm_name);
  const IndexRange range = (*iteration)++ == 0 ? IndexRange{0, tracers.NumParticles() - 1}
                                               : GetLastReceived(md, lukewarm_name);
  if (range.size() == 0) return TaskStatus::complete;

  // A tracer moves at most one cell per substep, so it stops soon after it enters a
  // block of another MeshData
  const Real dt_cell = tracers_package::MinCellWidth(md) / pkg->Param<Real>("speed");

  static auto desc =
      MakeLukewarmPackDescriptor<lukewarm::x, lukewarm::y, lukewarm::z, tracer::time,
                                 tracer::velocity>(md, lukewarm_name);
  auto pack = desc.GetPack(md);

  const Real t_end = t_start + dt;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PushTracers", DevExecSpace(), range.s, range.e,
      KOKKOS_LAMBDA(const int n) {
        Real &t = pack(tracer::time(), n);
        while (t < t_end) {
          const bool last = dt_cell >= t_end - t;
          const Real ds = last ? t_end - t : dt_cell;
          pack(lukewarm::x(), n) += pack(tracer::velocity(0), n) * ds;
          pack(lukewarm::y(), n) += pack(tracer::velocity(1), n) * ds;
          pack(lukewarm::z(), n) += pack(tracer::velocity(2), n) * ds;
          t = last ? t_end : t + ds;

          // Drop out for comms once the tracer has left this MeshData
          const auto loc = pack.Locate(n);
          if (!loc.InMeshData()) break;

          // The block and interior cell holding the tracer, e.g., to read a field there
          // through a field pack of md, whose block indices are the same:
          //   const int b = loc.Block();
          //   const auto [k, j, i] = pack.GetCellIndices(b, n);
          //   const Real value = field_pack(b, field::deposition(), k, j, i);
        }
      });

  return TaskStatus::complete;
}

TaskStatus CheckCompletion(MeshData<Real> *md, const Real t_end) {
  PARTHENON_INSTRUMENT
  // Only tracers the last exchange delivered can still have time left
  const auto range = GetLastReceived(md, lukewarm_name);
  if (range.size() == 0) return TaskStatus::complete;

  static auto desc = MakeLukewarmPackDescriptor<tracer::time>(md, lukewarm_name);
  auto pack = desc.GetPack(md);

  int num_unfinished = 0;
  parthenon::par_reduce(
      parthenon::loop_pattern_flatrange_tag, "CheckCompletion", DevExecSpace(), range.s,
      range.e,
      KOKKOS_LAMBDA(const int n, int &sum) { sum += pack(tracer::time(), n) < t_end; },
      Kokkos::Sum<int>(num_unfinished));

  return num_unfinished > 0 ? TaskStatus::iterate : TaskStatus::complete;
}

TaskStatus CheckTracersFinished(MeshData<Real> *md, const Real t_end) {
  static auto desc = MakeLukewarmPackDescriptor<tracer::time>(md, lukewarm_name);
  auto pack = desc.GetPack(md);

  int num_unfinished = 0;
  parthenon::par_reduce(
      parthenon::loop_pattern_flatrange_tag, "CheckTracersFinished", DevExecSpace(), 0,
      pack.GetNParticles() - 1,
      KOKKOS_LAMBDA(const int n, int &sum) { sum += pack(tracer::time(), n) != t_end; },
      Kokkos::Sum<int>(num_unfinished));
  PARTHENON_REQUIRE_THROWS(num_unfinished == 0,
                           std::to_string(num_unfinished) +
                               " tracers did not reach the end of the step. Raise "
                               "Tracers/max_iterations?");

  return TaskStatus::complete;
}

TaskStatus DepositTracers(MeshData<Real> *md) {
  auto pm = md->GetParentPointer();
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);

  static auto desc = MakePackDescriptor<field::deposition>(pm->resolved_packages.get());
  static auto pdesc = MakeLukewarmPackDescriptor<lukewarm::x>(md, lukewarm_name);
  auto vmesh = desc.GetPack(md);
  auto vpart = pdesc.GetPack(md);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Zero", parthenon::DevExecSpace(), 0, md->NumBlocks() - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        vmesh(b, field::deposition(), k, j, i) = 0.0;
      });

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

TaskListStatus TracerDriver::Step() {
  const auto status = StepTasks().Execute();
  int num_iterations = 0;
  for (const int n : push_iterations_) {
    num_iterations = std::max(num_iterations, n);
  }
  pmesh->packages.Get("tracers_package")->UpdateParam("num_iterations", num_iterations);
  return status;
}

TaskCollection TracerDriver::StepTasks() {
  TaskCollection tc;
  TaskID none(0);

  // The completion check runs once per region, so its verdict only covers every
  // MeshData if there is one per rank
  const int num_partitions = pmesh->DefaultNumPartitions();
  PARTHENON_REQUIRE(num_partitions == 1,
                    "The iterative tracer push needs one MeshData per rank, "
                    "parthenon/mesh/pack_size = -1");
  auto &pkg = pmesh->packages.Get("tracers_package");
  const int max_iterations = pkg->Param<int>("max_iterations");
  const Real t_end = tm.time + tm.dt;

  push_iterations_.resize(num_partitions);

  TaskRegion &region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = region[i];
    auto &base = pmesh->mesh_data.GetOrAdd("base", i);

    // A sublist only starts after a task it depends on
    auto start = tl.AddTask(
        none,
        [](int *iteration) {
          *iteration = 0;
          return TaskStatus::complete;
        },
        &push_iterations_[i]);

    // Keep pushing tracers until every one has reached the end of the step
    auto [itl, push] = tl.AddSublist(start, {1, max_iterations});
    auto push_tracers =
        itl.AddTask(none, PushTracers, base.get(), tm.time, tm.dt, &push_iterations_[i]);
    auto exchange =
        parthenon::AddLukewarmExchangeTasks(push_tracers, itl, base.get(), lukewarm_name);
    auto complete =
        itl.AddTask(TaskQualifier::once_per_region | TaskQualifier::global_sync |
                        TaskQualifier::completion,
                    exchange, CheckCompletion, base.get(), t_end);

    auto finished = tl.AddTask(push, CheckTracersFinished, base.get(), t_end);
    auto deposit = tl.AddTask(finished, DepositTracers, base.get());
  }

  // Timestep for the next step
  TaskRegion &dt_region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &base = pmesh->mesh_data.GetOrAdd("base", i);
    dt_region[i].AddTask(none, parthenon::Update::EstimateTimestep<MeshData<Real>>,
                         base.get());
  }

  return tc;
}

} // namespace lukewarm_transport
