#include "runner.hpp"

#include <chrono>

#include "kernels.hpp"
#include "loop_patterns.hpp"

namespace plb2 {

namespace {

template <int NITER, int SX, int SY, int SZ>
BenchmarkRow RunTypedCase(const CaseSpec &spec, const Dataset &dataset) {
  const auto alpha = MakeAlpha<NITER>();
  const auto beta = MakeBeta<NITER>();
  const int ni_mem = dataset.data.in.extent(4);
  const int nj_mem = dataset.data.in.extent(3);
  const std::array<int, 3> strides{ni_mem, nj_mem, 1};
  const auto dx = [] {
    std::array<int, SX> offsets{};
    for (int i = 0; i < SX; ++i) {
      offsets[i] = i + 1;
    }
    return offsets;
  }();
  const auto dy = [] {
    std::array<int, SY> offsets{};
    for (int i = 0; i < SY; ++i) {
      offsets[i] = i + 1;
    }
    return offsets;
  }();
  const auto dz = [] {
    std::array<int, SZ> offsets{};
    for (int i = 0; i < SZ; ++i) {
      offsets[i] = i + 1;
    }
    return offsets;
  }();

  const auto body_direct = KOKKOS_LAMBDA(const LoopData &data, int b, int v, int k, int j,
                                         int i) {
    return ComputeUnifiedCellDirect<NITER, SX, SY, SZ>(data.in, b, v, k, j, i, dx, dy, dz, alpha,
                                                       beta);
  };

  const auto build_access = KOKKOS_LAMBDA(const LoopData &data, int b, int v, int k, int j,
                                          int i) {
    return BuildUnifiedCellHoistedPointers<SX, SY, SZ>(data.in, b, v, k, j, i, strides, dx, dy,
                                                        dz);
  };

  const auto body_hoisted = KOKKOS_LAMBDA(const auto &access, int idx) {
    return ComputeUnifiedCellHoisted<NITER, SX, SY, SZ>(access, idx, alpha, beta);
  };

  const auto start = std::chrono::steady_clock::now();
  switch (spec.loop.kind) {
    case LoopKind::CpuFlatGhosts:
      RunCpuFlatGhosts(dataset, body_direct);
      break;
    case LoopKind::CpuBoviContiguous:
      RunCpuBoviContiguous(dataset, spec.loop.ninner, build_access, body_hoisted);
      break;
    case LoopKind::CpuBoviLogical:
      RunCpuBoviLogical(dataset, spec.loop.ninner, body_direct);
      break;
    case LoopKind::KokkosBoivFlat:
      RunKokkosBoivFlat(dataset, body_direct);
      break;
    case LoopKind::KokkosBoviTeamContiguous:
      RunKokkosBoviTeamContiguous(dataset, spec.loop.ninner, build_access, body_hoisted);
      break;
    case LoopKind::KokkosBoviTeamLogical:
      RunKokkosBoviTeamLogical(dataset, spec.loop.ninner, body_direct);
      break;
    case LoopKind::CpuBoivContiguous:
      RunCpuBoivContiguous(dataset, spec.loop.ninner, body_direct);
      break;
    case LoopKind::CpuBoivLogical:
      RunCpuBoivLogical(dataset, spec.loop.ninner, body_direct);
      break;
    case LoopKind::CpuBvoiContiguous:
      RunCpuBvoiContiguous(dataset, spec.loop.ninner, build_access, body_hoisted);
      break;
    case LoopKind::CpuBvoiLogical:
      RunCpuBvoiLogical(dataset, spec.loop.ninner, body_direct);
      break;
  }
  Kokkos::fence();
  const auto stop = std::chrono::steady_clock::now();

  BenchmarkRow row;
  row.loop_name = ToString(spec.loop.kind);
  row.backend = spec.backend;
  row.nblocks = spec.problem.nblocks;
  row.nvars = spec.problem.nvars;
  row.nz_interior = spec.problem.nz_interior;
  row.ny_interior = spec.problem.ny_interior;
  row.nx_interior = spec.problem.nx_interior;
  row.nghost = spec.problem.nghost;
  row.ninner = spec.loop.ninner;
  row.niter = spec.kernel.niter;
  row.stencil_x = spec.kernel.stencil_x;
  row.stencil_y = spec.kernel.stencil_y;
  row.stencil_z = spec.kernel.stencil_z;
  row.total_updates = CountUpdates(spec, dataset);
  row.min_seconds = std::chrono::duration<double>(stop - start).count();
  row.updates_per_second = static_cast<double>(row.total_updates) / row.min_seconds;
  return row;
}

}  // namespace

BenchmarkRow RunCase(const CaseSpec &spec) {
  Dataset dataset = BuildDataset(spec);
  PrepareDataset(spec, &dataset);
  switch (spec.kernel.niter) {
    case 0:
      return RunTypedCase<0, 1, 1, 1>(spec, dataset);
    case 4:
      return RunTypedCase<4, 1, 1, 1>(spec, dataset);
    case 8:
      return RunTypedCase<8, 1, 1, 1>(spec, dataset);
    default:
      return RunTypedCase<4, 1, 1, 1>(spec, dataset);
  }
}

}  // namespace plb2
