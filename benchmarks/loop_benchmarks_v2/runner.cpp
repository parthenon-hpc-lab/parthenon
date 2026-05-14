#include "runner.hpp"

#include <chrono>
#include <array>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

#include "kernels.hpp"
#include "loop_abstraction_kernel.hpp"
#include "loop_patterns.hpp"

namespace plb2 {

namespace {

#if defined(__GNUC__) || defined(__clang__)
#define PLB2_NOINLINE __attribute__((noinline))
#else
#define PLB2_NOINLINE
#endif

template <typename RunFn>
std::pair<double, double> TimeRepeatedRun(int warmup, int repeats, RunFn &&run_once) {
  for (int i = 0; i < warmup; ++i) {
    run_once();
    Kokkos::fence();
  }

  double total_seconds = 0.0;
  double best_seconds = std::numeric_limits<double>::infinity();
  for (int i = 0; i < repeats; ++i) {
    const auto start = std::chrono::steady_clock::now();
    run_once();
    Kokkos::fence();
    const auto stop = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(stop - start).count();
    total_seconds += elapsed;
    best_seconds = std::min(best_seconds, elapsed);
  }
  return {total_seconds / std::max(repeats, 1), best_seconds};
}

template <typename ViewType>
double ComputeOutputChecksum(const ViewType &out) {
  auto host_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  double checksum = 0.0;
  for (std::size_t b = 0; b < host_out.extent(0); ++b) {
    for (std::size_t v = 0; v < host_out.extent(1); ++v) {
      for (std::size_t k = 0; k < host_out.extent(2); ++k) {
        for (std::size_t j = 0; j < host_out.extent(3); ++j) {
          for (std::size_t i = 0; i < host_out.extent(4); ++i) {
            checksum += host_out(b, v, k, j, i);
          }
        }
      }
    }
  }
  return checksum;
}

template <loop_abstraction::loop_tag LOOP_TAG, loop_abstraction::inner_tag INNER_TAG,
          int SX, int SY, int SZ>
PLB2_NOINLINE
void RunLoopAbstractionCase(const CaseSpec &spec, const Dataset &dataset,
                            const std::array<int, SX> &dx, const std::array<int, SY> &dy,
                            const std::array<int, SZ> &dz,
                            const std::array<double, kMaxNiter> &alpha,
                            const std::array<double, kMaxNiter> &beta) {
  const std::optional<int> ninner =
      spec.loop.ninner > 0 ? std::optional<int>{spec.loop.ninner} : std::nullopt;
  const auto &problem = dataset.problem;
  RunUnifiedKernelWithLoopAbstraction<LOOP_TAG, INNER_TAG, SX, SY, SZ>(
      dataset.data.in, dataset.data.out, dataset.data.active_counts, problem.nblocks,
      problem.nx_interior, problem.ny_interior, problem.nz_interior, problem.nghost, dx, dy, dz,
      alpha, beta, spec.kernel.niter, ninner);
}

template <int SX, int SY, int SZ>
PLB2_NOINLINE
void RunCpuFlatGhostsCase(const CaseSpec &spec, const Dataset &dataset,
                          const std::array<int, SX> &dx, const std::array<int, SY> &dy,
                          const std::array<int, SZ> &dz,
                          const std::array<double, kMaxNiter> &alpha,
                          const std::array<double, kMaxNiter> &beta) {
  const int niter = spec.kernel.niter;
  const auto body_direct = KOKKOS_LAMBDA(const LoopData &data, int b, int v, int k, int j,
                                         int i) {
    return ComputeUnifiedCellDirect<SX, SY, SZ>(data.in, b, v, k, j, i, dx, dy, dz, alpha, beta,
                                                niter);
  };
  const auto build_access = KOKKOS_LAMBDA(const LoopData &data, int b, int v, int k, int j,
                                          int i) {
    return BuildUnifiedCellHoistedPointers<SX, SY, SZ>(data.in, b, v, k, j, i, dx, dy, dz);
  };
  const auto body_hoisted = KOKKOS_LAMBDA(const auto &access, int idx) {
    return ComputeUnifiedCellHoisted<SX, SY, SZ>(access, idx, alpha, beta, niter);
  };
  if (spec.loop.access_mode == "hoisted") {
    RunCpuFlatGhosts(dataset, build_access, body_hoisted);
  } else {
    RunCpuFlatGhosts(dataset, body_direct);
  }
}

template <int SX, int SY, int SZ>
PLB2_NOINLINE
void RunCpuBoviContiguousCase(const CaseSpec &spec, const Dataset &dataset,
                              const std::array<int, SX> &dx, const std::array<int, SY> &dy,
                              const std::array<int, SZ> &dz,
                              const std::array<double, kMaxNiter> &alpha,
                              const std::array<double, kMaxNiter> &beta) {
  const int niter = spec.kernel.niter;
  const auto body_direct = KOKKOS_LAMBDA(const LoopData &data, int b, int v, int k, int j,
                                         int i) {
    return ComputeUnifiedCellDirect<SX, SY, SZ>(data.in, b, v, k, j, i, dx, dy, dz, alpha, beta,
                                                niter);
  };
  const auto build_access = KOKKOS_LAMBDA(const LoopData &data, int b, int v, int k, int j,
                                          int i) {
    return BuildUnifiedCellHoistedPointers<SX, SY, SZ>(data.in, b, v, k, j, i, dx, dy, dz);
  };
  const auto body_hoisted = KOKKOS_LAMBDA(const auto &access, int idx) {
    return ComputeUnifiedCellHoisted<SX, SY, SZ>(access, idx, alpha, beta, niter);
  };
  if (spec.loop.access_mode == "hoisted") {
    RunCpuBoviContiguous(dataset, spec.loop.ninner, build_access, body_hoisted);
  } else {
    RunCpuBoviContiguousDirect(dataset, spec.loop.ninner, body_direct);
  }
}

template <int SX, int SY, int SZ>
PLB2_NOINLINE
void RunCpuBoviLogicalCase(const CaseSpec &spec, const Dataset &dataset,
                           const std::array<int, SX> &dx, const std::array<int, SY> &dy,
                           const std::array<int, SZ> &dz,
                           const std::array<double, kMaxNiter> &alpha,
                           const std::array<double, kMaxNiter> &beta) {
  const int niter = spec.kernel.niter;
  const auto body_direct = KOKKOS_LAMBDA(const LoopData &data, int b, int v, int k, int j,
                                         int i) {
    return ComputeUnifiedCellDirect<SX, SY, SZ>(data.in, b, v, k, j, i, dx, dy, dz, alpha, beta,
                                                niter);
  };
  RunCpuBoviLogical(dataset, spec.loop.ninner, body_direct);
}

template <int SX, int SY, int SZ>
PLB2_NOINLINE
void RunKokkosBoivFlatCase(const CaseSpec &spec, const Dataset &dataset,
                           const std::array<int, SX> &dx, const std::array<int, SY> &dy,
                           const std::array<int, SZ> &dz,
                           const std::array<double, kMaxNiter> &alpha,
                           const std::array<double, kMaxNiter> &beta) {
  const int niter = spec.kernel.niter;
  const auto body_direct = KOKKOS_LAMBDA(const LoopData &data, int b, int v, int k, int j,
                                         int i) {
    return ComputeUnifiedCellDirect<SX, SY, SZ>(data.in, b, v, k, j, i, dx, dy, dz, alpha, beta,
                                                niter);
  };
  RunKokkosBoivFlat(dataset, body_direct);
}

template <int SX, int SY, int SZ>
PLB2_NOINLINE
void RunKokkosBoviTeamContiguousCase(const CaseSpec &spec, const Dataset &dataset,
                                     const std::array<int, SX> &dx,
                                     const std::array<int, SY> &dy,
                                     const std::array<int, SZ> &dz,
                                     const std::array<double, kMaxNiter> &alpha,
                                     const std::array<double, kMaxNiter> &beta) {
  const int niter = spec.kernel.niter;
  const auto body_direct = KOKKOS_LAMBDA(const LoopData &data, int b, int v, int k, int j,
                                         int i) {
    return ComputeUnifiedCellDirect<SX, SY, SZ>(data.in, b, v, k, j, i, dx, dy, dz, alpha, beta,
                                                niter);
  };
  const auto build_access = KOKKOS_LAMBDA(const LoopData &data, int b, int v, int k, int j,
                                          int i) {
    return BuildUnifiedCellHoistedPointers<SX, SY, SZ>(data.in, b, v, k, j, i, dx, dy, dz);
  };
  const auto body_hoisted = KOKKOS_LAMBDA(const auto &access, int idx) {
    return ComputeUnifiedCellHoisted<SX, SY, SZ>(access, idx, alpha, beta, niter);
  };
  if (spec.loop.access_mode == "hoisted") {
    RunKokkosBoviTeamContiguous(dataset, spec.loop.ninner, build_access, body_hoisted);
  } else {
    RunKokkosBoviTeamContiguousDirect(dataset, spec.loop.ninner, body_direct);
  }
}

template <int SX, int SY, int SZ>
PLB2_NOINLINE
void RunKokkosBoviTeamLogicalCase(const CaseSpec &spec, const Dataset &dataset,
                                  const std::array<int, SX> &dx,
                                  const std::array<int, SY> &dy,
                                  const std::array<int, SZ> &dz,
                                  const std::array<double, kMaxNiter> &alpha,
                                  const std::array<double, kMaxNiter> &beta) {
  const int niter = spec.kernel.niter;
  const auto body_direct = KOKKOS_LAMBDA(const LoopData &data, int b, int v, int k, int j,
                                         int i) {
    return ComputeUnifiedCellDirect<SX, SY, SZ>(data.in, b, v, k, j, i, dx, dy, dz, alpha, beta,
                                                niter);
  };
  RunKokkosBoviTeamLogical(dataset, spec.loop.ninner, body_direct);
}

template <int SX, int SY, int SZ>
PLB2_NOINLINE
void RunCpuBoivContiguousCase(const CaseSpec &spec, const Dataset &dataset,
                              const std::array<int, SX> &dx, const std::array<int, SY> &dy,
                              const std::array<int, SZ> &dz,
                              const std::array<double, kMaxNiter> &alpha,
                              const std::array<double, kMaxNiter> &beta) {
  const int niter = spec.kernel.niter;
  const auto body_direct = KOKKOS_LAMBDA(const LoopData &data, int b, int v, int k, int j,
                                         int i) {
    return ComputeUnifiedCellDirect<SX, SY, SZ>(data.in, b, v, k, j, i, dx, dy, dz, alpha, beta,
                                                niter);
  };
  RunCpuBoivContiguous(dataset, spec.loop.ninner, body_direct);
}

template <int SX, int SY, int SZ>
PLB2_NOINLINE
void RunCpuBoivLogicalCase(const CaseSpec &spec, const Dataset &dataset,
                           const std::array<int, SX> &dx, const std::array<int, SY> &dy,
                           const std::array<int, SZ> &dz,
                           const std::array<double, kMaxNiter> &alpha,
                           const std::array<double, kMaxNiter> &beta) {
  const int niter = spec.kernel.niter;
  const auto body_direct = KOKKOS_LAMBDA(const LoopData &data, int b, int v, int k, int j,
                                         int i) {
    return ComputeUnifiedCellDirect<SX, SY, SZ>(data.in, b, v, k, j, i, dx, dy, dz, alpha, beta,
                                                niter);
  };
  RunCpuBoivLogical(dataset, spec.loop.ninner, body_direct);
}

template <int SX, int SY, int SZ>
PLB2_NOINLINE
void RunCpuBvoiContiguousCase(const CaseSpec &spec, const Dataset &dataset,
                              const std::array<int, SX> &dx, const std::array<int, SY> &dy,
                              const std::array<int, SZ> &dz,
                              const std::array<double, kMaxNiter> &alpha,
                              const std::array<double, kMaxNiter> &beta) {
  const int niter = spec.kernel.niter;
  const auto body_direct = KOKKOS_LAMBDA(const LoopData &data, int b, int v, int k, int j,
                                         int i) {
    return ComputeUnifiedCellDirect<SX, SY, SZ>(data.in, b, v, k, j, i, dx, dy, dz, alpha, beta,
                                                niter);
  };
  const auto build_access = KOKKOS_LAMBDA(const LoopData &data, int b, int v, int k, int j,
                                          int i) {
    return BuildUnifiedCellHoistedPointers<SX, SY, SZ>(data.in, b, v, k, j, i, dx, dy, dz);
  };
  const auto body_hoisted = KOKKOS_LAMBDA(const auto &access, int idx) {
    return ComputeUnifiedCellHoisted<SX, SY, SZ>(access, idx, alpha, beta, niter);
  };
  if (spec.loop.access_mode == "hoisted") {
    RunCpuBvoiContiguous(dataset, spec.loop.ninner, build_access, body_hoisted);
  } else {
    RunCpuBvoiContiguousDirect(dataset, spec.loop.ninner, body_direct);
  }
}

template <int SX, int SY, int SZ>
PLB2_NOINLINE
void RunCpuBvoiLogicalCase(const CaseSpec &spec, const Dataset &dataset,
                           const std::array<int, SX> &dx, const std::array<int, SY> &dy,
                           const std::array<int, SZ> &dz,
                           const std::array<double, kMaxNiter> &alpha,
                           const std::array<double, kMaxNiter> &beta) {
  const int niter = spec.kernel.niter;
  const auto body_direct = KOKKOS_LAMBDA(const LoopData &data, int b, int v, int k, int j,
                                         int i) {
    return ComputeUnifiedCellDirect<SX, SY, SZ>(data.in, b, v, k, j, i, dx, dy, dz, alpha, beta,
                                                niter);
  };
  RunCpuBvoiLogical(dataset, spec.loop.ninner, body_direct);
}

template <int SX, int SY, int SZ>
BenchmarkRow RunTypedCase(const CaseSpec &spec, const Dataset &dataset) {
  const auto alpha = MakeAlpha();
  const auto beta = MakeBeta();
  const auto dx = [&] {
    std::array<int, SX> offsets{};
    for (int i = 0; i < SX; ++i) {
      offsets[i] = spec.kernel.stencil_x[static_cast<std::size_t>(i)];
    }
    return offsets;
  }();
  const auto dy = [&] {
    std::array<int, SY> offsets{};
    for (int i = 0; i < SY; ++i) {
      offsets[i] = spec.kernel.stencil_y[static_cast<std::size_t>(i)];
    }
    return offsets;
  }();
  const auto dz = [&] {
    std::array<int, SZ> offsets{};
    for (int i = 0; i < SZ; ++i) {
      offsets[i] = spec.kernel.stencil_z[static_cast<std::size_t>(i)];
    }
    return offsets;
  }();

  const auto run_once = [&] {
    switch (spec.loop.kind) {
      case LoopKind::CpuFlatGhosts:
        RunCpuFlatGhostsCase<SX, SY, SZ>(spec, dataset, dx, dy, dz, alpha, beta);
        break;
      case LoopKind::CpuBoviContiguous:
        RunCpuBoviContiguousCase<SX, SY, SZ>(spec, dataset, dx, dy, dz, alpha, beta);
        break;
      case LoopKind::CpuBoviLogical:
        RunCpuBoviLogicalCase<SX, SY, SZ>(spec, dataset, dx, dy, dz, alpha, beta);
        break;
      case LoopKind::KokkosBoivFlat:
        RunKokkosBoivFlatCase<SX, SY, SZ>(spec, dataset, dx, dy, dz, alpha, beta);
        break;
      case LoopKind::KokkosBoviTeamContiguous:
        RunKokkosBoviTeamContiguousCase<SX, SY, SZ>(spec, dataset, dx, dy, dz, alpha, beta);
        break;
      case LoopKind::KokkosBoviTeamLogical:
        RunKokkosBoviTeamLogicalCase<SX, SY, SZ>(spec, dataset, dx, dy, dz, alpha, beta);
        break;
      case LoopKind::LoopAbstractionBoviMemory:
        RunLoopAbstractionCase<loop_abstraction::loop_tag::bovi,
                               loop_abstraction::inner_tag::memory, SX, SY, SZ>(
            spec, dataset, dx, dy, dz, alpha, beta);
        break;
      case LoopKind::LoopAbstractionBoviLogical:
        RunLoopAbstractionCase<loop_abstraction::loop_tag::bovi,
                               loop_abstraction::inner_tag::logical, SX, SY, SZ>(
            spec, dataset, dx, dy, dz, alpha, beta);
        break;
      case LoopKind::LoopAbstractionBoivLogical:
        RunLoopAbstractionCase<loop_abstraction::loop_tag::boiv,
                               loop_abstraction::inner_tag::logical, SX, SY, SZ>(
            spec, dataset, dx, dy, dz, alpha, beta);
        break;
      case LoopKind::LoopAbstractionBvoiMemory:
        RunLoopAbstractionCase<loop_abstraction::loop_tag::bvoi,
                               loop_abstraction::inner_tag::memory, SX, SY, SZ>(
            spec, dataset, dx, dy, dz, alpha, beta);
        break;
      case LoopKind::LoopAbstractionBvoiLogical:
        RunLoopAbstractionCase<loop_abstraction::loop_tag::bvoi,
                               loop_abstraction::inner_tag::logical, SX, SY, SZ>(
            spec, dataset, dx, dy, dz, alpha, beta);
        break;
      case LoopKind::CpuBoivContiguous:
        RunCpuBoivContiguousCase<SX, SY, SZ>(spec, dataset, dx, dy, dz, alpha, beta);
        break;
      case LoopKind::CpuBoivLogical:
        RunCpuBoivLogicalCase<SX, SY, SZ>(spec, dataset, dx, dy, dz, alpha, beta);
        break;
      case LoopKind::CpuBvoiContiguous:
        RunCpuBvoiContiguousCase<SX, SY, SZ>(spec, dataset, dx, dy, dz, alpha, beta);
        break;
      case LoopKind::CpuBvoiLogical:
        RunCpuBvoiLogicalCase<SX, SY, SZ>(spec, dataset, dx, dy, dz, alpha, beta);
        break;
    }
  };

  const auto [avg_seconds, min_seconds] =
      TimeRepeatedRun(spec.warmup, spec.repeats, run_once);

  BenchmarkRow row;
  row.loop_name = ToString(spec.loop.kind);
  row.backend = spec.backend;
  row.nblocks = spec.problem.nblocks;
  row.target_cells = spec.problem.target_cells;
  row.nvars = spec.problem.nvars;
  row.nz_interior = spec.problem.nz_interior;
  row.ny_interior = spec.problem.ny_interior;
  row.nx_interior = spec.problem.nx_interior;
  row.nghost = spec.problem.nghost;
  row.ninner = spec.loop.ninner;
  row.access_mode = spec.loop.access_mode;
  row.niter = spec.kernel.niter;
  row.stencil_x = FormatOffsetSet(spec.kernel.stencil_x);
  row.stencil_y = FormatOffsetSet(spec.kernel.stencil_y);
  row.stencil_z = FormatOffsetSet(spec.kernel.stencil_z);
  row.kernel_label = KernelLabel(spec);
  row.warmup = spec.warmup;
  row.repeats = spec.repeats;
  row.logical_cells_per_block = static_cast<std::uint64_t>(dataset.problem.logical_indexer.size());
  row.memory_cells_per_block = static_cast<std::uint64_t>(dataset.problem.memory_indexer.size());
  row.total_updates = CountUpdates(spec, dataset);
  row.touched_cells = CountTouchedCells(spec, dataset);
  row.avg_seconds = avg_seconds;
  row.min_seconds = min_seconds;
  row.updates_per_second = static_cast<double>(row.total_updates) / row.avg_seconds;
  row.touched_cells_per_second = static_cast<double>(row.touched_cells) / row.avg_seconds;
  if (spec.validate) {
    row.validation_checksum = ComputeOutputChecksum(dataset.data.out);
  }
  return row;
}

#undef PLB2_NOINLINE

}  // namespace

BenchmarkRow RunCase(const CaseSpec &spec) {
  Dataset dataset = BuildDataset(spec);
  PrepareDataset(spec, &dataset);
  const auto sx = spec.kernel.stencil_x.size();
  const auto sy = spec.kernel.stencil_y.size();
  const auto sz = spec.kernel.stencil_z.size();
  if (sx == 3 && sy == 1 && sz == 1) {
    return RunTypedCase<3, 1, 1>(spec, dataset);
  }
  if (sx == 1 && sy == 3 && sz == 1) {
    return RunTypedCase<1, 3, 1>(spec, dataset);
  }
  if (sx == 1 && sy == 1 && sz == 3) {
    return RunTypedCase<1, 1, 3>(spec, dataset);
  }
  if (sx == 1 && sy == 1 && sz == 1) {
    return RunTypedCase<1, 1, 1>(spec, dataset);
  }
  throw std::runtime_error("unsupported stencil shape: x{" + FormatOffsetSet(spec.kernel.stencil_x) +
                           "}y{" + FormatOffsetSet(spec.kernel.stencil_y) + "}z{" +
                           FormatOffsetSet(spec.kernel.stencil_z) + "}");
}

}  // namespace plb2
