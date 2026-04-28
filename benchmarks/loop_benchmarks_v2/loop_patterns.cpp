#include "loop_patterns.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>

#include <Kokkos_Core.hpp>

#include "kernels.hpp"

namespace plb2 {

namespace {

using TeamPolicy = Kokkos::TeamPolicy<>;
using TeamMember = TeamPolicy::member_type;

IndexRange MakeInteriorRange(int interior) { return {0, interior - 1}; }

IndexRange MakeMemoryRange(int interior, int nghost) { return {0, interior + 2 * nghost - 1}; }

int SelectNinner(const CaseSpec &spec) {
  if (spec.loop.ninner > 0) {
    return spec.loop.ninner;
  }
  return spec.problem.nx_interior * spec.problem.ny_interior;
}

int SelectNvarsForBlock(const CaseSpec &spec, int block) {
  if (block >= 0 && block < static_cast<int>(spec.problem.vars_per_block.size())) {
    return std::min(spec.problem.vars_per_block[block], spec.problem.nvars);
  }
  return spec.problem.nvars;
}

template <typename ViewType>
void InitializeDataViews(const CaseSpec &spec, const ViewType &in, const ViewType &aux,
                         const ViewType &out) {
  const auto problem = spec.problem;
  const int nz_mem = problem.nz_interior + 2 * problem.nghost;
  const int ny_mem = problem.ny_interior + 2 * problem.nghost;
  const int nx_mem = problem.nx_interior + 2 * problem.nghost;

  Kokkos::parallel_for(
      "InitializeDataV2",
      Kokkos::MDRangePolicy<Kokkos::Rank<5>>({0, 0, 0, 0, 0},
                                             {problem.nblocks, problem.nvars, nz_mem, ny_mem,
                                              nx_mem}),
      KOKKOS_LAMBDA(const int b, const int v, const int z, const int y, const int x) {
        const double seed = static_cast<double>(1 + x + 17 * y + 31 * z + 101 * v + 1009 * b);
        in(b, v, z, y, x) = 0.25 + 0.001 * seed;
        aux(b, v, z, y, x) = 0.75 + 0.002 * seed;
        out(b, v, z, y, x) = 0.0;
      });
  Kokkos::fence();
}

template <typename Body>
void RunCpuFlatGhosts(const Dataset &dataset, Body body) {
  const auto &spec = dataset.problem;
  const auto &data = dataset.data;
  const int nz_mem = spec.nz_interior + 2 * spec.nghost;
  const int ny_mem = spec.ny_interior + 2 * spec.nghost;
  const int nx_mem = spec.nx_interior + 2 * spec.nghost;

  for (int b = 0; b < spec.nblocks; ++b) {
    for (int v = 0; v < spec.nvars; ++v) {
      for (int z = 0; z < nz_mem; ++z) {
        for (int y = 0; y < ny_mem; ++y) {
#pragma omp simd
          for (int x = 0; x < nx_mem; ++x) {
            data.out(b, v, z, y, x) = body(b, v, z, y, x);
          }
        }
      }
    }
  }
}

template <typename Body>
void RunCpuHierarchicalContiguous(const Dataset &dataset, int ninner, Body body) {
  const auto &spec = dataset.problem;
  const auto &data = dataset.data;
  const int nz_mem = spec.nz_interior + 2 * spec.nghost;
  const int ny_mem = spec.ny_interior + 2 * spec.nghost;
  const int nx_mem = spec.nx_interior + 2 * spec.nghost;
  const int cells_per_block = nz_mem * ny_mem * nx_mem;
  const int nouter = (cells_per_block + ninner - 1) / ninner;

  for (int b = 0; b < spec.nblocks; ++b) {
    for (int outer = 0; outer < nouter; ++outer) {
      const int start = outer * ninner;
      const int stop = std::min(start + ninner, cells_per_block);
      for (int v = 0; v < spec.nvars; ++v) {
        const int nvars_block = SelectNvarsForBlock(spec, b);
        if (v >= nvars_block) {
          continue;
        }
        const double *const in = &data.in(b, v, 0, 0, 0);
        const double *const aux = &data.aux(b, v, 0, 0, 0);
        double *const out = &data.out(b, v, 0, 0, 0);
#pragma omp simd
        for (int idx = start; idx < stop; ++idx) {
          out[idx] = body(in, aux, idx);
        }
      }
    }
  }
}

template <typename Body>
void RunCpuHierarchicalLogical(const Dataset &dataset, int ninner, Body body) {
  const auto &spec = dataset.problem;
  const auto &data = dataset.data;
  const int z0 = spec.nghost;
  const int y0 = spec.nghost;
  const int x0 = spec.nghost;
  const int cells_per_block = spec.nz_interior * spec.ny_interior * spec.nx_interior;
  const int nouter = (cells_per_block + ninner - 1) / ninner;

  for (int b = 0; b < spec.nblocks; ++b) {
    const int nvars_block = SelectNvarsForBlock(spec, b);
    for (int outer = 0; outer < nouter; ++outer) {
      const int start = outer * ninner;
      const int stop = std::min(start + ninner, cells_per_block);
      for (int v = 0; v < nvars_block; ++v) {
#pragma omp simd
        for (int idx = start; idx < stop; ++idx) {
          const int z = z0 + idx / (spec.ny_interior * spec.nx_interior);
          const int rem = idx % (spec.ny_interior * spec.nx_interior);
          const int y = y0 + rem / spec.nx_interior;
          const int x = x0 + rem % spec.nx_interior;
          data.out(b, v, z, y, x) = body(b, v, z, y, x);
        }
      }
    }
  }
}

template <typename Body>
void RunKokkosBoivFlat(const Dataset &dataset, Body body) {
  const auto &spec = dataset.problem;
  const auto &data = dataset.data;
  const int nz_mem = spec.nz_interior + 2 * spec.nghost;
  const int ny_mem = spec.ny_interior + 2 * spec.nghost;
  const int nx_mem = spec.nx_interior + 2 * spec.nghost;
  const std::int64_t total = static_cast<std::int64_t>(spec.nblocks) * nz_mem * ny_mem * nx_mem;

  Kokkos::parallel_for(
      "KokkosBoivFlat", Kokkos::RangePolicy<>(0, total), KOKKOS_LAMBDA(const std::int64_t flat) {
        const int x = static_cast<int>(flat % nx_mem);
        const int y = static_cast<int>((flat / nx_mem) % ny_mem);
        const int z = static_cast<int>((flat / (nx_mem * ny_mem)) % nz_mem);
        const int b = static_cast<int>(flat / (nx_mem * ny_mem * nz_mem));
        for (int v = 0; v < SelectNvarsForBlock(spec, b); ++v) {
          data.out(b, v, z, y, x) = body(b, v, z, y, x);
        }
      });
}

template <typename Body>
void RunKokkosBoviTeamContiguous(const Dataset &dataset, int ninner, Body body) {
  const auto &spec = dataset.problem;
  const auto &data = dataset.data;
  const int nz_mem = spec.nz_interior + 2 * spec.nghost;
  const int ny_mem = spec.ny_interior + 2 * spec.nghost;
  const int nx_mem = spec.nx_interior + 2 * spec.nghost;
  const int cells_per_block = nz_mem * ny_mem * nx_mem;
  const int nouter = (cells_per_block + ninner - 1) / ninner;
  const int league_size = spec.nblocks * nouter;
  const TeamPolicy policy(league_size, Kokkos::AUTO);

  Kokkos::parallel_for(
      "KokkosBoviTeamContiguous", policy, KOKKOS_LAMBDA(const TeamMember &member) {
        const int league = member.league_rank();
        const int b = league / nouter;
        const int outer = league % nouter;
        const int start = outer * ninner;
        const int stop = std::min(start + ninner, cells_per_block);
        const int nvars_block = SelectNvarsForBlock(spec, b);

        for (int v = 0; v < nvars_block; ++v) {
          const double *const in = &data.in(b, v, 0, 0, 0);
          const double *const aux = &data.aux(b, v, 0, 0, 0);
          double *const out = &data.out(b, v, 0, 0, 0);
          Kokkos::parallel_for(Kokkos::TeamThreadRange(member, start, stop), [&](const int idx) {
            out[idx] = body(in, aux, idx);
          });
          member.team_barrier();
        }
      });
}

template <typename Body>
void RunKokkosBoviTeamLogical(const Dataset &dataset, int ninner, Body body) {
  const auto &spec = dataset.problem;
  const auto &data = dataset.data;
  const int cells_per_block = spec.nz_interior * spec.ny_interior * spec.nx_interior;
  const int nouter = (cells_per_block + ninner - 1) / ninner;
  const int league_size = spec.nblocks * nouter;
  const TeamPolicy policy(league_size, Kokkos::AUTO);
  const int z0 = spec.nghost;
  const int y0 = spec.nghost;
  const int x0 = spec.nghost;

  Kokkos::parallel_for(
      "KokkosBoviTeamLogical", policy, KOKKOS_LAMBDA(const TeamMember &member) {
        const int league = member.league_rank();
        const int b = league / nouter;
        const int outer = league % nouter;
        const int start = outer * ninner;
        const int stop = std::min(start + ninner, cells_per_block);
        const int nvars_block = SelectNvarsForBlock(spec, b);

        for (int v = 0; v < nvars_block; ++v) {
          Kokkos::parallel_for(Kokkos::TeamThreadRange(member, start, stop), [&](const int idx) {
            const int z = z0 + idx / (spec.ny_interior * spec.nx_interior);
            const int rem = idx % (spec.ny_interior * spec.nx_interior);
            const int y = y0 + rem / spec.nx_interior;
            const int x = x0 + rem % spec.nx_interior;
            body(data, b, v, z, y, x);
          });
          member.team_barrier();
        }
      });
}

template <int NITER, int SX, int SY, int SZ>
BenchmarkRow RunTypedCase(const CaseSpec &spec, const Dataset &dataset) {
  const auto alpha = MakeAlpha<NITER>();
  const auto beta = MakeBeta<NITER>();
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

  auto run_body_hoisted = [&](const double *in, const double *aux, int idx) {
    const double *const x_ptrs[SX] = {in + idx + 1};
    const double *const y_ptrs[SY] = {in + idx + spec.problem.nx_interior};
    const double *const z_ptrs[SZ] = {in + idx + spec.problem.nx_interior * spec.problem.ny_interior};
    std::array<const double *, SX> x{};
    std::array<const double *, SY> y{};
    std::array<const double *, SZ> z{};
    for (int i = 0; i < SX; ++i) x[i] = x_ptrs[i];
    for (int i = 0; i < SY; ++i) y[i] = y_ptrs[i];
    for (int i = 0; i < SZ; ++i) z[i] = z_ptrs[i];
    return ComputeUnifiedCellHoisted<NITER, SX, SY, SZ>(in[idx] + aux[idx], x, y, z, alpha, beta);
  };

  auto run_body_direct = [&](const LoopData &data, int b, int v, int z, int y, int x) {
    return ComputeUnifiedCellDirect<NITER, SX, SY, SZ>(data.in, b, v, z, y, x, dx, dy, dz, alpha,
                                                       beta);
  };

  const auto start = std::chrono::steady_clock::now();
  switch (spec.loop.kind) {
    case LoopKind::CpuFlatGhosts:
      RunCpuFlatGhosts(dataset, [&](int b, int v, int z, int y, int x) {
        return ComputeUnifiedCellDirect<NITER, SX, SY, SZ>(dataset.data.in, b, v, z, y, x, dx,
                                                           dy, dz, alpha, beta);
      });
      break;
    case LoopKind::CpuBoivContiguous:
    case LoopKind::CpuBoivLogical:
      RunCpuHierarchicalContiguous(dataset, SelectNinner(spec), [&](const double *in,
                                                                    const double *aux, int idx) {
        return ComputeUnifiedCellHoisted<NITER, SX, SY, SZ>(in[idx] + aux[idx], {}, {}, {}, alpha,
                                                             beta);
      });
      break;
    case LoopKind::CpuBoviContiguous:
    case LoopKind::CpuBoviLogical:
      RunCpuHierarchicalContiguous(dataset, SelectNinner(spec), [&](const double *in,
                                                                    const double *aux, int idx) {
        return ComputeUnifiedCellHoisted<NITER, SX, SY, SZ>(in[idx] + aux[idx], {}, {}, {}, alpha,
                                                             beta);
      });
      break;
    case LoopKind::KokkosBoivFlat:
      RunKokkosBoivFlat(dataset, [&](int b, int v, int z, int y, int x) {
        return ComputeUnifiedCellDirect<NITER, SX, SY, SZ>(dataset.data.in, b, v, z, y, x, dx,
                                                           dy, dz, alpha, beta);
      });
      break;
    case LoopKind::KokkosBoviTeamContiguous:
      RunKokkosBoviTeamContiguous(dataset, SelectNinner(spec), run_body_hoisted);
      break;
    case LoopKind::KokkosBoviTeamLogical:
      RunKokkosBoviTeamLogical(dataset, SelectNinner(spec), run_body_direct);
      break;
  }
  Kokkos::fence();
  const auto stop = std::chrono::steady_clock::now();

  const std::chrono::duration<double> elapsed = stop - start;
  BenchmarkRow row;
  row.loop_name = ToString(spec.loop.kind);
  row.backend = spec.backend;
  row.nblocks = spec.problem.nblocks;
  row.nvars = spec.problem.nvars;
  row.nz_interior = spec.problem.nz_interior;
  row.ny_interior = spec.problem.ny_interior;
  row.nx_interior = spec.problem.nx_interior;
  row.nghost = spec.problem.nghost;
  row.ninner = SelectNinner(spec);
  row.niter = spec.kernel.niter;
  row.stencil_x = spec.kernel.stencil_x;
  row.stencil_y = spec.kernel.stencil_y;
  row.stencil_z = spec.kernel.stencil_z;
  row.total_updates = CountUpdates(spec);
  row.min_seconds = elapsed.count();
  row.updates_per_second = static_cast<double>(row.total_updates) / row.min_seconds;
  return row;
}

template <int NITER, int SX, int SY, int SZ>
BenchmarkRow DispatchKernel(const CaseSpec &spec, const Dataset &dataset) {
  return RunTypedCase<NITER, SX, SY, SZ>(spec, dataset);
}

BenchmarkRow DispatchByKernelSpec(const CaseSpec &spec, const Dataset &dataset) {
  switch (spec.kernel.niter) {
    case 0:
      return DispatchKernel<0, 1, 1, 1>(spec, dataset);
    case 4:
      return DispatchKernel<4, 1, 1, 1>(spec, dataset);
    case 8:
      return DispatchKernel<8, 1, 1, 1>(spec, dataset);
    default:
      return DispatchKernel<4, 1, 1, 1>(spec, dataset);
  }
}

}  // namespace

Dataset BuildDataset(const CaseSpec &spec) {
  Dataset dataset;
  dataset.problem = spec.problem;
  dataset.interior_z = MakeInteriorRange(spec.problem.nz_interior);
  dataset.interior_y = MakeInteriorRange(spec.problem.ny_interior);
  dataset.interior_x = MakeInteriorRange(spec.problem.nx_interior);
  dataset.memory_z = MakeMemoryRange(spec.problem.nz_interior, spec.problem.nghost);
  dataset.memory_y = MakeMemoryRange(spec.problem.ny_interior, spec.problem.nghost);
  dataset.memory_x = MakeMemoryRange(spec.problem.nx_interior, spec.problem.nghost);

  const int nz_mem = spec.problem.nz_interior + 2 * spec.problem.nghost;
  const int ny_mem = spec.problem.ny_interior + 2 * spec.problem.nghost;
  const int nx_mem = spec.problem.nx_interior + 2 * spec.problem.nghost;
  dataset.data.in = View5D("in", spec.problem.nblocks, spec.problem.nvars, nz_mem, ny_mem, nx_mem);
  dataset.data.aux = View5D("aux", spec.problem.nblocks, spec.problem.nvars, nz_mem, ny_mem, nx_mem);
  dataset.data.out = View5D("out", spec.problem.nblocks, spec.problem.nvars, nz_mem, ny_mem, nx_mem);
  dataset.data.active_counts = Kokkos::View<int *>("active_counts", spec.problem.nblocks);
  return dataset;
}

void PrepareDataset(const CaseSpec &spec, Dataset *dataset) {
  auto host = Kokkos::create_mirror_view(dataset->data.active_counts);
  for (int b = 0; b < spec.problem.nblocks; ++b) {
    host(b) = SelectNvarsForBlock(spec, b);
  }
  Kokkos::deep_copy(dataset->data.active_counts, host);
  InitializeDataViews(spec, dataset->data.in, dataset->data.aux, dataset->data.out);
}

std::uint64_t CountUpdates(const CaseSpec &spec) {
  const std::uint64_t cells_per_block =
      static_cast<std::uint64_t>(spec.problem.nz_interior) *
      static_cast<std::uint64_t>(spec.problem.ny_interior) *
      static_cast<std::uint64_t>(spec.problem.nx_interior);
  std::uint64_t updates = 0;
  for (int b = 0; b < spec.problem.nblocks; ++b) {
    updates += cells_per_block * static_cast<std::uint64_t>(SelectNvarsForBlock(spec, b));
  }
  return updates;
}

BenchmarkRow RunCase(const CaseSpec &spec) {
  Dataset dataset = BuildDataset(spec);
  PrepareDataset(spec, &dataset);
  for (int i = 0; i < spec.warmup; ++i) {
    (void)DispatchByKernelSpec(spec, dataset);
  }
  BenchmarkRow row = DispatchByKernelSpec(spec, dataset);
  return row;
}

}  // namespace plb2
