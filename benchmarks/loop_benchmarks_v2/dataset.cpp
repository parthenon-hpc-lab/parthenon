#include "dataset.hpp"

#include <algorithm>
#include <cmath>

#include <Kokkos_Core.hpp>

namespace plb2 {

namespace {

int SelectNvarsForBlock(const CaseSpec &spec, int block) {
  if (block >= 0 && block < static_cast<int>(spec.problem.vars_per_block.size())) {
    return std::min(spec.problem.vars_per_block[block], spec.problem.nvars);
  }
  return spec.problem.nvars;
}

void NormalizeProblemSpec(ProblemSpec *problem) {
  if (problem == nullptr) {
    return;
  }

  if (problem->nblocks <= 0 && problem->target_cells > 0) {
    const std::uint64_t cells_per_block =
        static_cast<std::uint64_t>(problem->nz_interior) *
        static_cast<std::uint64_t>(problem->ny_interior) *
        static_cast<std::uint64_t>(problem->nx_interior);
    if (cells_per_block > 0) {
      const auto derived =
          static_cast<std::uint64_t>(std::llround(static_cast<double>(problem->target_cells) /
                                                  static_cast<double>(cells_per_block)));
      problem->nblocks = static_cast<int>(std::max<std::uint64_t>(1, derived));
    }
  }
  if (problem->nblocks <= 0) {
    problem->nblocks = 1;
  }
  if (problem->vars_per_block.empty()) {
    problem->vars_per_block.assign(problem->nblocks, problem->nvars);
  }
  if (static_cast<int>(problem->vars_per_block.size()) < problem->nblocks) {
    problem->vars_per_block.resize(problem->nblocks, problem->nvars);
  }
}

void InitializeDataViews(const CaseSpec &spec, const LoopData &data) {
  const auto memory_indexer = spec.problem.memory_indexer;
  const int nmem = static_cast<int>(memory_indexer.size());

  for (int b = 0; b < spec.problem.nblocks; ++b) {
    for (int v = 0; v < spec.problem.nvars; ++v) {
#pragma omp simd
      for (int flat = 0; flat < nmem; ++flat) {
        const auto [k, j, i] = memory_indexer(flat);
        const double seed = static_cast<double>(1 + i + 17 * j + 31 * k + 101 * v + 1009 * b);
        data.in(b, v, k, j, i) = 0.25 + 0.001 * seed;
        data.aux(b, v, k, j, i) = 0.75 + 0.002 * seed;
        data.out(b, v, k, j, i) = 0.0;
      }
    }
  }
}

}  // namespace

void NormalizeCaseSpec(CaseSpec *spec) {
  if (spec == nullptr) {
    return;
  }
  NormalizeProblemSpec(&spec->problem);
}

Dataset BuildDataset(const CaseSpec &spec) {
  Dataset dataset;
  dataset.problem = spec.problem;
  NormalizeProblemSpec(&dataset.problem);

  const auto logical_k = parthenon::IndexRange{spec.problem.nghost,
                                                spec.problem.nghost + spec.problem.nz_interior - 1};
  const auto logical_j = parthenon::IndexRange{spec.problem.nghost,
                                                spec.problem.nghost + spec.problem.ny_interior - 1};
  const auto logical_i = parthenon::IndexRange{spec.problem.nghost,
                                                spec.problem.nghost + spec.problem.nx_interior - 1};
  const auto memory_k = parthenon::IndexRange{0, spec.problem.nz_interior + 2 * spec.problem.nghost - 1};
  const auto memory_j = parthenon::IndexRange{0, spec.problem.ny_interior + 2 * spec.problem.nghost - 1};
  const auto memory_i = parthenon::IndexRange{0, spec.problem.nx_interior + 2 * spec.problem.nghost - 1};

  dataset.problem.logical_indexer =
      parthenon::Indexer3D(logical_k, logical_j, logical_i);
  dataset.problem.memory_indexer =
      parthenon::Indexer3D(memory_k, memory_j, memory_i);

  const int nk_mem = memory_k.size();
  const int nj_mem = memory_j.size();
  const int ni_mem = memory_i.size();
  dataset.data.in = View5D("in", spec.problem.nblocks, spec.problem.nvars, nk_mem, nj_mem, ni_mem);
  dataset.data.aux = View5D("aux", spec.problem.nblocks, spec.problem.nvars, nk_mem, nj_mem, ni_mem);
  dataset.data.out = View5D("out", spec.problem.nblocks, spec.problem.nvars, nk_mem, nj_mem, ni_mem);
  dataset.data.active_counts = Kokkos::View<int *>("active_counts", spec.problem.nblocks);
  return dataset;
}

void PrepareDataset(const CaseSpec &spec, Dataset *dataset) {
  auto host = Kokkos::create_mirror_view(dataset->data.active_counts);
  for (int b = 0; b < spec.problem.nblocks; ++b) {
    host(b) = SelectNvarsForBlock(spec, b);
  }
  Kokkos::deep_copy(dataset->data.active_counts, host);
  InitializeDataViews(spec, dataset->data);
}

std::uint64_t CountUpdates(const CaseSpec &spec, const Dataset &dataset) {
  const auto &problem = dataset.problem;
  if (problem.nblocks <= 0) {
    return 0;
  }

  const std::uint64_t per_block_cells = (spec.loop.kind == LoopKind::CpuFlatGhosts)
                                            ? static_cast<std::uint64_t>(problem.memory_indexer.size())
                                            : static_cast<std::uint64_t>(problem.logical_indexer.size());

  std::uint64_t updates = 0;
  for (int b = 0; b < problem.nblocks; ++b) {
    const int nvars = (b >= 0 && b < static_cast<int>(spec.problem.vars_per_block.size()))
                          ? std::min(spec.problem.vars_per_block[b], spec.problem.nvars)
                          : spec.problem.nvars;
    updates += per_block_cells * static_cast<std::uint64_t>(nvars);
  }
  return updates;
}

}  // namespace plb2
