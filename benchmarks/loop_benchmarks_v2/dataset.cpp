#include "dataset.hpp"

#include <algorithm>

#include <Kokkos_Core.hpp>

namespace plb2 {

namespace {

int SelectNvarsForBlock(const CaseSpec &spec, int block) {
  if (block >= 0 && block < static_cast<int>(spec.problem.vars_per_block.size())) {
    return std::min(spec.problem.vars_per_block[block], spec.problem.nvars);
  }
  return spec.problem.nvars;
}

void InitializeDataViews(const CaseSpec &spec, const LoopData &data) {
  const auto memory_indexer = spec.problem.memory_indexer;
  Kokkos::parallel_for(
      "InitializeDataV2",
      Kokkos::RangePolicy<>(0, static_cast<int>(memory_indexer.size())),
      KOKKOS_LAMBDA(const int flat) {
        const auto [b, v, z, y, x] = memory_indexer(flat);
        const double seed = static_cast<double>(1 + x + 17 * y + 31 * z + 101 * v + 1009 * b);
        data.in(b, v, z, y, x) = 0.25 + 0.001 * seed;
        data.aux(b, v, z, y, x) = 0.75 + 0.002 * seed;
        data.out(b, v, z, y, x) = 0.0;
      });
  Kokkos::fence();
}

}  // namespace

Dataset BuildDataset(const CaseSpec &spec) {
  Dataset dataset;
  dataset.problem = spec.problem;

  const auto block_range = parthenon::IndexRange{0, spec.problem.nblocks - 1};
  const auto var_range = parthenon::IndexRange{0, spec.problem.nvars - 1};
  const auto logical_k = parthenon::IndexRange{spec.problem.nghost,
                                                spec.problem.nghost + spec.problem.nz_interior - 1};
  const auto logical_j = parthenon::IndexRange{spec.problem.nghost,
                                                spec.problem.nghost + spec.problem.ny_interior - 1};
  const auto logical_i = parthenon::IndexRange{spec.problem.nghost,
                                                spec.problem.nghost + spec.problem.nx_interior - 1};
  const auto ghost_k =
      parthenon::IndexRange{0, spec.problem.nz_interior + 2 * spec.problem.nghost - 1};
  const auto ghost_j =
      parthenon::IndexRange{0, spec.problem.ny_interior + 2 * spec.problem.nghost - 1};
  const auto ghost_i =
      parthenon::IndexRange{0, spec.problem.nx_interior + 2 * spec.problem.nghost - 1};

  dataset.problem.memory_indexer =
      parthenon::Indexer5D(block_range, var_range, ghost_k, ghost_j, ghost_i);
  dataset.problem.logical_indexer =
      parthenon::Indexer4D(block_range, logical_k, logical_j, logical_i);
  dataset.problem.ghost_indexer =
      parthenon::Indexer4D(block_range, ghost_k, ghost_j, ghost_i);

  const int nk_mem = ghost_k.size();
  const int nj_mem = ghost_j.size();
  const int ni_mem = ghost_i.size();
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
  if (problem.memory_indexer.size() == 0 || problem.logical_indexer.size() == 0) {
    return 0;
  }

  if (spec.loop.kind == LoopKind::CpuFlatGhosts) {
    return static_cast<std::uint64_t>(problem.memory_indexer.size());
  }

  std::uint64_t updates = 0;
  const std::uint64_t cells_per_block =
      static_cast<std::uint64_t>(problem.logical_indexer.size() / problem.nblocks);
  for (int b = 0; b < problem.nblocks; ++b) {
    const int nvars = (b >= 0 && b < static_cast<int>(spec.problem.vars_per_block.size()))
                          ? std::min(spec.problem.vars_per_block[b], spec.problem.nvars)
                          : spec.problem.nvars;
    updates += cells_per_block * static_cast<std::uint64_t>(nvars);
  }
  return updates;
}

}  // namespace plb2
