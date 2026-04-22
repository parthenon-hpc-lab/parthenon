#pragma once

#include <cstddef>
#include <cstdint>

#include <Kokkos_Core.hpp>

#include "benchmark_driver.hpp"
#include "ragged_metadata.hpp"
#include "raw_memory_indexer.hpp"

namespace plb {

using View5D = Kokkos::View<double *****, Kokkos::LayoutRight>;

struct ProblemShape {
  int blocks = 0;
  int variables = 0;
  int nk = 0;
  int nj = 0;
  int ni = 0;
  int ndim = 0;
  int nghost = 0;
  parthenon::IndexRange interior_k;
  parthenon::IndexRange interior_j;
  parthenon::IndexRange interior_i;
  parthenon::IndexRange domain_k;
  parthenon::IndexRange domain_j;
  parthenon::IndexRange domain_i;
  parthenon::IndexRange memory_k;
  parthenon::IndexRange memory_j;
  parthenon::IndexRange memory_i;
  parthenon::Indexer4D cell_indexer;
  RawMemoryIndexer ij_indexer;
  RawMemoryIndexer tuned_indexer;
};

struct LoopData {
  View5D in;
  View5D aux;
  View5D out;
  View5D fx_up;
  View5D fx_lo;
  View5D fy_up;
  View5D fy_lo;
  View5D fz_up;
  View5D fz_lo;
  Kokkos::View<int *> active_counts;
};

struct Dataset {
  ProblemShape problem;
  LoopData data;
};

Dataset BuildDataset(const BenchmarkConfig &config);
void PrepareDataset(const BenchmarkConfig &config, const RaggedMetadata &metadata, Dataset *dataset);
void ExecuteLoopPattern(const BenchmarkConfig &config, const RaggedMetadata &metadata, Dataset *dataset);
std::uint64_t CountUpdates(const BenchmarkConfig &config, const RaggedMetadata &metadata);
int EffectiveInnerChunkLength(const BenchmarkConfig &config);
double EstimatedBytesPerUpdate(KernelKind kind);

}  // namespace plb
