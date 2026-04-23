#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace plb {

enum class KernelKind { Light, Flux, Stencil, Heavy };
enum class VariantKind {
  KokkosFlatKJI,    // Kokkos flat 1D policy over logical active cells
  KokkosMDRangeKJI, // Kokkos MDRange policy over logical active cells
  KokkosRawspanOVI, // Kokkos chunked raw-memory traversal: (outer, v, inner)
  KokkosLogicalOVI, // Kokkos chunked logical traversal: (outer, v, inner)
  CpuLogicalKJI,    // CPU logical traversal: (v, kji)
  CpuRawspanOVI,    // CPU chunked raw-memory traversal: (outer, v, inner)
  CpuRawspanVOI,    // CPU chunked raw-memory traversal: (v, outer, inner)
  CpuLogicalOVI     // CPU chunked logical traversal: (outer, v, inner)
};

struct BenchmarkConfig {
  std::string backend = "Serial";
  KernelKind kernel = KernelKind::Flux;
  VariantKind variant = VariantKind::KokkosFlatKJI;
  int blocks = 8;
  int variables = 8;
  int nk = 8;
  int nj = 16;
  int ni = 32;
  int ghost_zones = 2;
  int repeats = 5;
  int warmup = 1;
  bool ragged = false;
  int active_min = 1;
  int active_max = 8;
  int inner_chunk_length = -1;
  std::string team_size_mode = "auto";
  int explicit_team_size = 0;
  int heavy_iterations = 12;
  std::string csv_path;
};

struct BenchmarkRow {
  std::string backend;
  std::string variant;
  std::string kernel;
  bool ragged = false;
  int blocks = 0;
  int variables = 0;
  int nk = 0;
  int nj = 0;
  int ni = 0;
  int ghost_zones = 0;
  int active_min = 0;
  int active_max = 0;
  int inner_chunk_length = 0;
  std::string team_size_mode;
  int explicit_team_size = 0;
  int repeats = 0;
  double min_seconds = 0.0;
  double median_seconds = 0.0;
  double mean_seconds = 0.0;
  double updates_per_second = 0.0;
  double estimated_bandwidth_gb_s = 0.0;
  std::uint64_t total_updates = 0;
};

int RunBenchmark(const BenchmarkConfig &config);
std::string ToString(KernelKind kind);
std::string ToString(VariantKind kind);
bool ParseKernelKind(const std::string &text, KernelKind *kind);
bool ParseVariantKind(const std::string &text, VariantKind *kind);
std::string Usage();
bool ParseArgs(int argc, char **argv, BenchmarkConfig *config, std::string *error);
std::string FormatStdoutSummary(const BenchmarkRow &row);

}  // namespace plb
