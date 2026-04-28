#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

namespace parthenon {

struct IndexRange {
  int s = 0;
  int e = 0;
  KOKKOS_INLINE_FUNCTION int size() const { return e - s + 1; }
};

}  // namespace parthenon

#include "utils/indexer.hpp"

namespace plb2 {

enum class LoopKind {
  CpuFlatGhosts,
  CpuBoivContiguous,
  CpuBoivLogical,
  CpuBoviContiguous,
  CpuBoviLogical,
  CpuBvoiContiguous,
  CpuBvoiLogical,
  KokkosBoivFlat,
  KokkosBoviTeamContiguous,
  KokkosBoviTeamLogical,
};

struct ProblemSpec {
  int nblocks = 1;
  int nvars = 1;
  int nz_interior = 1;
  int ny_interior = 1;
  int nx_interior = 1;
  int nghost = 0;
  std::vector<int> vars_per_block;
  std::uint64_t target_cells = 0;
  parthenon::Indexer3D memory_indexer;
  parthenon::Indexer3D logical_indexer;
};

struct LoopSpec {
  LoopKind kind = LoopKind::CpuBoivContiguous;
  int ninner = -1;
};

struct KernelSpec {
  int niter = 0;
  int stencil_x = 1;
  int stencil_y = 1;
  int stencil_z = 1;
};

struct CaseSpec {
  ProblemSpec problem;
  LoopSpec loop;
  KernelSpec kernel;
  std::string backend = "Serial";
  int repeats = 5;
  int warmup = 1;
};

struct BenchmarkRow {
  std::string loop_name;
  std::string backend;
  int nblocks = 0;
  int nvars = 0;
  int nz_interior = 0;
  int ny_interior = 0;
  int nx_interior = 0;
  int nghost = 0;
  int ninner = 0;
  int niter = 0;
  int stencil_x = 0;
  int stencil_y = 0;
  int stencil_z = 0;
  int warmup = 0;
  int repeats = 0;
  std::uint64_t total_updates = 0;
  double avg_seconds = 0.0;
  double min_seconds = 0.0;
  double updates_per_second = 0.0;
};

LoopKind ParseLoopKind(const std::string &text);
std::string ToString(LoopKind kind);

}  // namespace plb2
