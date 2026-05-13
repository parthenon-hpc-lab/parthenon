#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

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
  LoopAbstractionBoviMemory,
  LoopAbstractionBoviLogical,
  LoopAbstractionBoivLogical,
  LoopAbstractionBoivLogicalDirect,
  LoopAbstractionBvoiMemory,
  LoopAbstractionBvoiLogical,
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
  std::string access_mode = "direct";
};

struct KernelSpec {
  int niter = 0;
  std::vector<int> stencil_x{0};
  std::vector<int> stencil_y{0};
  std::vector<int> stencil_z{0};
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
  std::uint64_t target_cells = 0;
  int nvars = 0;
  int nz_interior = 0;
  int ny_interior = 0;
  int nx_interior = 0;
  int nghost = 0;
  int ninner = 0;
  std::string access_mode;
  int niter = 0;
  std::string stencil_x;
  std::string stencil_y;
  std::string stencil_z;
  std::string kernel_label;
  int warmup = 0;
  int repeats = 0;
  std::uint64_t memory_cells_per_block = 0;
  std::uint64_t logical_cells_per_block = 0;
  std::uint64_t total_updates = 0;
  std::uint64_t touched_cells = 0;
  double avg_seconds = 0.0;
  double min_seconds = 0.0;
  double updates_per_second = 0.0;
  double touched_cells_per_second = 0.0;
};

LoopKind ParseLoopKind(const std::string &text);
std::string ToString(LoopKind kind);
void NormalizeCaseSpec(CaseSpec *spec);
std::string KernelLabel(const CaseSpec &spec);
std::string FormatOffsetSet(const std::vector<int> &offsets);

}  // namespace plb2
