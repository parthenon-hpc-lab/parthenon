#include "benchmark_driver.hpp"

#include <algorithm>
#include <iostream>

#include "runner.hpp"

namespace plb2 {

namespace {

bool ParseIntArg(const std::string &value, int *output) {
  try {
    std::size_t pos = 0;
    const int parsed = std::stoi(value, &pos);
    if (pos != value.size()) {
      return false;
    }
    *output = parsed;
    return true;
  } catch (...) {
    return false;
  }
}

}  // namespace

std::string ToString(LoopKind kind) {
  switch (kind) {
    case LoopKind::CpuFlatGhosts:
      return "cpu_flat_ghosts";
    case LoopKind::CpuBoivContiguous:
      return "cpu_boiv_contiguous";
    case LoopKind::CpuBoivLogical:
      return "cpu_boiv_logical";
    case LoopKind::CpuBoviContiguous:
      return "cpu_bovi_contiguous";
    case LoopKind::CpuBoviLogical:
      return "cpu_bovi_logical";
    case LoopKind::CpuBvoiContiguous:
      return "cpu_bvoi_contiguous";
    case LoopKind::CpuBvoiLogical:
      return "cpu_bvoi_logical";
    case LoopKind::KokkosBoivFlat:
      return "kokkos_boiv_flat";
    case LoopKind::KokkosBoviTeamContiguous:
      return "kokkos_bovi_team_contiguous";
    case LoopKind::KokkosBoviTeamLogical:
      return "kokkos_bovi_team_logical";
  }
  return "unknown";
}

LoopKind ParseLoopKind(const std::string &text) {
  if (text == "cpu_flat_ghosts") return LoopKind::CpuFlatGhosts;
  if (text == "cpu_boiv_contiguous") return LoopKind::CpuBoivContiguous;
  if (text == "cpu_boiv_logical") return LoopKind::CpuBoivLogical;
  if (text == "cpu_bovi_contiguous") return LoopKind::CpuBoviContiguous;
  if (text == "cpu_bovi_logical") return LoopKind::CpuBoviLogical;
  if (text == "cpu_bvoi_contiguous") return LoopKind::CpuBvoiContiguous;
  if (text == "cpu_bvoi_logical") return LoopKind::CpuBvoiLogical;
  if (text == "kokkos_boiv_flat") return LoopKind::KokkosBoivFlat;
  if (text == "kokkos_bovi_team_contiguous") return LoopKind::KokkosBoviTeamContiguous;
  if (text == "kokkos_bovi_team_logical") return LoopKind::KokkosBoviTeamLogical;
  return LoopKind::CpuBoivContiguous;
}

std::string Usage() {
  return
      "Usage: loop-benchmarks-v2 [options]\n"
      "  --loop NAME\n"
      "  --backend NAME\n"
      "  --nblocks N --nvars N --nz N --ny N --nx N --nghost N\n"
      "  --ninner N\n"
      "  --warmup N --repeats N\n"
      "  --niter N\n"
      "  --stencil-x N --stencil-y N --stencil-z N\n";
}

bool ParseArgs(int argc, char **argv, CaseSpec *spec, std::string *error) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const std::string &name) -> const char * {
      if (i + 1 >= argc) {
        if (error != nullptr) {
          *error = "missing value for " + name;
        }
        return nullptr;
      }
      return argv[++i];
    };

    if (arg == "--help" || arg == "-h") {
      if (error != nullptr) {
        *error = Usage();
      }
      return false;
    } else if (arg == "--loop") {
      const char *value = require_value(arg);
      if (value == nullptr) return false;
      spec->loop.kind = ParseLoopKind(value);
    } else if (arg == "--backend") {
      const char *value = require_value(arg);
      if (value == nullptr) return false;
      spec->backend = value;
    } else if (arg == "--nblocks") {
      const char *value = require_value(arg);
      if (value == nullptr || !ParseIntArg(value, &spec->problem.nblocks)) return false;
    } else if (arg == "--nvars") {
      const char *value = require_value(arg);
      if (value == nullptr || !ParseIntArg(value, &spec->problem.nvars)) return false;
    } else if (arg == "--nz") {
      const char *value = require_value(arg);
      if (value == nullptr || !ParseIntArg(value, &spec->problem.nz_interior)) return false;
    } else if (arg == "--ny") {
      const char *value = require_value(arg);
      if (value == nullptr || !ParseIntArg(value, &spec->problem.ny_interior)) return false;
    } else if (arg == "--nx") {
      const char *value = require_value(arg);
      if (value == nullptr || !ParseIntArg(value, &spec->problem.nx_interior)) return false;
    } else if (arg == "--nghost") {
      const char *value = require_value(arg);
      if (value == nullptr || !ParseIntArg(value, &spec->problem.nghost)) return false;
    } else if (arg == "--ninner") {
      const char *value = require_value(arg);
      if (value == nullptr || !ParseIntArg(value, &spec->loop.ninner)) return false;
    } else if (arg == "--warmup") {
      const char *value = require_value(arg);
      if (value == nullptr || !ParseIntArg(value, &spec->warmup)) return false;
    } else if (arg == "--repeats") {
      const char *value = require_value(arg);
      if (value == nullptr || !ParseIntArg(value, &spec->repeats)) return false;
    } else if (arg == "--niter") {
      const char *value = require_value(arg);
      if (value == nullptr || !ParseIntArg(value, &spec->kernel.niter)) return false;
    } else if (arg == "--stencil-x") {
      const char *value = require_value(arg);
      if (value == nullptr || !ParseIntArg(value, &spec->kernel.stencil_x)) return false;
    } else if (arg == "--stencil-y") {
      const char *value = require_value(arg);
      if (value == nullptr || !ParseIntArg(value, &spec->kernel.stencil_y)) return false;
    } else if (arg == "--stencil-z") {
      const char *value = require_value(arg);
      if (value == nullptr || !ParseIntArg(value, &spec->kernel.stencil_z)) return false;
    } else {
      if (error != nullptr) {
        *error = "unknown argument: " + arg;
      }
      return false;
    }
  }

  if (spec->warmup < 0 || spec->repeats < 1) {
    if (error != nullptr) {
      *error = "warmup must be >= 0 and repeats must be >= 1";
    }
    return false;
  }
  if (spec->problem.vars_per_block.empty()) {
    spec->problem.vars_per_block.assign(spec->problem.nblocks, spec->problem.nvars);
  }
  if (static_cast<int>(spec->problem.vars_per_block.size()) < spec->problem.nblocks) {
    spec->problem.vars_per_block.resize(spec->problem.nblocks, spec->problem.nvars);
  }
  return true;
}

int RunBenchmark(const CaseSpec &spec) {
  const BenchmarkRow row = RunCase(spec);
  std::cout << row.loop_name << " "
            << "backend=" << row.backend << " "
            << "warmup=" << row.warmup << " "
            << "repeats=" << row.repeats << " "
            << "avg_seconds=" << row.avg_seconds << " "
            << "min_seconds=" << row.min_seconds << " "
            << "updates_per_second=" << row.updates_per_second << '\n';
  return 0;
}

}  // namespace plb2
