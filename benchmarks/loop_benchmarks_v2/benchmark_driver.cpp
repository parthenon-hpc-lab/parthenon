#include "benchmark_driver.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>

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

std::vector<int> ParseOffsetArg(std::string value) {
  std::vector<int> offsets;
  for (char &ch : value) {
    if (ch == ';') {
      ch = ',';
    }
  }
  std::stringstream ss(value);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty()) {
      continue;
    }
    int parsed = 0;
    if (ParseIntArg(item, &parsed)) {
      offsets.push_back(parsed);
    }
  }
  return offsets;
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
    case LoopKind::LoopAbstractionBoviMemory:
      return "loop_abstraction_bovi_memory";
    case LoopKind::LoopAbstractionBoviLogical:
      return "loop_abstraction_bovi_logical";
    case LoopKind::LoopAbstractionBoivLogical:
      return "loop_abstraction_boiv_logical";
    case LoopKind::LoopAbstractionBvoiMemory:
      return "loop_abstraction_bvoi_memory";
    case LoopKind::LoopAbstractionBvoiLogical:
      return "loop_abstraction_bvoi_logical";
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
  if (text == "loop_abstraction_bovi_memory") return LoopKind::LoopAbstractionBoviMemory;
  if (text == "loop_abstraction_bovi_logical") return LoopKind::LoopAbstractionBoviLogical;
  if (text == "loop_abstraction_boiv_logical") return LoopKind::LoopAbstractionBoivLogical;
  if (text == "loop_abstraction_bvoi_memory") return LoopKind::LoopAbstractionBvoiMemory;
  if (text == "loop_abstraction_bvoi_logical") return LoopKind::LoopAbstractionBvoiLogical;
  return LoopKind::CpuBoivContiguous;
}

std::string Usage() {
  return
      "Usage: loop-benchmarks-v2 [options]\n"
      "  --loop NAME\n"
      "  --backend NAME\n"
      "  --access-mode direct|hoisted\n"
      "  --nblocks N --target-cells N --nvars N --nz N --ny N --nx N --nghost N\n"
      "  --ninner N\n"
      "  --warmup N --repeats N\n"
      "  --validate\n"
      "  --niter N\n"
      "  --stencil-x OFFSETS --stencil-y OFFSETS --stencil-z OFFSETS\n"
      "    OFFSETS may be a single integer or a comma/semicolon-separated list.\n";
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
    } else if (arg == "--access-mode") {
      const char *value = require_value(arg);
      if (value == nullptr) return false;
      spec->loop.access_mode = value;
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
    } else if (arg == "--target-cells") {
      const char *value = require_value(arg);
      if (value == nullptr) return false;
      try {
        std::size_t pos = 0;
        const std::uint64_t parsed = std::stoull(value, &pos);
        if (pos != std::strlen(value)) return false;
        spec->problem.target_cells = parsed;
      } catch (...) {
        return false;
      }
    } else if (arg == "--ninner") {
      const char *value = require_value(arg);
      if (value == nullptr || !ParseIntArg(value, &spec->loop.ninner)) return false;
    } else if (arg == "--warmup") {
      const char *value = require_value(arg);
      if (value == nullptr || !ParseIntArg(value, &spec->warmup)) return false;
    } else if (arg == "--repeats") {
      const char *value = require_value(arg);
      if (value == nullptr || !ParseIntArg(value, &spec->repeats)) return false;
    } else if (arg == "--validate") {
      spec->validate = true;
    } else if (arg == "--niter") {
      const char *value = require_value(arg);
      if (value == nullptr || !ParseIntArg(value, &spec->kernel.niter)) return false;
    } else if (arg == "--stencil-x") {
      const char *value = require_value(arg);
      if (value == nullptr) return false;
      spec->kernel.stencil_x = ParseOffsetArg(value);
      if (spec->kernel.stencil_x.empty()) return false;
    } else if (arg == "--stencil-y") {
      const char *value = require_value(arg);
      if (value == nullptr) return false;
      spec->kernel.stencil_y = ParseOffsetArg(value);
      if (spec->kernel.stencil_y.empty()) return false;
    } else if (arg == "--stencil-z") {
      const char *value = require_value(arg);
      if (value == nullptr) return false;
      spec->kernel.stencil_z = ParseOffsetArg(value);
      if (spec->kernel.stencil_z.empty()) return false;
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
  if (spec->loop.access_mode.empty()) {
    spec->loop.access_mode =
        (spec->loop.kind == LoopKind::CpuBoviContiguous ||
         spec->loop.kind == LoopKind::CpuBvoiContiguous ||
         spec->loop.kind == LoopKind::KokkosBoviTeamContiguous ||
         spec->loop.kind == LoopKind::LoopAbstractionBoviMemory ||
         spec->loop.kind == LoopKind::LoopAbstractionBoviLogical ||
         spec->loop.kind == LoopKind::LoopAbstractionBoivLogical ||
         spec->loop.kind == LoopKind::LoopAbstractionBvoiMemory ||
         spec->loop.kind == LoopKind::LoopAbstractionBvoiLogical)
            ? "hoisted"
            : "direct";
  }
  NormalizeCaseSpec(spec);
  return true;
}

int RunBenchmark(const CaseSpec &spec) {
  const BenchmarkRow row = RunCase(spec);
  std::cout << row.loop_name << " "
            << "backend=" << row.backend << " "
            << "access_mode=" << row.access_mode << " "
            << "niter=" << row.niter << " "
            << "stencil_x=" << row.stencil_x << " "
            << "stencil_y=" << row.stencil_y << " "
            << "stencil_z=" << row.stencil_z << " "
            << "kernel_label=" << row.kernel_label << " "
            << "warmup=" << row.warmup << " "
            << "repeats=" << row.repeats << " "
            << "avg_seconds=" << row.avg_seconds << " "
            << "min_seconds=" << row.min_seconds << " "
            << "updates_per_second=" << row.updates_per_second << " "
            << "touched_cells_per_second=" << row.touched_cells_per_second << " "
            << "validation_checksum=" << row.validation_checksum << '\n';
  return 0;
}

}  // namespace plb2
