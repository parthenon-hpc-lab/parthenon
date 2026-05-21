#include "benchmark_driver.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

#include "loop_patterns.hpp"
#include "output.hpp"
#include "ragged_metadata.hpp"

namespace plb {

namespace {

double Median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  const std::size_t mid = samples.size() / 2;
  if (samples.size() % 2 == 0) {
    return 0.5 * (samples[mid - 1] + samples[mid]);
  }
  return samples[mid];
}

BenchmarkRow Execute(const BenchmarkConfig &config) {
  Dataset dataset = BuildDataset(config);
  RaggedMetadata metadata = BuildRaggedMetadata(config.blocks, config.variables,
                                                config.active_min, config.active_max);
  PrepareDataset(config, metadata, &dataset);
  auto run_once = [&]() { ExecuteLoopPattern(config, metadata, &dataset); };

  for (int i = 0; i < config.warmup; ++i) {
    run_once();
  }

  std::vector<double> samples;
  samples.reserve(config.repeats);
  for (int repeat = 0; repeat < config.repeats; ++repeat) {
    const auto start = std::chrono::steady_clock::now();
    run_once();
    const auto stop = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = stop - start;
    samples.push_back(elapsed.count());
  }

  const std::uint64_t total_updates = CountUpdates(config, metadata);
  const int effective_active_min = config.ragged
                                       ? *std::min_element(metadata.active_counts.begin(),
                                                           metadata.active_counts.end())
                                       : config.variables;
  const int effective_active_max = config.ragged
                                       ? *std::max_element(metadata.active_counts.begin(),
                                                           metadata.active_counts.end())
                                       : config.variables;
  const double min_seconds = *std::min_element(samples.begin(), samples.end());
  const double mean_seconds = std::accumulate(samples.begin(), samples.end(), 0.0) /
                              static_cast<double>(samples.size());
  const double median_seconds = Median(samples);
  const double updates_per_second = static_cast<double>(total_updates) / min_seconds;
  const double bytes_per_update = EstimatedBytesPerUpdate(config.kernel);
  const double flops_per_update =
      EstimatedFlopsPerUpdate(config.kernel, config.heavy_iterations);
  const double bandwidth = (updates_per_second * bytes_per_update) / 1.0e9;
  const double arithmetic_intensity = flops_per_update / bytes_per_update;

  BenchmarkRow row;
  row.backend = config.backend;
  row.variant = ToString(config.variant);
  row.kernel = ToString(config.kernel);
  row.ragged = config.ragged;
  row.blocks = config.blocks;
  row.variables = config.variables;
  row.nk = config.nk;
  row.nj = config.nj;
  row.ni = config.ni;
  row.ghost_zones = config.ghost_zones;
  row.active_min = effective_active_min;
  row.active_max = effective_active_max;
  row.inner_chunk_length = EffectiveInnerChunkLength(config);
  row.team_size_mode = config.team_size_mode;
  row.explicit_team_size = config.explicit_team_size;
  row.heavy_iterations = config.heavy_iterations;
  row.repeats = config.repeats;
  row.min_seconds = min_seconds;
  row.median_seconds = median_seconds;
  row.mean_seconds = mean_seconds;
  row.updates_per_second = updates_per_second;
  row.estimated_bandwidth_gb_s = bandwidth;
  row.estimated_flops_per_update = flops_per_update;
  row.arithmetic_intensity_flops_per_byte = arithmetic_intensity;
  row.total_updates = total_updates;
  return row;
}

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

} // namespace

std::string ToString(KernelKind kind) {
  switch (kind) {
  case KernelKind::Light:
    return "light";
  case KernelKind::Flux:
    return "flux";
  case KernelKind::Stencil:
    return "stencil";
  case KernelKind::Heavy:
    return "heavy";
  }
  return "unknown";
}

std::string ToString(VariantKind kind) {
  switch (kind) {
  case VariantKind::KokkosFlatKJI:
    return "kokkos_flat_kji";
  case VariantKind::KokkosMDRangeKJI:
    return "kokkos_mdrange_kji";
  case VariantKind::KokkosDenseFlatBVKJI:
    return "kokkos_dense_flat_bvkji";
  case VariantKind::KokkosRawspanOVI:
    return "kokkos_rawspan_ovi";
  case VariantKind::KokkosRawspanViewOVI:
    return "kokkos_rawspan_view_ovi";
  case VariantKind::KokkosLogicalOVI:
    return "kokkos_logical_ovi";
  case VariantKind::CpuDenseFlatBVKJI:
    return "cpu_dense_flat_bvkji";
  case VariantKind::CpuLogicalKJI:
    return "cpu_logical_kji";
  case VariantKind::CpuRawspanOVI:
    return "cpu_rawspan_ovi";
  case VariantKind::CpuRawspanVOI:
    return "cpu_rawspan_voi";
  case VariantKind::CpuLogicalOVI:
    return "cpu_logical_ovi";
  }
  return "unknown";
}

bool ParseKernelKind(const std::string &text, KernelKind *kind) {
  if (text == "light") {
    *kind = KernelKind::Light;
    return true;
  }
  if (text == "flux") {
    *kind = KernelKind::Flux;
    return true;
  }
  if (text == "stencil") {
    *kind = KernelKind::Stencil;
    return true;
  }
  if (text == "heavy") {
    *kind = KernelKind::Heavy;
    return true;
  }
  return false;
}

bool ParseVariantKind(const std::string &text, VariantKind *kind) {
  if (text == "kokkos_flat_kji" || text == "flat") {
    *kind = VariantKind::KokkosFlatKJI;
    return true;
  }
  if (text == "kokkos_mdrange_kji" || text == "mdrange") {
    *kind = VariantKind::KokkosMDRangeKJI;
    return true;
  }
  if (text == "kokkos_dense_flat_bvkji" || text == "dense_flat" ||
      text == "dense_kokkos") {
    *kind = VariantKind::KokkosDenseFlatBVKJI;
    return true;
  }
  if (text == "kokkos_rawspan_ovi" || text == "hierarchical") {
    *kind = VariantKind::KokkosRawspanOVI;
    return true;
  }
  if (text == "kokkos_rawspan_view_ovi" || text == "rawspan_view") {
    *kind = VariantKind::KokkosRawspanViewOVI;
    return true;
  }
  if (text == "kokkos_logical_ovi") {
    *kind = VariantKind::KokkosLogicalOVI;
    return true;
  }
  if (text == "tuned") {
    *kind = VariantKind::KokkosRawspanOVI;
    return true;
  }
  if (text == "cpu_logical_kji" || text == "cpu_simd") {
    *kind = VariantKind::CpuLogicalKJI;
    return true;
  }
  if (text == "cpu_dense_flat_bvkji" || text == "cpu_dense" || text == "dense_cpu") {
    *kind = VariantKind::CpuDenseFlatBVKJI;
    return true;
  }
  if (text == "cpu_rawspan_ovi" || text == "cpu_hierarchical") {
    *kind = VariantKind::CpuRawspanOVI;
    return true;
  }
  if (text == "cpu_rawspan_voi" || text == "cpu_coalesced_outer_var") {
    *kind = VariantKind::CpuRawspanVOI;
    return true;
  }
  if (text == "cpu_logical_ovi") {
    *kind = VariantKind::CpuLogicalOVI;
    return true;
  }
  return false;
}

std::string Usage() {
  return "Usage: parthenon_loop_bench [options]\n"
         "  --kernel {light|flux|stencil|heavy}\n"
         "  --variant "
         "{kokkos_flat_kji|kokkos_mdrange_kji|kokkos_dense_flat_bvkji|kokkos_rawspan_ovi|"
         "kokkos_rawspan_view_ovi|kokkos_logical_ovi|cpu_dense_flat_bvkji|cpu_logical_"
         "kji|cpu_rawspan_ovi|cpu_rawspan_voi|cpu_logical_ovi}\n"
         "  --backend NAME\n"
         "  --blocks N --vars N --nk N --nj N --ni N\n"
         "  --ghosts N\n"
         "  --repeats N --warmup N\n"
         "  --ragged {on|off} --active-min N --active-max N\n"
         "  --inner-chunk-length N\n"
         "  --team-size {auto|explicit}\n"
         "  --explicit-team-size N\n"
         "  --heavy-iterations N\n"
         "  --csv PATH\n";
}

bool ParseArgs(int argc, char **argv, BenchmarkConfig *config, std::string *error) {
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
    } else if (arg == "--kernel") {
      const char *value = require_value(arg);
      if (value == nullptr || !ParseKernelKind(value, &config->kernel)) {
        if (error != nullptr) {
          *error = "invalid kernel";
        }
        return false;
      }
    } else if (arg == "--variant") {
      const char *value = require_value(arg);
      if (value == nullptr || !ParseVariantKind(value, &config->variant)) {
        if (error != nullptr) {
          *error = "invalid variant";
        }
        return false;
      }
    } else if (arg == "--backend") {
      const char *value = require_value(arg);
      if (value == nullptr) {
        return false;
      }
      config->backend = value;
    } else if (arg == "--blocks") {
      const char *value = require_value(arg);
      if (value == nullptr || !ParseIntArg(value, &config->blocks)) {
        if (error != nullptr) {
          *error = "invalid blocks value";
        }
        return false;
      }
    } else if (arg == "--vars") {
      const char *value = require_value(arg);
      if (value == nullptr || !ParseIntArg(value, &config->variables)) {
        if (error != nullptr) {
          *error = "invalid vars value";
        }
        return false;
      }
    } else if (arg == "--nk") {
      const char *value = require_value(arg);
      if (value == nullptr || !ParseIntArg(value, &config->nk)) {
        if (error != nullptr) {
          *error = "invalid nk value";
        }
        return false;
      }
    } else if (arg == "--nj") {
      const char *value = require_value(arg);
      if (value == nullptr || !ParseIntArg(value, &config->nj)) {
        if (error != nullptr) {
          *error = "invalid nj value";
        }
        return false;
      }
    } else if (arg == "--ni") {
      const char *value = require_value(arg);
      if (value == nullptr || !ParseIntArg(value, &config->ni)) {
        if (error != nullptr) {
          *error = "invalid ni value";
        }
        return false;
      }
    } else if (arg == "--ghosts") {
      const char *value = require_value(arg);
      if (value == nullptr || !ParseIntArg(value, &config->ghost_zones)) {
        if (error != nullptr) {
          *error = "invalid ghosts value";
        }
        return false;
      }
    } else if (arg == "--repeats") {
      const char *value = require_value(arg);
      if (value == nullptr || !ParseIntArg(value, &config->repeats)) {
        if (error != nullptr) {
          *error = "invalid repeats value";
        }
        return false;
      }
    } else if (arg == "--warmup") {
      const char *value = require_value(arg);
      if (value == nullptr || !ParseIntArg(value, &config->warmup)) {
        if (error != nullptr) {
          *error = "invalid warmup value";
        }
        return false;
      }
    } else if (arg == "--active-min") {
      const char *value = require_value(arg);
      if (value == nullptr || !ParseIntArg(value, &config->active_min)) {
        if (error != nullptr) {
          *error = "invalid active-min value";
        }
        return false;
      }
    } else if (arg == "--active-max") {
      const char *value = require_value(arg);
      if (value == nullptr || !ParseIntArg(value, &config->active_max)) {
        if (error != nullptr) {
          *error = "invalid active-max value";
        }
        return false;
      }
    } else if (arg == "--inner-chunk-length") {
      const char *value = require_value(arg);
      if (value == nullptr || !ParseIntArg(value, &config->inner_chunk_length)) {
        if (error != nullptr) {
          *error = "invalid inner-chunk-length value";
        }
        return false;
      }
    } else if (arg == "--team-size") {
      const char *value = require_value(arg);
      if (value == nullptr) {
        return false;
      }
      config->team_size_mode = value;
    } else if (arg == "--explicit-team-size") {
      const char *value = require_value(arg);
      if (value == nullptr || !ParseIntArg(value, &config->explicit_team_size)) {
        if (error != nullptr) {
          *error = "invalid explicit-team-size value";
        }
        return false;
      }
    } else if (arg == "--heavy-iterations") {
      const char *value = require_value(arg);
      if (value == nullptr || !ParseIntArg(value, &config->heavy_iterations)) {
        if (error != nullptr) {
          *error = "invalid heavy-iterations value";
        }
        return false;
      }
    } else if (arg == "--ragged") {
      const char *value = require_value(arg);
      if (value == nullptr) {
        return false;
      }
      const std::string text = value;
      if (text == "on" || text == "true") {
        config->ragged = true;
      } else if (text == "off" || text == "false") {
        config->ragged = false;
      } else {
        if (error != nullptr) {
          *error = "invalid ragged value";
        }
        return false;
      }
    } else if (arg == "--csv") {
      const char *value = require_value(arg);
      if (value == nullptr) {
        return false;
      }
      config->csv_path = value;
    } else {
      if (error != nullptr) {
        *error = "unknown argument: " + arg;
      }
      return false;
    }
  }

  if (config->blocks <= 0 || config->variables <= 0 || config->nk <= 0 ||
      config->nj <= 0 || config->ni <= 0 || config->ghost_zones < 0 ||
      config->repeats <= 0 || config->warmup < 0) {
    if (error != nullptr) {
      *error = "all sizes must be positive, ghosts must be non-negative, and warmup must "
               "be non-negative";
    }
    return false;
  }
  if (config->active_min <= 0 || config->active_max <= 0 ||
      config->active_min > config->active_max) {
    if (error != nullptr) {
      *error = "ragged active-min/active-max values are invalid";
    }
    return false;
  }
  return true;
}

std::string FormatStdoutSummary(const BenchmarkRow &row) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(6);
  out << "backend=" << row.backend << '\n';
  out << "variant=" << row.variant << '\n';
  out << "kernel=" << row.kernel << '\n';
  out << "ragged=" << (row.ragged ? "true" : "false") << '\n';
  out << "shape=blocks:" << row.blocks << " vars:" << row.variables << " nk:" << row.nk
      << " nj:" << row.nj << " ni:" << row.ni << '\n';
  out << "ghost_zones=" << row.ghost_zones << '\n';
  out << "active_range=" << row.active_min << ".." << row.active_max << '\n';
  out << "inner_chunk_length=" << row.inner_chunk_length << '\n';
  out << "team_size_mode=" << row.team_size_mode << '\n';
  out << "explicit_team_size=" << row.explicit_team_size << '\n';
  out << "heavy_iterations=" << row.heavy_iterations << '\n';
  out << "min_seconds=" << row.min_seconds << '\n';
  out << "median_seconds=" << row.median_seconds << '\n';
  out << "mean_seconds=" << row.mean_seconds << '\n';
  out << "updates_per_second=" << row.updates_per_second << '\n';
  out << "estimated_bandwidth_gb_s=" << row.estimated_bandwidth_gb_s << '\n';
  out << "estimated_flops_per_update=" << row.estimated_flops_per_update << '\n';
  out << "arithmetic_intensity_flops_per_byte=" << row.arithmetic_intensity_flops_per_byte
      << '\n';
  out << "total_updates=" << row.total_updates << '\n';
  return out.str();
}

int RunBenchmark(const BenchmarkConfig &config) {
  try {
    BenchmarkRow row = Execute(config);
    std::string error;
    if (!AppendCsvRow(config.csv_path, row, &error)) {
      std::cerr << error << '\n';
      return 1;
    }
    std::cout << FormatStdoutSummary(row);
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << "benchmark failed: " << ex.what() << '\n';
    return 1;
  }
}

} // namespace plb
