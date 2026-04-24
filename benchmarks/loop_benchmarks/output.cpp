#include "output.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>

namespace plb {

namespace {

std::string CsvEscape(const std::string &value) {
  if (value.find_first_of(",\"\n") == std::string::npos) {
    return value;
  }
  std::string escaped = "\"";
  for (char ch : value) {
    if (ch == '"') {
      escaped += "\"\"";
    } else {
      escaped += ch;
    }
  }
  escaped += "\"";
  return escaped;
}

}  // namespace

bool AppendCsvRow(const std::string &path, const BenchmarkRow &row, std::string *error) {
  namespace fs = std::filesystem;
  if (path.empty()) {
    return true;
  }

  std::error_code ec;
  const bool exists = fs::exists(path, ec);
  if (ec) {
    if (error != nullptr) {
      *error = "failed to inspect CSV path: " + ec.message();
    }
    return false;
  }

  std::ofstream out(path, std::ios::app);
  if (!out) {
    if (error != nullptr) {
      *error = "failed to open CSV output path: " + path;
    }
    return false;
  }

  if (!exists) {
    out << "backend,variant,kernel,ragged,blocks,variables,nk,nj,ni,ghost_zones,active_min,active_max,"
           "inner_chunk_length,team_size_mode,explicit_team_size,heavy_iterations,repeats,min_seconds,"
           "median_seconds,mean_seconds,updates_per_second,estimated_bandwidth_gb_s,"
           "estimated_flops_per_update,arithmetic_intensity_flops_per_byte,total_updates\n";
  }

  out << CsvEscape(row.backend) << ','
      << CsvEscape(row.variant) << ','
      << CsvEscape(row.kernel) << ','
      << (row.ragged ? "true" : "false") << ','
      << row.blocks << ','
      << row.variables << ','
      << row.nk << ','
      << row.nj << ','
      << row.ni << ','
      << row.ghost_zones << ','
      << row.active_min << ','
      << row.active_max << ','
      << row.inner_chunk_length << ','
      << CsvEscape(row.team_size_mode) << ','
      << row.explicit_team_size << ','
      << row.heavy_iterations << ','
      << row.repeats << ','
      << row.min_seconds << ','
      << row.median_seconds << ','
      << row.mean_seconds << ','
      << row.updates_per_second << ','
      << row.estimated_bandwidth_gb_s << ','
      << row.estimated_flops_per_update << ','
      << row.arithmetic_intensity_flops_per_byte << ','
      << row.total_updates << '\n';

  return true;
}

}  // namespace plb
