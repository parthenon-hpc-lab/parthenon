#include "multicase.hpp"

#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>

#include "runner.hpp"

namespace plb2 {

namespace {

std::string Trim(const std::string &text) {
  const auto begin = text.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    return "";
  }
  const auto end = text.find_last_not_of(" \t\r\n");
  return text.substr(begin, end - begin + 1);
}

std::vector<std::string> SplitCsvLine(const std::string &line) {
  std::vector<std::string> fields;
  std::string current;
  bool in_quotes = false;
  for (std::size_t i = 0; i < line.size(); ++i) {
    const char ch = line[i];
    if (in_quotes) {
      if (ch == '"' && i + 1 < line.size() && line[i + 1] == '"') {
        current.push_back('"');
        ++i;
      } else if (ch == '"') {
        in_quotes = false;
      } else {
        current.push_back(ch);
      }
    } else if (ch == ',') {
      fields.push_back(current);
      current.clear();
    } else if (ch == '"') {
      in_quotes = true;
    } else {
      current.push_back(ch);
    }
  }
  fields.push_back(current);
  return fields;
}

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

bool ParseUInt64(const std::string &value, std::uint64_t *output) {
  try {
    std::size_t pos = 0;
    const auto parsed = std::stoull(value, &pos);
    if (pos != value.size()) {
      return false;
    }
    *output = parsed;
    return true;
  } catch (...) {
    return false;
  }
}

bool ParseInt(const std::string &value, int *output) {
  try {
    std::size_t pos = 0;
    const auto parsed = std::stoi(value, &pos);
    if (pos != value.size()) {
      return false;
    }
    *output = parsed;
    return true;
  } catch (...) {
    return false;
  }
}

std::vector<int> ParseVarsPerBlock(const std::string &value) {
  std::vector<int> vars;
  std::stringstream ss(value);
  std::string item;
  while (std::getline(ss, item, ';')) {
    item = Trim(item);
    if (item.empty()) {
      continue;
    }
    int parsed = 0;
    if (ParseInt(item, &parsed)) {
      vars.push_back(parsed);
    }
  }
  return vars;
}

std::vector<int> ParseOffsetSet(const std::string &value) {
  std::vector<int> offsets;
  std::string normalized = value;
  for (char &ch : normalized) {
    if (ch == ';') {
      ch = ',';
    }
  }
  std::stringstream ss(normalized);
  std::string item;
  while (std::getline(ss, item, ',')) {
    item = Trim(item);
    if (item.empty()) {
      continue;
    }
    int parsed = 0;
    if (ParseInt(item, &parsed)) {
      offsets.push_back(parsed);
    }
  }
  return offsets;
}

bool WriteResultsCsv(const std::string &results_csv, const BenchmarkRow &row, bool append,
                     std::string *error) {
  namespace fs = std::filesystem;
  if (results_csv.empty()) {
    return true;
  }

  std::error_code ec;
  const bool exists = fs::exists(results_csv, ec);
  if (ec) {
    if (error != nullptr) {
      *error = "failed to inspect results path: " + ec.message();
    }
    return false;
  }

  std::ofstream out(results_csv, std::ios::app);
  if (!out) {
    if (error != nullptr) {
      *error = "failed to open results path: " + results_csv;
    }
    return false;
  }

  if (!exists || !append) {
    out << "loop,backend,nblocks,target_cells,nvars,nz_interior,ny_interior,nx_interior,nghost,"
           "ninner,access_mode,niter,stencil_x,stencil_y,stencil_z,kernel_label,warmup,repeats,"
           "logical_cells_per_block,memory_cells_per_block,total_updates,touched_cells,"
           "avg_seconds,min_seconds,updates_per_second,touched_cells_per_second\n";
  }

  out << CsvEscape(row.loop_name) << ','
      << CsvEscape(row.backend) << ','
      << row.nblocks << ','
      << row.target_cells << ','
      << row.nvars << ','
      << row.nz_interior << ','
      << row.ny_interior << ','
      << row.nx_interior << ','
      << row.nghost << ','
      << row.ninner << ','
      << CsvEscape(row.access_mode) << ','
      << row.niter << ','
      << CsvEscape(row.stencil_x) << ','
      << CsvEscape(row.stencil_y) << ','
      << CsvEscape(row.stencil_z) << ','
      << CsvEscape(row.kernel_label) << ','
      << row.warmup << ','
      << row.repeats << ','
      << row.logical_cells_per_block << ','
      << row.memory_cells_per_block << ','
      << row.total_updates << ','
      << row.touched_cells << ','
      << row.avg_seconds << ','
      << row.min_seconds << ','
      << row.updates_per_second << ','
      << row.touched_cells_per_second << '\n';

  return true;
}

bool ParseCaseRow(const std::unordered_map<std::string, std::size_t> &columns,
                  const std::vector<std::string> &fields, CaseSpec *spec, std::string *error) {
  auto get = [&](const char *name) -> const std::string & {
    static const std::string empty;
    const auto it = columns.find(name);
    if (it == columns.end() || it->second >= fields.size()) {
      return empty;
    }
    return fields[it->second];
  };

  const std::string loop = Trim(get("loop"));
  if (!loop.empty()) {
    spec->loop.kind = ParseLoopKind(loop);
  }
  const std::string backend = Trim(get("backend"));
  if (!backend.empty()) {
    spec->backend = backend;
  }

  if (!Trim(get("nblocks")).empty() && !ParseInt(get("nblocks"), &spec->problem.nblocks)) {
    if (error != nullptr) *error = "bad nblocks value";
    return false;
  }
  if (!Trim(get("target_cells")).empty() &&
      !ParseUInt64(get("target_cells"), &spec->problem.target_cells)) {
    if (error != nullptr) *error = "bad target_cells value";
    return false;
  }
  if (!Trim(get("nvars")).empty() && !ParseInt(get("nvars"), &spec->problem.nvars)) {
    if (error != nullptr) *error = "bad nvars value";
    return false;
  }
  if (!Trim(get("nz")).empty() && !ParseInt(get("nz"), &spec->problem.nz_interior)) {
    if (error != nullptr) *error = "bad nz value";
    return false;
  }
  if (!Trim(get("ny")).empty() && !ParseInt(get("ny"), &spec->problem.ny_interior)) {
    if (error != nullptr) *error = "bad ny value";
    return false;
  }
  if (!Trim(get("nx")).empty() && !ParseInt(get("nx"), &spec->problem.nx_interior)) {
    if (error != nullptr) *error = "bad nx value";
    return false;
  }
  if (!Trim(get("nghost")).empty() && !ParseInt(get("nghost"), &spec->problem.nghost)) {
    if (error != nullptr) *error = "bad nghost value";
    return false;
  }
  if (!Trim(get("ninner")).empty() && !ParseInt(get("ninner"), &spec->loop.ninner)) {
    if (error != nullptr) *error = "bad ninner value";
    return false;
  }
  const std::string access_mode = Trim(get("access_mode"));
  if (!access_mode.empty()) {
    spec->loop.access_mode = access_mode;
  }
  if (!Trim(get("warmup")).empty() && !ParseInt(get("warmup"), &spec->warmup)) {
    if (error != nullptr) *error = "bad warmup value";
    return false;
  }
  if (!Trim(get("repeats")).empty() && !ParseInt(get("repeats"), &spec->repeats)) {
    if (error != nullptr) *error = "bad repeats value";
    return false;
  }
  if (!Trim(get("niter")).empty() && !ParseInt(get("niter"), &spec->kernel.niter)) {
    if (error != nullptr) *error = "bad niter value";
    return false;
  }
  if (!Trim(get("stencil_x")).empty()) {
    spec->kernel.stencil_x = ParseOffsetSet(get("stencil_x"));
    if (spec->kernel.stencil_x.empty()) {
      if (error != nullptr) *error = "bad stencil_x value";
      return false;
    }
  }
  if (!Trim(get("stencil_y")).empty()) {
    spec->kernel.stencil_y = ParseOffsetSet(get("stencil_y"));
    if (spec->kernel.stencil_y.empty()) {
      if (error != nullptr) *error = "bad stencil_y value";
      return false;
    }
  }
  if (!Trim(get("stencil_z")).empty()) {
    spec->kernel.stencil_z = ParseOffsetSet(get("stencil_z"));
    if (spec->kernel.stencil_z.empty()) {
      if (error != nullptr) *error = "bad stencil_z value";
      return false;
    }
  }

  const std::string vars_per_block = Trim(get("vars_per_block"));
  if (!vars_per_block.empty()) {
    spec->problem.vars_per_block = ParseVarsPerBlock(vars_per_block);
  }

  NormalizeCaseSpec(spec);
  return true;
}

}  // namespace

bool RunCaseMatrix(const std::string &cases_csv, const std::string &results_csv,
                   std::string *error) {
  std::ifstream in(cases_csv);
  if (!in) {
    if (error != nullptr) {
      *error = "failed to open cases csv: " + cases_csv;
    }
    return false;
  }

  std::string header_line;
  if (!std::getline(in, header_line)) {
    if (error != nullptr) {
      *error = "cases csv is empty: " + cases_csv;
    }
    return false;
  }

  const auto headers = SplitCsvLine(header_line);
  std::unordered_map<std::string, std::size_t> columns;
  for (std::size_t i = 0; i < headers.size(); ++i) {
    columns[Trim(headers[i])] = i;
  }

  std::vector<CaseSpec> cases;
  std::string line;
  while (std::getline(in, line)) {
    if (Trim(line).empty()) {
      continue;
    }
    const auto fields = SplitCsvLine(line);
    CaseSpec spec;
    spec.backend = "Serial";
    spec.warmup = 2;
    spec.repeats = 5;
    if (!ParseCaseRow(columns, fields, &spec, error)) {
      return false;
    }
    cases.push_back(std::move(spec));
  }

  namespace fs = std::filesystem;
  std::error_code ec;
  const bool exists = fs::exists(results_csv, ec);
  if (ec) {
    if (error != nullptr) {
      *error = "failed to inspect results path: " + ec.message();
    }
    return false;
  }

  for (std::size_t i = 0; i < cases.size(); ++i) {
    const BenchmarkRow row = RunCase(cases[i]);
    const std::uint64_t logical_cells_total =
        row.logical_cells_per_block * static_cast<std::uint64_t>(row.nblocks);
    const std::uint64_t memory_cells_total =
        row.memory_cells_per_block * static_cast<std::uint64_t>(row.nblocks);
    const bool append = exists || i > 0;
    if (!WriteResultsCsv(results_csv, row, append, error)) {
      return false;
    }
    std::cout << "[" << (i + 1) << "/" << cases.size() << "] "
              << ToString(cases[i].loop.kind) << " "
              << "access_mode=" << row.access_mode << " "
              << "blocks=" << row.nblocks << " "
              << "edge=" << row.nx_interior << " "
              << "logical_cells=" << logical_cells_total << " "
              << "memory_cells=" << memory_cells_total << " "
              << "ninner=" << row.ninner << " "
              << "niter=" << row.niter << " "
              << "stencil_x=" << row.stencil_x << " "
              << "stencil_y=" << row.stencil_y << " "
              << "stencil_z=" << row.stencil_z << " "
              << "avg_seconds=" << row.avg_seconds << " "
              << "min_seconds=" << row.min_seconds << " "
              << "updates_per_second=" << row.updates_per_second << std::endl;
  }
  return true;
}

}  // namespace plb2
