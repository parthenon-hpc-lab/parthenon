#pragma once

#include <string>

#include "benchmark_driver.hpp"

namespace plb {

bool AppendCsvRow(const std::string &path, const BenchmarkRow &row, std::string *error);

} // namespace plb
