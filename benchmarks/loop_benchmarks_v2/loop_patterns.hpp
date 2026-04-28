#pragma once

#include <cstdint>

#include "config.hpp"

namespace plb2 {

Dataset BuildDataset(const CaseSpec &spec);
void PrepareDataset(const CaseSpec &spec, Dataset *dataset);

std::uint64_t CountUpdates(const CaseSpec &spec);
BenchmarkRow RunCase(const CaseSpec &spec);

}  // namespace plb2
