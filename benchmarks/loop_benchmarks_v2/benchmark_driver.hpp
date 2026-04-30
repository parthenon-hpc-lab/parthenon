#pragma once

#include <string>

#include "problem_spec.hpp"

namespace plb2 {

bool ParseArgs(int argc, char **argv, CaseSpec *spec, std::string *error);
std::string Usage();
int RunBenchmark(const CaseSpec &spec);

}  // namespace plb2
