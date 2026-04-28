#pragma once

#include <string>

namespace plb2 {

bool RunCaseMatrix(const std::string &cases_csv, const std::string &results_csv,
                   std::string *error);

}  // namespace plb2
