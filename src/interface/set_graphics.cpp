//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================

#include <algorithm>
#include <list>
#include <string>

#include "metadata.hpp"

namespace parthenon {

namespace strip_string {
constexpr char WHITESPACE[] = " \n\r\t\f\v";

std::string ltrim(const std::string &s) {
  size_t start = s.find_first_not_of(WHITESPACE);
  return (start == std::string::npos) ? "" : s.substr(start);
}

std::string rtrim(const std::string &s) {
  size_t end = s.find_last_not_of(WHITESPACE);
  return (end == std::string::npos) ? "" : s.substr(0, end + 1);
}

std::string trim(const std::string &s) { return rtrim(ltrim(s)); }
} // namespace strip_string

OutputFlags SetOutputFlags(std::unique_ptr<ParameterInput> &pin, Packages_t &packages) {
  OutputFlags output_flags;
  InputBlock *pib = pin->first_block;
  while(pib != nullptr) {
    if (pib->block_name.compare(0, 16, "parthenon/output") == 0) {
      if (pin->DoesParameterExist("variables")) {
        std::string s = pin->GetString(pib->block_name), "variables");
        std::string delimiter = ",";
        size_t pos = 0;
        std::string token;
        std::list<std::string> fields;
        while ((pos = s.find(delimiter)) != std::string::npos) {
          token = s.substr(0, pos);
          fields.push_front(strip_string::trim(token));
          s.erase(0, pos + delimiter.length());
        }
        fields.push_front(strip_string::trim(s));

        // make a new Metadata flag for this output
        MetadataFlag const new_output_flag = Metadata::AllocateNewFlag(pib->block_name);
        output_flags.push_back(new_output_flag)

        for (auto &pkg : packages) {
          for (auto &q : pkg.second->AllFields()) {
            auto it = std::find(fields.begin(), fields.end(), q.first);
            if (it != fields.end()) {
              q.second.Set(new_output_flag);
              fields.erase(it);
            }
          }
          for (auto &q : pkg.second->AllSparseFields()) {
            auto it = std::find(fields.begin(), fields.end(), q.first);
            if (it != fields.end()) {
              for (auto &m : q.second) {
                m.Set(new_output_flag);
              }
              fields.erase(it);
            }
          }
        }

        if (fields.size() != 0) {
          std::cerr << "These variables listed in "
                    << pib->block_name
                    << "/variables do not exist:" << std::endl;
          for (auto const &field : fields) {
            std::cerr << field << std::endl;
          }
        }
      }
    }
  }
  return output_flags;
}

} // namespace parthenon
