//========================================================================================
// (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
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

#include "string_utils.hpp"

#include <sstream>

#include "error_checking.hpp"

namespace parthenon {
namespace string_utils {

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

std::string PackStrings(const std::vector<std::string> &strs, char delimiter) {
  std::string pack;
  for (const auto &s : strs) {
    pack += s + delimiter;
  }
  return pack;
}

std::vector<std::string> UnpackStrings(const std::string &pack, char delimiter) {
  if (pack[pack.size() - 1] != delimiter) {
    std::stringstream msg;
    msg << "### ERROR: Pack string does not end with delimiter" << std::endl;
    PARTHENON_FAIL(msg);
  }

  std::vector<std::string> unpack;
  std::stringstream stm(pack);
  std::string token;

  while (std::getline(stm, token, delimiter)) {
    unpack.emplace_back(token);
  }

  return unpack;
}

} // namespace string_utils
} // namespace parthenon
