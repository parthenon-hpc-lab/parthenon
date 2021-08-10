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

#ifndef UTILS_STRING_UTILS_HPP_
#define UTILS_STRING_UTILS_HPP_

#include <string>
#include <vector>

namespace parthenon {
namespace string_utils {

// trim whitespace
std::string ltrim(const std::string &s);
std::string rtrim(const std::string &s);
std::string trim(const std::string &s);

// pack/unpack strings (basically join and split with a given delimiter)
std::string PackStrings(const std::vector<std::string> &strs, char delimiter);
std::vector<std::string> UnpackStrings(const std::string &pack, char delimiter);

} // namespace string_utils
} // namespace parthenon

#endif // UTILS_STRING_UTILS_HPP_
