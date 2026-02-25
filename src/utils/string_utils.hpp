//========================================================================================
// (C) (or copyright) 2020-2025. Triad National Security, LLC. All rights reserved.
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

// This file was made in part with generative AI

#ifndef UTILS_STRING_UTILS_HPP_
#define UTILS_STRING_UTILS_HPP_

#include <fstream>
#include <string>
#include <vector>

#include "basic_types.hpp"

namespace parthenon {
namespace string_utils {

// Return type for ParseAsciiTable
// could alternatively be a HostArray2D
template <typename T>
struct Table2D {
  std::vector<T> data;
  std::size_t rows = 0;
  std::size_t cols = 0;

  T &operator()(std::size_t r, std::size_t c) { return data.at(r * cols + c); }
  const T &operator()(std::size_t r, std::size_t c) const {
    return data.at(r * cols + c);
  }
};

// trim whitespace
std::string ltrim(const std::string &s);
std::string rtrim(const std::string &s);
std::string trim(const std::string &s);

// pack/unpack strings (basically join and split with a given delimiter)
std::string PackStrings(const std::vector<std::string> &strs, char delimiter);
std::vector<std::string> UnpackStrings(const std::string &pack, char delimiter
);

template <typename T=Real>
Table2D<T> ParseAsciiTable(std::istream &in);
extern template Table2D<double> ParseAsciiTable(std::istream &);
extern template Table2D<float> ParseAsciiTable(std::istream &);
extern template Table2D<int> ParseAsciiTable(std::istream &);
extern template Table2D<std::size_t> ParseAsciiTable(std::istream &);

template<typename T = Real>
inline Table2D<T> ParseAsciiTable(const std::string &filename) {
  std::ifstream in(filename);
  if (!in) {
    PARTHENON_THROW("Failed to open file " + filename);
  }
  return ParseAsciiTable(filename);
}
} // namespace string_utils
} // namespace parthenon

#endif // UTILS_STRING_UTILS_HPP_
