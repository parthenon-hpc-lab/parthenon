//========================================================================================
// (C) (or copyright) 2020-2026. Triad National Security, LLC. All rights reserved.
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
#include "kokkos_types.hpp"

namespace parthenon {
namespace string_utils {

// trim whitespace
std::string ltrim(const std::string &s);
std::string rtrim(const std::string &s);
std::string trim(const std::string &s);

// pack/unpack strings (basically join and split with a given delimiter)
std::string PackStrings(const std::vector<std::string> &strs, char delimiter);
std::vector<std::string> UnpackStrings(const std::string &pack, char delimiter);

// A mechanism to read an ascii file on a single rank and MPI
// broadcast it to all ranks. May be used for parsing in a massively
// parallel context where all ranks accessing a file might kill the
// filesystem. Note this idea could be generalized.
std::string BroadcastFileString(const std::string &filename);

template <typename T = Real>
HostArray2D<T> ParseAsciiTable(std::istream &in);
extern template HostArray2D<double> ParseAsciiTable<double>(std::istream &);
extern template HostArray2D<float> ParseAsciiTable<float>(std::istream &);
extern template HostArray2D<int> ParseAsciiTable<int>(std::istream &);
extern template HostArray2D<std::size_t> ParseAsciiTable<std::size_t>(std::istream &);

template <typename T = Real>
HostArray2D<T> ParseAsciiTable(const std::string &filename);
extern template HostArray2D<double> ParseAsciiTable<double>(const std::string &);
extern template HostArray2D<float> ParseAsciiTable<float>(const std::string &);
extern template HostArray2D<int> ParseAsciiTable<int>(const std::string &);
extern template HostArray2D<std::size_t>
ParseAsciiTable<std::size_t>(const std::string &);

} // namespace string_utils
} // namespace parthenon

#endif // UTILS_STRING_UTILS_HPP_
