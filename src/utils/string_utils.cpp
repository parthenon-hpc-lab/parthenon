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

#include "string_utils.hpp"

#include <sstream>
#include <string>
#include <vector>

#include "config.hpp"
#include "error_checking.hpp"
#include "globals.hpp"
#include "kokkos_types.hpp"
#include "parthenon_mpi.hpp"

namespace parthenon {
namespace string_utils {

constexpr char WHITESPACE[] = " \n\r\t\f\v";

std::string ltrim(const std::string &s) {
  std::size_t start = s.find_first_not_of(WHITESPACE);
  return (start == std::string::npos) ? "" : s.substr(start);
}

std::string rtrim(const std::string &s) {
  std::size_t end = s.find_last_not_of(WHITESPACE);
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
  std::vector<std::string> unpack;
  if (pack.size() == 0) {
    return unpack;
  }

  if (pack[pack.size() - 1] != delimiter) {
    std::stringstream msg;
    msg << "### ERROR: Pack string does not end with delimiter" << std::endl;
    PARTHENON_FAIL(msg);
  }

  std::stringstream stm(pack);
  std::string token;

  while (std::getline(stm, token, delimiter)) {
    unpack.emplace_back(token);
  }

  return unpack;
}

std::string BroadcastFileString(const std::string &filename) {
  std::uint64_t strlen;
  std::string str;

  if (Globals::my_rank == 0) {
    std::ifstream in(filename);
    if (!in) {
      PARTHENON_THROW("Failed to open file " + filename);
    }

    // Figure out length of file. Careful with tellg error code.
    in.seekg(0, std::ios::end);
    auto maybe_strlen = static_cast<std::int64_t>(in.tellg());
    PARTHENON_REQUIRE(maybe_strlen > 0, "File has menaingful length");
    strlen = static_cast<std::uint64_t>(maybe_strlen);

    // allocate memory for string upfront
    str.reserve(strlen);
    in.seekg(0, std::ios::beg);

    str.assign((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  }

#ifdef MPI_PARALLEL
  MPI_Bcast(&strlen, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  if (Globals::my_rank != 0) {
    str.resize(strlen);
  }
  MPI_Bcast(str.data(), strlen, MPI_BYTE, 0, MPI_COMM_WORLD);
#endif // MPI_PARALLEL
  return str;
}

template <typename T>
HostArray2D<T> ParseAsciiTable(std::istream &in) {
  std::vector<T> data;

  std::string line;
  std::size_t rows = 0;
  std::size_t cols = 0;

  while (std::getline(in, line)) {
    // Strip comments...
    if (auto pos = line.find('#'); pos != std::string::npos) {
      line.erase(pos);
    }

    // ...and whitespace
    line = trim(line);
    if (line.empty()) continue;

    std::istringstream iss(line);
    T value;
    std::size_t row_count = 0;

    while (iss >> value) {
      data.push_back(value);
      ++row_count;
    }

    if (!iss.eof()) {
      PARTHENON_THROW("ASCII parser error on line: " + std::to_string(rows) +
                      "! Incorrect type.");
    }
    if (row_count == 0) continue; // should not happen after trim, but safe

    if (rows == 0) {
      cols = row_count;
      if (cols == 0) { // table is empty. We can just return.
        break;
      }
    } else if (row_count != cols) {
      PARTHENON_THROW("Parsed ASCII table is ragged.");
    }
    ++rows;
  }

  // JMM: Thought about doing this by just copying the data, but doing
  // it this way safeguards against HostArray2D having a different
  // layout than row-major ordering.
  HostArray2D<T> out("Parsed ascii table", rows, cols);
  {
    std::size_t idx = 0;
    for (std::size_t row = 0; row < rows; ++row) {
      for (std::size_t col = 0; col < cols; ++col) {
        out(row, col) = data[idx++];
      }
    }
  }

  return out;
}
template HostArray2D<double> ParseAsciiTable<double>(std::istream &);
template HostArray2D<float> ParseAsciiTable<float>(std::istream &);
template HostArray2D<int> ParseAsciiTable<int>(std::istream &);
template HostArray2D<std::size_t> ParseAsciiTable<std::size_t>(std::istream &);

template <typename T>
HostArray2D<T> ParseAsciiTable(const std::string &filename) {
  std::string str = BroadcastFileString(filename);
  std::istringstream stream(str);
  return ParseAsciiTable<T>(stream);
}
template HostArray2D<double> ParseAsciiTable<double>(const std::string &);
template HostArray2D<float> ParseAsciiTable<float>(const std::string &);
template HostArray2D<int> ParseAsciiTable<int>(const std::string &);
template HostArray2D<std::size_t> ParseAsciiTable<std::size_t>(const std::string &);

} // namespace string_utils
} // namespace parthenon
