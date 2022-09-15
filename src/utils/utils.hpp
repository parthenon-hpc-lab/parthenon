//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
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
#ifndef UTILS_UTILS_HPP_
#define UTILS_UTILS_HPP_
//! \file utils.hpp
//  \brief prototypes of functions and class definitions for utils/*.cpp files

#include <cctype>
#include <csignal>
#include <cstdint>
#include <sstream>
#include <string>
#include <type_traits>

#include "constants.hpp"
#include "error_checking.hpp"
#include "kokkos_abstraction.hpp"

namespace parthenon {

void ChangeRunDir(const char *pdir);
void ShowConfig();

//----------------------------------------------------------------------------------------
//! SignalHandler
//  \brief static data and functions that implement a simple signal handling system
namespace SignalHandler {

enum class OutputSignal { none, now, final };
constexpr int nsignal = 3;
// using the +1 for signaling based on "output_now" trigger
static volatile int signalflag[nsignal + 1];
const int ITERM = 0, IINT = 1, IALRM = 2;
static sigset_t mask;
void SignalHandlerInit();
OutputSignal CheckSignalFlags();
int GetSignalFlag(int s);
void SetSignalFlag(int s);
void SetWallTimeAlarm(int t);
void CancelWallTimeAlarm();
void Report();

} // namespace SignalHandler

//----------------------------------------------------------------------------------------
//! Env
//  \brief static function to check and retrieve environment settings
namespace Env {

namespace Impl {
// TODO(the person bumping standard to C++17) Clean up this mess and use constexpr if

// TODO(JMM): We use template specialization below to parse
// information from environment variables
// where we template on return type. We hit an issue with HDF5's hsize_t, becuase:
// (a) sometimes HDF5 is disabled, so hsize_t isn't always present
// (b) hsize_t's type is system/compiler-dependent. Sometimes it's the same as size_t,
//     but it doesn't have to be. If you define a specialization for size_t
//     and hsize_t on a system where they are not the same,
//     the compiler will see a duplicate function and complain.
// To resolve this issue, we use SFINAE. The std::enable_ifs mean the
// templated functions are enabled only if the appropriate conditions are met.
// In particular, the prototype is enabled only for types that are
// not size_t or hsize_t.
// Then the implementation for size_t is always enabled and the implementation
// for hsize_t is enabled ONLY if HDF5 is available, and hsize_t != size_t.
template <typename T,
          typename std::enable_if<!std::is_same<T, size_t>::value, bool>::type = true
#ifdef ENABLE_HDF5
          ,
          typename std::enable_if<!std::is_same<T, hsize_t>::value, bool>::type = true>
#else
          >
#endif // ENABLE_HDF5

T parse_value(std::string &strvalue);

// Parse env. variable expected to hold a bool value allowing for different conventions.
// Note, the input strvalue will be modified by this fuction (converted to upper case).
template <>
inline bool parse_value<bool>(std::string &strvalue) {
  for (char &c : strvalue) {
    c = toupper(c);
  }
  return (strvalue == "TRUE") || (strvalue == "ON") || (strvalue == "1");
}

template <>
inline std::string parse_value<std::string>(std::string &strvalue) {
  return strvalue;
}

// Ensure that the environment variable only contains non-negative integers
template <typename T>
T parse_unsigned(const std::string &strvalue) {
  for (const char &c : strvalue) {
    PARTHENON_REQUIRE_THROWS(std::isdigit(c), "Parsed environment variable '" + strvalue +
                                                  "' contains non-digits.");
  }

  T res;
  std::istringstream(strvalue) >> res;
  return res;
}

template <typename T,
          typename std::enable_if<std::is_same<T, size_t>::value, bool>::type = true>
inline T parse_value(std::string &strvalue) {
  return parse_unsigned<size_t>(strvalue);
}

#ifdef ENABLE_HDF5
template <typename T,
          typename std::enable_if<std::is_same<T, hsize_t>::value, bool>::type = true,
          typename std::enable_if<!std::is_same<T, size_t>::value, bool>::type = true>
inline T parse_value(std::string &strvalue) {
  return parse_unsigned<hsize_t>(strvalue);
}
#endif // ifdef ENABLE_HDF5
} // namespace Impl

// Get environment variables of various types (with checks).
// If variable does not exist or exists but is not set, `defaultval` will be returned.
// Parameter "exists" is set depending on whether the variable is found at all in env.
template <typename T>
static T get(const char *name, T defaultval, bool &exists) {
  exists = true;
  const char *value = std::getenv(name);

  // Environment variable is not set
  if (value == nullptr) {
    exists = false;
    return defaultval;
  }

  std::string strvalue(value);

  // Environment variable is set but no value is set, use the default
  if (strvalue.empty()) {
    return defaultval;
  }

  // Environment variable is set and value is set
  return Impl::parse_value<T>(strvalue);
}
} // namespace Env
} // namespace parthenon

#endif // UTILS_UTILS_HPP_
