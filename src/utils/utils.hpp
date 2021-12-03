//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
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
#ifndef UTILS_UTILS_HPP_
#define UTILS_UTILS_HPP_
//! \file utils.hpp
//  \brief prototypes of functions and class definitions for utils/*.cpp files

#include <cctype>
#include <csignal>
#include <cstdint>
#include <sstream>
#include <string>

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
template <typename T>
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

template <>
inline size_t parse_value<size_t>(std::string &strvalue) {
  return parse_unsigned<size_t>(strvalue);
}

#ifdef ENABLE_HDF5
template <>
inline hsize_t parse_value<hsize_t>(std::string &strvalue) {
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
