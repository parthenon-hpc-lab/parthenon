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
#ifndef UTILS_ERROR_CHECKING_HPP_
#define UTILS_ERROR_CHECKING_HPP_
//! \file error_checking.hpp
//  \brief utility macros for error checking

#include <Kokkos_Core.hpp>
#include <iostream>
#include <stdexcept>
#include <string>

#define PARTHENON_REQUIRE(condition, message)                                            \
  if (!(condition)) {                                                                    \
    parthenon::ErrorChecking::require(#condition, message, __FILE__, __LINE__);          \
  }

#define PARTHENON_FAIL(message)                                                          \
  parthenon::ErrorChecking::fail(message, __FILE__, __LINE__);

#ifdef NDEBUG
#define PARTHENON_DEBUG_REQUIRE(condition, message)
#else
#define PARTHENON_DEBUG_REQUIRE(condition, message) PARTHENON_REQUIRE(condition, message)
#endif

#ifdef NDEBUG
#define PARTHENON_DEBUG_FAIL(message)
#else
#define PARTHENON_DEBUG_FAIL(message) PARTHENON_FAIL(message)
#endif

namespace parthenon {
namespace ErrorChecking {

KOKKOS_INLINE_FUNCTION
void require(std::string const &condition, std::string const &message,
             std::string const &filename, int const linenumber) {
  fprintf(stderr,
          "### PARTHENON ERROR\n  Condition:   %s\n  Message:     %s\n  File:        "
          "%s\n  Line number: %i\n",
          condition.c_str(), message.c_str(), filename.c_str(), linenumber);
  exit(EXIT_FAILURE);
}

KOKKOS_INLINE_FUNCTION
void fail(std::string const &message, std::string const &filename, int const linenumber) {
  fprintf(
      stderr,
      "### PARTHENON ERROR\n  Message:     %s\n  File:        %s\n  Line number: %i\n",
      message.c_str(), filename.c_str(), linenumber);
  exit(EXIT_FAILURE);
}

} // namespace ErrorChecking
} // namespace parthenon

#endif // UTILS_ERROR_CHECKING_HPP_
