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

#include <iostream>

#include <Kokkos_Core.hpp>

#define PARTHENON_REQUIRE(condition, message)                                            \
  if (!(condition)) {                                                                    \
    parthenon::ErrorChecking::require(#condition, message, __FILE__, __LINE__);          \
  }

#define PARTHENON_FAIL(message)                                                          \
  parthenon::ErrorChecking::fail(message, __FILE__, __LINE__);

#define PARTHENON_WARN(message)                                                          \
  parthenon::ErrorChecking::warn(message, __FILE__, __LINE__);

#ifdef NDEBUG
#define PARTHENON_DEBUG_REQUIRE(condition, message) ((void)0)
#else
#define PARTHENON_DEBUG_REQUIRE(condition, message) PARTHENON_REQUIRE(condition, message)
#endif

#ifdef NDEBUG
#define PARTHENON_DEBUG_FAIL(message) ((void)0)
#else
#define PARTHENON_DEBUG_FAIL(message) PARTHENON_FAIL(message)
#endif

#ifdef NDEBUG
#define PARTHENON_DEBUG_WARN(message) ((void)0)
#else
#define PARTHENON_DEBUG_WARN(message) PARTHENON_WARN(message)
#endif

namespace parthenon {
namespace ErrorChecking {

KOKKOS_INLINE_FUNCTION
void require(const char *const condition, const char *const message,
             const char *const filename, int const linenumber) {
  printf("### PARTHENON ERROR\n  Condition:   %s\n  Message:     %s\n  File:        "
         "%s\n  Line number: %i\n",
         condition, message, filename, linenumber);
  Kokkos::abort(message);
}

KOKKOS_INLINE_FUNCTION
void fail(const char *const message, const char *const filename, int const linenumber) {
  printf("### PARTHENON ERROR\n  Message:     %s\n  File:        %s\n  Line number: %i\n",
         message, filename, linenumber);
  Kokkos::abort(message);
}

inline void fail(std::stringstream const &message, const char *const filename,
                 int const linenumber) {
  fail(message.str().c_str(), filename, linenumber);
}

KOKKOS_INLINE_FUNCTION
void warn(const char *const message, const char *const filename, int const linenumber) {
  printf("### PARTHENON WARNING\n  Message:     %s\n  File:        %s\n  Line number: %i\n",
         message, filename, linenumber);
}

inline void warn(std::stringstream const &message, const char *const filename,
                 int const linenumber) {
  warn(message.str().c_str(), filename, linenumber);
}

} // namespace ErrorChecking
} // namespace parthenon

#endif // UTILS_ERROR_CHECKING_HPP_
