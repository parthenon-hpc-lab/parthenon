//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
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
#include <stdexcept>
#include <string>

#include <config.hpp>

#include <Kokkos_Core.hpp>
#ifdef HDF5OUTPUT
#include <hdf5.h>
#endif

#include <parthenon_mpi.hpp>

#define PARTHENON_REQUIRE(condition, message)                                            \
  if (!(condition)) {                                                                    \
    parthenon::ErrorChecking::require(#condition, message, __FILE__, __LINE__);          \
  }

#define PARTHENON_REQUIRE_THROWS(condition, message)                                     \
  if (!(condition)) {                                                                    \
    parthenon::ErrorChecking::require_throws(#condition, message, __FILE__, __LINE__);   \
  }

#define PARTHENON_FAIL(message)                                                          \
  parthenon::ErrorChecking::fail(message, __FILE__, __LINE__);

#define PARTHENON_THROW(message)                                                         \
  parthenon::ErrorChecking::fail_throws(message, __FILE__, __LINE__);

#ifdef MPI_PARALLEL
#define PARTHENON_MPI_CHECK(expr)                                                        \
  do {                                                                                   \
    int parthenon_mpi_check_status = (expr);                                             \
    if (MPI_SUCCESS != parthenon_mpi_check_status) {                                     \
      ::parthenon::ErrorChecking::fail_throws_mpi(parthenon_mpi_check_status, #expr,     \
                                                  __FILE__, __LINE__);                   \
    }                                                                                    \
  } while (false)
#endif

#ifdef HDF5OUTPUT
#define PARTHENON_HDF5_CHECK(expr)                                                       \
  ([&]() -> herr_t {                                                                     \
    herr_t const parthenon_hdf5_check_err = (expr);                                      \
    if (parthenon_hdf5_check_err < 0) {                                                  \
      ::parthenon::ErrorChecking::fail_throws_hdf5(parthenon_hdf5_check_err, #expr,      \
                                                   __FILE__, __LINE__);                  \
    }                                                                                    \
    return parthenon_hdf5_check_err;                                                     \
  })()
#endif

#define PARTHENON_WARN(message)                                                          \
  parthenon::ErrorChecking::warn(message, __FILE__, __LINE__);

#ifdef NDEBUG
#define PARTHENON_DEBUG_REQUIRE(condition, message) ((void)0)
#else
#define PARTHENON_DEBUG_REQUIRE(condition, message) PARTHENON_REQUIRE(condition, message)
#endif

#ifdef NDEBUG
#define PARTHENON_DEBUG_REQUIRE_THROWS(condition, message) ((void)0)
#else
#define PARTHENON_DEBUG_REQUIRE_THROWS(condition, message)                               \
  PARTHENON_REQUIRE_THROWS(condition, message)
#endif

#ifdef NDEBUG
#define PARTHENON_DEBUG_FAIL(message) ((void)0)
#else
#define PARTHENON_DEBUG_FAIL(message) PARTHENON_FAIL(message)
#endif

#ifdef NDEBUG
#define PARTHENON_DEBUG_THROW(message) ((void)0)
#else
#define PARTHENON_DEBUG_THROW(message) PARTHENON_THROW(message)
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

inline void require(const char *const condition, std::string const &message,
                    const char *const filename, int const linenumber) {
  require(condition, message.c_str(), filename, linenumber);
}

inline void require(const char *const condition, std::stringstream const &message,
                    const char *const filename, int const linenumber) {
  require(condition, message.str().c_str(), filename, linenumber);
}

// TODO(JMM): should we define our own parthenon error class? Or is
// std::runtime_error good enough?
inline void require_throws(const char *const condition, const char *const message,
                           const char *const filename, int const linenumber) {
  std::stringstream msg;
  msg << "### PARTHENON ERROR\n  Condition:   " << condition
      << "\n  Message:     " << message << "\n  File:        " << filename
      << "\n  Line number: " << linenumber << std::endl;
  throw std::runtime_error(msg.str().c_str());
}

inline void require_throws(const char *const condition, std::string const &message,
                           const char *const filename, int const linenumber) {
  require_throws(condition, message.c_str(), filename, linenumber);
}

inline void require_throws(const char *const condition, std::stringstream const &message,
                           const char *const filename, int const linenumber) {
  require_throws(condition, message.str().c_str(), filename, linenumber);
}

[[noreturn]] KOKKOS_INLINE_FUNCTION void
fail(const char *const message, const char *const filename, int const linenumber) {
  printf("### PARTHENON ERROR\n  Message:     %s\n  File:        %s\n  Line number: %i\n",
         message, filename, linenumber);
  Kokkos::abort(message);
  // Kokkos::abort ends control flow, but is not marked as `[[noreturn]]`, so we need this
  // loop to supress a warning that the function does not return.
  while (true) {
  }
}

[[noreturn]] inline void fail(std::stringstream const &message,
                              const char *const filename, int const linenumber) {
  fail(message.str().c_str(), filename, linenumber);
}

[[noreturn]] inline void fail_throws(const char *const message,
                                     const char *const filename, int const linenumber) {
  std::stringstream msg;
  msg << "### PARTHENON ERROR\n  Message:     " << message
      << "\n  File:        " << filename << "\n  Line number: " << linenumber
      << std::endl;
  throw std::runtime_error(msg.str().c_str());
}

[[noreturn]] inline void fail_throws(std::string const &message,
                                     const char *const filename, int const linenumber) {
  fail_throws(message.c_str(), filename, linenumber);
}

[[noreturn]] inline void fail_throws(std::stringstream const &message,
                                     const char *const filename, int const linenumber) {
  fail_throws(message.str().c_str(), filename, linenumber);
}

KOKKOS_INLINE_FUNCTION
void warn(const char *const message, const char *const filename, int const linenumber) {
  printf("### PARTHENON WARNING\n  Message:     %s\n  File:        %s\n  Line number: "
         "%i\n",
         message, filename, linenumber);
}

inline void warn(std::stringstream const &message, const char *const filename,
                 int const linenumber) {
  warn(message.str().c_str(), filename, linenumber);
}

#ifdef MPI_PARALLEL
[[noreturn]] void fail_throws_mpi(int const status, char const *const expr,
                                  char const *const filename, int const linenumber);
#endif

#ifdef HDF5OUTPUT
[[noreturn]] void fail_throws_hdf5(herr_t err, char const *const expr,
                                   char const *const filename, int const linenumber);
#endif
} // namespace ErrorChecking
} // namespace parthenon

#endif // UTILS_ERROR_CHECKING_HPP_
