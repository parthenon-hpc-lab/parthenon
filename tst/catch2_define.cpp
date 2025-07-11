//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2022 The Parthenon collaboration
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

#include <string>
#define CATCH_CONFIG_RUNNER

#include <algorithm>
#include <iostream>

#ifdef CATCH2_MPI_PARALLEL
#include <mpi.h>
#endif // CATCH2_MPI_PARALLEL

#include <catch2/catch.hpp>

#include <Kokkos_Core.hpp>

#ifdef CATCH2_MPI_PARALLEL
template <class T>
bool HasMPITests(const T &config) {
  // Used to check if a given test in the suite matches the requsted
  // test specification
  const auto &test_spec = config.testSpec();

  const auto &all_test_cases = Catch::getAllTestCasesSorted(config);
  for (auto const &test_case : all_test_cases) {
    auto &tags = test_case.getTestCaseInfo().tags;
    if (test_spec.matches(test_case) &&
        std::find(tags.begin(), tags.end(), "MPI") != tags.end()) {
      return true;
    }
  }
  return false;
}
#endif // CATCH2_MPI_PARALLEL

int main(int argc, char *argv[]) {
  // With Catch2 >2.13.4 catch_discover_tests() is used to discover tests by calling the
  // test executable with `--list-test-names-only` and parsing the results.
  // However, we have to init Kokkos first, which potentially shows warnings that are
  // incorrectly parsed as test. Thus, we here disable the warning for when the tests are
  // parsed.
  for (auto i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--list-test-names-only") {
      setenv("KOKKOS_DISABLE_WARNINGS", "TRUE", 1);
    }
  }

  Catch::Session session;

  int returnCode = session.applyCommandLine(argc, argv);
  if (returnCode != 0) {
    return returnCode;
  }
  const auto &config = session.config();
#ifdef CATCH2_MPI_PARALLEL
  bool running_mpi_tests = HasMPITests(config);
  if (running_mpi_tests) {
    int already_initialized;
    MPI_Initialized(&already_initialized);
    if (!already_initialized && (MPI_SUCCESS != MPI_Init(&argc, &argv))) {
      std::cerr << "### FATAL ERROR in ParthenonInit" << std::endl
                << "MPI Initialization failed." << std::endl;
      return -1;
    }
  }
#endif // CATCH2_MPI_PARALLEL

  Kokkos::initialize(argc, argv);

  int result;
  {
    result = session.run();

    // global clean-up...
  }

  Kokkos::finalize();

#ifdef CATCH2_MPI_PARALLEL
  if (running_mpi_tests) {
    int mpi_finalized;
    MPI_Finalized(&mpi_finalized);
    if (!mpi_finalized) MPI_Finalize();
  }
#endif // CATCH2_MPI_PARALLEL

  return result;
}
