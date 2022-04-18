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

#include <catch2/catch.hpp>

#include <Kokkos_Core.hpp>

int main(int argc, char *argv[]) {
  // With Catch2 >2.13.4 catch_discover_tests() is used to discover tests by calling the
  // test executable with `--list-test-names-only` and parsing the results.
  // However, we have to init Kokkos first, which potentially shows warnings that are
  // incorrectly parsed as test. Thus, we here disable the warning for when the tests are
  // parsed.
  for (auto i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--list-test-names-only") {
      setenv("KOKKOS_DISABLE_WARNINGS", "ON", 1);
    }
  }

  // global setup...
  Kokkos::initialize(argc, argv);

  int result;
  {
    result = Catch::Session().run(argc, argv);

    // global clean-up...
  }

  Kokkos::finalize();
  return result;
}
