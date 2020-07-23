//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001
// for Los Alamos National Laboratory (LANL), which is operated by Triad
// National Security, LLC for the U.S. Department of Energy/National Nuclear
// Security Administration. All rights in the program are reserved by Triad
// National Security, LLC, and the U.S. Department of Energy/National Nuclear
// Security Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
// in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do
// so.
//========================================================================================

#include <catch2/catch.hpp>

#include "utils/error_checking.hpp"

TEST_CASE("Parthenon Error Checking", "[ErrorChecking][Kokkos][coverage]") {
  SECTION("PARTHENON_REQUIRE passes if condition true") {
    PARTHENON_REQUIRE(true, "This shouldn't fail");
  }
#ifdef PARTHENON_TEST_ERROR_CHECKING
  SECTION("PARTHENON_REQUIRE fails if condition false") {
    PARTHENON_REQUIRE(false, "This should fail");
  }
  SECTION("PARTHENON FAIL causes the code to die") { PARTHENON_FAIL("This should die"); }
  SECTION("PARTHENON_DEBUG_REQUIRE does nothing if NDEBUG is defined") {
    PARTHENON_DEBUG_REQUIRE(false, "This should only die if NDEBUG is not defined");
  }
  SECTION("PARTHENON_DEBUG_FAIL does nothing if NDEBUG is defined") {
    PARTHENON_DEBUG_FAIL("This should only die if NDEBUG is not defined");
  }
#endif // PARTHENON_TEST_ERROR_CHECKING
}
