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

#include <string>

#include <catch2/catch.hpp>

#include "tasks/tasks.hpp"

using parthenon::Task;
using parthenon::TaskID;

TEST_CASE("Just check everything", "[|][GetIDs][empty]") {
  GIVEN("Some TaskIDs") {
    Task ta,tb;
    TaskID a(&ta);
    TaskID b(&tb);
    TaskID c = a | b;
    TaskID none;

    REQUIRE(none.empty() == true);
    REQUIRE(c.GetIDs().size() == 2);
  }
}
