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

#include "task_list/tasks.hpp"

using parthenon::TaskID;

TEST_CASE("Just check everything", "[CheckDependencies][SetFinished][equal][or]") {
  GIVEN("Some TaskIDs") {
    TaskID a(1);
    TaskID b(2);
    TaskID c(BITBLOCK + 1); // make sure we get a task with more than one block
    TaskID complete;

    TaskID ac = (a | c);
    bool should_be_false = ac.CheckDependencies(b);
    bool should_be_truea = ac.CheckDependencies(a);
    bool should_be_truec = ac.CheckDependencies(c);
    TaskID abc = (a | b | c);
    complete.SetFinished(abc);
    bool equal_true = (complete == abc);
    bool equal_false = (complete == ac);

    REQUIRE(should_be_false == false);
    REQUIRE(should_be_truea == true);
    REQUIRE(should_be_truec == true);
    REQUIRE(equal_true == true);
    REQUIRE(equal_false == false);

    WHEN("a negative number is passed") {
      REQUIRE_THROWS_AS(a.Set(-1), std::invalid_argument);
    }
  }
}
