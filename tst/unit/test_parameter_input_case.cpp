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

#include <iostream>
#include <istream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <catch2/catch.hpp>

#include "parameter_input.hpp"

using parthenon::ParameterInput;

TEST_CASE("Test case sensitivity in input parsing.", "[ParameterInput][coverage]") {
  GIVEN("A ParameterInput object already populated") {
    ParameterInput in;
    std::stringstream ss;
    ss << "<block1>" << std::endl
       << "var0 = 0   # comment" << std::endl
       << "<BlOcK2>" << std::endl
       << "vaR3 = 3" << std::endl;

    std::istringstream s(ss.str());
    in.LoadFromStream(s);
    int a = in.GetInteger("block1", "var0");

    WHEN("We try to pull out parameters with the wrong case") {
      THEN("It should just work") {
        REQUIRE(0 == in.GetInteger("Block1", "Var0"));
        REQUIRE(0 == in.GetInteger("block1", "VAR0"));
        REQUIRE(3 == in.GetInteger("block2", "var3"));
        REQUIRE(3 == in.GetInteger("bLoCk2", "VAr3"));
      }
    }

    ParameterInput broken_in;

    std::stringstream ss2;
    ss2 << "<block1>" << std::endl
        << "var0 = 0   # comment" << std::endl
        << "<block2>" << std::endl
        << "Var0 = 3" << std::endl
        << "<Block1>" << std::endl
        << "var1 = 1" << std::endl
        << "<blocK2>" << std::endl
        << "var0 = 4" << std::endl;

    std::istringstream s2(ss2.str());

    WHEN("An input file is parsed with duplicate fields") {
      THEN("The code should exit with the appropriate error message") {
        REQUIRE_THROWS_AS(broken_in.LoadFromStream(s2), std::runtime_error);
      }
    }
  }
}
