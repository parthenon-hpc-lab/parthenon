//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
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

TEST_CASE("Test required/desired checking from inputs", "[ParameterInput]") {
  GIVEN("A ParameterInput object already populated") {
    ParameterInput in;
    std::stringstream ss;
    ss << "<block1>" << std::endl
       << "var1 = 0   # comment" << std::endl
       << "var2 = 1,  & # another comment" << std::endl
       << "       2" << std::endl
       << "<block2>" << std::endl
       << "var3 = 3" << std::endl
       << "# comment" << std::endl
       << "var4 = 4" << std::endl
       << "var_default = 5 # Default value added at run time" << std::endl;

    std::istringstream s(ss.str());
    in.LoadFromStream(s);

    // capture all std::cout
    std::stringstream cout_cap;
    std::streambuf *cout = std::cout.rdbuf(cout_cap.rdbuf());

    WHEN("We require a paramter that has been provided") {
      THEN("Nothing should happen") {
        REQUIRE_NOTHROW(in.CheckRequired("block1", "var1"));
        REQUIRE_NOTHROW(in.CheckRequired("block2", "var4"));
        REQUIRE_NOTHROW(in.CheckRequired("block1", "var2"));
      }
    }
    AND_WHEN("We require missing parameters") {
      THEN("The check should throw a runtime error") {
        REQUIRE_THROWS_AS(in.CheckRequired("block2", "var9"), std::runtime_error);
      }
    }
    AND_WHEN("We require a parameter that is set by a code default") {
      THEN("The check should throw a runtime error") {
        REQUIRE_THROWS_AS(in.CheckRequired("block2", "var_default"), std::runtime_error);
      }
    }
    AND_WHEN("We desire missing parameters") {
      cout_cap.clear();
      THEN("The check should print warnings") {
        in.CheckDesired("block2", "var2");
        in.CheckDesired("block3", "var4");
        ss.str("");
        ss << std::endl
           << "### WARNING in CheckDesired:" << std::endl
           << "Parameter file missing desired field <block2>/var2" << std::endl
           << std::endl
           << "### WARNING in CheckDesired:" << std::endl
           << "Parameter file missing desired field <block3>/var4" << std::endl;
        REQUIRE(cout_cap.str() == ss.str());
      }
    }
    std::cout.rdbuf(cout);
  }
  GIVEN("An invalid input deck") {
    ParameterInput in;
    std::stringstream ss;
    ss << "<block1>" << std::endl
       << "var1 = 0   # comment" << std::endl
       << "var2 = 1,  & 2.5 # another comment" << std::endl
       << "       2" << std::endl
       << "<block2>" << std::endl
       << "var3 = 3" << std::endl
       << "# comment" << std::endl
       << "var4 = 4" << std::endl
       << "var_default = 5 # Default value added at run time" << std::endl;
    WHEN("it is parsed") {
      std::istringstream s(ss.str());
      REQUIRE_THROWS_AS(in.LoadFromStream(s), std::runtime_error);
    }
  }
}
