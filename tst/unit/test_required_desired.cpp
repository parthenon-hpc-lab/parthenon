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

using parthenon::AppInputs_t;
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

    // capture all std::cerr
    std::stringstream cerr_cap;
    std::streambuf *cerr = std::cerr.rdbuf(cerr_cap.rdbuf());

    WHEN("We require missing parameters") {
      AppInputs_t req;
      req["block1"].push_back("var1");
      req["block1"].push_back("var2");
      req["block2"].push_back("var3");
      req["block2"].push_back("var9");
      AppInputs_t des;
      THEN("The check should throw a runtime error") {
        REQUIRE_THROWS_AS(in.CheckRequiredDesired(req, des), std::runtime_error);
      }
    }
    WHEN("We require a parameter that is set by a code default") {
      AppInputs_t req;
      req["block2"].push_back("var_default");
      AppInputs_t des;
      THEN("The check should throw a runtime error") {
        REQUIRE_THROWS_AS(in.CheckRequiredDesired(req, des), std::runtime_error);
      }
    }
    AND_WHEN("We desire missing parameters") {
      AppInputs_t req;
      AppInputs_t des;
      des["block3"].push_back("var4");
      des["block2"].push_back("var2");
      THEN("The check should print warnings") {
        // cout_cap.clear();
        in.CheckRequiredDesired(req, des);
        std::stringstream ss;
        ss << std::endl
           << "Parameter file missing suggested field <block2>/var2" << std::endl
           << std::endl
           << std::endl
           << "Parameter file missing suggested field <block3>/var4" << std::endl
           << std::endl;
        REQUIRE(cerr_cap.str() == ss.str());
      }
    }
    std::cerr.rdbuf(cerr);
  }
}
