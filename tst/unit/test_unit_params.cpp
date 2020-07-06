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

#include "interface/params.hpp"

using parthenon::Params;

TEST_CASE("Add and Get is called", "[Add,Get][coverage]") {
  GIVEN("A key") {
    Params params;
    std::string key = "test_key";
    double value = -2.0;
    params.Add(key, value);
    double output = params.Get<double>(key);
    REQUIRE(output == Approx(value));
    WHEN("the same key is provided a second time") {
      REQUIRE_THROWS_AS(params.Add(key, value), std::invalid_argument);
    }

    WHEN("attempting to get the key but casting to a different type") {
      REQUIRE_THROWS_AS(params.Get<int>(key), std::invalid_argument);
    }
  }

  GIVEN("An empty params structure") {
    Params params;
    WHEN(" attempting to get a key that does not exist ") {
      std::string non_existent_key = "key";
      REQUIRE_THROWS_AS(params.Get<double>(non_existent_key), std::invalid_argument);
    }
  }
}

TEST_CASE("reset is called", "[reset]") {
  GIVEN("A key is added") {
    Params params;
    std::string key = "test_key";
    double value = -2.0;
    params.Add(key, value);
    WHEN("the params are reset") {
      params.reset();
      REQUIRE_THROWS_AS(params.Get<double>(key), std::invalid_argument);
    }
  }
}

TEST_CASE("when hasKey is called", "[hasKey]") {
  GIVEN("A key is added") {
    Params params;
    std::string key = "test_key";
    double value = -2.0;
    params.Add(key, value);

    REQUIRE(params.hasKey(key) == true);

    WHEN("the params are reset") {
      params.reset();
      REQUIRE(params.hasKey(key) == false);
    }
  }
}
