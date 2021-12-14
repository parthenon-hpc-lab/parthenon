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

TEST_CASE("Add, Get, and Update are called", "[Add,Get,Update]") {
  GIVEN("A key with some value") {
    Params params;
    std::string key = "test_key";
    double value = -2.0;
    THEN("we can add it to Params") {
      params.Add(key, value);
      AND_THEN("and retreive it with Get") {
        double output = params.Get<double>(key);
        REQUIRE(output == Approx(value));
      }
      AND_THEN("we can update the value") {
        params.Update<double>(key, 2.0);
        REQUIRE(params.Get<double>(key) == Approx(2.0));
      }
      WHEN("trying to Update with a wrong type") {
        THEN("an error is thrown") {
          REQUIRE_THROWS_AS(params.Update<int>(key, 2), std::runtime_error);
        }
      }
      WHEN("the same key is provided a second time") {
        THEN("an error is thrown") {
          REQUIRE_THROWS_AS(params.Add(key, value), std::runtime_error);
        }
      }
      WHEN("attempting to get the key but casting to a different type") {
        THEN("an error is thrown") {
          REQUIRE_THROWS_AS(params.Get<int>(key), std::runtime_error);
        }
      }
      WHEN("attempting to get the pointer with GetVolatile") {
        THEN("an error is thrown") {
          REQUIRE_THROWS_AS(params.GetVolatile<double>(key), std::runtime_error);
        }
      }
    }
    WHEN("We add it to params as volatile") {
      params.Add(key, value, true);
      THEN("We can retrieve the pointer to the object with GetVolatile") {
        double *pval = params.GetVolatile<double>(key);
        REQUIRE(*pval == Approx(value));
        AND_THEN("We can modify the value by dereferencing the pointer") {
          double new_val = 5;
          *pval = new_val;
          AND_THEN("params.get reflects the new value") {
            double output = params.Get<double>(key);
            REQUIRE(output == Approx(new_val));
          }
        }
      }
    }
  }

  GIVEN("An empty params structure") {
    Params params;
    std::string non_existent_key = "key";
    WHEN(" attempting to get a key that does not exist ") {
      THEN("an error is thrown") {
        REQUIRE_THROWS_AS(params.Get<double>(non_existent_key), std::runtime_error);
      }
    }
    WHEN(" attempting to update a key that does not exist ") {
      THEN("an error is thrown") {
        REQUIRE_THROWS_AS(params.Update<double>(non_existent_key, 2.0),
                          std::runtime_error);
      }
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
      REQUIRE_THROWS_AS(params.Get<double>(key), std::runtime_error);
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
