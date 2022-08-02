//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2022 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2021-2022. Triad National Security, LLC. All rights reserved.
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
#include <string>

#include <catch2/catch.hpp>

#include "utils/nan_payload_tag.hpp"

TEST_CASE("NaN payload tagging", "[NaN payload]") {
  GIVEN("An NaN with a tag") {
    double qnan = std::numeric_limits<double>::quiet_NaN(); 
    double flag1 = parthenon::GetNaNWithPayloadTag<double>(1); 
    double flag2 = parthenon::GetNaNWithPayloadTag<double>(2);
    
    REQUIRE(std::isnan(qnan));
    REQUIRE(std::isnan(flag1));
    REQUIRE(std::isnan(flag2)); 
    
    REQUIRE(!parthenon::BitwiseCompare(flag1, qnan));
    REQUIRE(!parthenon::BitwiseCompare(flag1, flag2));
    REQUIRE(parthenon::BitwiseCompare(flag1, flag1));

    // Test tagged NaN propagation
    double val1 = flag1; 
    double val2 = 20.0 + flag1; 
    double val3 = 1/0.0 + flag1; 

    REQUIRE(parthenon::BitwiseCompare(flag1, val1));
    REQUIRE(parthenon::BitwiseCompare(flag1, val2));
    REQUIRE(parthenon::BitwiseCompare(flag1, val3));
  }
}
