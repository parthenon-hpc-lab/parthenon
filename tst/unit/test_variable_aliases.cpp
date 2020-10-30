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
#include <vector>

#include <catch2/catch.hpp>

#include "basic_types.hpp"
#include "config.hpp"
#include "defs.hpp"
#include "interface/container.hpp"
#include "interface/metadata.hpp"
#include "interface/variable.hpp"
#include "interface/variable_pack.hpp"
#include "kokkos_abstraction.hpp"
#include "parthenon_arrays.hpp"

using parthenon::Container;
using parthenon::Metadata;
using parthenon::Real;

TEST_CASE("We can add an alias to a variable in a container", "[ContainerAlias]") {
  GIVEN("A container with a variable in it") {
    Container<Real> rc;
    std::vector<int> block_size = {16, 16, 16};
    Metadata m({Metadata::Independent, Metadata::FillGhost}, block_size);
    rc.Add("var", m);

    WHEN("We add an alias") {
      rc.Add("alias", "var", m);
      THEN("We can extract the variable via either the alias or the original name") {
        REQUIRE(rc.Get("var").label() == "var");
        REQUIRE(rc.Get("alias").label() == "var");
      }
      THEN("The variable is only counted once in a variable pack") {
        auto pack = rc.PackVariables();
        REQUIRE(pack.GetDim(4) == 1);
      }
      THEN("We can pack based on the original name") {
        auto pack = rc.PackVariables(std::vector<std::string>{"var"});
        REQUIRE(pack.GetDim(4) == 1);
      }
      THEN("We can pack based on the aliased name") {
        auto pack = rc.PackVariables(std::vector<std::string>{"alias"});
        REQUIRE(pack.GetDim(4) == 1);
      }
      THEN("Packing both alias and the original names throws an error") {
        REQUIRE_THROWS(rc.PackVariables(std::vector<std::string>{"var", "alias"}));
      }
    }
  }
}
