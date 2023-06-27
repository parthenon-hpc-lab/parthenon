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

#include <bitset>
#include <iostream>
#include <string>

#include <catch2/catch.hpp>

#include "defs.hpp"

using namespace parthenon; 

void RefineLocation(LogicalLocation loc, std::map<LogicalLocation, int> &leaves) { 
  bool deleted = leaves.erase(loc);
  if (deleted) { 
    auto daughters = loc.GetDaughters();
    for (auto &daughter : daughters) {
      leaves.insert({daughter, -1});
    }
  }
}

TEST_CASE("Logical Location", "[Logical Location]") {
  GIVEN("A refinement structure") {
    std::map<LogicalLocation, int> leaves; 
    leaves.insert({LogicalLocation(), -1});    
    RefineLocation(LogicalLocation(), leaves); 
    RefineLocation(LogicalLocation(1, 0, 0, 0), leaves);
    RefineLocation(LogicalLocation(1, 1, 1, 1), leaves);

    int gid = 0; 
    for (auto & [leaf, id] : leaves) { 
      id = gid; 
      printf("(level = %i, lx3 = %li, lx2 = %li lx1 = %li) gid = %2i morton = ", 
             leaf.level(), leaf.lx3(), leaf.lx2(), leaf.lx1(), 
             gid);
      std::cout << std::bitset<64>(leaf.morton().most << 1) << " (" << leaf.morton().most << ")" << std::endl;
      gid++;
    }
  }
}
