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
    RefineLocation(LogicalLocation(2, 3, 3, 3), leaves);

    THEN("LogicalLocations store the correct Morton numbers") {
      int gid = 0;
      uint64_t last_morton = 0; 
      for (auto & [leaf, id] : leaves) { 
        id = gid; 
        // Build the Morton number of this logical location by hand
        std::bitset<64> hand_morton;
        auto lx3 = leaf.lx3();
        auto lx2 = leaf.lx2();
        auto lx1 = leaf.lx1();
        for (int i = 0; i < leaf.level(); ++i) {
          // This is just 2^(leaf.level() - 1 - i) and we use this place by place to 
          // extract the digits of the binary representation of lx*  
          uint64_t cur_place = 1 << (leaf.level() - 1 - i);

          if (lx3 / cur_place == 1) {
            // We start at 62 because only the last 63 bits of the Morton number held 
            // in LogicalLocation store information (the leftmost bit should always be zero)
            hand_morton[62 - (3 * i + 0)] = 1; 
          }
          lx3 = lx3 % cur_place;
          
          if (lx2 / cur_place == 1) {
            hand_morton[62 - (3 * i + 1)] = 1; 
          }
          lx2 = lx2 % cur_place;      
          
          if (lx1 / cur_place == 1) {
            hand_morton[62 - (3 * i + 2)] = 1; 
          }
          lx1 = lx1 % cur_place;      
        }
        // Check that we have the correct Morton number 
        REQUIRE(hand_morton.to_ullong() == leaf.morton().most);

        // Check that the map is in Morton order
        REQUIRE(((leaf.morton().most > last_morton) || (leaf.morton().most == 0)));
        last_morton = leaf.morton().most;
        gid++;
      }
    }
  }
}
