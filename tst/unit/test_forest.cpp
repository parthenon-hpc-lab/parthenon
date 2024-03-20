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

#include "kokkos_abstraction.hpp"
#include "mesh/forest/forest.hpp"

using namespace parthenon::forest; 

TEST_CASE("Forest construction", "[forest]") {
  auto tree1 = Tree::create(1, 2, 0);
  auto tree2 = Tree::create(2, 2, 0);
  
  // Periodic connectivity to self
  tree1->AddNeighborTree(CellCentOffsets({0,  1, 0}), tree1, RelativeOrientation());
  tree1->AddNeighborTree(CellCentOffsets({0, -1, 0}), tree1, RelativeOrientation());
  tree2->AddNeighborTree(CellCentOffsets({0,  1, 0}), tree2, RelativeOrientation());
  tree2->AddNeighborTree(CellCentOffsets({0, -1, 0}), tree2, RelativeOrientation());

  // Connectivity to the other tree (both periodic and internal)
  for (int offy : {-1, 0, 1}) {
      tree1->AddNeighborTree(CellCentOffsets({ 1, offy, 0}), tree2, RelativeOrientation());
      tree1->AddNeighborTree(CellCentOffsets({-1, offy, 0}), tree2, RelativeOrientation());
  }
  for (int offy : {-1, 0, 1}) {
      tree2->AddNeighborTree(CellCentOffsets({ 1, offy, 0}), tree1, RelativeOrientation());
      tree2->AddNeighborTree(CellCentOffsets({-1, offy, 0}), tree1, RelativeOrientation());
  }

  Forest forest; 
  forest.AddTree(tree1);
  forest.AddTree(tree2);

  GIVEN("A periodic forest with two trees") {
    REQUIRE(forest.GetMeshBlockListAndResolveGids().size() == 2);
  }
}
