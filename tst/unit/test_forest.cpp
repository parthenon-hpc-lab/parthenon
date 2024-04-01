//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2024 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2021-2024. Triad National Security, LLC. All rights reserved.
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

void DerefineAllPossibleLocations(Forest &forest) {
  auto locs = forest.GetMeshBlockListAndResolveGids();
  std::vector<parthenon::LogicalLocation> deref_locs;
  for (const auto &l : locs) {
    auto parent = l.GetParent();
    const int ndim = 2;
    int ndaught{0};
    for (const auto &d : parent.GetDaughters(ndim)) {
      ndaught += forest.count(d);
    }
    if (ndaught == std::pow(2, ndim)) deref_locs.push_back(parent);
  }
  std::sort(deref_locs.begin(), deref_locs.end(),
            [](auto &l, auto &r) { return l.level() > r.level(); });
  for (const auto &d : deref_locs) {
    forest.Derefine(d);
  }
}

TEST_CASE("Forest construction", "[forest]") {
  // Create two trees in two dimensions that both have a single block
  auto tree1 = Tree::create(1, 2, 0);
  auto tree2 = Tree::create(2, 2, 0);

  // Periodic connectivity to self
  for (int offy : {-1, 1}) {
    tree1->AddNeighborTree(CellCentOffsets({0, offy, 0}), tree1, RelativeOrientation());
    tree2->AddNeighborTree(CellCentOffsets({0, offy, 0}), tree2, RelativeOrientation());
  }
  // Connectivity to the other tree (both periodic and internal)
  for (int offy : {-1, 0, 1}) {
    for (int offx : {-1, 1}) {
      tree1->AddNeighborTree(CellCentOffsets({offx, offy, 0}), tree2,
                             RelativeOrientation());
      tree2->AddNeighborTree(CellCentOffsets({offx, offy, 0}), tree1,
                             RelativeOrientation());
    }
  }

  // Create a forest from the two trees
  Forest forest;
  forest.AddTree(tree1);
  forest.AddTree(tree2);

  GIVEN("A periodic forest with two trees") {
    auto locs = forest.GetMeshBlockListAndResolveGids();
    REQUIRE(locs.size() == 2);

    // Refine the lower left block repeatedly and check that there are the correct
    // number of blocks after each refinement when things are properly nested. Numbers
    // were determined by hand.
    forest.Refine(locs[0]);
    locs = forest.GetMeshBlockListAndResolveGids();
    REQUIRE(locs.size() == 5);

    forest.Refine(locs[0]);
    locs = forest.GetMeshBlockListAndResolveGids();
    REQUIRE(locs.size() == 11);

    forest.Refine(locs[0]);
    locs = forest.GetMeshBlockListAndResolveGids();
    REQUIRE(locs.size() == 23);

    // Now flag all blocks for refinement and derefine employing
    // proper nesting. Should just be reverse of previous refinement
    // operations.
    DerefineAllPossibleLocations(forest);
    locs = forest.GetMeshBlockListAndResolveGids();
    REQUIRE(locs.size() == 11);

    DerefineAllPossibleLocations(forest);
    locs = forest.GetMeshBlockListAndResolveGids();
    REQUIRE(locs.size() == 5);

    DerefineAllPossibleLocations(forest);
    locs = forest.GetMeshBlockListAndResolveGids();
    REQUIRE(locs.size() == 2);
  }
}
