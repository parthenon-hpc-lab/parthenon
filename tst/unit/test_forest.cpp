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

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <catch2/catch.hpp>

#include "kokkos_abstraction.hpp"
#include "mesh/forest/forest.hpp"

using namespace parthenon::forest;
namespace {
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

// Create an n-tree forest with one internal point of valence n and refine one tree
// twice at a location next to the n-valence point. This should result in a forest with
// 10 + 7 * (n - 1) blocks after properly nested refinement. The logical coordinate
// transformations between the produced trees are nontrivial. If nblocks_min !=
// nblocks_max, this creates a forest that is the combination of disconnected n-tree
// forests with n between nblocks_min and nblocks_max
Forest n_blocks(int nblocks_min, int nblocks_max) {
  std::unordered_map<uint64_t, std::shared_ptr<Node>> nodes;
  ForestDefinition forest_def;
  int nc = 0;
  int fc = 0;
  parthenon::Real xoffset = 0.0;
  for (int nblocks = nblocks_min; nblocks <= nblocks_max; ++nblocks) {
    for (int point = 0; point < 2 * nblocks; ++point) {
      nodes[nc + point] =
          Node::create(nc + point, {std::sin(point * M_PI / nblocks) + xoffset,
                                    std::cos(point * M_PI / nblocks)});
    }
    nodes[nc + 2 * nblocks] = Node::create(nc + 2 * nblocks, {0.0 + xoffset, 0.0});
    auto &n = nodes;
    for (int t = 0; t < nblocks; ++t)
      forest_def.AddFace(fc + t,
                         {n[nc + 2 * t + 1], n[nc + 2 * t],
                          n[nc + (2 * t + 2) % (2 * nblocks)], n[nc + 2 * nblocks]});
    nc += 2 * nblocks + 1;
    fc += nblocks;
    xoffset += 2.2;
  }
  auto forest = Forest::Make2D(forest_def);

  // Do some refinements that should propagate into all trees
  fc = 0;
  for (int nblocks = nblocks_min; nblocks <= nblocks_max; ++nblocks) {
    forest.Refine(parthenon::LogicalLocation(fc + 2, 0, 0, 0, 0));
    forest.Refine(parthenon::LogicalLocation(fc + 2, 1, 1, 1, 0));
    forest.Refine(parthenon::LogicalLocation(fc + 2, 2, 3, 3, 0));
    fc += nblocks;
  }

  return forest;
}
} // namespace

TEST_CASE("Simple forest construction", "[forest]") {
  // Create two trees in two dimensions that both have a single block
  auto tree1 = Tree::create(1, 2, 0);
  auto tree2 = Tree::create(2, 2, 0);

  // Periodic connectivity to self
  for (int offy : {-1, 1}) {
    tree1->AddNeighborTree(parthenon::CellCentOffsets(0, offy, 0), tree1,
                           LogicalCoordinateTransformation(), true);
    tree2->AddNeighborTree(parthenon::CellCentOffsets(0, offy, 0), tree2,
                           LogicalCoordinateTransformation(), true);
  }
  // Connectivity to the other tree (both periodic and internal)
  for (int offy : {-1, 0, 1}) {
    for (int offx : {-1, 1}) {
      bool tree1_p;
      bool tree2_p;
      if (offx == 1) {
        tree1_p = false;
        tree2_p = true;
      } else {
        tree1_p = true;
        tree2_p = false;
      }
      tree1->AddNeighborTree(parthenon::CellCentOffsets(offx, offy, 0), tree2,
                             LogicalCoordinateTransformation(), tree1_p);
      tree2->AddNeighborTree(parthenon::CellCentOffsets(offx, offy, 0), tree1,
                             LogicalCoordinateTransformation(), tree2_p);
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

TEST_CASE("Singular forest construction", "[forest]") {
  GIVEN("A forest with three trees and three-valent point") {
    auto forest = n_blocks(3, 3);
    auto locs = forest.GetMeshBlockListAndResolveGids();
    REQUIRE(locs.size() == 24);
  }
  GIVEN("A forest with four trees and four-valent point") {
    auto forest = n_blocks(4, 4);
    auto locs = forest.GetMeshBlockListAndResolveGids();
    REQUIRE(locs.size() == 31);
  }
  GIVEN("A forest with five trees and five-valent point") {
    auto forest = n_blocks(5, 5);
    auto locs = forest.GetMeshBlockListAndResolveGids();
    REQUIRE(locs.size() == 38);
  }
  GIVEN("A forest with three-, four-, and five-valent points") {
    auto forest = n_blocks(3, 5);
    auto locs = forest.GetMeshBlockListAndResolveGids();
    REQUIRE(locs.size() == 93);
  }
}
