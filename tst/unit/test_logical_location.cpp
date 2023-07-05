//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2023 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2021-2023. Triad National Security, LLC. All rights reserved.
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

TEST_CASE("Morton Numbers", "[Morton Numbers]") {
  constexpr int type_size = 8 * sizeof(uint64_t);
  GIVEN("some interleave constants") {
    std::map<int, uint64_t> ics_2d, ics_3d;

    for (int pow = 1; pow <= type_size; pow *= 2) {
      ics_2d[pow] = impl::GetInterleaveConstant<2>(pow);
      ics_3d[pow] = impl::GetInterleaveConstant<3>(pow);
    }

    THEN("the interleave constants have the correct bits") {
      for (const auto &[pow, ic] : ics_2d) {
        std::bitset<type_size> by_hand_constant;
        int idx = 0;
        do {
          for (int i = 0; i < pow; ++i) {
            if (idx < type_size) by_hand_constant[idx] = 1;
            idx++;
          }
          for (int i = 0; i < pow; ++i) {
            if (idx < type_size) by_hand_constant[idx] = 0;
            idx++;
          }
        } while (idx < type_size);
        REQUIRE(ic == by_hand_constant.to_ullong());
      }

      for (const auto &[pow, ic] : ics_3d) {
        std::bitset<type_size> by_hand_constant;
        int idx = 0;
        do {
          for (int i = 0; i < pow; ++i) {
            if (idx < type_size) by_hand_constant[idx] = 1;
            idx++;
          }
          for (int i = 0; i < pow * 2; ++i) {
            if (idx < type_size) by_hand_constant[idx] = 0;
            idx++;
          }
        } while (idx < type_size);
        REQUIRE(ic == by_hand_constant.to_ullong());
      }
    }
  }

  GIVEN("A number with all bits set") {
    uint64_t ones = ~0ULL;
    THEN("interleaving one zero produces the correct bit pattern") {
      constexpr int NVALID_BITS = 32;
      auto interleaved = InterleaveZeros<2, NVALID_BITS>(ones);
      std::bitset<type_size> bs_interleaved(interleaved);
      std::cout << "Single zero interleave : " << bs_interleaved << std::endl;
      int idx = 0;
      do {
        if (idx < 2 * NVALID_BITS) REQUIRE(bs_interleaved[idx] == 1);
        idx++;
        if (idx < 2 * NVALID_BITS) REQUIRE(bs_interleaved[idx] == 0);
        idx++;
      } while (idx < 2 * NVALID_BITS);
    }

    THEN("interleaving two zeros produces the correct bit pattern") {
      constexpr int NVALID_BITS = 21;
      auto interleaved = InterleaveZeros<3, NVALID_BITS>(ones);
      std::bitset<type_size> bs_interleaved(interleaved);
      std::cout << "Double zero interleave : " << bs_interleaved << std::endl;
      int idx = 0;
      do {
        if (idx < 3 * NVALID_BITS) REQUIRE(bs_interleaved[idx] == 1);
        idx++;
        if (idx < 3 * NVALID_BITS) REQUIRE(bs_interleaved[idx] == 0);
        idx++;
        if (idx < 3 * NVALID_BITS) REQUIRE(bs_interleaved[idx] == 0);
        idx++;
      } while (idx < 3 * NVALID_BITS);
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
    int gid = 0;
    for (auto &[n, leaf_id] : leaves)
      leaf_id = gid++;

    // Create a hash map of the leaves
    std::unordered_map<LogicalLocation, int> hash_leaves;
    hash_leaves.insert(std::begin(leaves), std::end(leaves));

    // Create a set of the leaves
    std::set<LogicalLocation> set_leaves;
    for (const auto &[k, v] : leaves)
      set_leaves.insert(k);

    THEN("LogicalLocations store the correct Morton numbers and the map is in Morton "
         "order") {
      uint64_t last_morton = 0;
      for (const auto &[leaf, leaf_id] : leaves) {
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
            // in LogicalLocation store information (the leftmost bit should always be
            // zero)
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
        REQUIRE(hand_morton.to_ullong() == leaf.morton().bits[0]);

        // Check that the map is in Morton order
        REQUIRE(((leaf.morton().bits[0] > last_morton) || (leaf.morton().bits[0] == 0)));
        last_morton = leaf.morton().bits[0];
      }
    }

    THEN("We can correctly find the blocks adjacent to a face") {
      auto base_loc = LogicalLocation(2, 2, 3, 3);

      auto possible_neighbors =
          base_loc.GetPossibleBlocksSurroundingTopologicalElement(1, 0, 0);
      std::set<LogicalLocation> by_hand_elements, automatic_elements;
      // There should be five total neighboring blocks of this face since one neighbor is
      // refined
      by_hand_elements.insert(LogicalLocation(2, 2, 3, 3));
      by_hand_elements.insert(LogicalLocation(3, 6, 6, 6));
      by_hand_elements.insert(LogicalLocation(3, 6, 6, 7));
      by_hand_elements.insert(LogicalLocation(3, 6, 7, 6));
      by_hand_elements.insert(LogicalLocation(3, 6, 7, 7));

      for (auto &n : possible_neighbors) {
        if (hash_leaves.count(n) > 0) {
          automatic_elements.insert(n);
        }
      }
      REQUIRE(by_hand_elements == automatic_elements);
    }

    THEN("We can correctly find the blocks adjacent to an edge") {
      auto base_loc = LogicalLocation(2, 2, 3, 3);

      auto possible_neighbors =
          base_loc.GetPossibleBlocksSurroundingTopologicalElement(1, -1, 0);
      std::set<LogicalLocation> by_hand_elements, automatic_elements;
      // There should be five total neighboring blocks of this edge since one neighbor is
      // refined
      by_hand_elements.insert(LogicalLocation(2, 2, 2, 3));
      by_hand_elements.insert(LogicalLocation(2, 3, 2, 3));
      by_hand_elements.insert(LogicalLocation(2, 2, 3, 3));
      by_hand_elements.insert(LogicalLocation(3, 6, 6, 6));
      by_hand_elements.insert(LogicalLocation(3, 6, 6, 7));

      for (auto &n : possible_neighbors) {
        if (hash_leaves.count(n) > 0) {
          automatic_elements.insert(n);
        }
      }
      REQUIRE(by_hand_elements == automatic_elements);
    }

    THEN("We can correctly find the blocks adjacent to a node") {
      auto base_loc = LogicalLocation(2, 2, 3, 3);

      auto possible_neighbors =
          base_loc.GetPossibleBlocksSurroundingTopologicalElement(1, -1, -1);
      std::set<LogicalLocation> by_hand_elements, automatic_elements;
      // There should be eight total neighboring blocks for this node
      by_hand_elements.insert(LogicalLocation(2, 2, 2, 3));
      by_hand_elements.insert(LogicalLocation(2, 2, 2, 2));
      by_hand_elements.insert(LogicalLocation(2, 3, 2, 3));
      by_hand_elements.insert(LogicalLocation(2, 3, 2, 2));
      by_hand_elements.insert(LogicalLocation(2, 2, 3, 3));
      by_hand_elements.insert(LogicalLocation(2, 2, 3, 2));
      by_hand_elements.insert(LogicalLocation(3, 6, 6, 6));
      by_hand_elements.insert(LogicalLocation(2, 3, 3, 2));

      for (auto &n : possible_neighbors) {
        if (hash_leaves.count(n) > 0) {
          automatic_elements.insert(n);
        }
      }
      REQUIRE(by_hand_elements == automatic_elements);
    }

    THEN("We can find the ownership array of a block") {
      LogicalLocation base_loc(2, 2, 3, 3);
      auto owns = DetermineOwnership(base_loc, set_leaves);

      // Determined by drawing and inspecting diagram
      block_ownership_t by_hand;
      for (int ox1 : {-1, 0, 1})
        for (int ox2 : {-1, 0, 1})
          for (int ox3 : {-1, 0, 1}) {
            by_hand(ox1, ox2, ox3) = true;
          }
      by_hand(1, 0, 0) = false;
      by_hand(1, 0, -1) = false;
      by_hand(1, 0, 1) = false;
      by_hand(1, -1, 0) = false;
      by_hand(1, 1, 0) = false;
      by_hand(1, -1, -1) = false;
      by_hand(1, 1, -1) = false;
      by_hand(1, -1, 1) = false;
      by_hand(1, 1, 1) = false;

      for (int ox3 : {-1, 0, 1}) {
        for (int ox2 : {-1, 0, 1}) {
          for (int ox1 : {-1, 0, 1}) {
            REQUIRE(by_hand(ox1, ox2, ox3) == owns(ox1, ox2, ox3));
          }
        }
      }
    }

    THEN("We can find the ownership array of another block") {
      LogicalLocation base_loc(2, 1, 1, 1);
      auto owns = DetermineOwnership(base_loc, set_leaves);

      // Determined by drawing and inspecting diagram
      block_ownership_t by_hand;
      for (int ox1 : {-1, 0, 1})
        for (int ox2 : {-1, 0, 1})
          for (int ox3 : {-1, 0, 1}) {
            by_hand(ox1, ox2, ox3) = true;
          }
      by_hand(1, 1, 1) = false;

      for (int ox3 : {-1, 0, 1}) {
        for (int ox2 : {-1, 0, 1}) {
          for (int ox1 : {-1, 0, 1}) {
            REQUIRE(by_hand(ox1, ox2, ox3) == owns(ox1, ox2, ox3));
          }
        }
      }
    }

    THEN("We can find the ownership array of yet another block") {
      LogicalLocation base_loc(2, 0, 0, 0);
      auto owns = DetermineOwnership(base_loc, set_leaves);

      // Determined by drawing and inspecting diagram, this should be the
      // ownership structure for every block in a uniform grid
      block_ownership_t by_hand;
      for (int ox1 : {-1, 0, 1})
        for (int ox2 : {-1, 0, 1})
          for (int ox3 : {-1, 0, 1}) {
            by_hand(ox1, ox2, ox3) = false;
          }
      by_hand(-1, -1, -1) = true;
      by_hand(0, -1, -1) = true;
      by_hand(-1, 0, -1) = true;
      by_hand(-1, -1, 0) = true;
      by_hand(-1, 0, 0) = true;
      by_hand(0, -1, 0) = true;
      by_hand(0, 0, -1) = true;
      by_hand(0, 0, 0) = true;
      for (int ox3 : {-1, 0, 1}) {
        for (int ox2 : {-1, 0, 1}) {
          for (int ox1 : {-1, 0, 1}) {
            REQUIRE(by_hand(ox1, ox2, ox3) == owns(ox1, ox2, ox3));
          }
        }
      }
    }

    THEN("We can find the ownership array of yet another block") {
      LogicalLocation base_loc(3, 7, 7, 7);
      auto owns = DetermineOwnership(base_loc, set_leaves);

      // Determined by drawing and inspecting diagram, this is
      // the upper rightmost block in the grid on the finest refinement
      // level so it should own everything
      block_ownership_t by_hand;
      for (int ox1 : {-1, 0, 1})
        for (int ox2 : {-1, 0, 1})
          for (int ox3 : {-1, 0, 1}) {
            by_hand(ox1, ox2, ox3) = true;
          }
      for (int ox3 : {-1, 0, 1}) {
        for (int ox2 : {-1, 0, 1}) {
          for (int ox1 : {-1, 0, 1}) {
            REQUIRE(by_hand(ox1, ox2, ox3) == owns(ox1, ox2, ox3));
          }
        }
      }
    }
  }
}
