//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2023 The Parthenon collaboration
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

#include <bitset>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "defs.hpp"
#include "mesh/forest/block_ownership.hpp"
#include "mesh/forest/logical_coordinate_transformation.hpp"
#include "utils/indexer.hpp"

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

    for (int power = 1; power <= type_size; power *= 2) {
      ics_2d[power] = impl::GetInterleaveConstant<2>(power);
      ics_3d[power] = impl::GetInterleaveConstant<3>(power);
    }

    THEN("the interleave constants have the correct bits") {
      for (const auto &[power, ic] : ics_2d) {
        std::bitset<type_size> by_hand_constant;
        int idx = 0;
        do {
          for (int i = 0; i < power; ++i) {
            if (idx < type_size) by_hand_constant[idx] = 1;
            idx++;
          }
          for (int i = 0; i < power; ++i) {
            if (idx < type_size) by_hand_constant[idx] = 0;
            idx++;
          }
        } while (idx < type_size);
        REQUIRE(ic == by_hand_constant.to_ullong());
      }

      for (const auto &[power, ic] : ics_3d) {
        std::bitset<type_size> by_hand_constant;
        int idx = 0;
        do {
          for (int i = 0; i < power; ++i) {
            if (idx < type_size) by_hand_constant[idx] = 1;
            idx++;
          }
          for (int i = 0; i < power * 2; ++i) {
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
    std::unordered_set<LogicalLocation> set_leaves;
    for (const auto &[k, v] : leaves)
      set_leaves.insert(k);

    // Create neighbor blocks from the leaves
    std::vector<parthenon::forest::NeighborLocation> neighbor_locs;
    for (const auto &[k, v] : leaves) {
      neighbor_locs.emplace_back(k, k,
                                 parthenon::forest::LogicalCoordinateTransformation());
    }

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
        // Check that we have the correct Morton number, least significant bits
        // should be zero
        REQUIRE(hand_morton.to_ullong() == leaf.morton().bits[0]);
        REQUIRE(0ULL == leaf.morton().bits[1]);
        REQUIRE(0ULL == leaf.morton().bits[2]);

        // Check that the map is in Morton order
        REQUIRE(((leaf.morton().bits[0] > last_morton) || (leaf.morton().bits[0] == 0)));
        last_morton = leaf.morton().bits[0];
      }
    }

    THEN("We can find the ownership array of a block") {
      LogicalLocation base_loc(2, 2, 3, 3);
      auto owns = DetermineOwnership(base_loc, neighbor_locs);

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
      auto owns = DetermineOwnership(base_loc, neighbor_locs);

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
      auto owns = DetermineOwnership(base_loc, neighbor_locs);

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
      auto owns = DetermineOwnership(base_loc, neighbor_locs);

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

    GIVEN("An ownership array of a block") {
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

      // Make a corner that would not be owned by an interior block in a uniform grid
      // owned
      by_hand(-1, 1, 1) = true;

      const int N = 3;
      THEN("We can build the correct index range for setting the buffer") {
        auto owns =
            GetIndexRangeMaskFromOwnership(TopologicalElement::F1, by_hand, -1, 0, 0);
        using p_t = std::pair<int, int>;
        SpatiallyMaskedIndexer6D idxer(owns, p_t{0, 0}, p_t{0, 0}, p_t{0, 0}, p_t{0, N},
                                       p_t{0, N}, p_t{0, N});
        for (int idx = 0; idx < idxer.size(); ++idx) {
          const auto [t, u, v, k, j, i] = idxer(idx);
          REQUIRE(idxer.IsActive(k, j, i));
        }
      }

      THEN("We can build the correct index range for setting the buffer") {
        auto owns =
            GetIndexRangeMaskFromOwnership(TopologicalElement::F2, by_hand, -1, 0, 0);
        using p_t = std::pair<int, int>;
        SpatiallyMaskedIndexer6D idxer(owns, p_t{0, 0}, p_t{0, 0}, p_t{0, 0}, p_t{0, N},
                                       p_t{0, N}, p_t{0, N});
        for (int idx = 0; idx < idxer.size(); ++idx) {
          const auto [t, u, v, k, j, i] = idxer(idx);
          if (idxer.IsActive(k, j, i)) REQUIRE(j != N);
          if (!idxer.IsActive(k, j, i)) REQUIRE(j == N);
        }
      }

      THEN("We can build the correct index range for setting the buffer") {
        auto owns =
            GetIndexRangeMaskFromOwnership(TopologicalElement::F3, by_hand, -1, 0, 0);
        using p_t = std::pair<int, int>;
        SpatiallyMaskedIndexer6D idxer(owns, p_t{0, 0}, p_t{0, 0}, p_t{0, 0}, p_t{0, N},
                                       p_t{0, N}, p_t{0, N});
        for (int idx = 0; idx < idxer.size(); ++idx) {
          const auto [t, u, v, k, j, i] = idxer(idx);
          if (idxer.IsActive(k, j, i)) REQUIRE(k != N);
          if (!idxer.IsActive(k, j, i)) REQUIRE(k == N);
        }
      }

      THEN("We can build the correct index range for setting the buffer") {
        // Imagine that we have a z-edge field that is being communicated across the left
        // x-face of the sender in a uniform grid so that the sender owns the left edge of
        // the face, the interior of the face, but another block owns the right edge of
        // the face For a z-edge, ownership should be independent of the z-direction since
        // the z-coordinate is centered. This is generic I think for centered coordinates
        // of elements

        // For passing an edge oriented in the z-direction along the x-face of a block,
        // given the ownership status of the block given above, all indices at the upper
        // end of the y-index range should be masked out but everything else should be
        // unmasked
        auto owns =
            GetIndexRangeMaskFromOwnership(TopologicalElement::E3, by_hand, -1, 0, 0);
        using p_t = std::pair<int, int>;
        SpatiallyMaskedIndexer6D idxer(owns, p_t{0, 0}, p_t{0, 0}, p_t{0, 0}, p_t{0, N},
                                       p_t{0, N}, p_t{0, N});
        for (int idx = 0; idx < idxer.size(); ++idx) {
          const auto [t, u, v, k, j, i] = idxer(idx);
          if (idxer.IsActive(k, j, i)) REQUIRE(j != N);
          if (!idxer.IsActive(k, j, i)) REQUIRE(j == N);
        }
      }

      THEN("We can build the correct index range for setting the buffer") {
        auto owns =
            GetIndexRangeMaskFromOwnership(TopologicalElement::E2, by_hand, -1, 0, 0);
        using p_t = std::pair<int, int>;
        SpatiallyMaskedIndexer6D idxer(owns, p_t{0, 0}, p_t{0, 0}, p_t{0, 0}, p_t{0, N},
                                       p_t{0, N}, p_t{0, N});
        for (int idx = 0; idx < idxer.size(); ++idx) {
          const auto [t, u, v, k, j, i] = idxer(idx);
          if (idxer.IsActive(k, j, i)) REQUIRE(k != N);
          if (!idxer.IsActive(k, j, i)) REQUIRE(k == N);
        }
      }

      THEN("We can build the correct index range for setting the buffer") {
        auto owns =
            GetIndexRangeMaskFromOwnership(TopologicalElement::E1, by_hand, -1, 0, 0);
        using p_t = std::pair<int, int>;
        SpatiallyMaskedIndexer6D idxer(owns, p_t{0, 0}, p_t{0, 0}, p_t{0, 0}, p_t{0, N},
                                       p_t{0, N}, p_t{0, N});
        for (int idx = 0; idx < idxer.size(); ++idx) {
          const auto [t, u, v, k, j, i] = idxer(idx);
          if (idxer.IsActive(k, j, i)) REQUIRE((k != N && j != N));
          if (!idxer.IsActive(k, j, i)) REQUIRE((k == N || j == N));
        }
      }

      THEN("We can build the correct index range for setting the buffer") {
        auto owns =
            GetIndexRangeMaskFromOwnership(TopologicalElement::NN, by_hand, -1, 0, 0);
        using p_t = std::pair<int, int>;
        SpatiallyMaskedIndexer6D idxer(owns, p_t{0, 0}, p_t{0, 0}, p_t{0, 0}, p_t{0, N},
                                       p_t{0, N}, p_t{0, N});
        for (int idx = 0; idx < idxer.size(); ++idx) {
          const auto [t, u, v, k, j, i] = idxer(idx);
          if (idxer.IsActive(k, j, i))
            REQUIRE(((k != N && j != N) || (i == 0 && j == N && k == N)));
          if (!idxer.IsActive(k, j, i))
            REQUIRE(((k == N || j == N) && !(i == 0 && j == N && k == N)));
        }
      }
    }
  }
}
