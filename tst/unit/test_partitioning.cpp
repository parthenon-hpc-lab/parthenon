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

#include <list>
#include <stdexcept>
#include <vector>

#include <catch2/catch.hpp>

#include "utils/partition_stl_containers.hpp"

using parthenon::partition::Partition_t;
using parthenon::partition::partition_impl::IntCeil;

inline void check_partitions_even(Partition_t<int> partitions, int nparts,
                                  int elements_per_part) {
  assert(nparts > 0);
  assert(elements_per_part > 0);
  REQUIRE(partitions.size() == (decltype(partitions.size()))nparts);
  for (size_t p = 0; p < partitions.size(); p++) {
    REQUIRE(partitions[p].size() == (decltype(partitions[p].size()))elements_per_part);
  }

  int n_incorrect = 0;
  for (size_t p = 0; p < partitions.size(); p++) {
    for (int i = 0; i < elements_per_part; i++) {
      if (partitions[p][i] != (int)p * elements_per_part + i) {
        n_incorrect++;
      }
    }
  }
  REQUIRE(n_incorrect == 0);
}

TEST_CASE("Check that integer division, rounded up, works", "[IntCeil]") {
  GIVEN("A set of pairs of integers with a common denominator") {
    constexpr int max_fac = 5;
    THEN("Division works as expected") {
      int n_wrong = 0;
      for (int denom = 1; denom <= max_fac; denom++) {
        for (int factor = 1; factor <= max_fac; factor++) {
          int x = factor * denom;
          int y = denom;
          if (IntCeil(x, y) != factor) n_wrong++;
        }
      }
      REQUIRE(n_wrong == 0);
    }
  }
  GIVEN("A set of pairs of integers with non-common denominators") {
    constexpr int max_fac = 5;
    THEN("Division rounds up") {
      int n_wrong = 0;
      for (int denom = 1; denom <= max_fac; denom++) {
        for (int factor = 1; factor <= max_fac; factor++) {
          for (int offset = 1; offset < denom; offset++) {
            int x = factor * denom + offset;
            int y = denom;
            if (IntCeil(x, y) != factor + 1) n_wrong++;
          }
        }
      }
      REQUIRE(n_wrong == 0);
    }
  }
}

TEST_CASE("Check that partitioning a container works", "[Partition]") {
  GIVEN("An attempt to partition an empty container") {
    std::vector<int> v(0);
    int psize = 4;
    THEN("We get zero partitions") {
      auto partitions = parthenon::partition::ToSizeN(v, psize);
      REQUIRE(partitions.size() == 0);
    }
  }
  GIVEN("An attempt to partition into zero partitions") {
    std::vector<int> v = {1, 2, 3};
    int psize = 0;
    THEN("The partition attempt throws an error") {
      REQUIRE_THROWS_AS(parthenon::partition::ToSizeN(v, psize), std::runtime_error);
    }
  }
  GIVEN("An attempt to partition 3 elements into partitions of size 4") {
    constexpr int psize = 4;
    std::vector<int> v = {1, 2, 3};
    THEN("We get a single partition of size 3") {
      auto partitions = parthenon::partition::ToSizeN(v, psize);
      REQUIRE(partitions.size() == 1);
      REQUIRE(partitions[0].size() == 3);
    }
  }

  GIVEN("A list of integers of size 15") {
    constexpr int nelements = 15;
    constexpr int nparts = 3;
    constexpr int elements_per_part = 5;
    std::list<int> l;
    for (int i = 0; i < nelements; i++) {
      l.push_back(i);
    }
    THEN("We can partition the list into 3 partitions of size 5 via Partition::ToSizeN") {
      auto partitions = parthenon::partition::ToSizeN(l, elements_per_part);

      check_partitions_even(partitions, nparts, elements_per_part);
    }
    THEN("We can partition the list into 3 partitions of size 5 via "
         "Partition::ToNPartitions") {
      auto partitions = parthenon::partition::ToNPartitions(l, nparts);

      check_partitions_even(partitions, nparts, elements_per_part);
    }
  }
  GIVEN("A list of size 17") {
    constexpr int nelements = 17;
    constexpr int nparts = 5;
    constexpr int elements_per_part = 4;
    constexpr int leftover = 1;
    std::list<int> l;
    for (int i = 0; i < nelements; i++) {
      l.push_back(i);
    }
    THEN("We can partition the list into 5 partitions of size 4 via Partition::ToSizeN") {
      auto partitions = parthenon::partition::ToSizeN(l, elements_per_part);

      REQUIRE(partitions.size() == nparts);
      AND_THEN("The first 4 partitions are of size 4") {
        for (size_t p = 0; p < partitions.size() - 1; p++) {
          REQUIRE(partitions[p].size() == elements_per_part);
        }
        AND_THEN("The final partition is smaller, reflecting the mismatched sizes") {
          REQUIRE(partitions.back().size() == leftover);
        }
      }
      AND_THEN("The elements are all correct for the first 4 partitions") {
        int n_incorrect = 0;
        for (size_t p = 0; p < partitions.size() - 1; p++) {
          for (int i = 0; i < elements_per_part; i++) {
            if (partitions[p][i] != (int)p * elements_per_part + i) {
              n_incorrect++;
            }
          }
        }
        REQUIRE(n_incorrect == 0);
        AND_THEN("The elements are correct for the final partition") {
          const int p = partitions.size() - 1;
          for (int i = 0; i < leftover; i++) {
            REQUIRE(partitions[p][i] == p * elements_per_part + i);
          }
        }
      }
      AND_THEN("ToNPartitions and ToSizeN agree") {
        auto partitions_v2 = parthenon::partition::ToNPartitions(l, nparts);
        REQUIRE(partitions.size() == partitions_v2.size());
        for (size_t p = 0; p < partitions.size(); p++) {
          REQUIRE(partitions[p].size() == partitions_v2[p].size());
        }
        int n_incorrect = 0;
        for (size_t p = 0; p < partitions.size(); p++) {
          for (size_t i = 0; i < partitions[p].size(); i++) {
            if (partitions[p][i] != partitions_v2[p][i]) {
              n_incorrect++;
            }
          }
        }
        REQUIRE(n_incorrect == 0);
      }
    }
  }
}
