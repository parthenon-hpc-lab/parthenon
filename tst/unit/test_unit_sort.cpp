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

#include <algorithm>
#include <iostream>
#include <string>

#include <catch2/catch.hpp>

#include "kokkos_abstraction.hpp"
#include "parthenon_arrays.hpp"
#include "utils/sort.hpp"

using parthenon::ParArray1D;
using parthenon::sort;

constexpr int N = 100;

struct Key {
  KOKKOS_INLINE_FUNCTION
  Key() {}
  KOKKOS_INLINE_FUNCTION
  Key(const int key, const int value) : key_(key), value_(value) {}

  int key_;
  int value_;
};
struct KeyComparator {
  KOKKOS_INLINE_FUNCTION
  bool operator()(const Key &s1, const Key &s2) { return s1.key_ < s2.key_; }
};

TEST_CASE("Sorting", "[sort]") {
// No sort support on HIP at the moment
#if !defined(KOKKOS_ENABLE_HIP)
  GIVEN("An unordered list of integers") {
    ParArray1D<int> data("Data to sort", N);

    parthenon::par_for(
        parthenon::loop_pattern_flatrange_tag, "initial data", parthenon::DevExecSpace(),
        0, N - 1, KOKKOS_LAMBDA(const int n) { data(n) = 2 * N - n; });

    sort(data);

    auto data_h = Kokkos::create_mirror_view(data);
    Kokkos::deep_copy(data_h, data);

    for (int n = 0; n < N; n++) {
      REQUIRE(data_h(n) == 101 + n);
    }
  }

  GIVEN("An unordered list of key-value pairs") {
    ParArray1D<Key> data("Data to sort", 5);

    parthenon::par_for(
        parthenon::loop_pattern_flatrange_tag, "initial data", parthenon::DevExecSpace(),
        0, 5 - 1, KOKKOS_LAMBDA(const int n) {
          if (n == 0) {
            data(n) = Key(5, 5);
          } else if (n == 1) {
            data(n) = Key(4, 4);
          } else if (n == 3) {
            data(n) = Key(3, 3);
          } else if (n == 2) {
            data(n) = Key(2, 2);
          } else if (n == 4) {
            data(n) = Key(1, 1);
          }
        });

    sort(data, KeyComparator());

    auto data_h = Kokkos::create_mirror_view(data);
    Kokkos::deep_copy(data_h, data);

    REQUIRE(data_h(0).value_ == 1);
    REQUIRE(data_h(1).value_ == 2);
    REQUIRE(data_h(2).value_ == 3);
    REQUIRE(data_h(3).value_ == 4);
    REQUIRE(data_h(4).value_ == 5);
  }
#endif // !defined(KOKKOS_ENABLE_HIP)
}
