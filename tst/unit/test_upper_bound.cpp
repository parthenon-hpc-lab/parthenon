//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2023 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2023. Triad National Security, LLC. All rights reserved.
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
#include <string>
#include <vector>

#include <catch2/catch.hpp>

#include "Kokkos_Core.hpp"
#include "utils/sort.hpp"

TEST_CASE("upper_bound", "[between][out of bounds][on edges]") {
  GIVEN("A sorted list") {
    const std::vector<double> data{-1, 0, 1e-2, 5, 10};

    Kokkos::View<double *, Kokkos::DefaultExecutionSpace> arr("arr", data.size());
    auto arr_h = Kokkos::create_mirror_view(arr);

    for (int i = 0; i < data.size(); i++) {
      arr_h(i) = data[i];
    }

    Kokkos::deep_copy(arr, arr_h);

    WHEN("a value between entries is given") {
      int result;
      double val = 0.001;
      Kokkos::parallel_reduce(
          "unit::upper_bound::between", 1,
          KOKKOS_LAMBDA(int /*i*/, int &lres) {
            lres = parthenon::upper_bound(arr, val);
          },
          result);
      THEN("then the next index is returned") { REQUIRE(result == 2); }
      THEN("it matches the stl result") {
        REQUIRE(result == std::upper_bound(data.begin(), data.end(), val) - data.begin());
      }
    }
    WHEN("a value below the lower bound is given") {
      int result;
      double val = -1.1;
      Kokkos::parallel_reduce(
          "unit::upper_bound::below", 1,
          KOKKOS_LAMBDA(int /*i*/, int &lres) {
            lres = parthenon::upper_bound(arr, val);
          },
          result);
      THEN("then the first index is returned") { REQUIRE(result == 0); }
      THEN("it matches the stl result") {
        REQUIRE(result == std::upper_bound(data.begin(), data.end(), val) - data.begin());
      }
    }
    WHEN("a value above the upper bound is given") {
      int result;
      double val = 10.01;
      Kokkos::parallel_reduce(
          "unit::upper_bound::above", 1,
          KOKKOS_LAMBDA(int /*i*/, int &lres) {
            lres = parthenon::upper_bound(arr, val);
          },
          result);
      THEN("then the length of the array is returned") { REQUIRE(result == data.size()); }
      THEN("it matches the stl result") {
        REQUIRE(result == std::upper_bound(data.begin(), data.end(), val) - data.begin());
      }
    }
    WHEN("a value on the left edge is given") {
      int result;
      double val = -1;
      Kokkos::parallel_reduce(
          "unit::upper_bound::left", 1,
          KOKKOS_LAMBDA(int /*i*/, int &lres) {
            lres = parthenon::upper_bound(arr, val);
          },
          result);
      THEN("then the second index is returned") { REQUIRE(result == 1); }
      THEN("it matches the stl result") {
        REQUIRE(result == std::upper_bound(data.begin(), data.end(), val) - data.begin());
      }
    }
    WHEN("a value on the right edge is given") {
      int result;
      double val = 10;
      Kokkos::parallel_reduce(
          "unit::upper_bound::right", 1,
          KOKKOS_LAMBDA(int /*i*/, int &lres) {
            lres = parthenon::upper_bound(arr, val);
          },
          result);
      THEN("then the length of the array is returned") { REQUIRE(result == data.size()); }
      THEN("it matches the stl result") {
        REQUIRE(result == std::upper_bound(data.begin(), data.end(), val) - data.begin());
      }
    }
  }
}
