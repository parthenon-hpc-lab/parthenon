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
#include "parthenon_arrays.hpp"
#include "utils/nan_payload_tag.hpp"

using real_t = double;
using policy2d = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
using arr2d_t = parthenon::ParArray2D<real_t>;

TEST_CASE("NaN payload tagging", "[NaN payload]") {
  real_t qnan = std::numeric_limits<real_t>::quiet_NaN();
  real_t flag1 = parthenon::GetNaNWithPayloadTag<real_t>(1);
  real_t flag2 = parthenon::GetNaNWithPayloadTag<real_t>(2);
  GIVEN("An two NaNs with tags") {

    // This should at least warn us if the platform will not
    // support NaN based tagging of fields. The other test
    // should still succeed in that case.
    CHECK_NOFAIL(std::numeric_limits<real_t>::is_iec559);

    REQUIRE(std::isnan(qnan));
    REQUIRE(std::isnan(flag1));
    REQUIRE(std::isnan(flag2));

    if (std::numeric_limits<real_t>::is_iec559) {
      REQUIRE(!parthenon::BitwiseCompare(flag1, qnan));
      REQUIRE(!parthenon::BitwiseCompare(flag1, flag2));
    }

    REQUIRE(parthenon::BitwiseCompare(flag1, flag1));

    // Test tagged NaN propagation
    real_t val1 = flag1;
    real_t val2 = 20.0 + flag1;
    real_t val3 = 1 / 0.0 + flag1;

    REQUIRE(parthenon::BitwiseCompare(flag1, val1));
    REQUIRE(parthenon::BitwiseCompare(flag1, val2));
    REQUIRE(parthenon::BitwiseCompare(flag1, val3));
  }

  GIVEN("An array filled with tags") {
    const int N = 50;
    arr2d_t arr2d("Test array", N, N);

    Kokkos::parallel_for(
        policy2d({0, 0}, {N, N}),
        KOKKOS_LAMBDA(const int j, const int i) { arr2d(j, i) = flag1; });

    int num_flag1 = 0;
    int num_flag2 = 0;
    Kokkos::parallel_reduce(
        policy2d({0, 0}, {N, N}),
        KOKKOS_LAMBDA(const int j, const int i, int &l_num_flag1, int &l_num_flag2) {
          if (parthenon::BitwiseCompare(flag1, arr2d(i, j))) ++l_num_flag1;
          if (parthenon::BitwiseCompare(flag2, arr2d(i, j))) ++l_num_flag2;
        },
        num_flag1, num_flag2);

    REQUIRE(num_flag1 == N * N);
    if (std::numeric_limits<real_t>::is_iec559) {
      REQUIRE(num_flag2 == 0);
    }
  }
}
