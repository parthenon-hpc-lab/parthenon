//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
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
#include <memory>
#include <string>
#include <vector>

#include <catch2/catch.hpp>

#include "defs.hpp"
#include "interface/metadata.hpp"
#include "interface/variable.hpp"
#include "kokkos_abstraction.hpp"

using parthenon::DevExecSpace;
using parthenon::FaceVariable;
using parthenon::loop_pattern_mdrange_tag;
using parthenon::Metadata;
using parthenon::par_for;
using parthenon::Real;

TEST_CASE("Can create a vector-valued face-variable",
          "[FaceVariable][Constructor][Get][Set]") {
  GIVEN("One-copy, vector metadata, meshblock size, and vector shape") {
    constexpr int blockShape[] = {14, 12, 10}; // arbitrary
    std::vector<int> array_size({1});          // 1-vector
    std::string name("Test Variable");
    Metadata m({Metadata::Face, Metadata::Vector, Metadata::Derived, Metadata::OneCopy},
               array_size);

    WHEN("We construct a FaceVariable") {
      std::array<int, 6> dims(
          {blockShape[0], blockShape[1], blockShape[2], array_size[0], 1, 1});
      // std::array<int,6>
      // dims({1,1,array_size[0],blockShape[2],blockShape[1],blockShape[0]});
      FaceVariable<int> f(name, dims, m);
      THEN("Each ParArrayND in the variable has the right shape") {
        REQUIRE(f.Get(1).GetDim(1) == blockShape[0] + 1);
        REQUIRE(f.Get(1).GetDim(2) == blockShape[1]);
        REQUIRE(f.Get(1).GetDim(3) == blockShape[2]);
        REQUIRE(f.Get(2).GetDim(1) == blockShape[0]);
        REQUIRE(f.Get(2).GetDim(2) == blockShape[1] + 1);
        REQUIRE(f.Get(2).GetDim(3) == blockShape[2]);
        REQUIRE(f.Get(3).GetDim(1) == blockShape[0]);
        REQUIRE(f.Get(3).GetDim(2) == blockShape[1]);
        REQUIRE(f.Get(3).GetDim(3) == blockShape[2] + 1);
        for (int d = 1; d <= 3; d++) {
          REQUIRE(f.Get(d).GetDim(4) == array_size[0]);
        }
        AND_THEN("The metadata is correct") { REQUIRE(f.metadata() == m); }
        AND_THEN("We can set array elements") {
          auto space = DevExecSpace();
          auto x1f = f.Get(1);
          auto x2f = f.Get(2);
          auto x3f = f.Get(3);
          par_for(
              loop_pattern_mdrange_tag, "set array elements", space, 0, blockShape[2] - 1,
              0, blockShape[1] - 1, 0, blockShape[0] - 1,
              KOKKOS_LAMBDA(const int k, const int j, const int i) {
                x1f(k, j, i) = 1 + k + j + i;
                x2f(k, j, i) = 2 + k + j + i;
                x3f(k, j, i) = 3 + k + j + i;
              });
          // boundaries
          int idx = blockShape[0];
          par_for(
              loop_pattern_mdrange_tag, "set boundaries 0", space, 0, blockShape[2] - 1,
              0, blockShape[1] - 1,
              KOKKOS_LAMBDA(const int k, const int j) { x1f(k, j, idx) = -1; });
          idx = blockShape[1];
          par_for(
              loop_pattern_mdrange_tag, "set boundaries 1", space, 0, blockShape[2] - 1,
              0, blockShape[0] - 1,
              KOKKOS_LAMBDA(const int k, const int i) { x2f(k, idx, i) = -1; });
          idx = blockShape[2];
          par_for(
              loop_pattern_mdrange_tag, "set boundaries 2", space, 0, blockShape[1] - 1,
              0, blockShape[0] - 1,
              KOKKOS_LAMBDA(const int j, const int i) { x3f(idx, j, i) = -1; });
          AND_THEN("We can read them back") {
            int num_incorrect = 1; // != 0
            using policy3D = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
            Kokkos::parallel_reduce(
                policy3D({0, 0, 0}, {blockShape[2], blockShape[1], blockShape[0]}),
                KOKKOS_LAMBDA(const int k, const int j, const int i, int &update) {
                  bool correct = x1f(k, j, i) == 1 + k + j + i;
                  update += correct ? 0 : 1;
                  correct = x2f(k, j, i) == 2 + k + j + i;
                  update += correct ? 0 : 1;
                  correct = x3f(k, j, i) == 3 + k + j + i;
                },
                num_incorrect);
            REQUIRE(num_incorrect == 0);
            // boundaries
            using policy2D = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
            idx = blockShape[0];
            Kokkos::parallel_reduce(
                policy2D({0, 0}, {blockShape[2], blockShape[1]}),
                KOKKOS_LAMBDA(const int k, const int j, int &update) {
                  bool correct = x1f(k, j, idx) == -1;
                  update += correct ? 0 : 1;
                },
                num_incorrect);
            REQUIRE(num_incorrect == 0);
            idx = blockShape[1];
            Kokkos::parallel_reduce(
                policy2D({0, 0}, {blockShape[2], blockShape[0]}),
                KOKKOS_LAMBDA(const int k, const int i, int &update) {
                  bool correct = x2f(k, idx, i) == -1;
                  update += correct ? 0 : 1;
                },
                num_incorrect);
            REQUIRE(num_incorrect == 0);
            idx = blockShape[2];
            Kokkos::parallel_reduce(
                policy2D({0, 0}, {blockShape[1], blockShape[0]}),
                KOKKOS_LAMBDA(const int j, const int i, int &update) {
                  bool correct = x3f(idx, j, i) == -1;
                  update += correct ? 0 : 1;
                },
                num_incorrect);
            REQUIRE(num_incorrect == 0);
          }
        }
      }
    }
  }
}
