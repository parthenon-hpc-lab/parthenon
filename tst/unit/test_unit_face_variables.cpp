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
#include <iostream> // debug
#include <memory>
#include <string>
#include <vector>
#include <catch2/catch.hpp>

#include "athena.hpp"
#include "interface/Metadata.hpp"
#include "interface/Variable.hpp"

using parthenon::FaceVariable;
using parthenon::Real;
using parthenon::Metadata;

TEST_CASE("Can create a vector-valued face-variable",
          "[FaceVariable,Constructor,Get,Set]") {
  GIVEN("One-copy, vector metadata, meshblock size, and vector shape") {
    constexpr int blockShape[] = {14, 12, 10}; // arbitrary
    std::vector<int> array_size({3}); // 3-vector
    std::string name("Test Variable");
    Metadata m({Metadata::face, Metadata::vector,
          Metadata::derived, Metadata::oneCopy}, array_size);

    WHEN("We construct a FaceVariable") {
      std::array<int,6> dims({blockShape[0],blockShape[1],blockShape[2],
            array_size[0],1,1});
      FaceVariable f(name,m,dims);
      THEN("Each AthenaArray in the variable has the right shape") {
        REQUIRE(f.Get(1).GetDim1() == blockShape[0] + 1);
        REQUIRE(f.Get(1).GetDim2() == blockShape[1]);
        REQUIRE(f.Get(1).GetDim3() == blockShape[2]);
        REQUIRE(f.Get(2).GetDim1() == blockShape[0]);
        REQUIRE(f.Get(2).GetDim2() == blockShape[1] + 1);
        REQUIRE(f.Get(2).GetDim3() == blockShape[2]);
        REQUIRE(f.Get(3).GetDim1() == blockShape[0]);
        REQUIRE(f.Get(3).GetDim2() == blockShape[1]);
        REQUIRE(f.Get(3).GetDim3() == blockShape[2] + 1);
        for (int d = 1; d <= 3; d++) {
          REQUIRE( f.Get(d).GetDim4() == array_size[0] );
        }
        AND_THEN("The metadata is correct") {
          REQUIRE( f.metadata() == m );
        }
        AND_THEN("We can set array elements") {
          int s = 0;
          for (int d = 1; d <= 3; d++) {
            for (int k = 0; k < blockShape[2]; k++) {
              for (int j = 0; j < blockShape[1]; j++) {
                for (int i = 0; i < blockShape[0]; i++) {
                  f(d,k,j,i) = s;
                  s++;
                }
              }
            }
          }
          // boundaries
          for (int k = 0; k < blockShape[2]; k++) {
            for (int j = 0; j < blockShape[1]; j++) {
              f(1,k,j,blockShape[0]) = -1;
            }
          }
          for (int k = 0; k < blockShape[2]; k++) {
            for (int i = 0; i < blockShape[0]; i++) {
              f(2,k,blockShape[1],i) = -1;
            }
          }
          for (int j = 0; j < blockShape[1]; j++) {
            for (int i = 0; i < blockShape[0]; i++) {
              f(3,blockShape[2],j,i) = -1;
            }
          }
          AND_THEN("We can read them back") {
            int s = 0;
            bool read_correct = true;
            for (int d = 1; d <= 3; d++) {
              for (int k = 0; k < blockShape[2]; k++) {
                for (int j = 0; j < blockShape[1]; j++) {
                  for (int i = 0; i < blockShape[0]; i++) {
                    read_correct = read_correct && f(d,k,j,i) == s;
                    s++;
                  }
                }
              }
            }
            // boundaries
            for (int k = 0; k < blockShape[2]; k++) {
              for (int j = 0; j < blockShape[1]; j++) {
                read_correct = read_correct && f(1,k,j,blockShape[0]) == -1;
              }
            }
            for (int k = 0; k < blockShape[2]; k++) {
              for (int i = 0; i < blockShape[0]; i++) {
                read_correct = read_correct && f(2,k,blockShape[1],i) == -1;
              }
            }
            for (int j = 0; j < blockShape[1]; j++) {
              for (int i = 0; i < blockShape[0]; i++) {
                read_correct = read_correct && f(3,blockShape[2],j,i) == -1;
              }
            }
            REQUIRE(read_correct);
          }
        }
      }
    }
  }
}
