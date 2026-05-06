//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2024 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

#include <catch2/catch.hpp>

#include "interface/var_id.hpp"

using parthenon::InvalidSparseID;
using parthenon::MakeVarLabel;
using parthenon::SparseID;
using parthenon::SparseIDHasher;

TEST_CASE("SparseID basics", "[SparseID]") {
  GIVEN("Scalar, pair, and invalid sparse ids") {
    const auto scalar = SparseID::Scalar(7);
    const auto pair = SparseID::Pair(7, 11);
    const auto invalid = InvalidSparseID;

    THEN("Scalar and pair accessors behave as expected") {
      REQUIRE(scalar() == 7);
      REQUIRE(scalar(0) == 7);
      REQUIRE(scalar(1) == parthenon::InvalidSparseIDValue);
      REQUIRE(pair(0) == 7);
      REQUIRE(pair(1) == 11);
    }

    THEN("Validity is based on the first component") {
      REQUIRE(parthenon::IsValidSparseID(scalar));
      REQUIRE(parthenon::IsValidSparseID(pair));
      REQUIRE_FALSE(parthenon::IsValidSparseID(invalid));
    }

    THEN("Labels preserve the current scalar format") {
      REQUIRE(MakeVarLabel("foo", invalid) == "foo");
      REQUIRE(MakeVarLabel("foo", scalar) == "foo_7");
      REQUIRE(MakeVarLabel("foo", pair) == "foo_7_11");
    }

    THEN("Ordering is lexicographic in the two components") {
      std::vector<SparseID> ids = {pair, SparseID::Pair(2, 0), scalar,
                                   SparseID::Scalar(3), SparseID::Pair(7, 1)};
      std::sort(ids.begin(), ids.end());

      std::vector<SparseID> expected = {SparseID::Pair(2, 0), SparseID::Scalar(3), scalar,
                                        SparseID::Pair(7, 1), pair};
      REQUIRE(ids == expected);
    }

    THEN("Hashing and equality work in associative containers") {
      std::unordered_set<SparseID, SparseIDHasher> ids;
      ids.insert(scalar);
      ids.insert(pair);
      ids.insert(SparseID::Scalar(7));
      ids.insert(SparseID::Pair(7, 11));

      REQUIRE(ids.size() == 2);
      REQUIRE(ids.count(SparseID::Scalar(7)) == 1);
      REQUIRE(ids.count(SparseID::Pair(7, 11)) == 1);

      std::map<SparseID, std::string> labels;
      labels[scalar] = "scalar";
      labels[pair] = "pair";

      REQUIRE(labels.at(SparseID::Scalar(7)) == "scalar");
      REQUIRE(labels.at(SparseID::Pair(7, 11)) == "pair");
    }
  }
}
