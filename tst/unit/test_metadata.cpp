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

// Third Party Includes
#include <catch2/catch.hpp>

// Parthenon includes
#include <interface/Metadata.hpp>

using parthenon::Metadata;

TEST_CASE("Built-in flags are registered", "[Metadata]") {
    GIVEN("The Built-In Flags") {
#define PARTHENON_INTERNAL_FOR_FLAG(name) REQUIRE(#name == Metadata::name.Name());
    PARTHENON_INTERNAL_FOREACH_BUILTIN_FLAG
#undef PARTHENON_INTERNAL_FOR_FLAG
    }
}

TEST_CASE("A Metadata flag is allocated", "[Metadata]") {
    GIVEN("A User Flag") {
        auto const f = Metadata::AllocateNewFlag("TestFlag");
        // Note: `parthenon::internal` is subject to change, and so this test may rightfully break
        // later - this test needn't be maintained if so.
        //
        // Checks that the first allocated flag is equal to `Max` - the final built-in flag + 1.
        REQUIRE(f.InternalFlagValue() == static_cast<int>(parthenon::internal::MetadataInternal::Max));
        REQUIRE("TestFlag" == f.Name());

        // It should throw an error if you try to allocate a new flag with the same name.
        REQUIRE_THROWS_AS(Metadata::AllocateNewFlag("TestFlag"), std::runtime_error);
    }
}