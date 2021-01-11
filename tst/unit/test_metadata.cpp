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

#include <catch2/catch.hpp>

#include "interface/metadata.hpp"

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
    // Note: `parthenon::internal` is subject to change, and so this test may
    // rightfully break later - this test needn't be maintained if so.
    //
    // Checks that the first allocated flag is equal to `Max` - the final built-in
    // flag + 1.
    REQUIRE(f.InternalFlagValue() ==
            static_cast<int>(parthenon::internal::MetadataInternal::Max));
    REQUIRE("TestFlag" == f.Name());

    // It should throw an error if you try to allocate a new flag with the same name.
    REQUIRE_THROWS_AS(Metadata::AllocateNewFlag("TestFlag"), std::runtime_error);
  }
}

TEST_CASE("A Metadata struct is created", "[Metadata]") {
  GIVEN("A default Metadata struct") {
    Metadata m;

#define PARTHENON_INTERNAL_FOR_FLAG(name) REQUIRE(!m.IsSet(Metadata::name));
    PARTHENON_INTERNAL_FOREACH_BUILTIN_FLAG
#undef PARTHENON_INTERNAL_FOR_FLAG
  }

  GIVEN("Setting an arbitrary Metadata flag only sets that flag") {
    Metadata m;

    m.Set(Metadata::FillGhost);

#define PARTHENON_INTERNAL_FOR_FLAG(name)                                                \
  if (Metadata::name != Metadata::FillGhost) REQUIRE(!m.IsSet(Metadata::name));

    PARTHENON_INTERNAL_FOREACH_BUILTIN_FLAG
#undef PARTHENON_INTERNAL_FOR_FLAG

    REQUIRE(m.IsSet(Metadata::FillGhost));
  }

  GIVEN("Two Equivalent Metadata Structs Are Compared") {
    Metadata a({Metadata::Cell}), b({Metadata::Face});

    b.Unset(Metadata::Face);
    b.Set(Metadata::Cell);

    REQUIRE(a == b);
  }

  GIVEN("Two Metadata Structs Are Equivalent But Initialized Differently") {
    Metadata a, b;

    a.Set(Metadata::Cell);
    a.Set(Metadata::Derived);

    b.Set(Metadata::Derived);
    b.Set(Metadata::Cell);

    REQUIRE(a == b);
  }

  GIVEN("Two Different Metadata Structs Are Compared") {
    REQUIRE(Metadata({Metadata::Cell}) != Metadata({Metadata::Face}));
    REQUIRE(Metadata({Metadata::Cell, Metadata::Derived}) !=
            Metadata({Metadata::Face, Metadata::Derived}));
    REQUIRE(Metadata({Metadata::Cell, Metadata::Derived}) !=
            Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy}));
  }
}

TEST_CASE("Metadata created with a sparse ID must be sparse", "[Metadata]") {
  WHEN("We add metadata with a sparse ID but the sparse flag unset") {
    THEN("We raise an error") { REQUIRE_THROWS(Metadata({Metadata::Cell}, 10)); }
  }
}
