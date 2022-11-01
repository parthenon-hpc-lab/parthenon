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

#include <catch2/catch.hpp>

#include "coordinates/coordinates.hpp"
#include "interface/metadata.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/domain.hpp"
#include "prolong_restrict/pr_ops.hpp"
#include "prolong_restrict/prolong_restrict.hpp"

using parthenon::Coordinates_t;
using parthenon::IndexRange;
using parthenon::Metadata;
using parthenon::ParArray6D;
using parthenon::Real;

// Some fake ops classes
template <int DIM>
struct MyProlongOp {
  KOKKOS_FORCEINLINE_FUNCTION static void
  Do(const int l, const int m, const int n, const int k, const int j, const int i,
     const IndexRange &ckb, const IndexRange &cjb, const IndexRange &cib,
     const IndexRange &kb, const IndexRange &jb, const IndexRange &ib,
     const Coordinates_t &coords, const Coordinates_t &coarse_coords,
     const ParArray6D<Real> *pcoarse, const ParArray6D<Real> *pfine) {
    return; // stub
  }
};
template <int DIM>
struct MyRestrictOp {
  KOKKOS_FORCEINLINE_FUNCTION static void
  Do(const int l, const int m, const int n, const int ck, const int cj, const int ci,
     const IndexRange &ckb, const IndexRange &cjb, const IndexRange &cib,
     const IndexRange &kb, const IndexRange &jb, const IndexRange &ib,
     const Coordinates_t &coords, const Coordinates_t &coarse_coords,
     const ParArray6D<Real> *pcoarse, const ParArray6D<Real> *pfine) {
    return; // stub
  }
};

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

TEST_CASE("Refinement Information in Metadata", "[Metadata]") {
  GIVEN("A metadata struct with relevant flags set") {
    Metadata m({Metadata::Cell, Metadata::FillGhost});
    THEN("It knows it's registered for refinement") { REQUIRE(m.IsRefined()); }
    THEN("It has the default Prolongation/Restriction ops") {
      const auto cell_funcs = parthenon::refinement::RefinementFunctions_t::RegisterOps<
          parthenon::refinement_ops::ProlongateCellMinMod,
          parthenon::refinement_ops::RestrictCellAverage>();
      REQUIRE(m.GetRefinementFunctions() == cell_funcs);
    }
    WHEN("We register new operations") {
      m.RegisterRefinementOps<MyProlongOp, MyRestrictOp>();
      THEN("The refinement func must be set to our custom ops") {
        const auto my_funcs =
            parthenon::refinement::RefinementFunctions_t::RegisterOps<MyProlongOp,
                                                                      MyRestrictOp>();
        REQUIRE(m.GetRefinementFunctions() == my_funcs);
      }
    }
  }
  // JMM: I also wanted to test registration of refinement operations
  // but this turns out to be impossible because Catch2 macros are not
  // careful with commas, and the macro interprets commas within the
  // template as separate arguments.
  GIVEN("A metadata struct without the relevant flags set") {
    Metadata m;
    WHEN("We try to request refinement functions") {
      THEN("It should fail") {
        REQUIRE_THROWS_AS(m.GetRefinementFunctions(), std::runtime_error);
      }
    }
  }
}
