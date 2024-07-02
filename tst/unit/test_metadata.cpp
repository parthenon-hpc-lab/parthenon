//========================================================================================
// (C) (or copyright) 2020-2024. Triad National Security, LLC. All rights reserved.
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

#include "basic_types.hpp"
#include "coordinates/coordinates.hpp"
#include "interface/metadata.hpp"
#include "interface/variable_state.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/domain.hpp"
#include "prolong_restrict/pr_ops.hpp"
#include "prolong_restrict/prolong_restrict.hpp"

using parthenon::Coordinates_t;
using parthenon::IndexRange;
using parthenon::Metadata;
using parthenon::ParArrayND;
using parthenon::Real;
using parthenon::TopologicalElement;
using parthenon::VariableState;

// Some fake ops classes
struct MyProlongOp {
  static constexpr bool OperationRequired(TopologicalElement fel,
                                          TopologicalElement cel) {
    return fel == cel;
  }
  template <int DIM, TopologicalElement EL = TopologicalElement::CC,
            TopologicalElement /*CEL*/ = TopologicalElement::CC>
  KOKKOS_FORCEINLINE_FUNCTION static void
  Do(const int l, const int m, const int n, const int k, const int j, const int i,
     const IndexRange &ckb, const IndexRange &cjb, const IndexRange &cib,
     const IndexRange &kb, const IndexRange &jb, const IndexRange &ib,
     const Coordinates_t &coords, const Coordinates_t &coarse_coords,
     const ParArrayND<Real, VariableState> *pcoarse,
     const ParArrayND<Real, VariableState> *pfine) {
    return; // stub
  }
};
struct MyRestrictOp {
  static constexpr bool OperationRequired(TopologicalElement fel,
                                          TopologicalElement cel) {
    return fel == cel;
  }
  template <int DIM, TopologicalElement EL = TopologicalElement::CC,
            TopologicalElement /*CEL*/ = TopologicalElement::CC>
  KOKKOS_FORCEINLINE_FUNCTION static void
  Do(const int l, const int m, const int n, const int ck, const int cj, const int ci,
     const IndexRange &ckb, const IndexRange &cjb, const IndexRange &cib,
     const IndexRange &kb, const IndexRange &jb, const IndexRange &ib,
     const Coordinates_t &coords, const Coordinates_t &coarse_coords,
     const ParArrayND<Real, VariableState> *pcoarse,
     const ParArrayND<Real, VariableState> *pfine) {
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
    const std::string name = "TestFlag";
    auto const f = Metadata::AddUserFlag(name);
    // Note: `parthenon::internal` is subject to change, and so this test may
    // rightfully break later - this test needn't be maintained if so.
    //
    // Checks that an allocated flag was ended to the end of existing flags
    REQUIRE(f.InternalFlagValue() == static_cast<int>(Metadata::num_flags) - 1);
    REQUIRE(name == f.Name());

    // Metadata should be able to report that this flag exists and
    // nonexistent flags don't.
    REQUIRE(Metadata::FlagNameExists(name));
    REQUIRE(!(Metadata::FlagNameExists("NoCanDoBuddy")));

    // The identical flag should be retrievable
    auto const f2 = Metadata::GetUserFlag(name);
    REQUIRE(f == f2);

    // It should throw an error if you try to allocate a new flag with the same name.
    REQUIRE_THROWS_AS(Metadata::AddUserFlag(name), std::runtime_error);

    // We can get or add a flag
    auto const f3 = Metadata::GetOrAddFlag("ImFlexible");
    auto const f4 = Metadata::GetOrAddFlag("ImFlexible");
    REQUIRE(f3 == f4);
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

TEST_CASE("Metadata FlagCollection", "[Metadata]") {
  GIVEN("Some metadata flag sets") {
    using parthenon::MetadataFlag;
    using FS_t = Metadata::FlagCollection;
    FS_t set1(std::vector<MetadataFlag>{Metadata::Cell, Metadata::Face}, true);
    FS_t set2;
    set2.TakeUnion(Metadata::Independent, Metadata::FillGhost);
    FS_t set3(Metadata::Requires, Metadata::Overridable);
    WHEN("We take the union") {
      auto s = set1 || set2;
      THEN("We get the flags we expect") {
        const auto &su = s.GetUnions();
        REQUIRE(su.count(Metadata::Cell) > 0);
        REQUIRE(su.count(Metadata::Face) > 0);
        REQUIRE(su.count(Metadata::Independent) > 0);
        REQUIRE(su.count(Metadata::FillGhost) > 0);
        const auto &si = s.GetIntersections();
        REQUIRE(si.empty());
        const auto &se = s.GetExclusions();
        REQUIRE(se.empty());
      }
    }
    WHEN("We take the intersection") {
      auto s = set1 && set3;
      THEN("We get the flags we expect") {
        REQUIRE(s.GetUnions() == set1.GetUnions());
        REQUIRE(s.GetIntersections() == set3.GetIntersections());
        REQUIRE(s.GetExclusions().empty());
      }
    }
    WHEN("We exclude some flags") {
      auto s = set1;
      s.Exclude({Metadata::Requires, Metadata::Overridable});
      THEN("We get the flags we expect") {
        REQUIRE(s.GetUnions() == set1.GetUnions());
        REQUIRE(s.GetIntersections() == set1.GetIntersections());
        REQUIRE(s.GetExclusions() == set3.GetIntersections());
      }
    }
    WHEN("We perform more complicated set arithmetic") {
      auto s = (FS_t({Metadata::Cell}, true) + FS_t({Metadata::Face}, true)) *
                   FS_t(Metadata::Requires) * FS_t(Metadata::Overridable) -
               set2;
      THEN("We get the expected flags") {
        REQUIRE(s.GetUnions() == set1.GetUnions());
        REQUIRE(s.GetIntersections() == set3.GetIntersections());
        REQUIRE(s.GetExclusions() == set2.GetUnions());
      }
    }
  }
}

TEST_CASE("Refinement Information in Metadata", "[Metadata]") {
  GIVEN("A metadata struct with relevant flags set") {
    Metadata m({Metadata::Cell, Metadata::FillGhost});
    THEN("It knows it's registered for refinement") { REQUIRE(m.HasRefinementOps()); }
    THEN("It has the default Prolongation/Restriction ops") {
      const auto cell_funcs = parthenon::refinement::RefinementFunctions_t::RegisterOps<
          parthenon::refinement_ops::ProlongateSharedMinMod,
          parthenon::refinement_ops::RestrictAverage>();
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
  GIVEN("A simple metadata object") {
    using FlagVec = std::vector<parthenon::MetadataFlag>;
    Metadata m(FlagVec{Metadata::Derived, Metadata::OneCopy});
    THEN("It's valid") { REQUIRE(m.IsValid()); }
    // TODO(JMM): This test should go away when issue #844 is resolved
    WHEN("We improperly set prolongation/restriction") {
      m.Set(Metadata::FillGhost);
      THEN("The metadata is no longer valid") { REQUIRE(!m.IsValid()); }
    }
  }
}
