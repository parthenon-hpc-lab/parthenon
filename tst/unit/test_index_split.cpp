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
#include <memory>
#include <string>
#include <vector>

#include <catch2/catch.hpp>

#include <Kokkos_Core.hpp>

#include "basic_types.hpp"
#include "globals.hpp"
#include "interface/data_collection.hpp"
#include "interface/mesh_data.hpp"
#include "interface/meshblock_data.hpp"
#include "interface/metadata.hpp"
#include "interface/sparse_pack.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/meshblock.hpp"
#include "parthenon/package.hpp"
#include "utils/index_split.hpp"

// TODO(jcd): can't call the MeshBlock constructor without mesh_refinement.hpp???
#include "mesh/mesh_refinement.hpp"

using namespace parthenon::package::prelude;
using parthenon::BlockList_t;
using parthenon::DevExecSpace;
using parthenon::IndexDomain;
using parthenon::IndexSplit;
using parthenon::MeshBlock;
using parthenon::MeshBlockData;
using parthenon::MeshData;
using parthenon::Metadata;
using parthenon::PackIndexMap;
using parthenon::par_for;
using parthenon::Real;
using parthenon::StateDescriptor;

namespace {
BlockList_t MakeBlockList(const std::shared_ptr<StateDescriptor> pkg, const int NBLOCKS,
                          const int NSIDE, const int NDIM) {
  BlockList_t block_list;
  block_list.reserve(NBLOCKS);
  for (int i = 0; i < NBLOCKS; ++i) {
    auto pmb = std::make_shared<MeshBlock>(NSIDE, NDIM);
    auto &pmbd = pmb->meshblock_data.Get();
    pmbd->Initialize(pkg, pmb);
    block_list.push_back(pmb);
  }
  return block_list;
}
// JMM: Variables aren't really needed for this test but...
struct v1 : public parthenon::variable_names::base_t<false> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION v1(Ts &&...args)
      : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}
  static std::string name() { return "v1"; }
};
struct v3 : public parthenon::variable_names::base_t<false, 3> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION v3(Ts &&...args)
      : parthenon::variable_names::base_t<false, 3>(std::forward<Ts>(args)...) {}
  static std::string name() { return "v3"; }
};
struct v5 : public parthenon::variable_names::base_t<false> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION v5(Ts &&...args)
      : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}
  static std::string name() { return "v5"; }
};
} // namespace

TEST_CASE("IndexSplit", "[IndexSplit]") {
  GIVEN("A set of meshblocks and meshblock and mesh data") {
    constexpr int N = 6;
    constexpr int NDIM = 3;
    constexpr int NBLOCKS = 9;
    const std::vector<int> scalar_shape{N, N, N};
    const std::vector<int> vector_shape{N, N, N, 3};

    Metadata m({Metadata::Independent, Metadata::WithFluxes}, scalar_shape);
    Metadata m_vector({Metadata::Independent, Metadata::WithFluxes, Metadata::Vector},
                      vector_shape);
    auto pkg = std::make_shared<StateDescriptor>("Test package");
    pkg->AddField(v1::name(), m);
    pkg->AddField(v3::name(), m_vector);
    pkg->AddField(v5::name(), m);
    BlockList_t block_list = MakeBlockList(pkg, NBLOCKS, N, NDIM);

    MeshData<Real> mesh_data("base");
    mesh_data.Set(block_list, nullptr, NDIM);

    WHEN("We initialize an IndexSplit with all outer k and no outer j") {
      IndexSplit sp(&mesh_data, IndexDomain::interior, IndexSplit::all_outer,
                    IndexSplit::no_outer);
      THEN("The outer range should be appropriate") { REQUIRE(sp.outer_size() == N); }
      THEN("The inner ranges should be appropriate") {
        using atomic_view = Kokkos::MemoryTraits<Kokkos::Atomic>;
        Kokkos::View<int *, atomic_view> nwrong("nwrong", 1);
        parthenon::par_for_outer(
            DEFAULT_OUTER_LOOP_PATTERN, "Test IndexSplit", DevExecSpace(), 0, 0, 0,
            sp.outer_size() - 1, // N * N - 1
            KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int outer_idx) {
              const auto krange = sp.GetBoundsK(outer_idx);
              const auto jrange = sp.GetBoundsJ(outer_idx);
              const auto irange = sp.GetInnerBounds(jrange);
              // JMM: Note that these are little cleaner without ghosts
              if (!(krange.s == outer_idx)) nwrong(0) += 1;
              if (!(krange.e == outer_idx)) nwrong(0) += 1;
              if (!(jrange.s == 0)) nwrong(0) += 1;
              if (!(jrange.e == N - 1)) nwrong(0) += 1;
              if (!(irange.s == 0)) nwrong(0) += 1;
              if (!(irange.e == (N * N - 1))) nwrong(0) += 1;
            });
        auto nwrong_h = Kokkos::create_mirror_view(nwrong);
        Kokkos::deep_copy(nwrong_h, nwrong);
        REQUIRE(nwrong_h(0) == 0);
      }
    }
    WHEN("We initialize an IndexSplit with outer k and outer j") {
      IndexSplit sp(&mesh_data, IndexDomain::interior, IndexSplit::all_outer,
                    IndexSplit::all_outer);
      THEN("the outer index range should be appropriate") {
        REQUIRE(sp.outer_size() == (N * N));
      }
      THEN("The inner index ranges should be appropriate") {
        using atomic_view = Kokkos::MemoryTraits<Kokkos::Atomic>;
        Kokkos::View<int *, atomic_view> nwrong("nwrong", 1);
        parthenon::par_for_outer(
            DEFAULT_OUTER_LOOP_PATTERN, "Test IndexSplit", DevExecSpace(), 0, 0, 0,
            sp.outer_size() - 1,
            KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int outer_idx) {
              const auto krange = sp.GetBoundsK(outer_idx);
              const auto jrange = sp.GetBoundsJ(outer_idx);
              const auto irange = sp.GetInnerBounds(jrange);
              if (!(krange.s == krange.e)) nwrong(0) += 1;
              if (!(jrange.s == jrange.e)) nwrong(0) += 1;
              if (!(irange.s == 0)) nwrong(0) += 1;
              if (!(irange.e == N - 1)) nwrong(0) += 1;
            });
        auto nwrong_h = Kokkos::create_mirror_view(nwrong);
        Kokkos::deep_copy(nwrong_h, nwrong);
        REQUIRE(nwrong_h(0) == 0);
      }
    }

    WHEN("We initialize with nkp > NK") {
      constexpr int NKP = N + 1;
      REQUIRE(NKP > N);
      IndexSplit sp(&mesh_data, IndexDomain::interior, NKP, IndexSplit::no_outer);
      THEN("The outer index range should not overrun the mesh domain") {
        REQUIRE(sp.outer_size() == N);
      }
    }

    WHEN("We initialize with nkp*njp > NK*NJ") {
      constexpr int NTOOBIG = N + 1;
      REQUIRE(NTOOBIG > N);
      IndexSplit sp(&mesh_data, IndexDomain::interior, NTOOBIG, NTOOBIG);
      THEN("The outer index range should not overrun the mesh domain") {
        REQUIRE(sp.outer_size() == N * N);
      }
    }

    WHEN("We initialize an IndexSplit so that work and nj are evenly divisible") {
      constexpr int NJP = 3;
      REQUIRE(N % NJP == 0);
      IndexSplit sp(&mesh_data, IndexDomain::interior, IndexSplit::all_outer, NJP);
      THEN("The outer index range should be appropriate") {
        REQUIRE(sp.outer_size() == NJP * N);
      }
      THEN("The inner index ranges should be appropriate") {
        using atomic_view = Kokkos::MemoryTraits<Kokkos::Atomic>;
        Kokkos::View<int *, atomic_view> nwrong("nwrong", 1);
        parthenon::par_for_outer(
            DEFAULT_OUTER_LOOP_PATTERN, "Test IndexSplit", DevExecSpace(), 0, 0, 0,
            sp.outer_size() - 1,
            KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int outer_idx) {
              const auto krange = sp.GetBoundsK(outer_idx);
              const auto jrange = sp.GetBoundsJ(outer_idx);
              const auto irange = sp.GetInnerBounds(jrange);
              if (!(krange.s == krange.e)) nwrong(0) += 1;
              if (!(jrange.e == jrange.s + 1)) nwrong(0) += 1;
              if (!((irange.e - irange.s + 1) == (N / NJP) * N)) nwrong(0) += 1;
            });
        auto nwrong_h = Kokkos::create_mirror_view(nwrong);
        Kokkos::deep_copy(nwrong_h, nwrong);
        REQUIRE(nwrong_h(0) == 0);
      }
    }

    WHEN("We initialize an IndexSplit so that work and nk are evenly divisible") {
      constexpr int NKP = 3;
      REQUIRE(N % NKP == 0);
      IndexSplit sp(&mesh_data, IndexDomain::interior, NKP, IndexSplit::no_outer);
      THEN("The outer index range should be appropriate") {
        REQUIRE(sp.outer_size() == NKP);
      }
      THEN("The inner index ranges should be appropriate") {
        using atomic_view = Kokkos::MemoryTraits<Kokkos::Atomic>;
        Kokkos::View<int *, atomic_view> nwrong("nwrong", 1);
        parthenon::par_for_outer(
            DEFAULT_OUTER_LOOP_PATTERN, "Test IndexSplit", DevExecSpace(), 0, 0, 0,
            sp.outer_size() - 1,
            KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int outer_idx) {
              const auto krange = sp.GetBoundsK(outer_idx);
              const auto jrange = sp.GetBoundsJ(outer_idx);
              const auto irange = sp.GetInnerBounds(jrange);
              // The user is expected to loop over k manually between
              // the outer loop and the inner.
              if (!((krange.e - krange.s + 1) == (N / NKP))) nwrong(0) += 1;
              if (!((jrange.e - jrange.s + 1) == N)) nwrong(0) += 1;
              if (!((irange.e - irange.s + 1) == (N * N))) nwrong(0) += 1;
            });
        auto nwrong_h = Kokkos::create_mirror_view(nwrong);
        Kokkos::deep_copy(nwrong_h, nwrong);
        REQUIRE(nwrong_h(0) == 0);
      }
    }

    WHEN("We initialize an IndexSplit so the work and nj aren't evenly divisible") {
      constexpr int NJP = 4;
      REQUIRE(N % NJP > 0);
      IndexSplit sp(&mesh_data, IndexDomain::interior, IndexSplit::all_outer, NJP);
      THEN("The outer index range should be appropriate") {
        REQUIRE(sp.outer_size() == NJP * N);
      }
      THEN("The inner index ranges should be appropriate") {
        int total_work = 0;
        const int outer_size = sp.outer_size();
        parthenon::par_reduce(
            DEFAULT_LOOP_PATTERN, "Test IndexSplit", DevExecSpace(), 0,  outer_size - 1,
            KOKKOS_LAMBDA(const int outer_idx, int &total_work) {
              const auto krange = sp.GetBoundsK(outer_idx);
              const auto jrange = sp.GetBoundsJ(outer_idx);
              const auto irange = sp.GetInnerBounds(jrange);
              const int local_work = (krange.e - krange.s + 1) * (irange.e - irange.s + 1);
              total_work += local_work;
            }, Kokkos::Sum<int>(total_work));
        REQUIRE(total_work == N * N * N);
      }
    }

    WHEN("We initialize an IndexSplit so the work and nk aren't evenly divisible") {
      constexpr int NKP = 4;
      REQUIRE(N % NKP > 0);
      IndexSplit sp(&mesh_data, IndexDomain::interior, NKP, IndexSplit::no_outer);
      THEN("The outer index range should be appropriate") {
        REQUIRE(sp.outer_size() == NKP);
      }
      THEN("The inner index ranges should be appropriate") {
        int total_work = 0;
        parthenon::par_reduce(
            DEFAULT_LOOP_PATTERN, "Test IndexSplit", DevExecSpace(), 0, sp.outer_size() - 1,
            KOKKOS_LAMBDA(const int outer_idx, int &total_work) {
              const auto krange = sp.GetBoundsK(outer_idx);
              const auto jrange = sp.GetBoundsJ(outer_idx);
              const auto irange = sp.GetInnerBounds(jrange);
              total_work += (krange.e - krange.s + 1) * (irange.e - irange.s + 1);
            }, Kokkos::Sum<int>(total_work));
        REQUIRE(total_work == N * N * N);
      }
    }
  }
}
