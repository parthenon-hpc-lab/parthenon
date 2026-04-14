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
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include <Kokkos_Core.hpp>

#include "basic_types.hpp"
#include "globals.hpp"
#include "interface/data_collection.hpp"
#include "interface/mesh_data.hpp"
#include "interface/meshblock_data.hpp"
#include "interface/metadata.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/meshblock.hpp"
#include "pack/sparse_pack/sparse_pack.hpp"
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
// Helper to create blocks with asymmetric dimensions
BlockList_t MakeBlockListAsymmetric(const std::shared_ptr<StateDescriptor> pkg,
                                    const int NBLOCKS, const int nx1, const int nx2,
                                    const int nx3) {
  BlockList_t block_list;
  block_list.reserve(NBLOCKS);
  for (int i = 0; i < NBLOCKS; ++i) {
    auto pmb = std::make_shared<MeshBlock>();
    // Directly set cellbounds since it's public
    if (nx3 > 0) {
      pmb->cellbounds = parthenon::IndexShape(nx3, nx2, nx1, parthenon::Globals::nghost);
    } else if (nx2 > 0) {
      pmb->cellbounds = parthenon::IndexShape(nx2, nx1, parthenon::Globals::nghost);
    } else {
      pmb->cellbounds = parthenon::IndexShape(nx1, parthenon::Globals::nghost);
    }
    auto &pmbd = pmb->meshblock_data.Get();
    pmbd->Initialize(pkg, pmb);
    block_list.push_back(pmb);
  }
  return block_list;
}

// Test configuration struct
struct TestConfig {
  int ndim;
  int nx1, nx2, nx3;
  int nghost;
  int nkp, njp;
  std::string description;

  int get_nk() const { return (ndim >= 3) ? nx3 : 1; }
  int get_nj() const { return (ndim >= 2) ? nx2 : 1; }
  int get_ni() const { return nx1; }
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
    mesh_data.Initialize(block_list, nullptr);

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
            parthenon::loop_pattern_flatrange_tag, "Test IndexSplit", DevExecSpace(), 0,
            outer_size - 1,
            KOKKOS_LAMBDA(const int outer_idx, int &total_work) {
              const auto krange = sp.GetBoundsK(outer_idx);
              const auto jrange = sp.GetBoundsJ(outer_idx);
              const auto irange = sp.GetInnerBounds(jrange);
              const int local_work =
                  (krange.e - krange.s + 1) * (irange.e - irange.s + 1);
              total_work += local_work;
            },
            Kokkos::Sum<int>(total_work));
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
            parthenon::loop_pattern_flatrange_tag, "Test IndexSplit", DevExecSpace(), 0,
            sp.outer_size() - 1,
            KOKKOS_LAMBDA(const int outer_idx, int &total_work) {
              const auto krange = sp.GetBoundsK(outer_idx);
              const auto jrange = sp.GetBoundsJ(outer_idx);
              const auto irange = sp.GetInnerBounds(jrange);
              total_work += (krange.e - krange.s + 1) * (irange.e - irange.s + 1);
            },
            Kokkos::Sum<int>(total_work));
        REQUIRE(total_work == N * N * N);
      }
    }
  }
}

TEST_CASE("IndexSplit Comprehensive", "[IndexSplit][comprehensive]") {
  // Save original nghost value
  const int original_nghost = parthenon::Globals::nghost;

  // Define test configurations to cover the matrix from the refactor plan
  std::vector<TestConfig> configs = {
      // 1D cases
      {1, 4, 0, 0, 0, IndexSplit::all_outer, IndexSplit::no_outer, "1D small all_outer"},
      {1, 4, 0, 0, 0, 1, 1, "1D small nkp=1 njp=1"},
      {1, 4, 0, 0, 2, IndexSplit::all_outer, IndexSplit::no_outer, "1D small ng=2 all_outer"},
      {1, 16, 0, 0, 0, 4, 1, "1D medium nkp=4 (divides evenly)"},
      {1, 16, 0, 0, 0, 5, 1, "1D medium nkp=5 (doesn't divide)"},

      // 2D cases
      {2, 4, 4, 0, 0, IndexSplit::all_outer, IndexSplit::no_outer, "2D small all_outer,no_outer"},
      {2, 4, 4, 0, 0, IndexSplit::all_outer, IndexSplit::all_outer, "2D small all_outer,all_outer"},
      {2, 4, 4, 0, 2, IndexSplit::all_outer, IndexSplit::all_outer, "2D small ng=2 all_outer,all_outer"},
      {2, 6, 6, 0, 0, 3, 1, "2D medium nkp=3 (divides evenly)"},
      {2, 6, 6, 0, 0, 4, 1, "2D medium nkp=4 (doesn't divide)"},
      {2, 6, 6, 0, 0, 1, 3, "2D medium njp=3 (divides evenly)"},
      {2, 6, 6, 0, 0, 1, 4, "2D medium njp=4 (doesn't divide)"},
      {2, 6, 6, 0, 0, 3, 2, "2D medium nkp=3 njp=2"},

      // 3D cases
      {3, 4, 4, 4, 0, IndexSplit::all_outer, IndexSplit::no_outer, "3D small all_outer,no_outer"},
      {3, 4, 4, 4, 0, IndexSplit::all_outer, IndexSplit::all_outer, "3D small all_outer,all_outer"},
      {3, 4, 4, 4, 0, IndexSplit::no_outer, IndexSplit::no_outer, "3D small no_outer,no_outer"},
      {3, 4, 4, 4, 2, IndexSplit::all_outer, IndexSplit::all_outer, "3D small ng=2 all_outer,all_outer"},
      {3, 6, 6, 6, 0, 3, 1, "3D medium nkp=3 (divides evenly)"},
      {3, 6, 6, 6, 0, 4, 1, "3D medium nkp=4 (doesn't divide)"},
      {3, 6, 6, 6, 0, 1, 3, "3D medium njp=3 (divides evenly)"},
      {3, 6, 6, 6, 0, 1, 4, "3D medium njp=4 (doesn't divide)"},
      {3, 6, 6, 6, 0, 3, 2, "3D medium nkp=3 njp=2"},
      {3, 6, 6, 6, 2, 2, 2, "3D medium ng=2 nkp=2 njp=2"},

      // Asymmetric 3D case
      {3, 4, 8, 16, 0, 4, 2, "3D asymmetric 4x8x16"},
  };

  // Setup package for all tests
  const std::vector<int> scalar_shape{16, 16, 16};
  Metadata m({Metadata::Independent, Metadata::WithFluxes}, scalar_shape);
  auto pkg = std::make_shared<StateDescriptor>("Test package");
  pkg->AddField(v1::name(), m);

  for (const auto &config : configs) {
    GIVEN(config.description) {
      // Set nghost for this test
      parthenon::Globals::nghost = config.nghost;

      // Create mesh blocks
      constexpr int NBLOCKS = 3;
      BlockList_t block_list;
      if (config.nx2 == config.nx1 && config.nx3 == config.nx1) {
        // Symmetric case - use simpler helper
        block_list = MakeBlockList(pkg, NBLOCKS, config.nx1, config.ndim);
      } else {
        // Asymmetric case - use specialized helper
        block_list = MakeBlockListAsymmetric(pkg, NBLOCKS, config.nx1, config.nx2,
                                            config.nx3);
      }

      MeshData<Real> mesh_data("base");
      mesh_data.Initialize(block_list, nullptr);

      WHEN("Using IndexDomain constructor") {
        IndexSplit sp(&mesh_data, IndexDomain::interior, config.nkp, config.njp);

        // Get expected bounds for verification
        auto kb = mesh_data.GetBoundsK(IndexDomain::interior);
        auto jb = mesh_data.GetBoundsJ(IndexDomain::interior);
        auto ib = mesh_data.GetBoundsI(IndexDomain::interior);

        THEN("outer_size() is correct") {
          // Compute expected outer_size based on resolved nkp and njp
          int expected_nkp = config.nkp;
          int expected_njp = config.njp;
          const int total_k = kb.e - kb.s + 1;
          const int total_j = jb.e - jb.s + 1;

          if (expected_nkp == IndexSplit::all_outer) expected_nkp = total_k;
          else if (expected_nkp == IndexSplit::no_outer) expected_nkp = 1;
          else if (expected_nkp == 0) {
#ifdef PARTHENON_ENABLE_GPU
            expected_nkp = total_k;
#else
            expected_nkp = 1;
#endif
          }
          expected_nkp = std::min(expected_nkp, total_k);

          if (expected_njp == IndexSplit::all_outer) expected_njp = total_j;
          else if (expected_njp == IndexSplit::no_outer) expected_njp = 1;
          else if (expected_njp == 0) {
#ifdef PARTHENON_ENABLE_GPU
            expected_njp = total_j; // Simplified - actual code is more complex
#else
            expected_njp = 1;
#endif
          }
          expected_njp = std::min(expected_njp, total_j);

          REQUIRE(sp.outer_size() == expected_nkp * expected_njp);
        }

        THEN("GetBounds methods cover the domain correctly") {
          // Verify (k, j) pairs are covered exactly once
          const int nk = kb.e - kb.s + 1;
          const int nj = jb.e - jb.s + 1;
          std::vector<std::vector<int>> kj_coverage(nk, std::vector<int>(nj, 0));

          for (int p = 0; p < sp.outer_size(); ++p) {
            auto krange = sp.GetBoundsK(p);
            auto jrange = sp.GetBoundsJ(p);

            // Verify ranges are within bounds
            REQUIRE(krange.s >= kb.s);
            REQUIRE(krange.e <= kb.e);
            REQUIRE(jrange.s >= jb.s);
            REQUIRE(jrange.e <= jb.e);

            // Mark all (k,j) pairs for this chunk
            for (int k = krange.s; k <= krange.e; ++k) {
              for (int j = jrange.s; j <= jrange.e; ++j) {
                int kidx = k - kb.s;
                int jidx = j - jb.s;
                kj_coverage[kidx][jidx]++;
              }
            }
          }

          // Verify every (k,j) pair is covered exactly once
          for (int kidx = 0; kidx < nk; ++kidx) {
            for (int jidx = 0; jidx < nj; ++jidx) {
              REQUIRE(kj_coverage[kidx][jidx] == 1);
            }
          }

          // Verify i-range is consistent
          auto ib_check = sp.GetBoundsI(0);
          REQUIRE(ib_check.s == ib.s);
          REQUIRE(ib_check.e == ib.e);
        }

        THEN("get_i and get_deltaj decode inner indices correctly") {
          auto kb_entire = mesh_data.GetBoundsK(IndexDomain::entire);
          auto jb_entire = mesh_data.GetBoundsJ(IndexDomain::entire);
          auto ib_entire = mesh_data.GetBoundsI(IndexDomain::entire);
          const int ni_entire = ib_entire.e - ib_entire.s + 1;

          // Test for first outer index
          if (sp.outer_size() > 0) {
            auto jrange = sp.GetBoundsJ(0);
            auto inner = sp.GetInnerBounds(jrange);

            // Test at start, middle, and end of inner range
            std::vector<int> test_indices = {inner.s};
            if (inner.e > inner.s) {
              test_indices.push_back((inner.s + inner.e) / 2);
              test_indices.push_back(inner.e);
            }

            for (int idx : test_indices) {
              int i = sp.get_i(idx);
              int deltaj = sp.get_deltaj(idx);

              // Verify i is in valid range
              REQUIRE(i >= 0);
              REQUIRE(i < ni_entire);

              // Verify deltaj is in valid range
              int expected_max_deltaj = jrange.e - jrange.s + 1;
              REQUIRE(deltaj >= 0);
              REQUIRE(deltaj < expected_max_deltaj);

              // Verify round-trip: reconstruct idx from i and deltaj
              int reconstructed_idx = deltaj * ni_entire + i;
              REQUIRE(reconstructed_idx == idx);
            }
          }
        }

        THEN("get_max_ni/nj/nk methods return reasonable values") {
          auto ib_entire = mesh_data.GetBoundsI(IndexDomain::entire);
          auto jb_entire = mesh_data.GetBoundsJ(IndexDomain::entire);
          auto kb_entire = mesh_data.GetBoundsK(IndexDomain::entire);

          REQUIRE(sp.get_max_ni() == ib_entire.e - ib_entire.s + 1);
          REQUIRE(sp.get_max_nj() > 0);
          REQUIRE(sp.get_max_nk() > 0);
          REQUIRE(sp.get_max_nij() == sp.get_max_ni() * sp.get_max_nj());
        }

        THEN("Total work across all chunks equals domain volume") {
          int total_work = 0;
          for (int p = 0; p < sp.outer_size(); ++p) {
            auto krange = sp.GetBoundsK(p);
            auto jrange = sp.GetBoundsJ(p);
            auto irange = sp.GetBoundsI(p);
            int local_work = (krange.e - krange.s + 1) *
                            (jrange.e - jrange.s + 1) *
                            (irange.e - irange.s + 1);
            total_work += local_work;
          }

          int expected_work = (kb.e - kb.s + 1) *
                             (jb.e - jb.s + 1) *
                             (ib.e - ib.s + 1);
          REQUIRE(total_work == expected_work);
        }
      }

      WHEN("Using explicit IndexRange constructor") {
        auto kb = mesh_data.GetBoundsK(IndexDomain::interior);
        auto jb = mesh_data.GetBoundsJ(IndexDomain::interior);
        auto ib = mesh_data.GetBoundsI(IndexDomain::interior);

        IndexSplit sp_explicit(&mesh_data, kb, jb, ib, config.nkp, config.njp);
        IndexSplit sp_domain(&mesh_data, IndexDomain::interior, config.nkp, config.njp);

        THEN("Results match IndexDomain constructor") {
          REQUIRE(sp_explicit.outer_size() == sp_domain.outer_size());

          for (int p = 0; p < sp_explicit.outer_size(); ++p) {
            auto kb_exp = sp_explicit.GetBoundsK(p);
            auto kb_dom = sp_domain.GetBoundsK(p);
            REQUIRE(kb_exp.s == kb_dom.s);
            REQUIRE(kb_exp.e == kb_dom.e);

            auto jb_exp = sp_explicit.GetBoundsJ(p);
            auto jb_dom = sp_domain.GetBoundsJ(p);
            REQUIRE(jb_exp.s == jb_dom.s);
            REQUIRE(jb_exp.e == jb_dom.e);

            auto ib_exp = sp_explicit.GetBoundsI(p);
            auto ib_dom = sp_domain.GetBoundsI(p);
            REQUIRE(ib_exp.s == ib_dom.s);
            REQUIRE(ib_exp.e == ib_dom.e);
          }

          REQUIRE(sp_explicit.get_max_ni() == sp_domain.get_max_ni());
          REQUIRE(sp_explicit.get_max_nj() == sp_domain.get_max_nj());
          REQUIRE(sp_explicit.get_max_nk() == sp_domain.get_max_nk());
        }
      }
    }
  }

  // Restore original nghost
  parthenon::Globals::nghost = original_nghost;
}
