//========================================================================================
// (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
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

#include "basic_types.hpp"
#include "interface/data_collection.hpp"
#include "interface/mesh_data.hpp"
#include "interface/meshblock_data.hpp"
#include "interface/metadata.hpp"
#include "interface/sparse_pack.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/meshblock.hpp"

// TODO(jcd): can't call the MeshBlock constructor without mesh_refinement.hpp???
#include "mesh/mesh_refinement.hpp"

using parthenon::BlockList_t;
using parthenon::DevExecSpace;
using parthenon::IndexDomain;
using parthenon::loop_pattern_mdrange_tag;
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

TEST_CASE("Test behavior of sparse packs", "[SparsePack]") {
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

    MeshData<Real> mesh_data;
    mesh_data.Set(block_list, "base");

    WHEN("We initialize the independent variables by hand and deallocate one") {
      auto ib = block_list[0]->cellbounds.GetBoundsI(IndexDomain::entire);
      auto jb = block_list[0]->cellbounds.GetBoundsJ(IndexDomain::entire);
      auto kb = block_list[0]->cellbounds.GetBoundsK(IndexDomain::entire);
      const std::vector<std::string> all_indep{"v1", "v3", "v5"};
      for (int b = 0; b < NBLOCKS; ++b) {
        auto &pmb = block_list[b];
        auto &pmbd = pmb->meshblock_data.Get();
        for (int v = 0; v < all_indep.size(); ++v) {
          auto &vnam = all_indep[v];
          auto var = pmbd->Get(vnam);
          auto var4 = var.data.Get<4>();
          int num_components = var.GetDim(4);
          par_for(
              loop_pattern_mdrange_tag, "initialize " + vnam, DevExecSpace(), kb.s, kb.e,
              jb.s, jb.e, ib.s, ib.e, KOKKOS_LAMBDA(int k, int j, int i) {
                for (int c = 0; c < num_components; ++c) {
                  Real n = i + 1e1 * j + 1e2 * k + 1e4 * c + 1e5 * v + 1e3 * b;
                  var4(c, k, j, i) = n;
                }
              });
        }
      }
      // Deallocate a variable on an arbitrary block
      block_list[2]->DeallocateSparse("v3");

      THEN("A sparse pack correctly loads this data and can be read from v3 on all "
           "blocks") {
        // Create a pack use type variables
        auto sparse_pack =
            parthenon::SparsePack<v5, v3>::Get<MeshData<Real>, Real>(&mesh_data, {Metadata::WithFluxes});

        // Create the same pack using strings
        auto tup = parthenon::SparsePack<>::Get<MeshData<Real>, Real>(
            &mesh_data, std::vector<std::string>{"v5", "v3"},
            std::vector<parthenon::MetadataFlag>{Metadata::WithFluxes});
        parthenon::SparsePack<> sparse_pack_notype = std::get<0>(tup);
        auto pack_map = std::get<1>(tup);
        parthenon::PackIdx iv3(pack_map["v3"]);

        // Make sure that we have only cached one pack, since these should be the
        // same base pack
        REQUIRE(mesh_data.GetSparsePackCache().size() == 1);

        const int v = 1; // v3 is the second variable in the loop above so v = 1 there
        int nwrong = 0;
        par_reduce(
            loop_pattern_mdrange_tag, "check vector", DevExecSpace(), 0,
            sparse_pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA(int b, int k, int j, int i, int &ltot) {
              int lo = sparse_pack.GetLowerBound(b, v3());
              int hi = sparse_pack.GetUpperBound(b, v3());
              for (int c = 0; c <= hi - lo; ++c) {
                Real n = i + 1e1 * j + 1e2 * k + 1e4 * c + 1e5 * v + 1e3 * b;
                if (n != sparse_pack(b, lo + c, k, j, i)) ltot += 1;
                if (n != sparse_pack(b, v3(c), k, j, i)) ltot += 1;
              }
              lo = sparse_pack_notype.GetLowerBound(b, iv3);
              hi = sparse_pack_notype.GetUpperBound(b, iv3);
              for (int c = 0; c <= hi - lo; ++c) {
                Real n = i + 1e1 * j + 1e2 * k + 1e4 * c + 1e5 * v + 1e3 * b;
                if (n != sparse_pack_notype(b, lo + c, k, j, i)) ltot += 1;
                if (n != sparse_pack_notype(b, iv3 + c, k, j, i)) ltot += 1;
              }
            },
            nwrong);
        REQUIRE(nwrong == 0);
      }

      THEN("A sparse pack correctly loads this data and can be read from v3 on a single "
           "block") {
        auto sparse_pack =
            parthenon::SparsePack<v5, v3>::Get<MeshBlockData<Real>, Real>(block_list[0]->meshblock_data.Get().get());

        const int v = 1; // v3 is the second variable in the loop above so v = 1 there
        int nwrong = 0;
        int b = 0;
        par_reduce(
            loop_pattern_mdrange_tag, "check vector", DevExecSpace(), kb.s, kb.e, jb.s,
            jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA(int k, int j, int i, int &ltot) {
              int lo = sparse_pack.GetLowerBound(b, v3());
              int hi = sparse_pack.GetUpperBound(b, v3());
              for (int c = 0; c <= hi - lo; ++c) {
                Real n = i + 1e1 * j + 1e2 * k + 1e4 * c + 1e5 * v + 1e3 * b;
                if (n != sparse_pack(b, lo + c, k, j, i)) ltot += 1;
              }
            },
            nwrong);
        REQUIRE(nwrong == 0);
      }

      THEN("A sparse pack correctly reads based on a regex variable") {
        auto sparse_pack =
            parthenon::SparsePack<parthenon::variable_names::any>::Get<MeshData<Real>, Real>(&mesh_data);

        auto tup = parthenon::SparsePack<>::Get<MeshData<Real>, Real>(
            &mesh_data, std::vector<std::pair<std::string, bool>>{{".*", true}});
        auto sparse_pack_notype = std::get<0>(tup);
        auto pack_map = std::get<1>(tup);
        parthenon::PackIdx iall(pack_map[".*"]);

        int nwrong = 0;
        par_reduce(
            loop_pattern_mdrange_tag, "check all", DevExecSpace(), 0, NBLOCKS - 1, kb.s,
            kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA(int b, int k, int j, int i, int &ltot) {
              int lo = sparse_pack.GetLowerBound(b, parthenon::variable_names::any());
              int hi = sparse_pack.GetUpperBound(b, parthenon::variable_names::any());
              for (int c = 0; c <= hi - lo; ++c) {
                Real n = i + 1e1 * j + 1e2 * k + 1e3 * b;
                if (std::abs(n - std::fmod(sparse_pack(b, lo + c, k, j, i), 1e4)) >
                    1.e-12)
                  ltot += 1;
              }
              lo = sparse_pack_notype.GetLowerBound(b, iall);
              hi = sparse_pack_notype.GetUpperBound(b, iall);
              for (int c = 0; c <= hi - lo; ++c) {
                Real n = i + 1e1 * j + 1e2 * k + 1e3 * b;
                if (std::abs(n - std::fmod(sparse_pack_notype(b, lo + c, k, j, i), 1e4)) >
                    1.e-12)
                  ltot += 1;
                sparse_pack_notype(b, lo + c, k, j, i) = 0.0;
              }
            },
            nwrong);
        REQUIRE(nwrong == 0);
      }
    }
  }
}
