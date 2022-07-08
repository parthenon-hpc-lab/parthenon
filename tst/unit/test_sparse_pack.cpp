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
#include <memory>
#include <string>
#include <vector>

#include <catch2/catch.hpp>

#include "basic_types.hpp"
#include "interface/data_collection.hpp"
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

struct v1 : public parthenon::variable_t<false> {
  static std::string name() { return "v1"; }
};

struct v3 : public parthenon::variable_t<false, 3> {
  static std::string name() { return "v3"; }
};

struct v5 : public parthenon::variable_t<false> {
  static std::string name() { return "v5"; }
};

struct vall : public parthenon::variable_t<true> {
  static std::string name() { return "v[0-9]+"; }
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
        parthenon::SparsePack<v1, v3, v5> sparse_pack(&mesh_data);

        const int v = 1;
        int nwrong = 0;
        par_reduce(
            loop_pattern_mdrange_tag, "check imap, scalar", DevExecSpace(), 0,
            NBLOCKS - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA(int b, int k, int j, int i, int &ltot) {
              int lo = sparse_pack.GetLowerBound(v3(), b);
              int hi = sparse_pack.GetUpperBound(v3(), b);
              for (int c = 0; c <= hi - lo; ++c) {
                Real n = i + 1e1 * j + 1e2 * k + 1e4 * c + 1e5 * v + 1e3 * b;
                if (n != sparse_pack(b, lo + c, k, j, i)) ltot += 1;
              }
            },
            nwrong);
        REQUIRE(nwrong == 0);
      }

      THEN("A sparse pack correctly reads based on a regex variable") {
        parthenon::SparsePack<vall> sparse_pack(&mesh_data);

        int nwrong = 0;
        par_reduce(
            loop_pattern_mdrange_tag, "check imap, scalar", DevExecSpace(), 0,
            NBLOCKS - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA(int b, int k, int j, int i, int &ltot) {
              int lo = sparse_pack.GetLowerBound(vall(), b);
              int hi = sparse_pack.GetUpperBound(vall(), b);
              for (int c = 0; c <= hi - lo; ++c) {
                Real n = i + 1e1 * j + 1e2 * k + 1e3 * b;
                if (n != std::fmod(sparse_pack(b, lo + c, k, j, i), 1e4)) ltot += 1;
              }
            },
            nwrong);
        REQUIRE(nwrong == 0);
      }
    }
  }
}
