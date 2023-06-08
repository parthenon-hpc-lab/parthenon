//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
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

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <catch2/catch.hpp>

#include "basic_types.hpp"
#include "interface/mesh_data.hpp"
#include "interface/meshblock_data.hpp"
#include "interface/metadata.hpp"
#include "interface/state_descriptor.hpp"
#include "interface/variable_pack.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/domain.hpp"
#include "mesh/meshblock.hpp"

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

TEST_CASE("MeshData works as expected for simple packs", "[MeshData]") {
  GIVEN("A set of meshblocks and meshblock and mesh data") {
    constexpr int N = 6;
    constexpr int NDIM = 3;
    constexpr int NBLOCKS = 9;
    const std::vector<int> scalar_shape{N, N, N};
    const std::vector<int> vector_shape{N, N, N, 3};

    Metadata m({Metadata::Independent, Metadata::WithFluxes}, scalar_shape);
    Metadata m_vector({Metadata::Independent, Metadata::WithFluxes, Metadata::Vector},
                      vector_shape);
    Metadata m_face({Metadata::Face, Metadata::Independent, Metadata::WithFluxes});
    auto pkg = std::make_shared<StateDescriptor>("Test package");
    pkg->AddField("v1", m);
    pkg->AddField("v3", m_vector);
    pkg->AddField("v5", m);
    pkg->AddField("v6", m_face);
    BlockList_t block_list = MakeBlockList(pkg, NBLOCKS, N, NDIM);

    MeshData<Real> mesh_data;
    mesh_data.Set(block_list, "base");

    THEN("The number of blocks is correct") { REQUIRE(mesh_data.NumBlocks() == NBLOCKS); }

    WHEN("We initialize the independent variables by hand") {
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
                  Real n = i + 1e1 * j + 1e2 * k + 1e3 * c + 1e4 * v + 1e5 * b;
                  var4(c, k, j, i) = n;
                }
              });
        }
        auto &var = pmbd->Get("v6");
        par_for(
            loop_pattern_mdrange_tag, "initialize v6", DevExecSpace(), kb.s, kb.e + 1,
            jb.s, jb.e + 1, ib.s, ib.e + 1, KOKKOS_LAMBDA(int k, int j, int i) {
              Real n = i + 1e1 * j + 1e2 * k + 1e3 * 0 + 1e4 * 4 + 1e5 * b;
              if (k <= kb.e && j <= jb.e) var(0, 0, 0, 0, k, j, i) = 3 * n + 0;
              if (k <= kb.e && i <= ib.e) var(1, 0, 0, 0, k, j, i) = 3 * n + 1;
              if (j <= jb.e && i <= ib.e) var(2, 0, 0, 0, k, j, i) = 3 * n + 2;
            });
      }

      THEN("A pack for a scalar with a PackIndexMap works") {
        PackIndexMap imap;
        const std::vector<std::string> vlist{"v5"};
        auto pack = mesh_data.PackVariables(vlist, imap);
        const int vlo = imap["v5"].first;
        const int vhi = imap["v5"].second;
        REQUIRE(vlo == 0);
        REQUIRE(vhi == vlo);

        const int c = 0;
        const int v = 2;
        int nwrong = 0;
        par_reduce(
            loop_pattern_mdrange_tag, "check imap, scalar", DevExecSpace(), 0,
            pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA(int b, int k, int j, int i, int &ltot) {
              Real n = i + 1e1 * j + 1e2 * k + 1e3 * c + 1e4 * v + 1e5 * b;
              if (n != pack(b, vlo, k, j, i)) ltot += 1;
            },
            nwrong);
        REQUIRE(nwrong == 0);
      }

      THEN("A pack for a scalar and a face field with a PackIndexMap works") {
        PackIndexMap imap;
        const std::vector<std::string> vlist{"v5", "v6"};
        auto pack = mesh_data.PackVariables(vlist, imap);
        const int v5lo = imap["v5"].first;
        const int v6lo = imap["v6"].first;
        const int v6hi = imap["v6"].second;

        const int c = 0;
        const int v = 2;
        int nwrong = 0;
        using TE = parthenon::TopologicalElement;
        par_reduce(
            loop_pattern_mdrange_tag, "check imap, scalar", DevExecSpace(), 0,
            pack.GetDim(5) - 1, kb.s, kb.e + 1, jb.s, jb.e + 1, ib.s, ib.e + 1,
            KOKKOS_LAMBDA(int b, int k, int j, int i, int &ltot) {
              Real n = i + 1e1 * j + 1e2 * k + 1e3 * c + 1e4 * v + 1e5 * b;
              if (k <= kb.e && j <= jb.e && i <= ib.e && n != pack(b, v5lo, k, j, i))
                ltot += 1;
              n = i + 1e1 * j + 1e2 * k + 1e3 * c + 1e4 * 4 + 1e5 * b;
              if (3 * n + 0 != pack(b, TE::F1, v6lo, k, j, i) && k <= kb.e && j <= jb.e)
                ltot += 1;
              if (3 * n + 1 != pack(b, TE::F2, v6lo, k, j, i) && k <= kb.e && i <= ib.e)
                ltot += 1;
              if (3 * n + 2 != pack(b, TE::F3, v6lo, k, j, i) && j <= jb.e && i <= ib.e)
                ltot += 1;
            },
            nwrong);
        REQUIRE(nwrong == 0);
      }

      THEN("A hardcoded pack for a scalar with no PackIndexMap works") {
        const std::vector<std::string> vlist{"v5"};
        auto pack = mesh_data.PackVariables(vlist);

        const int c = 0;
        const int v = 2;
        int nwrong = 0;
        par_reduce(
            loop_pattern_mdrange_tag, "check hardcoded, scalar", DevExecSpace(), 0,
            pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA(int b, int k, int j, int i, int &ltot) {
              Real n = i + 1e1 * j + 1e2 * k + 1e3 * c + 1e4 * v + 1e5 * b;
              if (n != pack(b, 0, k, j, i)) ltot += 1;
            },
            nwrong);
        REQUIRE(nwrong == 0);
      }
    }
  }
}
