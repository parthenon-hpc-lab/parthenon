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
#include "mesh/meshblock.hpp"

// TODO(jcd): can't call the MeshBlock constructor without mesh_refinement.hpp???
#include "mesh/mesh_refinement.hpp"

using parthenon::DataCollection;
using parthenon::MeshBlock;
using parthenon::MeshBlockData;
using parthenon::MeshData;
using parthenon::Metadata;
using parthenon::Real;

TEST_CASE("Adding MeshBlockData objects to a DataCollection", "[DataCollection]") {
  GIVEN("An DataCollection with a base MeshBlockData with some variables") {
    DataCollection<MeshBlockData<Real>> d;
    auto pmb = std::make_shared<MeshBlock>();
    auto &mbd = d.Get();
    mbd->SetBlockPointer(pmb);
    std::vector<int> size(6, 1);
    Metadata m_ind({Metadata::Independent});
    Metadata m_one({Metadata::OneCopy});
    mbd->Add("var1", m_ind, size);
    mbd->Add("var2", m_one, size);
    mbd->Add("var3", m_ind, size);
    auto &v1 = mbd->Get("var1");
    auto &v2 = mbd->Get("var2");
    auto &v3 = mbd->Get("var3");
    v1(0) = 111;
    v2(0) = 222;
    v3(0) = 333;
    WHEN("We add a MeshBlockData to the container") {
      auto x = d.Add("full", mbd);
      auto &xv1 = x->Get("var1");
      auto &xv2 = x->Get("var2");
      auto &xv3 = x->Get("var3");
      xv1(0) = 11;
      xv2(0) = 22;
      xv3(0) = 33;
      THEN("Independent variables should have new storage") {
        REQUIRE(xv1(0) != v1(0));
        REQUIRE(xv3(0) != v3(0));
      }
      AND_THEN("OneCopy variables should not have new storage") {
        REQUIRE(xv2(0) == v2(0));
      }
    }
    AND_WHEN("We want only a subset of variables in a new MeshBlockData") {
      // reset vars
      v1(0) = 111;
      v2(0) = 222;
      v3(0) = 333;
      auto x = d.Add("part", mbd, {"var2", "var3"});
      THEN("Requesting the missing variables should throw") {
        REQUIRE_THROWS(x->Get("var1"));
      }
      AND_THEN("Requesting the specified variables should work as expected") {
        auto &xv2 = x->Get("var2");
        auto &xv3 = x->Get("var3");
        xv2(0) = 22;
        xv3(0) = 33;
        REQUIRE(xv3(0) != v3(0));
        REQUIRE(xv2(0) == v2(0));
      }
    }
  }
}
