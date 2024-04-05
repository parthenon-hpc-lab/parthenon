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
#include "kokkos_abstraction.hpp"
#include "mesh/meshblock.hpp"

// TODO(jcd): can't call the MeshBlock constructor without mesh_refinement.hpp???
#include "mesh/mesh_refinement.hpp"

using parthenon::DataCollection;
using parthenon::DevExecSpace;
using parthenon::loop_pattern_flatrange_tag;
using parthenon::MeshBlock;
using parthenon::MeshBlockData;
using parthenon::MeshData;
using parthenon::Metadata;
using parthenon::par_for;
using parthenon::Real;
using parthenon::StateDescriptor;

TEST_CASE("Adding MeshBlockData objects to a DataCollection", "[DataCollection]") {
  GIVEN("An DataCollection with a base MeshBlockData with some variables") {
    DataCollection<MeshBlockData<Real>> d;
    auto pmb = std::make_shared<MeshBlock>();

    std::vector<int> size(6, 1);
    Metadata m_ind({Metadata::Independent}, size);
    Metadata m_one({Metadata::OneCopy}, size);

    auto pgk = std::make_shared<StateDescriptor>("DataCollection test");
    pgk->AddField("var1", m_ind);
    pgk->AddField("var2", m_one);
    pgk->AddField("var3", m_ind);

    auto &mbd = d.Get();
    mbd->Initialize(pgk, pmb);

    auto &v1 = mbd->Get("var1").data;
    auto &v2 = mbd->Get("var2").data;
    auto &v3 = mbd->Get("var3").data;
    par_for(
        loop_pattern_flatrange_tag, "init vars", DevExecSpace(), 0, 0,
        KOKKOS_LAMBDA(const int i) {
          v1(0) = 111;
          v2(0) = 222;
          v3(0) = 333;
        });
    WHEN("We add a MeshBlockData to the container") {
      auto x = d.Add("full", mbd);
      auto &xv1 = x->Get("var1").data;
      auto &xv2 = x->Get("var2").data;
      auto &xv3 = x->Get("var3").data;
      par_for(
          loop_pattern_flatrange_tag, "init vars", DevExecSpace(), 0, 0,
          KOKKOS_LAMBDA(const int i) {
            xv1(0) = 11;
            xv2(0) = 22;
            xv3(0) = 33;
          });
      auto hv1 = v1.GetHostMirrorAndCopy();
      auto hv2 = v2.GetHostMirrorAndCopy();
      auto hv3 = v3.GetHostMirrorAndCopy();
      auto hxv1 = xv1.GetHostMirrorAndCopy();
      auto hxv2 = xv2.GetHostMirrorAndCopy();
      auto hxv3 = xv3.GetHostMirrorAndCopy();
      THEN("Independent variables should have new storage") {
        REQUIRE(hxv1(0) != hv1(0));
        REQUIRE(hxv3(0) != hv3(0));
      }
      AND_THEN("OneCopy variables should not have new storage") {
        REQUIRE(hxv2(0) == hv2(0));
      }
    }
    AND_WHEN("We want only a subset of variables in a new MeshBlockData") {
      // reset vars
      par_for(
          loop_pattern_flatrange_tag, "init vars", DevExecSpace(), 0, 0,
          KOKKOS_LAMBDA(const int i) { v2(0) = 222; });
      auto x = d.Add("part", mbd, {"var2", "var3"});
      THEN("Requesting the missing variables should throw") {
        // This no longer call PARTHENON_REQUIRE_THROWS, it just call PARTHENON_REQUIRE
        // since throwing does not work nicely with threads
        //REQUIRE_THROWS(x->Get("var1"));
      }
      AND_THEN("Requesting the specified variables should work as expected") {
        auto &xv2 = x->Get("var2").data;
        auto &xv3 = x->Get("var3").data;
        par_for(
            loop_pattern_flatrange_tag, "init vars", DevExecSpace(), 0, 0,
            KOKKOS_LAMBDA(const int i) {
              xv2(0) = 22;
              xv3(0) = 33;
            });
        auto hv2 = v2.GetHostMirrorAndCopy();
        auto hv3 = v3.GetHostMirrorAndCopy();
        auto hxv2 = xv2.GetHostMirrorAndCopy();
        auto hxv3 = xv3.GetHostMirrorAndCopy();
        REQUIRE(hxv3(0) != hv3(0));
        REQUIRE(hxv2(0) == hv2(0));
      }
    }
  }
}
