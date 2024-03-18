//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2024. Triad National Security, LLC. All rights reserved.
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
#include "globals.hpp"
#include "interface/mesh_data.hpp"
#include "interface/meshblock_data.hpp"
#include "interface/metadata.hpp"
#include "interface/state_descriptor.hpp"
#include "interface/variable_pack.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/domain.hpp"
#include "mesh/meshblock.hpp"
#include "outputs/output_utils.hpp"

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
using parthenon::IndexShape;

using namespace parthenon::OutputUtils;

TEST_CASE("The VarInfo object produces appropriate ranges", "[VarInfo][OutputUtils]") {
  GIVEN("A MeshBlock with some vars on it") {
    constexpr int NG = 2;
    constexpr int NSIDE = 16;
    constexpr int NDIM = 3;

    constexpr auto interior = parthenon::IndexDomain::interior;
    constexpr auto entire = parthenon::IndexDomain::entire;
    parthenon::Globals::nghost = NG;

    const std::string scalar_cell = "scalar_cell";

    Metadata m({Metadata::Cell, Metadata::Independent});
    auto pkg = std::make_shared<StateDescriptor>("Test package");
    pkg->AddField("scalar_cell", m);

    auto pmb = std::make_shared<MeshBlock>(NSIDE, NDIM);
    auto pmbd = pmb->meshblock_data.Get();
    pmbd->Initialize(pkg, pmb);

    IndexShape cellbounds = pmb->cellbounds;
    THEN("The CellBounds object is reasonable") {
      REQUIRE(cellbounds.ncellsk(entire) == NSIDE + 2*NG);
      REQUIRE(cellbounds.ncellsk(interior) == NSIDE);
      REQUIRE(cellbounds.ncellsj(entire) == NSIDE + 2*NG);
      REQUIRE(cellbounds.ncellsj(interior) == NSIDE);
      REQUIRE(cellbounds.ncellsi(entire) == NSIDE + 2*NG);
      REQUIRE(cellbounds.ncellsi(interior) == NSIDE);
    }

    WHEN("We Initialize VarInfo on a scalar cell var") {
      auto v = pmbd->GetVarPtr(scalar_cell);
      VarInfo info(v, cellbounds);

      THEN("The shape is correct over both interior and entire") {
        std::vector<int> shape(10);
        int ndim = info.FillShape<int>(interior, shape.data());
        REQUIRE(ndim == 3 + 1); // block index + k,j,i
        for (int i = 0; i < ndim - 1; ++i) { // don't think about block index
          REQUIRE(shape[i] == NSIDE);
        }
      }
    }
  }
}
