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
using parthenon::IndexShape;
using parthenon::loop_pattern_mdrange_tag;
using parthenon::MeshBlock;
using parthenon::MeshBlockData;
using parthenon::MeshData;
using parthenon::Metadata;
using parthenon::PackIndexMap;
using parthenon::par_for;
using parthenon::Real;
using parthenon::StateDescriptor;

using namespace parthenon::OutputUtils;

TEST_CASE("The VarInfo object produces appropriate ranges", "[VarInfo][OutputUtils]") {
  GIVEN("A MeshBlock with some vars on it") {
    constexpr int NG = 2;
    constexpr int NSIDE = 16;
    constexpr int NDIM = 3;
    constexpr int NFULL = (NSIDE + 2 * NG);

    constexpr auto interior = parthenon::IndexDomain::interior;
    constexpr auto entire = parthenon::IndexDomain::entire;

    // JMM: This needs to be reset to 0 when we're done, because other
    // tests assume it's unset, thus zero-initialized.
    parthenon::Globals::nghost = NG;

    auto pkg = std::make_shared<StateDescriptor>("Test package");

    const std::string scalar_cell = "scalar_cell";
    Metadata m({Metadata::Cell, Metadata::Independent});
    pkg->AddField(scalar_cell, m);

    const std::string tensor_cell = "tensor_cell";
    m = Metadata({Metadata::Cell, Metadata::Independent}, std::vector<int>{3, 4});
    pkg->AddField(tensor_cell, m);

    const std::string tensor_none = "tensor_none";
    m = Metadata({Metadata::None, Metadata::Independent}, std::vector<int>{3, 4});
    pkg->AddField(tensor_none, m);

    // four-vector-valued var with a vector at each face
    const std::string vector_face = "four_vector_face";
    m = Metadata({Metadata::Face, Metadata::Independent}, std::vector<int>{4});
    pkg->AddField(vector_face, m);

    // vector-valued var with a single value at each edge
    const std::string scalar_edge = "scalar_edge";
    m = Metadata({Metadata::Edge, Metadata::Independent});
    pkg->AddField(scalar_edge, m);

    const std::vector<std::string> var_names = {scalar_cell, tensor_cell, tensor_none,
                                                vector_face, scalar_edge};

    auto pmb = std::make_shared<MeshBlock>(NSIDE, NDIM);
    auto pmbd = pmb->meshblock_data.Get();
    pmbd->Initialize(pkg, pmb);

    IndexShape cellbounds = pmb->cellbounds;
    THEN("The CellBounds object is reasonable") {
      REQUIRE(cellbounds.ncellsk(entire) == NSIDE + 2 * NG);
      REQUIRE(cellbounds.ncellsk(interior) == NSIDE);
      REQUIRE(cellbounds.ncellsj(entire) == NSIDE + 2 * NG);
      REQUIRE(cellbounds.ncellsj(interior) == NSIDE);
      REQUIRE(cellbounds.ncellsi(entire) == NSIDE + 2 * NG);
      REQUIRE(cellbounds.ncellsi(interior) == NSIDE);
    }

    WHEN("We initialize VarInfo on a scalar cell var") {
      auto v = pmbd->GetVarPtr(scalar_cell);
      VarInfo info(v, cellbounds);

      THEN("The shape is correct over both interior and entire") {
        std::vector<int> shape(10);
        int ndim = info.FillShape<int>(interior, shape.data());
        REQUIRE(ndim == 3);
        for (int i = 0; i < ndim; ++i) {
          REQUIRE(shape[i] == NSIDE);
        }

        ndim = info.FillShape<int>(entire, shape.data());
        REQUIRE(ndim == 3);
        for (int i = 0; i < ndim; ++i) {
          REQUIRE(shape[i] == NSIDE + 2 * NG);
        }
      }
      THEN("The size and tensorsize are correct") {
        REQUIRE(info.Size() == NFULL * NFULL * NFULL);
        REQUIRE(info.TensorSize() * info.ntop_elems == 1);
      }
    }

    WHEN("We initialize VarInfo on a tensor cell var") {
      auto v = pmbd->GetVarPtr(tensor_cell);
      VarInfo info(v, cellbounds);

      THEN("The shape is correct over both interior and entire") {
        std::vector<int> shape(10);
        int ndim = info.FillShape<int>(interior, shape.data());
        REQUIRE(ndim == 5);
        REQUIRE(shape[0] == 4);
        REQUIRE(shape[1] == 3);
        for (int i = 2; i < ndim; ++i) {
          REQUIRE(shape[i] == NSIDE);
        }

        ndim = info.FillShape<int>(entire, shape.data());
        REQUIRE(ndim == 5);
        REQUIRE(shape[0] == 4);
        REQUIRE(shape[1] == 3);
        for (int i = 2; i < ndim; ++i) {
          REQUIRE(shape[i] == NSIDE + 2 * NG);
        }
      }
      THEN("The size and tensorsize are correct") {
        REQUIRE(info.Size() == 3 * 4 * NFULL * NFULL * NFULL);
        REQUIRE(info.TensorSize() * info.ntop_elems == 3 * 4);
      }
    }

    WHEN("We initialize VarInfo on a tensor no-centering var") {
      auto v = pmbd->GetVarPtr(tensor_none);
      VarInfo info(v, cellbounds);

      THEN("The shape is correct over both interior and entire") {
        std::vector<int> shape(10);
        int ndim = info.FillShape<int>(interior, shape.data());
        REQUIRE(ndim == 2);
        REQUIRE(shape[0] == 4);
        REQUIRE(shape[1] == 3);

        ndim = info.FillShape<int>(entire, shape.data());
        REQUIRE(ndim == 2);
        REQUIRE(shape[0] == 4);
        REQUIRE(shape[1] == 3);
      }
      THEN("The size and tensorsize are correct") {
        REQUIRE(info.Size() == 3 * 4);
        REQUIRE(info.TensorSize() * info.ntop_elems == 3 * 4);
      }
    }

    WHEN("We initialize VarInfo on a vector face var") {
      auto v = pmbd->GetVarPtr(vector_face);
      VarInfo info(v, cellbounds);

      THEN("The shape is correct over both interior and entire") {
        std::vector<int> shape(10);
        int ndim = info.FillShape<int>(interior, shape.data());
        REQUIRE(ndim == 5);
        REQUIRE(shape[0] == 3);
        REQUIRE(shape[1] == 4);
        for (int i = 2; i < ndim; ++i) {
          REQUIRE(shape[i] == NSIDE + 1);
        }
        info.FillShape<int>(entire, shape.data());
        for (int i = 2; i < ndim; ++i) {
          REQUIRE(shape[i] == NSIDE + 2 * NG + 1);
        }
      }
      THEN("The size and tensorsize are correct") {
        REQUIRE(info.Size() == 3 * 4 * (NFULL + 1) * (NFULL + 1) * (NFULL + 1));
        REQUIRE(info.TensorSize() * info.ntop_elems == 3 * 4);
      }
      THEN("Requesting reversed padded shape provides correctly shaped object") {
        constexpr int ND = VarInfo::VNDIM;
        auto padded_shape = info.GetPaddedShapeReversed(interior);
        REQUIRE(padded_shape.size() == ND);
        REQUIRE(padded_shape[ND - 1] == NSIDE + 1);
        REQUIRE(padded_shape[ND - 2] == NSIDE + 1);
        REQUIRE(padded_shape[ND - 3] == NSIDE + 1);
        REQUIRE(padded_shape[ND - 4] == 4);
        REQUIRE(padded_shape[0] == 3);
        for (int i = 1; i < ND - 5; ++i) {
          REQUIRE(padded_shape[i] == 1);
        }
      }
      THEN("Requesting padded shape provides correctly shaped object") {
        constexpr int ND = VarInfo::VNDIM;
        auto padded_shape = info.GetPaddedShape(interior);
        REQUIRE(padded_shape.size() == ND);
        REQUIRE(padded_shape[0] == NSIDE + 1);
        REQUIRE(padded_shape[1] == NSIDE + 1);
        REQUIRE(padded_shape[2] == NSIDE + 1);
        REQUIRE(padded_shape[3] == 4);
        for (int i = 4; i < ND - 1; ++i) {
          REQUIRE(padded_shape[i] == 1);
        }
        REQUIRE(padded_shape[ND - 1] == 3);
      }

      THEN("The padded bounds are correct") {
        auto [kb, jb, ib] = info.GetPaddedBoundsKJI(interior);
        REQUIRE(kb.s == NG);
        REQUIRE(kb.e == NSIDE + NG);
        REQUIRE(jb.s == NG);
        REQUIRE(jb.e == NSIDE + NG);
        REQUIRE(ib.s == NG);
        REQUIRE(ib.e == NSIDE + NG);
      }
    }

    WHEN("We initialize VarInfo on a scaler edge var") {
      auto v = pmbd->GetVarPtr(scalar_edge);
      VarInfo info(v, cellbounds);

      THEN("The shape is correct over both interior and entire") {
        std::vector<int> shape(10);
        int ndim = info.FillShape<int>(interior, shape.data());
        REQUIRE(ndim == 4);
        REQUIRE(shape[0] == 3);
        for (int i = 1; i < ndim; ++i) {
          REQUIRE(shape[i] == NSIDE + 1);
        }
        info.FillShape<int>(entire, shape.data());
        for (int i = 1; i < ndim; ++i) {
          REQUIRE(shape[i] == NSIDE + 2 * NG + 1);
        }
      }
      THEN("The size and tensorsize are correct") {
        REQUIRE(info.Size() == 3 * (NFULL + 1) * (NFULL + 1) * (NFULL + 1));
        REQUIRE(info.TensorSize() * info.ntop_elems == 3);
      }
    }

    WHEN("We request info from all vars") {
      auto vars = parthenon::GetAnyVariables(pmbd->GetVariableVector(),
                                             {parthenon::Metadata::Independent});
      auto all_info = VarInfo::GetAll(vars, cellbounds);
      THEN("The labels are all present") {
        for (const std::string &name : var_names) {
          auto pinfo = std::find(all_info.begin(), all_info.end(), name);
          REQUIRE(pinfo != all_info.end());
        }
      }
    }

    // JMM: This needs to be reset to 0 when we're done, because other
    // tests assume it's unset, thus zero-initialized.
    parthenon::Globals::nghost = 0;
  }
}
