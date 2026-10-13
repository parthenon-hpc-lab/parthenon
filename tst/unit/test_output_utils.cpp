//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2024 The Parthenon collaboration
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
// This file was made in part with generative AI.

#include <cstdio>
#include <filesystem>
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
#ifdef ENABLE_HDF5
#include "outputs/parthenon_hdf5.hpp"
#include "outputs/restart_hdf5.hpp"
#endif
#ifdef PARTHENON_ENABLE_OPENPMD
#include "outputs/parthenon_opmd.hpp"
#include "outputs/restart_opmd.hpp"
#endif

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

#ifdef ENABLE_HDF5
namespace {
class TemporaryHDF5File {
 public:
  explicit TemporaryHDF5File(std::string filename) : filename_(std::move(filename)) {}
  ~TemporaryHDF5File() { std::filesystem::remove(filename_); }

  const std::string &filename() const { return filename_; }

 private:
  std::string filename_;
};

void CreateFieldCatalogFixture(const std::string &filename,
                               const std::vector<std::string> *field_names) {
  using namespace parthenon::HDF5;
  H5F file = H5F::FromHIDCheck(
      H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));
  auto info = MakeGroup(file, "Info");
  auto params = MakeGroup(file, "Params");
  HDF5WriteAttribute("IncludesGhost", 0, info);
  HDF5WriteAttribute("OutputFormatVersion", OUTPUT_VERSION_FORMAT, info);
  if (field_names != nullptr) {
    HDF5WriteAttribute("OutputDatasetNames", *field_names, info);
  }

  const hsize_t dimensions[] = {1};
  const H5S dataspace = H5S::FromHIDCheck(H5Screate_simple(1, dimensions, dimensions));
  for (const auto &name : {"advected", "pool_7", "SparseInfo", "Levels"}) {
    H5D dataset = H5D::FromHIDCheck(H5Dcreate2(file, name, H5T_NATIVE_DOUBLE, dataspace,
                                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
  }
  auto ignored_group = MakeGroup(file, "ignored_group");
}
} // namespace

TEST_CASE("HDF5 restart readers discover analysis fields", "[OutputUtils][HDF5]") {
  using parthenon::RestartReaderHDF5;

  SECTION("OutputDatasetNames is authoritative and filterable") {
    TemporaryHDF5File file("analysis_fields_metadata.phdf");
    const std::vector<std::string> fields{"my_derived_var", "advected", "pool_7"};
    CreateFieldCatalogFixture(file.filename(), &fields);

    RestartReaderHDF5 reader(file.filename().c_str());
    REQUIRE(reader.GetFieldNames() == fields);
    REQUIRE(reader.GetFieldNames({"missing", "pool_7", "pool_7", "advected"}) ==
            std::vector<std::string>{"advected", "pool_7"});
  }

  SECTION("Root datasets provide a legacy compatibility catalog") {
    TemporaryHDF5File file("analysis_fields_legacy.phdf");
    CreateFieldCatalogFixture(file.filename(), nullptr);

    RestartReaderHDF5 reader(file.filename().c_str());
    REQUIRE(reader.GetFieldNames() == std::vector<std::string>{"advected", "pool_7"});
    REQUIRE(reader.GetFieldNames({"missing", "pool_7"}) ==
            std::vector<std::string>{"pool_7"});
  }
}
#endif

#ifdef PARTHENON_ENABLE_OPENPMD
TEST_CASE("OpenPMD mesh record names recover logical field labels",
          "[OutputUtils][OpenPMD]") {
  using parthenon::OpenPMDUtils::GetFieldLabelFromMeshRecordName;

  REQUIRE(GetFieldLabelFromMeshRecordName("advected_lvl0", 2) == "advected");
  REQUIRE(GetFieldLabelFromMeshRecordName("my_derived_var_lvl12", 2) == "my_derived_var");
  REQUIRE(GetFieldLabelFromMeshRecordName("face_field_F1_lvl2", 2) == "face_field");
  REQUIRE(GetFieldLabelFromMeshRecordName("edge_field_E3_lvl1", 2) == "edge_field");
  REQUIRE(GetFieldLabelFromMeshRecordName("node_field_NN_lvl0", 2) == "node_field");
  REQUIRE_THROWS(GetFieldLabelFromMeshRecordName("advected_advected_lvl0", 1));
  REQUIRE_THROWS(GetFieldLabelFromMeshRecordName("advected", 2));
  REQUIRE_THROWS(GetFieldLabelFromMeshRecordName("advected_lvlx", 2));
}

TEST_CASE("OpenPMD restart readers discover analysis fields", "[OutputUtils][OpenPMD]") {
  using parthenon::RestartReaderOPMD;

  SECTION("Field metadata is authoritative and filterable") {
    const std::string filename = "analysis_fields_metadata.%05T.bp";
    {
      auto series = openPMD::Series(filename, openPMD::Access::CREATE);
      series.setIterationEncoding(openPMD::IterationEncoding::fileBased);
      auto iteration = series.iterations[0];
      iteration.setAttribute("OutputFormatVersion", 2);
      iteration.setAttribute(
          "OutputDatasetNames",
          std::vector<std::string>{"advected", "my_derived_var", "sparse_7"});
    }

    RestartReaderOPMD reader(filename.c_str());
    REQUIRE(reader.GetFieldNames() ==
            std::vector<std::string>{"advected", "my_derived_var", "sparse_7"});
    REQUIRE(reader.GetFieldNames({"missing", "sparse_7", "advected"}) ==
            std::vector<std::string>{"advected", "sparse_7"});
  }

  SECTION("Format-version 2 records provide a compatibility catalog") {
    const std::string filename = "analysis_fields_records.%05T.bp";
    {
      auto series = openPMD::Series(filename, openPMD::Access::CREATE);
      series.setIterationEncoding(openPMD::IterationEncoding::fileBased);
      auto iteration = series.iterations[0];
      iteration.setAttribute("OutputFormatVersion", 2);
      const auto dataset = openPMD::Dataset(openPMD::Datatype::DOUBLE, {1});
      auto advected_level_0 =
          iteration.meshes["advected_lvl0"][openPMD::MeshRecordComponent::SCALAR];
      advected_level_0.resetDataset(dataset);
      advected_level_0.makeConstant(0.0);
      auto advected_level_1 =
          iteration.meshes["advected_lvl1"][openPMD::MeshRecordComponent::SCALAR];
      advected_level_1.resetDataset(dataset);
      advected_level_1.makeConstant(0.0);
      auto face_field =
          iteration.meshes["face_field_F1_lvl0"][openPMD::MeshRecordComponent::SCALAR];
      face_field.resetDataset(dataset);
      face_field.makeConstant(0.0);
    }

    RestartReaderOPMD reader(filename.c_str());
    REQUIRE(reader.GetFieldNames() == std::vector<std::string>{"advected", "face_field"});
  }

  SECTION("Format-version 1 requires an explicit field list") {
    const std::string filename = "analysis_fields_legacy.%05T.bp";
    {
      auto series = openPMD::Series(filename, openPMD::Access::CREATE);
      series.setIterationEncoding(openPMD::IterationEncoding::fileBased);
      auto iteration = series.iterations[0];
      iteration.setAttribute("OutputFormatVersion", 1);
      const auto dataset = openPMD::Dataset(openPMD::Datatype::DOUBLE, {1});
      auto advected =
          iteration
              .meshes["advected_advected_lvl0"][openPMD::MeshRecordComponent::SCALAR];
      advected.resetDataset(dataset);
      advected.makeConstant(0.0);
    }

    RestartReaderOPMD reader(filename.c_str());
    REQUIRE_THROWS(reader.GetFieldNames());
    REQUIRE(reader.GetFieldNames({"missing", "advected"}) ==
            std::vector<std::string>{"advected"});
  }
}
#endif

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
    IndexShape f_cellbounds = pmb->f_cellbounds;
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
      auto all_info = VarInfo::GetAll(vars, cellbounds, f_cellbounds);
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

TEST_CASE("Sparse labeled components use base-name output labels",
          "[VarInfo][OutputUtils]") {
  auto pkg = std::make_shared<StateDescriptor>("Test sparse labels");

  Metadata sparse_scalar_md({Metadata::Cell, Metadata::Independent, Metadata::Sparse});
  Metadata sparse_vector_md(
      {Metadata::Cell, Metadata::Independent, Metadata::Sparse, Metadata::Vector},
      std::vector<int>{3}, std::vector<std::string>{"x", "y", "z"});
  Metadata dense_scalar_md({Metadata::Cell, Metadata::Independent});

  pkg->AddSparsePool("shape_shift", sparse_scalar_md, std::vector<int>{1},
                     std::vector<std::string>{"scalar"});
  pkg->AddSparsePool("velocity", sparse_vector_md, std::vector<int>{3});
  pkg->AddField("density", dense_scalar_md);

  auto pmb = std::make_shared<MeshBlock>(8, 3);
  auto pmbd = pmb->meshblock_data.Get();
  pmbd->Initialize(pkg, pmb);

  auto sparse_scalar = VarInfo(pmbd->GetVarPtr("shape_shift_1"), pmb->cellbounds);
  REQUIRE(sparse_scalar.label == "shape_shift_1");
  REQUIRE(sparse_scalar.component_labels ==
          std::vector<std::string>{"shape_shift_scalar"});

  auto sparse_vector = VarInfo(pmbd->GetVarPtr("velocity_3"), pmb->cellbounds);
  REQUIRE(sparse_vector.component_labels ==
          std::vector<std::string>{"velocity_3_x", "velocity_3_y", "velocity_3_z"});

  auto dense_scalar = VarInfo(pmbd->GetVarPtr("density"), pmb->cellbounds);
  REQUIRE(dense_scalar.component_labels == std::vector<std::string>{"density"});
}

TEST_CASE("Duplicate output component names fail metadata generation",
          "[VarInfo][OutputUtils]") {
  auto pkg = std::make_shared<StateDescriptor>("Duplicate sparse labels");

  Metadata sparse_scalar_md({Metadata::Cell, Metadata::Independent, Metadata::Sparse});
  pkg->AddSparsePool("shape_shift", sparse_scalar_md, std::vector<int>{1, 2},
                     std::vector<std::string>{"scalar", "scalar"});

  auto pmb = std::make_shared<MeshBlock>(8, 3);
  auto pmbd = pmb->meshblock_data.Get();
  pmbd->Initialize(pkg, pmb);

  auto vars = parthenon::GetAnyVariables(pmbd->GetVariableVector(),
                                         {parthenon::Metadata::Independent});
  REQUIRE_THROWS(VarInfo::GetAll(vars, pmb->cellbounds, pmb->f_cellbounds));
}
