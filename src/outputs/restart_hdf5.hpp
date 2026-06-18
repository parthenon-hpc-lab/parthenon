//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2022 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2026. Triad National Security, LLC. All rights reserved.
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

#ifndef OUTPUTS_RESTART_HDF5_HPP_
#define OUTPUTS_RESTART_HDF5_HPP_
//! \file io_wrapper.hpp
//  \brief defines a set of small wrapper functions for MPI versus Serial Output.

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "config.hpp"
#include "outputs/restart.hpp"
#ifdef ENABLE_HDF5
#include <hdf5.h>

#include "interface/metadata.hpp"
#include "outputs/parthenon_hdf5.hpp"
#include "outputs/parthenon_hdf5_types.hpp"

using namespace parthenon::HDF5;
#endif

#include "mesh/domain.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

class Mesh;
class Param;
class Swarm;
class MeshBlock;

class RestartReaderHDF5 : public RestartReader {
 public:
  explicit RestartReaderHDF5(const char *filename);

  [[nodiscard]] SparseInfo GetSparseInfo() const override;

  [[nodiscard]] MeshInfo GetMeshInfo() const override;

  [[nodiscard]] SimTime GetTimeInfo() const override;

  [[nodiscard]] std::string GetInputString() const override {
    return GetAttr<std::string>("Input", "File");
  };

  // Return output format version number. Return -1 if not existent.
  [[nodiscard]] int GetOutputFormatVersion() const override;

  [[nodiscard]] int HasGhost() const override { return has_ghost; };

 private:
#ifdef ENABLE_HDF5
  struct DatasetHandle {
    hid_t type;
    H5D dataset;
    H5S dataspace;
    int rank;
    hsize_t count;
    std::vector<hsize_t> dims;
  };

  // internal convenience function to open a dataset, perform some checks, and get
  // dimensions
  template <typename T>
  DatasetHandle OpenDataset(const std::string &name) const {
    DatasetHandle handle;

    // make sure dataset exists
    auto status = PARTHENON_HDF5_CHECK(H5Oexists_by_name(fh_, name.c_str(), H5P_DEFAULT));
    PARTHENON_REQUIRE_THROWS(
        status > 0, "Dataset '" + name + "' does not exist in HDF5 file " + filename_);

    // open dataset
    handle.dataset = H5D::FromHIDCheck(H5Dopen2(fh_, name.c_str(), H5P_DEFAULT));
    handle.dataspace = H5S::FromHIDCheck(H5Dget_space(handle.dataset));

    // get the HDF5 type from the template parameter and make sure it matches the dataset
    // type
    T *typepointer = nullptr;
    handle.type = getHDF5Type(typepointer);
    const H5T dset_type = H5T::FromHIDCheck(H5Dget_type(handle.dataset));
    status = PARTHENON_HDF5_CHECK(H5Tequal(handle.type, dset_type));
    PARTHENON_REQUIRE_THROWS(status > 0, "Type mismatch for dataset " + name);

    // get rank and dims
    const H5S filespace = H5S::FromHIDCheck(H5Dget_space(handle.dataset));
    handle.rank = PARTHENON_HDF5_CHECK(H5Sget_simple_extent_ndims(filespace));

    handle.dims.resize(handle.rank);
    PARTHENON_HDF5_CHECK(H5Sget_simple_extent_dims(filespace, handle.dims.data(), NULL));
    handle.count = 1;
    for (int idir = 0; idir < handle.rank; idir++) {
      handle.count = handle.count * handle.dims[idir];
    }

    return handle;
  }
#endif // ENABLE_HDF5

 public:
  // Gets data for all blocks on current rank.
  // Assumes blocks are contiguous
  // fills internal data for given pointer
  void ReadBlocks(const std::string &name, IndexRange range,
                  const OutputUtils::VarInfo &info, std::vector<Real> &dataVec,
                  Mesh *pmesh) const override;

  //  The PackOrUnpack logic requires knowledge of how data is stored and being read into
  //  the buffer. For HDF5 data is padded if needed (i.e., a face centered field has tims
  //  nx#+1 in all dimensions).
  [[nodiscard]] bool BlockdataIsPadded() const override { return true; };

  // Gets the data from a swarm var on current rank. Assumes all
  // blocks are contiguous. Fills dataVec based on shape from swarmvar
  // metadata.
  template <typename T>
  void ReadSwarmVar(const std::string &swarmname, const std::string &varname,
                    const std::size_t count, const std::size_t offset, const Metadata &m,
                    std::vector<T> &dataVec) {
#ifndef ENABLE_HDF5
    PARTHENON_FAIL("Restart functionality is not available because HDF5 is disabled");
#else
    auto hdl = OpenDataset<T>(swarmname + "/SwarmVars/" + varname);

    constexpr int CHUNK_MAX_DIM = 6;
    hsize_t h5_offset[CHUNK_MAX_DIM];
    hsize_t h5_count[CHUNK_MAX_DIM];
    const auto &shape = m.Shape();
    const auto rank = shape.size();
    const bool is_vector = m.IsSet(Metadata::Vector);
    std::size_t total_count = count;
    for (int i = 0; i < CHUNK_MAX_DIM; ++i) {
      h5_offset[i] = h5_count[i] = 0;
    }
    for (std::size_t i = 0; i < rank; ++i) {
      h5_count[i] = shape[rank - 1 - i];
      total_count *= shape[rank - 1 - i];
    }
    h5_count[rank] = count;
    h5_offset[rank] = offset;
    if (dataVec.size() < total_count) { // greedy re-alloc
      dataVec.resize(total_count);
    }
    PARTHENON_HDF5_CHECK(H5Sselect_hyperslab(hdl.dataspace, H5S_SELECT_SET, h5_offset,
                                             NULL, h5_count, NULL));
    const H5S memspace = H5S::FromHIDCheck(H5Screate_simple(rank + 1, h5_count, NULL));
    PARTHENON_HDF5_CHECK(H5Dread(hdl.dataset, hdl.type, memspace, hdl.dataspace,
                                 H5P_DEFAULT, dataVec.data()));
#endif // ENABLE_HDF5
  }

  // Reads an array dataset from file as a 1D vector.
  template <typename T>
  std::vector<T> ReadDataset(const std::string &name) const {
#ifndef ENABLE_HDF5
    PARTHENON_FAIL("Restart functionality is not available because HDF5 is disabled");
#else  // HDF5 enabled
    auto hdl = OpenDataset<T>(name);

    std::vector<T> data(hdl.count);
    const H5S memspace =
        H5S::FromHIDCheck(H5Screate_simple(hdl.rank, hdl.dims.data(), NULL));

    // Read data from file
    PARTHENON_HDF5_CHECK(H5Dread(hdl.dataset, hdl.type, memspace, hdl.dataspace,
                                 H5P_DEFAULT, static_cast<void *>(data.data())));

    return data;
#endif // ENABLE_HDF5
  }

  template <typename T>
  std::vector<T> GetAttrVec(const std::string &location, const std::string &name) const {
#ifndef ENABLE_HDF5
    PARTHENON_FAIL("Restart functionality is not available because HDF5 is disabled");
#else  // HDF5 enabled
    // check if the location exists in the file
    PARTHENON_HDF5_CHECK(H5Oexists_by_name(fh_, location.c_str(), H5P_DEFAULT));

    // open the object specified by the location path, this could be a dataset or group
    const H5O obj = H5O::FromHIDCheck(H5Oopen(fh_, location.c_str(), H5P_DEFAULT));

    return HDF5ReadAttributeVec<T>(obj, name);
#endif // ENABLE_HDF5
  }

  template <typename T>
  T GetAttr(const std::string &location, const std::string &name) const {
    // Note: We don't need a template specialization for std::string, since that case will
    // be handled by HDF5ReadAttributeVec
    auto res = GetAttrVec<T>(location, name);
    if (res.size() != 1) {
      PARTHENON_THROW("Expected a scalar attribute " + name +
                      ", but got a vector of length " + std::to_string(res.size()));
    }

    return res[0];
  }
  void ReadSwarmVar(const std::string &swarmname, const std::string &varname,
                    const std::size_t count, const std::size_t offset, const Metadata &m,
                    std::vector<Real> &dataVec) override {
    ReadSwarmVar<>(swarmname, varname, count, offset, m, dataVec);
  };
  void ReadSwarmVar(const std::string &swarmname, const std::string &varname,
                    const std::size_t count, const std::size_t offset, const Metadata &m,
                    std::vector<std::uint64_t> &dataVec) override {
    ReadSwarmVar<>(swarmname, varname, count, offset, m, dataVec);
  };
  void ReadSwarmVar(const std::string &swarmname, const std::string &varname,
                    const std::size_t count, const std::size_t offset, const Metadata &m,
                    std::vector<int> &dataVec) override {
    ReadSwarmVar<>(swarmname, varname, count, offset, m, dataVec);
  };

  // Gets the counts and offsets for MPI ranks for the meshblocks set
  // by the indexrange. Returns the total count on this rank.
  [[nodiscard]] std::size_t GetSwarmCounts(const std::string &swarm,
                                           const IndexRange &range,
                                           std::vector<std::size_t> &counts,
                                           std::vector<std::size_t> &offsets) override;

  void ReadParams(const std::string &name, Params &p) override;

  [[nodiscard]] bool VariableExists(const std::string &name, const DataType data_type,
                                    const std::string swarmvarname = ""

  ) const override {
#ifdef ENABLE_HDF5
    // Make sure dataset exists. Our HDF5 output does not differentiate between
    // fields and swarms, so we can ignore the data_type. Note, we may eventually
    // want to fix this as swarms and fields with the same name may cause issues.
    // disable error handling/printing while probing so missing datasets do not
    // spam the log, then restore the aborting handler.
    std::string full_name = name;
    if (data_type == DataType::SwarmVar) {
      full_name = name + "/SwarmVars/" + swarmvarname;
    }
    H5Eset_auto(H5E_DEFAULT, NULL, NULL);
    auto status =
        PARTHENON_HDF5_CHECK(H5Oexists_by_name(fh_, full_name.c_str(), H5P_DEFAULT));
    H5Eset_auto(H5E_DEFAULT, aborting_error_handler, NULL);
    return status > 0;
#else
    PARTHENON_FAIL("Restart functionality is not available because HDF5 is disabled");
    return false;
#endif // ENABLE_HDF5
  }
  // closes out the restart file
  // perhaps belongs in a destructor?
  void Close();

  // High-level template function to read all swarm variables of a given type from restart
  // file and distribute them to blocks
  template <typename T>
  void ReadSwarmVars(const std::shared_ptr<Swarm> &pswarm,
                     const std::vector<std::shared_ptr<MeshBlock>> &block_list,
                     const std::size_t count_on_rank, const std::size_t offset);
  const std::string filename_;

  // Does file have ghost cells?
  int has_ghost;

#ifdef ENABLE_HDF5
  // Currently all restarts are HDF5 files
  // when that changes, this will be revisited
  H5F fh_;
  H5G params_group_;
#endif // ENABLE_HDF5
};

// Include full definitions needed for template implementation
#include "interface/swarm.hpp"
#include "mesh/meshblock.hpp"
#include "utils/string_utils.hpp"

// Template implementation for ReadSwarmVars
// This needs to be in the header since it's a template function
template <typename T>
void RestartReaderHDF5::ReadSwarmVars(
    const std::shared_ptr<Swarm> &pswarm,
    const std::vector<std::shared_ptr<MeshBlock>> &block_list,
    const std::size_t count_on_rank, const std::size_t offset) {
  const std::string &swarmname = pswarm->label();
  std::vector<T> dataVec;
  for (const auto &var : pswarm->GetVariableVector<T>()) {
    const std::string &varname = var->label();
    const auto &m = var->metadata();
    auto arrdims = m.GetArrayDims(pswarm->GetBlockPointer(), false);

    auto var_missing_on_disk = !VariableExists(swarmname + "/SwarmVars/" + varname);

    // Backwards compatibility: try old position names if new ones missing
    std::string varname_to_read = varname;
    if (var_missing_on_disk) {
      varname_to_read = GetBackwardsCompatibleSwarmVarName(varname);
      // Check if the old name exists
      if (varname_to_read != varname) {
        var_missing_on_disk =
            !VariableExists(swarmname + "/SwarmVars/" + varname_to_read);
        if (!var_missing_on_disk && Globals::my_rank == 0) {
          std::cout << "SwarmVar: " << varname
                    << " using backwards-compatible name: " << varname_to_read << "\n";
        }
      }
    }

    if (Globals::my_rank == 0 && var_missing_on_disk) {
      std::cout << "SwarmVar: " << varname << " missing on disk\n";
    } else if (Globals::my_rank == 0 && varname_to_read == varname) {
      std::cout << "SwarmVar: " << varname << "\n";
    }

    if (var_missing_on_disk) {
      // TODO(JMM/PG) Add failed load list of "fail/needs fix" list
      continue;
    }

    try {
      ReadSwarmVar(swarmname, varname_to_read, count_on_rank, offset, m, dataVec);
    } catch (std::exception &ex) {
      // Variable does exist but could not be read. So we definitely want to fail here.
      PARTHENON_THROW(StringPrintf("[%d] WARNING: Failed to read Swarm %s Variable %s "
                                   "from restart file:\n%s",
                                   Globals::my_rank, swarmname.c_str(), varname.c_str(),
                                   ex.what()));
    }

    // Only safe because swarm starts completely defragged.
    // Note ordering here: block is second-inner-most loop.
    // If hdf5 output format changes, this needs to change too.
    std::size_t ivec = 0;
    for (int n6 = 0; n6 < arrdims[5]; ++n6) {
      for (int n5 = 0; n5 < arrdims[4]; ++n5) {
        for (int n4 = 0; n4 < arrdims[3]; ++n4) {
          for (int n3 = 0; n3 < arrdims[2]; ++n3) {
            for (int n2 = 0; n2 < arrdims[1]; ++n2) {
              for (auto &pmb : block_list) {
                // 1 deep copy per tensor component per swarmvar per
                // block, unfortunately. But only at initialization.
                auto swarm_container = pmb->meshblock_data.Get()->GetSwarmData();
                auto pswarm_blk = swarm_container->Get(swarmname);
                auto v = Kokkos::subview(pswarm_blk->Get<T>(varname).data, n6, n5, n4, n3,
                                         n2, Kokkos::ALL());
                auto v_h = Kokkos::create_mirror_view(v);
                for (int n1 = 0; n1 < pswarm_blk->GetNumActive(); ++n1) {
                  v_h(n1) = dataVec[ivec++];
                }
                Kokkos::deep_copy(v, v_h);
              }
            }
          }
        }
      }
    }
  }
}

} // namespace parthenon
#endif // OUTPUTS_RESTART_HDF5_HPP_
