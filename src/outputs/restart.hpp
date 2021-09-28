//========================================================================================
// (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
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
#ifndef OUTPUTS_RESTART_HPP_
#define OUTPUTS_RESTART_HPP_
//! \file io_wrapper.hpp
//  \brief defines a set of small wrapper functions for MPI versus Serial Output.

#include <cinttypes>
#include <string>
#include <vector>

#ifdef ENABLE_HDF5
#include <hdf5.h>

#include "outputs/parthenon_hdf5.hpp"

using namespace parthenon::HDF5;
#endif

#include "mesh/domain.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

class Mesh;

class RestartReader {
 public:
  explicit RestartReader(const char *theFile);

  struct SparseInfo {
    // labels of sparse fields (full label, i.e. base name and sparse id)
    std::vector<std::string> labels;

    // allocation status of sparse fields (2D array outer dimension: block, inner
    // dimension: sparse field)
    // can't use std::vector here because std::vector<hbool_t> is the same as
    // std::vector<bool> and it doesn't have .data() member
    std::unique_ptr<hbool_t[]> allocated;

    int num_blocks = 0;
    int num_sparse = 0;

    bool IsAllocated(int block, int sparse_field_idx) const {
      PARTHENON_REQUIRE_THROWS(allocated != nullptr,
                               "Tried to get allocation status but no data present");
      PARTHENON_REQUIRE_THROWS((block >= 0) && (block < num_blocks),
                               "Invalid block index in SparseInfo::IsAllocated");
      PARTHENON_REQUIRE_THROWS((sparse_field_idx >= 0) && (sparse_field_idx < num_sparse),
                               "Invalid sparse field index in SparseInfo::IsAllocated");

      return allocated[block * num_sparse + sparse_field_idx];
    }
  };

  SparseInfo GetSparseInfo() const;

 private:
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
#ifndef ENABLE_HDF5
    PARTHENON_FAIL("Restart functionality is not available because HDF5 is disabled");
#else  // HDF5 enabled
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
#endif // ENABLE_HDF5
  }

 public:
  // Gets data for all blocks on current rank.
  // Assumes blocks are contiguous
  // fills internal data for given pointer
  template <typename T>
  void ReadBlocks(const std::string &name, IndexRange range, std::vector<T> &dataVec,
                  const std::vector<size_t> &bsize, size_t vlen = 1) const {
#ifndef ENABLE_HDF5
    PARTHENON_FAIL("Restart functionality is not available because HDF5 is disabled");
#else  // HDF5 enabled
    auto hdl = OpenDataset<T>(name);

    PARTHENON_REQUIRE_THROWS(hdl.rank == 5, "Expected data set of rank 5, but dataset " +
                                                name + " has rank " +
                                                std::to_string(hdl.rank));

    /** Select hyperslab in dataset **/
    hsize_t offset[5] = {static_cast<hsize_t>(range.s), 0, 0, 0, 0};
    hsize_t count[5] = {static_cast<hsize_t>(range.e - range.s + 1), bsize[2], bsize[1],
                        bsize[0], vlen};

    hsize_t total_count = 1;
    for (int i = 0; i < 5; ++i) {
      total_count *= count[i];
    }

    PARTHENON_REQUIRE_THROWS(dataVec.size() >= total_count,
                             "Buffer (size " + std::to_string(dataVec.size()) +
                                 ") is too small for dataset " + name + " (size " +
                                 std::to_string(total_count) + ")");
    PARTHENON_HDF5_CHECK(
        H5Sselect_hyperslab(hdl.dataspace, H5S_SELECT_SET, offset, NULL, count, NULL));

    const H5S memspace = H5S::FromHIDCheck(H5Screate_simple(5, count, NULL));
    PARTHENON_HDF5_CHECK(
        H5Sselect_hyperslab(hdl.dataspace, H5S_SELECT_SET, offset, NULL, count, NULL));

    // Read data from file
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

  // closes out the restart file
  // perhaps belongs in a destructor?
  void Close();

  // Does file have ghost cells?
  int hasGhost;

 private:
  const std::string filename_;

#ifdef ENABLE_HDF5
  // Currently all restarts are HDF5 files
  // when that changes, this will be revisited
  H5F fh_;
#endif // ENABLE_HDF5
};

} // namespace parthenon
#endif // OUTPUTS_RESTART_HPP_
