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

  // Gets data for all blocks on current rank.
  // Assumes blocks are contiguous
  // fills internal data for given pointer
  // returns NBlocks on success, -1 on failure
  template <typename T>
  int ReadBlocks(const char *name, IndexRange range, std::vector<T> &dataVec,
                 const std::vector<size_t> &bsize, size_t vlen = 1) {
#ifndef ENABLE_HDF5
    PARTHENON_FAIL("Restart functionality is not available because HDF5 is disabled");
#else
    try {
      // dataVec is assumed to be of the correct size
      T *data = dataVec.data();

      // compute block size, probaby could cache this.
      hid_t const theHdfType = getHDF5Type(data);

      H5D const dataset = H5D::FromHIDCheck(H5Dopen2(fh_, name, H5P_DEFAULT));
      H5S const dataspace = H5S::FromHIDCheck(H5Dget_space(dataset));

      /** Define hyperslab in dataset **/
      hsize_t offset[5] = {static_cast<hsize_t>(range.s), 0, 0, 0, 0};
      hsize_t count[5] = {static_cast<hsize_t>(range.e - range.s + 1), bsize[2], bsize[1],
                          bsize[0], vlen};
      PARTHENON_HDF5_CHECK(
          H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset, NULL, count, NULL));

      /** Define memory dataspace **/
      H5S const memspace = H5S::FromHIDCheck(H5Screate_simple(5, count, NULL));
      PARTHENON_HDF5_CHECK(
          H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset, NULL, count, NULL));

      // Read data from file
      PARTHENON_HDF5_CHECK(
          H5Dread(dataset, H5T_NATIVE_DOUBLE, memspace, dataspace, H5P_DEFAULT, data));

      return static_cast<int>(count[0]);
    } catch (const std::exception &e) {
      std::cout << e.what();
      return -1;
    }
#endif
  }

  // Reads an array dataset from file as a 1D vector.
  // Returns number of items read in count if provided
  template <typename T>
  std::vector<T> ReadDataset(const char *name, size_t *count = nullptr) {
    // Returns entire 1D array.
    // status, never checked.  We should...
#ifndef ENABLE_HDF5
    PARTHENON_FAIL("Restart functionality is not available because HDF5 is disabled");
#else
    T *typepointer = nullptr;
    hid_t const theHdfType = getHDF5Type(typepointer);

    H5D const dataset = H5D::FromHIDCheck(H5Dopen2(fh_, name, H5P_DEFAULT));
    H5S const dataspace = H5S::FromHIDCheck(H5Dget_space(dataset));

    // Allocate array of correct size
    H5S const filespace = H5S::FromHIDCheck(H5Dget_space(dataset));

    int rank = PARTHENON_HDF5_CHECK(H5Sget_simple_extent_ndims(filespace));

    std::vector<hsize_t> dims(rank);
    PARTHENON_HDF5_CHECK(H5Sget_simple_extent_dims(filespace, dims.data(), NULL));
    hsize_t isize = 1;
    for (int idir = 0; idir < rank; idir++) {
      isize = isize * dims[idir];
    }
    if (count != nullptr) {
      *count = isize;
    }

    std::vector<T> data(isize);
    /** Define memory dataspace **/
    H5S const memspace = H5S::FromHIDCheck(H5Screate_simple(rank, dims.data(), NULL));

    // Read data from file
    PARTHENON_HDF5_CHECK(H5Dread(dataset, theHdfType, memspace, dataspace, H5P_DEFAULT,
                                 static_cast<void *>(data.data())));

    return data;
#endif
  }

  template <typename T>
  std::vector<T> GetAttrVec(const std::string &location, const std::string &name) {
#ifndef ENABLE_HDF5
    PARTHENON_FAIL("Restart functionality is not available because HDF5 is disabled");
#else
    // check if the location exists in the file
    PARTHENON_HDF5_CHECK(H5Oexists_by_name(fh_, location.c_str(), H5P_DEFAULT));

    // open the object specified by the location path, this could be a dataset or group
    const H5O obj = H5O::FromHIDCheck(H5Oopen(fh_, location.c_str(), H5P_DEFAULT));

    return HDF5ReadAttributeVec<T>(obj, name);
#endif
  }

  template <typename T>
  T GetAttr(const std::string &location, const std::string &name) {
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
#endif
};

} // namespace parthenon
#endif // OUTPUTS_RESTART_HPP_
