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
#ifndef OUTPUTS_RESTART_HPP_
#define OUTPUTS_RESTART_HPP_
//! \file io_wrapper.hpp
//  \brief defines a set of small wrapper functions for MPI versus Serial Output.

#ifdef HDF5OUTPUT
#include <hdf5.h>
#endif

#include <cinttypes>
#include <string>
#include <vector>

#include "outputs/parthenon_hdf5.hpp"
namespace parthenon {
class Mesh;

class RestartReader {
 public:
  explicit RestartReader(const char *theFile);

  // Gets single block data for variable name and
  // fills internal data for given pointer
  // returns 1 on success, -1 on failure
  //! \fn void RestartReader::ReadBlock(const char *name, const int blockID,
  //! std::vector<T>data)
  //  \brief Reads data for one block from restart file
  template <typename T>
  int ReadBlock(const char *name, const int blockID, std::vector<T> &dataVec,
                size_t vlen) {
    IndexRange range{blockID, blockID};
    return ReadBlocks(name, range, dataVec, vlen);
  }

  // Gets data for all blocks on current rank.
  // Assumes blocks are contiguous
  // fills internal data for given pointer
  // returns NBlocks on success, -1 on failure
  template <typename T>
  int ReadBlocks(const char *name, IndexRange range, std::vector<T> &dataVec,
                 size_t vlen = 1) {
#ifdef HDF5OUTPUT
    try {
      herr_t status;

      // dataVec is assumed to be of the correct size
      T *data = dataVec.data();

      // compute block size, probaby could cache this.
      hid_t theHdfType = getHdfType(data);

      hid_t dataset = H5Dopen2(fh_, name, H5P_DEFAULT);
      if (dataset < 0) {
        return -1;
      }
      hid_t dataspace = H5Dget_space(dataset);
      if (dataspace < 0) {
        H5Dclose(dataset);
        return -1;
      }

      /** Define hyperslab in dataset **/
      hsize_t offset[5] = {static_cast<hsize_t>(range.s), 0, 0, 0, 0};
      hsize_t count[5] = {static_cast<hsize_t>(range.e - range.s + 1), nx3_, nx2_, nx1_,
                          vlen};
      status = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset, NULL, count, NULL);
      if (status < 0) {
        H5Dclose(dataspace);
        H5Dclose(dataset);
        return -1;
      }

      /** Define memory dataspace **/
      hid_t memspace = H5Screate_simple(5, count, NULL);
      hsize_t offsetMem[5] = {0, 0, 0, 0, 0};
      status =
          H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offsetMem, NULL, count, NULL);

      // Read data from file
      status =
          H5Dread(dataset, H5T_NATIVE_DOUBLE, memspace, dataspace, H5P_DEFAULT, data);

      // CLose the dataspace and data set.
      H5Dclose(dataset);
      H5Sclose(memspace);
      H5Sclose(dataspace);
      if (status < 0) {
        return -1;
      } else {
        return static_cast<int>(count[0]);
      }
    } catch (const std::exception &e) {
      std::cout << e.what();
      return -1;
    }
#else
    return -1;
#endif
  }

  // Reads an array dataset from file as a 1D vector.
  // Returns number of items read in count if provided
  template <typename T>
  std::vector<T> ReadDataset(const char *name, size_t *count = nullptr) {
    // Returns entire 1D array.
    // status, never checked.  We should...
#ifdef HDF5OUTPUT
    herr_t status;

    T *typepointer = nullptr;
    hid_t theHdfType = getHdfType(typepointer);

    hid_t dataset = H5Dopen2(fh_, name, H5P_DEFAULT);
    hid_t dataspace = H5Dget_space(dataset);

    // Allocate array of correct size
    hid_t filespace = H5Dget_space(dataset);
    int rank = H5Sget_simple_extent_ndims(filespace);
    std::vector<hsize_t> dims(rank);
    status = H5Sget_simple_extent_dims(filespace, dims.data(), NULL);
    hsize_t isize = 1;
    for (int idir = 0; idir < rank; idir++) {
      isize = isize * dims[idir];
    }
    if (count != nullptr) {
      *count = isize;
    }

    std::vector<T> data(isize);
    /** Define memory dataspace **/
    hid_t memspace = H5Screate_simple(rank, dims.data(), NULL);

    // Read data from file
    status = H5Dread(dataset, theHdfType, memspace, dataspace, H5P_DEFAULT,
                     static_cast<void *>(data.data()));

    // CLose the dataspace and data set.
    H5Sclose(filespace);
    H5Sclose(dataspace);
    H5Dclose(dataset);
#else
    std::vector<T> data;
#endif
    return data;
  }

  // Reads a string attribute.
  // Returns number of items read in count if provided
  std::string ReadAttrString(const char *dataset, const char *name,
                             size_t *count = nullptr);

  // Type specific interface
  std::vector<Real> ReadAttr1DReal(const char *dataset, const char *name,
                                   size_t *count = nullptr) {
    return ReadAttrBytes_<Real>(dataset, name, count);
  }
  std::vector<int32_t> ReadAttr1DI32(const char *dataset, const char *name,
                                     size_t *count = nullptr) {
    return ReadAttrBytes_<int32_t>(dataset, name, count);
  }

  template <typename T>
  T GetAttr(const char *dataset, const char *name) {
    auto x = ReadAttrBytes_<T>(dataset, name);
    return x[0];
  }
  // closes out the restart file
  // perhaps belongs in a destructor?
  void Close();

 private:
  const std::string filename_;
  // // Reads an array attribute from file.
  // // Returns number of items read in count if provided
  //! \fn std::vector<T> RestartReader::ReadAttr1D(const char *dataset,
  //! const char *name, size_t *count = nullptr)
  //  \brief Reads a 1D array attribute for given dataset
  template <typename T>
  std::vector<T> ReadAttrBytes_(const char *dataset, const char *name,
                                size_t *count = nullptr) {
    // Returns entire 1D array.
    // status, never checked.  We should...
#ifdef HDF5OUTPUT
    herr_t status;

    T *typepointer = nullptr;
    hid_t theHdfType = getHdfType(typepointer);

    hid_t dset = H5Dopen2(fh_, dataset, H5P_DEFAULT);
    hid_t attr = H5Aopen(dset, name, H5P_DEFAULT);
    hid_t dataspace = H5Aget_space(attr);

    // Allocate array of correct size
    int rank = H5Sget_simple_extent_ndims(dataspace);
    std::vector<hsize_t> dims(rank);
    status = H5Sget_simple_extent_dims(dataspace, dims.data(), NULL);
    hsize_t isize = 1;
    for (int idir = 0; idir < rank; idir++) {
      isize = isize * dims[idir];
    }
    if (count != nullptr) {
      *count = isize;
    }

    std::vector<T> data(isize);

    // Read data from file
    status = H5Aread(attr, theHdfType, static_cast<void *>(data.data()));

    // CLose the dataspace and data set.
    H5Sclose(dataspace);
    H5Aclose(attr);
    H5Dclose(dset);
#else
    std::vector<T> data;
#endif
    return data;
  }

#ifdef HDF5OUTPUT
  // Currently all restarts are HDF5 files
  // when that changes, this will be revisited
  hid_t fh_;
  hsize_t nx1_, nx2_, nx3_;
#endif
};

} // namespace parthenon
#endif // OUTPUTS_RESTART_HPP_
