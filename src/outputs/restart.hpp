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
#include <cinttypes>
#ifdef HDF5OUTPUT
#include <hdf5.h>
#endif
#include <vector>

#include "outputs/parthenon_hdf5.hpp"
namespace parthenon {
class Mesh;

class RestartReader {
 public:
  explicit RestartReader(const char *theFile);

  // Gets single block data for variable name and
  // fills internal data for given pointer
  // returns 1 on success, negative on failure
  //! \fn void RestartReader::ReadBlock(const char *name, const int blockID, T *data)
  //  \brief Reads data for one block from restart file
  template <typename T>
  int ReadBlock(const char *name, const int blockID, T *data, size_t vlen) {
    IndexRange range{blockID, blockID};
    return ReadBlocks(name, range, data, vlen);
  }

  // Gets data for all blocks on current rank.
  // Assumes blocks are contiguous
  // fills internal data for given pointer
  // returns NBlocks on success, negative on failure
  template <typename T>
  int ReadBlocks(const char *name, IndexRange range, T *data, size_t vlen = 1) {
#ifdef HDF5OUTPUT
    try {
      herr_t status;

      // compute block size, probaby could cache this.
      hid_t theHdfType = getHdfType(data);

      hid_t dataset = H5Dopen2(fh_, name, H5P_DEFAULT);
      if (dataset < 0) {
        return -1;
      }
      hid_t dataspace = H5Dget_space(dataset);
      if (dataspace < 0) {
        H5Dclose(dataset);
        return -2;
      }

      /** Define hyperslab in dataset **/
      hsize_t offset[5] = {static_cast<hsize_t>(range.s), 0, 0, 0, 0};
      hsize_t count[5] = {static_cast<hsize_t>(range.e - range.s + 1), nx3_, nx2_, nx1_,
                          vlen};
      status = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset, NULL, count, NULL);
      if (status < 0) {
        H5Dclose(dataspace);
        H5Dclose(dataset);
        return -3;
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
        return -4;
      } else {
        return static_cast<int>(count[0]);
      }
    } catch (const std::exception &e) {
      std::cout << e.what();
      return -5;
    }
#else
    return -6;
#endif
  }

  // Reads an array dataset from file as a 1D vector.
  // Returns number of items read in count if provided
  template <typename T>
  std::vector<T> ReadDataset(const char *name, size_t *count = nullptr);

  // Reads a string attribute.
  // Returns number of items read in count if provided
  std::string ReadAttrString(const char *dataset, const char *name,
                             size_t *count = nullptr);

  // Type specific interface
  std::vector<Real> ReadAttr1DReal(const char *dataset, const char *name,
                                   size_t *count = nullptr);
  std::vector<int32_t> ReadAttr1DI32(const char *dataset, const char *name,
                                     size_t *count = nullptr);

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
  template <typename T>
  std::vector<T> ReadAttrBytes_(const char *dataset, const char *name,
                                size_t *count = nullptr);

#ifdef HDF5OUTPUT
  // Currently all restarts are HDF5 files
  // when that changes, this will be revisited
  hid_t fh_;
  hsize_t nx1_, nx2_, nx3_;
#endif
};
} // namespace parthenon
#endif
