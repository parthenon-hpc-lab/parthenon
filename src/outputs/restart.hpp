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

namespace parthenon {
class Mesh;

class RestartReader {
 public:
  explicit RestartReader(const char *theFile);

  // Gets single block data for variable name and
  // fills internal data for given pointer
  // returns 1 on success, negative on failure
  template <typename T>
  int ReadBlock(const char *name, const int blockID, T *data, size_t vlen = 1);

  // Gets data for all blocks on current rank.
  // Assumes blocks are contiguous
  // fills internal data for given pointer
  // returns NBlocks on success, negative on failure
  template <typename T>
  int ReadBlocks(const char *name, IndexRange range, T *data, size_t vlen = 1);

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
