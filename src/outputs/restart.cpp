//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
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
//! \file restart.cpp
//  \brief writes restart files

#include <memory>
#include <string>
#include <utility>

#include "H5Ppublic.h"
#include "H5Tpublic.h"
#include "H5public.h"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "outputs/outputs.hpp"
#include "outputs/restart.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

//----------------------------------------------------------------------------------------
//! \fn void RestartReader::RestartReader(const std::string filename)
//  \brief Opens the restart file and stores appropriate file handle in fh_
RestartReader::RestartReader(const char *filename) : filename_(filename) {
#ifndef HDF5OUTPUT
  std::stringstream msg;
  msg << "### FATAL ERROR in Restart (Reader) constructor" << std::endl
      << "Executable not configured for HDF5 outputs, but HDF5 file format "
      << "is required for restarts" << std::endl;
  PARTHENON_FAIL(msg);
#else
  // Open the HDF file in read only mode
  fh_ = H5F::FromHIDCheck(H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT));

  // populate block size from the file
  std::vector<int32_t> blockSize = ReadAttr1DI32("Mesh", "blockSize");
  hasGhost = GetAttr<int>("Mesh", "includesGhost");
  nx1_ = static_cast<hsize_t>(blockSize[0]);
  nx2_ = static_cast<hsize_t>(blockSize[1]);
  nx3_ = static_cast<hsize_t>(blockSize[2]);
#endif
}

//! \fn std::shared_ptr<std::vector<T>> RestartReader::ReadAttrString(const char *dataset,
//! const char *name, size_t *count = nullptr)
//  \brief Reads a string attribute for given dataset
std::string RestartReader::ReadAttrString(const char *dataset, const char *name,
                                          size_t *count) {
  // Returns entire 1D array.
  // status, never checked.  We should...
#ifdef HDF5OUTPUT
  hid_t theHdfType = H5T_C_S1;

  H5D const dset = H5D::FromHIDCheck(H5Dopen2(fh_, dataset, H5P_DEFAULT));
  H5A const attr = H5A::FromHIDCheck(H5Aopen(dset, name, H5P_DEFAULT));
  H5S const dataspace = H5S::FromHIDCheck(H5Aget_space(attr));

  // Allocate array of correct size
  H5T const filetype = H5T::FromHIDCheck(H5Aget_type(attr));
  hsize_t isize = H5Tget_size(filetype);
  isize++;
  if (count != nullptr) {
    *count = isize;
  }

  std::vector<char> s(isize + 1, '\0');
  // Read data from file
  //  H5Aread(attr, theHdfType, static_cast<void *>(s));
  H5T const memType = H5T::FromHIDCheck(H5Tcopy(H5T_C_S1));
  PARTHENON_HDF5_CHECK(H5Tset_size(memType, isize));
  PARTHENON_HDF5_CHECK(H5Aread(attr, memType, s.data()));

  return std::string(s.data());
#else
  return std::string("HDF5 NOT COMPILED IN");
#endif
}

} // namespace parthenon
