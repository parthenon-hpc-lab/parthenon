//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
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
//! \file restart.cpp
//  \brief writes restart files

#include <memory>
#include <string>
#include <utility>

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
#ifndef ENABLE_HDF5
  std::stringstream msg;
  msg << "### FATAL ERROR in Restart (Reader) constructor" << std::endl
      << "Executable not configured for HDF5 outputs, but HDF5 file format "
      << "is required for restarts" << std::endl;
  PARTHENON_FAIL(msg);
#else
  // Open the HDF file in read only mode
  fh_ = H5F::FromHIDCheck(H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT));

  hasGhost = GetAttr<int>("Info", "IncludesGhost");
#endif
}

} // namespace parthenon
