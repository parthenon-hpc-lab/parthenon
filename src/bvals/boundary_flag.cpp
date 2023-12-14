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
//! \file boundary_flag.cpp
//  \brief utilities for processing the user's input <parthenon/mesh> ixn_bc, oxn_bc
//  parameters and
// the associated internal BoundaryFlag enumerators

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "bvals/bvals.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

//----------------------------------------------------------------------------------------
//! \fn GetBoundaryFlag(std::string input_string)
//  \brief Parses input string to return scoped enumerator flag specifying boundary
//  condition. Typically called in Mesh() ctor and in pgen/*.cpp files.

BoundaryFlag GetBoundaryFlag(const std::string &input_string) {
  if (input_string == "periodic") {
    return BoundaryFlag::periodic;
  } else if (input_string == "none") {
    return BoundaryFlag::undef;
  } else if (input_string == "block") {
    return BoundaryFlag::block;
  } else {
    return BoundaryFlag::user;
  }
}

} // namespace parthenon
