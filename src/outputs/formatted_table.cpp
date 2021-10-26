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
//! \file formatted_table.cpp
//  \brief writes output data as a formatted table.  Should not be used to output large
//  3D data sets as this format is very slow and memory intensive.  Most useful for 1D
//  slices and/or sums.  Writes one file per Meshblock.

#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "coordinates/coordinates.hpp"
#include "defs.hpp"
#include "mesh/mesh.hpp"
#include "outputs/outputs.hpp"

namespace parthenon {

//----------------------------------------------------------------------------------------
//! \fn void FormattedTableOutput:::WriteOutputFile(Mesh *pm)
//  \brief writes OutputData to file in tabular format using C style std::fprintf
//         Writes one file per MeshBlock

void FormattedTableOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin, SimTime *tm,
                                           const SignalHandler::OutputSignal signal) {
  throw std::runtime_error(std::string(__func__) + " is not implemented");
}

} // namespace parthenon
