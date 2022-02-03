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
//! \file flux_correction_fc.cpp
//  \brief functions that perform flux correction for face-centered variables

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "parthenon_mpi.hpp"

#include "bvals/fc/bvals_fc.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "parthenon_arrays.hpp"
#include "utils/buffer_utils.hpp"

// TODO(felker): break-up the long functions in this file

namespace parthenon {
//----------------------------------------------------------------------------------------
//! \fn void FaceCenteredBoundaryVariable::SendEMFCorrection()
//  \brief Restrict, pack and send the flux correction to the coarse neighbor(s) if needed

void FaceCenteredBoundaryVariable::SendFluxCorrection(bool is_allocated) {
  throw std::runtime_error(
      "FaceCenteredBoundaryVariable::SendFluxCorrection not implemented yet");
}

//----------------------------------------------------------------------------------------
//! \fn void FaceCenteredBoundaryVariable::ReceiveFluxCorrection()
//  \brief Receive and Apply the flux correction to the coarse neighbor(s) if needed

bool FaceCenteredBoundaryVariable::ReceiveFluxCorrection(bool is_allocated) {
  throw std::runtime_error(
      "FaceCenteredBoundaryVariable::ReceiveFluxCorrection not implemented yet");
}

} // namespace parthenon
