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
//! \file globals.cpp
//  \brief namespace containing global variables.
//
// Yes, we all know global variables should NEVER be used, but in fact they are ideal for,
// e.g., global constants that are set once and never changed.  To prevent name collisions
// global variables are wrapped in their own namespace.

#include "globals.hpp"
#include "defs.hpp"

namespace parthenon {
namespace Globals {

int nghost;

// all of these global variables are set at the start of main():
int my_rank; // MPI rank of this process
int nranks;  // total number of MPI ranks

// sparse configuration values that are needed in various places
SparseConfig sparse_config;

// timeout (in seconds) for cell_centered_bvars::ReceiveBoundaryBuffers task
double receive_boundary_buffer_timeout;

// the total time (in seconds) the current task has been running, can be used to set
// timeouts for tasks
double current_task_runtime_sec;

} // namespace Globals
} // namespace parthenon
