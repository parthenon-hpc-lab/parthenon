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

// C headers

// C++ headers
#include <cmath>      // sqrt()
#include <csignal>    // ISO C/C++ signal() and sigset_t, sigemptyset() POSIX C extensions
#include <cstdint>    // int64_t
#include <cstdio>     // sscanf()
#include <cstdlib>    // strtol
#include <ctime>      // clock(), CLOCKS_PER_SEC, clock_t
#include <exception>  // exception
#include <iomanip>    // setprecision()
#include <iostream>   // cout, endl
#include <limits>     // max_digits10
#include <new>        // bad_alloc
#include <string>     // string
#include <vector>

// Parthenon headers
#include "parthenon_manager.hpp"
#include "athena.hpp"
#include "globals.hpp"
#include "argument_parser.hpp"
#include "mesh/mesh.hpp"
#include "outputs/io_wrapper.hpp"
#include "outputs/outputs.hpp"
#include "parameter_input.hpp"
#include "utils/utils.hpp"
#include "interface/Update.hpp"

// Application headers
#include "pi.hpp"

// MPI/OpenMP headers
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif
  
using namespace parthenon;

//----------------------------------------------------------------------------------------
//! \fn int main(int argc, char *argv[])
//  \brief Athena++ main program

int main(int argc, char *argv[]) {

  ParthenonManager pman;

  auto manager_status = pman.ParthenonInit(argc, argv);
  if (manager_status == ParthenonStatus::complete) {
    pman.ParthenonFinalize();
    return 0;
  }
  if (manager_status == ParthenonStatus::error) {
    pman.ParthenonFinalize();
    return 1;
  }

  CalculatePi driver(pman.pinput.get(), pman.pmesh.get(), pman.pouts.get());

  // start a timer
  pman.PreDriver();

  auto driver_status = driver.Execute();

  // Make final outputs, print diagnostics
  pman.PostDriver(driver_status);

  // call MPI_Finalize if necessary
  pman.ParthenonFinalize();

  return(0);
}
