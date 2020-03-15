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

  if (Globals::my_rank == 0) {
    std::cout << "\n"<<Globals::my_rank<<":Setup complete, entering main loop...\n" << std::endl;
  }

  clock_t tstart = clock();
#ifdef OPENMP_PARALLEL
  double omp_start_time = omp_get_wtime();
#endif

  auto driver_status = driver.Execute();

  // Make final outputs, print diagnostics, clean up and terminate

  if (Globals::my_rank == 0)
    SignalHandler::CancelWallTimeAlarm();

  pman.pouts->MakeOutputs(pman.pmesh.get(), pman.pinput.get());

  // Print diagnostic messages related to the end of the simulation
  if (Globals::my_rank == 0) {
    pman.pmesh->OutputCycleDiagnostics();
    SignalHandler::Report();
    if (driver_status == DriverStatus::complete) {
      std::cout << std::endl << "Driver completed." << std::endl;
    } else if (driver_status == DriverStatus::timeout) {
      std::cout << std::endl << "Driver timed out.  Restart to continue." << std::endl;
    } else if (driver_status == DriverStatus::failed) {
      std::cout << std::endl << "Driver failed." << std::endl;
    }

    std::cout << "time=" << pman.pmesh->time << " cycle=" << pman.pmesh->ncycle << std::endl;
    std::cout << "tlim=" << pman.pmesh->tlim << " nlim=" << pman.pmesh->nlim << std::endl;

    if (pman.pmesh->adaptive) {
      std::cout << std::endl << "Number of MeshBlocks = " << pman.pmesh->nbtotal
                << "; " << pman.pmesh->nbnew << "  created, " << pman.pmesh->nbdel
                << " destroyed during this simulation." << std::endl;
    }

    // Calculate and print the zone-cycles/cpu-second and wall-second
#ifdef OPENMP_PARALLEL
    double omp_time = omp_get_wtime() - omp_start_time;
#endif
    clock_t tstop = clock();
    double cpu_time = (tstop>tstart ? static_cast<double> (tstop-tstart) :
                       1.0)/static_cast<double> (CLOCKS_PER_SEC);
    std::uint64_t zonecycles =
        pman.pmesh->mbcnt*static_cast<std::uint64_t> (pman.pmesh->pblock->GetNumberOfMeshBlockCells());
    double zc_cpus = static_cast<double> (zonecycles) / cpu_time;

    std::cout << std::endl << "zone-cycles = " << zonecycles << std::endl;
    std::cout << "cpu time used  = " << cpu_time << std::endl;
    std::cout << "zone-cycles/cpu_second = " << zc_cpus << std::endl;
#ifdef OPENMP_PARALLEL
    double zc_omps = static_cast<double> (zonecycles) / omp_time;
    std::cout << std::endl << "omp wtime used = " << omp_time << std::endl;
    std::cout << "zone-cycles/omp_wsecond = " << zc_omps << std::endl;
#endif
  }

  pman.ParthenonFinalize();

  return(0);
}
