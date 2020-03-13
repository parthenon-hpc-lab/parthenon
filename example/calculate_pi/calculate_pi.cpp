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

int Parthenon_MPI_Init(int argc, char *argv[]);
void Parthenon_MPI_Finalize();

//----------------------------------------------------------------------------------------
//! \fn int main(int argc, char *argv[])
//  \brief Athena++ main program

int main(int argc, char *argv[]) {
  //--- Step 1. --------------------------------------------------------------------------
  // Initialize MPI environment, if necessary
  int status = Parthenon_MPI_Init(argc, argv);
  if (!status) return(0);

  //--- Step 2. --------------------------------------------------------------------------
  // Check for command line options and respond.
  ArgParse arg(argc, argv);
  if (arg.exit_flag) {
    Parthenon_MPI_Finalize();
    return(0);
  }

  // Set up the signal handler
  SignalHandler::SignalHandlerInit();
  if (Globals::my_rank == 0 && arg.wtlim > 0)
    SignalHandler::SetWallTimeAlarm(arg.wtlim);

  // Note steps 3-6 are protected by a simple error handler
  //--- Step 3. --------------------------------------------------------------------------
  // Construct object to store input parameters, then parse input file and command line.
  // With MPI, the input is read by every process in parallel using MPI-IO.

  ParameterInput *pinput;
  IOWrapper infile, restartfile;
#ifdef ENABLE_EXCEPTIONS
  try {
#endif
    pinput = new ParameterInput;
    if (arg.iarg_flag == 1) {
      // if both -r and -i are specified, override the parameters using the input file
      infile.Open(arg.input_filename, IOWrapper::FileMode::read);
      pinput->LoadFromFile(infile);
      infile.Close();
    }
    pinput->ModifyFromCmdline(argc ,argv);
#ifdef ENABLE_EXCEPTIONS
  }
  catch(std::bad_alloc& ba) {
    std::cout << "### FATAL ERROR in main" << std::endl
              << "memory allocation failed initializing class ParameterInput: "
              << ba.what() << std::endl;
    if (arg.res_flag == 1) restartfile.Close();
    Parthenon_MPI_Finalize();
    return(0);
  }
  catch(std::exception const& ex) {
    std::cout << ex.what() << std::endl;  // prints diagnostic message
    if (arg.res_flag == 1) restartfile.Close();
    Parthenon_MPI_Finalize();
    return(0);
  }
#endif // ENABLE_EXCEPTIONS

  //--- Step 4. --------------------------------------------------------------------------
  // Construct and initialize Mesh

  FillDerivedVariables::SetFillDerivedFunctions(nullptr,nullptr);

  Mesh *pmesh;
#ifdef ENABLE_EXCEPTIONS
  try {
#endif
    // load properties and physics
    std::vector<std::shared_ptr<PropertiesInterface>> properties;
    ProcessProperties(properties, pinput);

    std::map<std::string, std::shared_ptr<StateDescriptor>> physics;
    InitializePhysics(physics, pinput);

    if (arg.res_flag == 0) {
      pmesh = new Mesh(pinput, properties, physics, arg.mesh_flag);
    } else {
      pmesh = new Mesh(pinput, restartfile, properties, physics, arg.mesh_flag);
    }
#ifdef ENABLE_EXCEPTIONS
  }
  catch(std::bad_alloc& ba) {
    std::cout << "### FATAL ERROR in main" << std::endl
              << "memory allocation failed initializing class Mesh: "
              << ba.what() << std::endl;
    if (arg.res_flag == 1) restartfile.Close();
    Parthenon_MPI_Finalize();
    return(0);
  }
  catch(std::exception const& ex) {
    std::cout << ex.what() << std::endl;  // prints diagnostic message
    if (arg.res_flag == 1) restartfile.Close();
    Parthenon_MPI_Finalize();
    return(0);
  }
#endif // ENABLE_EXCEPTIONS

  //--- Step 5. --------------------------------------------------------------------------
  // Set initial conditions by calling problem generator, or reading restart file

#ifdef ENABLE_EXCEPTIONS
  try {
#endif
    pmesh->Initialize(arg.res_flag, pinput);
#ifdef ENABLE_EXCEPTIONS
  }
  catch(std::bad_alloc& ba) {
    std::cout << "### FATAL ERROR in main" << std::endl << "memory allocation failed "
              << "in problem generator " << ba.what() << std::endl;
    Parthenon_MPI_Finalize();
    return(0);
  }
  catch(std::exception const& ex) {
    std::cout << ex.what() << std::endl;  // prints diagnostic message
    Parthenon_MPI_Finalize();
    return(0);
  }
#endif // ENABLE_EXCEPTIONS

  //--- Step 6. --------------------------------------------------------------------------
  // Change to run directory, initialize outputs object, and make output of ICs

  Outputs *pouts;
#ifdef ENABLE_EXCEPTIONS
  try {
#endif
    ChangeRunDir(arg.prundir);
    pouts = new Outputs(pmesh, pinput);
    if (arg.res_flag == 0) pouts->MakeOutputs(pmesh, pinput);
#ifdef ENABLE_EXCEPTIONS
  }
  catch(std::bad_alloc& ba) {
    std::cout << "### FATAL ERROR in main" << std::endl
              << "memory allocation failed setting initial conditions: "
              << ba.what() << std::endl;
    Parthenon_MPI_Finalize();
    return(0);
  }
  catch(std::exception const& ex) {
    std::cout << ex.what() << std::endl;  // prints diagnostic message
    Parthenon_MPI_Finalize();
    return(0);
  }
#endif // ENABLE_EXCEPTIONS

  //=== Step 7. === START OF MAIN INTEGRATION LOOP =======================================
  // For performance, there is no error handler protecting this step (except outputs)

  CalculatePi driver(pinput, pmesh, pouts);

  if (Globals::my_rank == 0) {
    std::cout << "\n"<<Globals::my_rank<<":Setup complete, entering main loop...\n" << std::endl;
  }

  clock_t tstart = clock();
#ifdef OPENMP_PARALLEL
  double omp_start_time = omp_get_wtime();
#endif

  driver.Execute();

  // Make final outputs, print diagnostics, clean up and terminate

  if (Globals::my_rank == 0 && arg.wtlim > 0)
    SignalHandler::CancelWallTimeAlarm();

  //--- Step 9. --------------------------------------------------------------------------
  // Make the final outputs
#ifdef ENABLE_EXCEPTIONS
  try {
#endif
    pouts->MakeOutputs(pmesh,pinput,true);
#ifdef ENABLE_EXCEPTIONS
  }
  catch(std::bad_alloc& ba) {
    std::cout << "### FATAL ERROR in main" << std::endl
              << "memory allocation failed during output: " << ba.what() <<std::endl;
    Parthenon_MPI_Finalize();
    return(0);
  }
  catch(std::exception const& ex) {
    std::cout << ex.what() << std::endl;  // prints diagnostic message
    Parthenon_MPI_Finalize();
    return(0);
  }
#endif // ENABLE_EXCEPTIONS

  pmesh->UserWorkAfterLoop(pinput);

  //--- Step 10. -------------------------------------------------------------------------
  // Print diagnostic messages related to the end of the simulation
  if (Globals::my_rank == 0) {
    pmesh->OutputCycleDiagnostics();
    if (SignalHandler::GetSignalFlag(SIGTERM) != 0) {
      std::cout << std::endl << "Terminating on Terminate signal" << std::endl;
    } else if (SignalHandler::GetSignalFlag(SIGINT) != 0) {
      std::cout << std::endl << "Terminating on Interrupt signal" << std::endl;
    } else if (SignalHandler::GetSignalFlag(SIGALRM) != 0) {
      std::cout << std::endl << "Terminating on wall-time limit" << std::endl;
    } else if (pmesh->ncycle == pmesh->nlim) {
      std::cout << std::endl << "Terminating on cycle limit" << std::endl;
    } else {
      std::cout << std::endl << "Terminating on time limit" << std::endl;
    }

    std::cout << "time=" << pmesh->time << " cycle=" << pmesh->ncycle << std::endl;
    std::cout << "tlim=" << pmesh->tlim << " nlim=" << pmesh->nlim << std::endl;

    if (pmesh->adaptive) {
      std::cout << std::endl << "Number of MeshBlocks = " << pmesh->nbtotal
                << "; " << pmesh->nbnew << "  created, " << pmesh->nbdel
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
        pmesh->mbcnt*static_cast<std::uint64_t> (pmesh->pblock->GetNumberOfMeshBlockCells());
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

  delete pinput;
  delete pmesh;
  delete pouts;

  Parthenon_MPI_Finalize();

  return(0);
}


int Parthenon_MPI_Init(int argc, char *argv[]) {
  int status = 1;
#ifdef MPI_PARALLEL
#ifdef OPENMP_PARALLEL
  int mpiprv;
  if (MPI_SUCCESS != MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpiprv)) {
    std::cout << "### FATAL ERROR in main" << std::endl
              << "MPI Initialization failed." << std::endl;
    status = 0;
  }
  if (mpiprv != MPI_THREAD_MULTIPLE) {
    std::cout << "### FATAL ERROR in main" << std::endl
              << "MPI_THREAD_MULTIPLE must be supported for the hybrid parallelzation. "
              << MPI_THREAD_MULTIPLE << " : " << mpiprv
              << std::endl;
    MPI_Finalize();
    status = 0;
  }
#else  // no OpenMP
  if (MPI_SUCCESS != MPI_Init(&argc, &argv)) {
    std::cout << "### FATAL ERROR in main" << std::endl
              << "MPI Initialization failed." << std::endl;
    status = 0;
  }
#endif  // OPENMP_PARALLEL
  // Get process id (rank) in MPI_COMM_WORLD
  if (MPI_SUCCESS != MPI_Comm_rank(MPI_COMM_WORLD, &(Globals::my_rank))) {
    std::cout << "### FATAL ERROR in main" << std::endl
              << "MPI_Comm_rank failed." << std::endl;
    MPI_Finalize();
    status = 0;
  }

  // Get total number of MPI processes (ranks)
  if (MPI_SUCCESS != MPI_Comm_size(MPI_COMM_WORLD, &Globals::nranks)) {
    std::cout << "### FATAL ERROR in main" << std::endl
              << "MPI_Comm_size failed." << std::endl;
    MPI_Finalize();
    status = 0;
  }
#else  // no MPI
  Globals::my_rank = 0;
  Globals::nranks  = 1;
#endif  // MPI_PARALLEL
  return status;
}

void Parthenon_MPI_Finalize() {
#ifdef MPI_PARALLEL
    MPI_Finalize();
#endif
}
