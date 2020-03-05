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

// Third Party Includes
#include <mpi.h>

// Athena++ headers
#include "mesh/mesh.hpp"

//----------------------------------------------------------------------------------------
//! \fn int main(int argc, char *argv[])
//  \brief Parthenon Sample main program

int main(int argc, char *argv[]) {
  //--- Step 1. --------------------------------------------------------------------------
  // Initialize MPI environment, if necessary

#ifdef MPI_PARALLEL
#ifdef OPENMP_PARALLEL
  int mpiprv;
  if (MPI_SUCCESS != MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpiprv)) {
    std::cout << "### FATAL ERROR in main" << std::endl
              << "MPI Initialization failed." << std::endl;
    return 1;
  }
  if (mpiprv != MPI_THREAD_MULTIPLE) {
    std::cout << "### FATAL ERROR in main" << std::endl
              << "MPI_THREAD_MULTIPLE must be supported for the hybrid parallelzation. "
              << MPI_THREAD_MULTIPLE << " : " << mpiprv
              << std::endl;
    MPI_Finalize();
    return 2;
  }
#else  // no OpenMP
  if (MPI_SUCCESS != MPI_Init(&argc, &argv)) {
    std::cout << "### FATAL ERROR in main" << std::endl
              << "MPI Initialization failed." << std::endl;
    return 3;
  }
#endif  // OPENMP_PARALLEL
  // Get process id (rank) in MPI_COMM_WORLD
  if (MPI_SUCCESS != MPI_Comm_rank(MPI_COMM_WORLD, &(Globals::my_rank))) {
    std::cout << "### FATAL ERROR in main" << std::endl
              << "MPI_Comm_rank failed." << std::endl;
    MPI_Finalize();
    return 4;
  }

  // Get total number of MPI processes (ranks)
  if (MPI_SUCCESS != MPI_Comm_size(MPI_COMM_WORLD, &Globals::nranks)) {
    std::cout << "### FATAL ERROR in main" << std::endl
              << "MPI_Comm_size failed." << std::endl;
    MPI_Finalize();
    return 5;
  }
#else  // no MPI
  Globals::my_rank = 0;
  Globals::nranks  = 1;
#endif  // MPI_PARALLEL

  if (argc != 2) {
    if (Globals::my_rank == 0) {
      std::cout << "\nUsage: " << argv[0] << " input_file\n"
        << "\tTry this input file:\n"
        << "\tparthenon/example/parthinput.example"
        << std::endl;
    }
    return 0;
  }
  std::string inputFileName = argv[1];
  ParameterInput pin;
  IOWrapper inputFile;
  inputFile.Open(inputFileName.c_str(), IOWrapper::FileMode::read);
  pin.LoadFromFile(inputFile);
  inputFile.Close();

  if (Globals::my_rank == 0) {
    std::cout << "\ninput file = " << inputFileName << std::endl;
    if (pin.DoesParameterExist("mesh","nx1")) {
      std::cout << "nx1 = " << pin.GetInteger("mesh","nx1") << std::endl;
    }
    if (pin.DoesParameterExist("mesh","x1min")) {
      std::cout << "x1min = " << pin.GetReal("mesh","x1min") << std::endl;
    }
    if (pin.DoesParameterExist("mesh","x1max")) {
      std::cout << "x1max = " << pin.GetReal("mesh","x1max") << std::endl;
    }
    if (pin.DoesParameterExist("mesh", "ix1_bc")) {
      std::cout << "x1 inner boundary condition = "
        << pin.GetString("mesh","ix1_bc") << std::endl;
    }
    if (pin.DoesParameterExist("mesh", "ox1_bc")) {
      std::cout << "x1 outer boundary condition = "
        << pin.GetString("mesh","ox1_bc") << std::endl;
    }
  }

  std::vector<std::shared_ptr<PropertiesInterface>> mats;
  std::map<std::string, std::shared_ptr<StateDescriptor>> physics;
  Mesh m(&pin, mats, physics);

#ifdef MPI_PARALLEL
  MPI_Finalize();
#endif
}
