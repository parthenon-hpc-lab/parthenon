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

  ParameterInput pin;
  pin.SetInteger("mesh", "nx1", 50);
  pin.SetReal("mesh", "x1min", -0.5);
  pin.SetReal("mesh", "x1max", 0.5);

  pin.SetInteger("mesh", "nx2", 1);
  pin.SetReal("mesh", "x2min", -0.5);
  pin.SetReal("mesh", "x2max", 0.5);

  pin.SetInteger("mesh", "nx3", 1);
  pin.SetReal("mesh", "x3min", -0.5);
  pin.SetReal("mesh", "x3max", 0.5);

  pin.SetReal("time", "tlim", 1.0);

  std::vector<std::shared_ptr<MaterialPropertiesInterface>> mats;
  std::map<std::string, std::shared_ptr<StateDescriptor>> physics;
  Mesh m(&pin, mats, physics, [](Container<Real> &) {});

#ifdef MPI_PARALLEL
  MPI_Finalize();
#endif
}
