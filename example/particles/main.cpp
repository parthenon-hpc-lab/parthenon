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

#include "parthenon_manager.hpp"

#include "particles.hpp"

int main(int argc, char *argv[]) {
  using parthenon::ParthenonManager;
  using parthenon::ParthenonStatus;
  ParthenonManager pman;

  // call ParthenonInit to initialize MPI and Kokkos, parse the input deck, and set up
  auto manager_status = pman.ParthenonInit(argc, argv);
  if (manager_status == ParthenonStatus::complete) {
    pman.ParthenonFinalize();
    return 0;
  }
  if (manager_status == ParthenonStatus::error) {
    pman.ParthenonFinalize();
    return 1;
  }
  printf("File: %s Line: %i\n", __FILE__, __LINE__);
  // Now that ParthenonInit has been called and setup succeeded, the code can now
  // make use of MPI and Kokkos

  // Initialize the driver
  particles_example::ParticleDriver driver(pman.pinput.get(), pman.pmesh.get());
                                            //pman.pouts.get());
  printf("File: %s Line: %i\n", __FILE__, __LINE__);

  // start a timer
  pman.PreDriver();
  printf("File: %s Line: %i\n", __FILE__, __LINE__);

  // This line actually runs the simulation
  auto driver_status = driver.Execute();
  printf("File: %s Line: %i\n", __FILE__, __LINE__);

  // Make final outputs, print diagnostics
  pman.PostDriver(driver_status);
  printf("File: %s Line: %i\n", __FILE__, __LINE__);

  // call MPI_Finalize and Kokkos::finalize if necessary
  pman.ParthenonFinalize();

  // MPI and Kokkos can no longer be used

  return (0);
}
