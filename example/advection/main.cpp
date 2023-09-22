//========================================================================================
// (C) (or copyright) 2020-2023. Triad National Security, LLC. All rights reserved.
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

#include "advection_driver.hpp"

int main(int argc, char *argv[]) {
  using parthenon::ParthenonManager;
  using parthenon::ParthenonStatus;
  ParthenonManager pman;

  // Redefine parthenon defaults
  pman.app_input->ProcessPackages = advection_example::ProcessPackages;
  pman.app_input->ProblemGenerator = advection_example::ProblemGenerator;
  pman.app_input->UserWorkAfterLoop = advection_example::UserWorkAfterLoop;

  // call ParthenonInit to initialize MPI and Kokkos, parse the input deck, and set up
  auto manager_status = pman.ParthenonInitEnv(argc, argv);
  if (manager_status == ParthenonStatus::complete) {
    pman.ParthenonFinalize();
    return 0;
  }
  if (manager_status == ParthenonStatus::error) {
    pman.ParthenonFinalize();
    return 1;
  }

  // Now that ParthenonInit has been called and setup succeeded, the code can now
  // make use of MPI and Kokkos.
  // This needs to be scoped so that the driver object is destructed before Finalize
  pman.ParthenonInitPackagesAndMesh();
  {
    // Initialize the driver
    advection_example::AdvectionDriver driver(pman.pinput.get(), pman.app_input.get(),
                                              pman.pmesh.get());

    // This line actually runs the simulation
    auto driver_status = driver.Execute();
  }
  // call MPI_Finalize and Kokkos::finalize if necessary
  pman.ParthenonFinalize();

  // MPI and Kokkos can no longer be used

  return (0);
}
