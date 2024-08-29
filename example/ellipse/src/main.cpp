//========================================================================================
// (C) (or copyright) 2024. Triad National Security, LLC. All rights reserved.
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

#include <parthenon/driver.hpp>
#include <parthenon_manager.hpp>
using namespace parthenon::driver::prelude;

#include "driver.hpp"
#include "ellipse/ellipse.hpp"
#include "indicator/indicator.hpp"
#include "pgen.hpp"

int main(int argc, char *argv[]) {
  parthenon::ParthenonManager pman;

  // Set up kokkos and read pin
  auto manager_status = pman.ParthenonInitEnv(argc, argv);
  if (manager_status == ParthenonStatus::complete) {
    pman.ParthenonFinalize();
    return 0;
  }
  if (manager_status == ParthenonStatus::error) {
    pman.ParthenonFinalize();
    return 1;
  }

  // Redefine parthenon defaults
  pman.app_input->ProcessPackages = [](std::unique_ptr<ParameterInput> &pin) {
    Packages_t packages;
    packages.Add(Indicator::Initialize(pin.get()));
    packages.Add(Ellipse::Initialize(pin.get()));
    return packages;
  };
  pman.app_input->ProblemGenerator = SetupEllipse;

  // call ParthenonInit to set up the mesh
  // scope so that the mesh object, kokkos views, etc, all get cleaned
  // up before kokkos::finalize
  pman.ParthenonInitPackagesAndMesh();
  {

    // Initialize the driver
    ToyDriver driver(pman.pinput.get(), pman.app_input.get(), pman.pmesh.get());

    // This line actually runs the simulation
    auto driver_status = driver.Execute(); // unneeded here

    // call MPI_Finalize and Kokkos::finalize if necessary
  }
  pman.ParthenonFinalize();

  // MPI and Kokkos can no longer be used

  return (0);
}
