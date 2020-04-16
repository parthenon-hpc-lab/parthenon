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

// Parthenon Includes
#include <parthenon/app.hpp>

// Local Includes
#include "pi.hpp"

// Preludes
using namespace parthenon::app::prelude;

// Self namespace
using namespace calculate_pi;

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

  return (0);
}
