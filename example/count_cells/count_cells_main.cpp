//========================================================================================
// (C) (or copyright) 2022. Triad National Security, LLC. All rights reserved.
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

// C++ includes
#include <iostream>

// Parthenon includes
#include <parthenon/driver.hpp>

#include "count_cells.hpp"

using namespace parthenon::driver::prelude;

Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;
  packages.Add(count_cells::Initialize(pin.get()));
  return packages;
}

int main(int argc, char *argv;') {
  ParthenonManager pman;
  pman.app_input->ProcessPackages = ProcessPackages;

  // Generates the mesh, which should refine appropriately
  // despite no variables being set.
  auto manager_status = pman.ParthenonInit(argc, argv);
  if (manager_status == ParthenonStatus::complete) {
    pman.ParthenonFinalize();
    return 0;
  }
  if (manager_status == ParthenonStatus::error) {
    pman.ParthenonFinalize();
    return 1;
  }

  count_cells::CountCells(pman.pmesh.get());

  pman.ParthenonFinalize();
  return 0;
}

