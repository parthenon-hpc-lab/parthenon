//========================================================================================
// (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
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

//#include <parthenon/parthenon_manager.hpp>
#include "loop_advection.hpp"

int main(int argc, char *argv[]) {
  using loop_advection_example::LoopAdvectionDriver;
  using parthenon::ParthenonManager;
  using parthenon::ParthenonStatus;
  ParthenonManager pman;

  pman.app_input->ProcessPackages = loop_advection_example::ProcessPackages;
  pman.app_input->ProblemGenerator = loop_advection_example::ProblemGenerator;
  pman.app_input->UserWorkAfterLoop = loop_advection_example::UserWorkAfterLoop;

  auto status = pman.ParthenonInit(argc, argv);
  if (status == ParthenonStatus::complete || status == ParthenonStatus::error) {
    pman.ParthenonFinalize();
    return (status == ParthenonStatus::error) ? 1 : 0;
  }

  LoopAdvectionDriver driver(pman.pinput.get(), pman.app_input.get(), pman.pmesh.get());
  driver.Execute();
  pman.ParthenonFinalize();

  return 0;
}
