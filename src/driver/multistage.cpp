
//========================================================================================
// (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
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

#include "driver/multistage.hpp"

namespace parthenon {

TaskListStatus MultiStageDriver::Step() {
  Kokkos::Profiling::pushRegion("MultiStage_Step");
  using DriverUtils::ConstructAndExecuteTaskLists;
  TaskListStatus status;
  integrator->dt = tm.dt;
  for (int stage = 1; stage <= integrator->nstages; stage++) {
    // Clear any initialization info. We should be relying
    // on only the immediately preceding stage to contain
    // reasonable data
    pmesh->SetAllVariablesToInitialized();
    status = ConstructAndExecuteTaskLists<>(this, stage);
    if (status != TaskListStatus::complete) break;
  }
  Kokkos::Profiling::popRegion(); // MultiStage_Step
  return status;
}

TaskListStatus MultiStageBlockTaskDriver::Step() {
  Kokkos::Profiling::pushRegion("MultiStageBlockTask_Step");
  using DriverUtils::ConstructAndExecuteBlockTasks;
  TaskListStatus status;
  integrator->dt = tm.dt;
  for (int stage = 1; stage <= integrator->nstages; stage++) {
    status = ConstructAndExecuteBlockTasks<>(this, stage);
    if (status != TaskListStatus::complete) break;
  }
  Kokkos::Profiling::popRegion(); // MultiStageBlockTask_Step
  return status;
}

} // namespace parthenon
