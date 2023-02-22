
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

/*
 * These integrators are of the 2S form as described in
 * Ketchson, Jcomp 229 (2010) 1763-1773
 * See Equation 14.
 * They can be generalized to support more general methods with the
 * introduction of a delta term for a first averaging.
 */
StagedIntegrator::StagedIntegrator(ParameterInput *pin) {
  std::string integrator_name =
      pin->GetOrAddString("parthenon/time", "integrator", "rk2");

  if (integrator_name == "rk1") {
    nstages = 1;
    delta.resize(nstages);
    beta.resize(nstages);
    gam0.resize(nstages);
    gam1.resize(nstages);

    delta[0] = 1.0;
    beta[0] = 1.0;
    gam0[0] = 0.0;
    gam1[0] = 1.0;
  } else if (integrator_name == "rk2") {
    nstages = 2;
    delta.resize(nstages);
    beta.resize(nstages);
    gam0.resize(nstages);
    gam1.resize(nstages);

    delta[0] = 1.0;
    beta[0] = 1.0;
    gam0[0] = 0.0;
    gam1[0] = 1.0;

    delta[1] = 0.0;
    beta[1] = 0.5;
    gam0[1] = 0.5;
    gam1[1] = 0.5;
  } else if (integrator_name == "vl2") {
    nstages = 2;
    delta.resize(nstages);
    beta.resize(nstages);
    gam0.resize(nstages);
    gam1.resize(nstages);

    delta[0] = 1.0;
    beta[0] = 0.5;
    gam0[0] = 0.0;
    gam1[0] = 1.0;

    delta[1] = 0.0;
    beta[1] = 1.0;
    gam0[1] = 0.0;
    gam1[1] = 1.0;
  } else if (integrator_name == "rk3") {
    nstages = 3;
    delta.resize(nstages);
    beta.resize(nstages);
    gam0.resize(nstages);
    gam1.resize(nstages);

    delta[0] = 1.0;
    beta[0] = 1.0;
    gam0[0] = 0.0;
    gam1[0] = 1.0;

    delta[1] = 0.0;
    beta[1] = 0.25;
    gam0[1] = 0.25;
    gam1[1] = 0.75;

    delta[2] = 0.0;
    beta[2] = 2.0 / 3.0;
    gam0[2] = 2.0 / 3.0;
    gam1[2] = 1.0 / 3.0;
  } else if (integrator_name == "rk4") {
    // From Table 4 of Ketchson, Jcomp 229 (2010) 1763-1773
    nstages = 5;
    delta.resize(nstages);
    beta.resize(nstages);
    gam0.resize(nstages);
    gam1.resize(nstages);

    delta[0] = 1.0;
    beta[0] = 0.357534921136978;
    gam0[0] = 0.0;
    gam1[0] = 1.0;

    delta[1] = 0.0;
    beta[1] = 2.364680399061355;
    gam0[1] = -3.666545952121251;
    gam1[1] = 4.666545952121251;

    delta[2] = 0.0;
    beta[2] = 0.016239790859612;
    gam0[2] = 0.035802535958088;
    gam1[2] = 0.964197464041912;

    delta[3] = 0.0;
    beta[3] = 0.498173799587251;
    gam0[3] = 4.398279365655791;
    gam1[3] = -3.398279365655790;

    delta[4] = 0.0;
    beta[4] = 0.433334235669763;
    gam0[4] = 0.770411587328417;
    gam1[4] = 0.229588412671583;
  } else {
    throw std::invalid_argument("Invalid selection for the time integrator: " +
                                integrator_name);
  }
  stage_name.resize(nstages + 1);
  stage_name[0] = "base";
  for (int i = 1; i < nstages; i++) {
    stage_name[i] = std::to_string(i);
  }
  stage_name[nstages] = stage_name[0];
}

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
