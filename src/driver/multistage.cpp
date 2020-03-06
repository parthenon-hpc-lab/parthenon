
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

#include "multistage.hpp"

namespace parthenon {

MultiStageDriver::MultiStageDriver(ParameterInput *pin, Mesh *pm, Outputs *pout) : EvolutionDriver(pin,pm,pout) {
  pmesh = pm;
  std::string integrator_name = pin->GetOrAddString("time", "integrator", "rk2");

  int nstages;
  std::vector<Real> beta;
  if (!integrator_name.compare("rk1")) {
    nstages = 1;
    beta.resize(nstages);
    beta[0] = 1.0;
  } else if (!integrator_name.compare("rk2")) {
    nstages = 2;
    beta.resize(nstages);
    beta[0] = 1.0;
    beta[1] = 0.5;
  } else if (!integrator_name.compare("rk3")) {
    nstages = 3;
    beta.resize(nstages);
    beta[0] = 1.0;
    beta[1] = 0.25;
    beta[2] = 2.0/3.0;
  } else {
    // this should be an error
  }

  integrator = new Integrator(nstages, beta);
  stage_name.resize(nstages+1);
  stage_name[0] = "base";
  for (int i=1; i<nstages; i++) {
    stage_name[i] = std::to_string(i);
  }
  stage_name[nstages] = stage_name[0];

}

TaskListStatus MultiStageBlockTaskDriver::Step() {
  int nmb = pmesh->GetNumMeshBlocksThisRank(Globals::my_rank);
  std::vector<TaskList> task_lists;
  task_lists.resize(nmb);

  for (int stage=1; stage<=integrator->_nstages; stage++) {
    int i=0;
    MeshBlock *pmb = pmesh->pblock;
    while (pmb != nullptr) {
      task_lists[i] = MakeTaskList(pmb, stage);
      i++;
      pmb = pmb->next;
    }
    int complete_cnt = 0;
    while (complete_cnt != nmb) {
      for (auto & tl : task_lists) {
        if (!tl.IsComplete()) {
            auto status = tl.DoAvailable();
            if (status == TaskListStatus::complete) {
              complete_cnt++;
            }
        }
      }
    }
  }
  return TaskListStatus::complete;
}

} // namespace parthenon