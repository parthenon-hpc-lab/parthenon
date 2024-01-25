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

#ifndef DRIVER_MULTISTAGE_HPP_
#define DRIVER_MULTISTAGE_HPP_

#include <memory>
#include <string>
#include <vector>

#include "application_input.hpp"
#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "tasks/tasks.hpp"
#include "time_integration/staged_integrator.hpp"

namespace parthenon {

template <typename Integrator = LowStorageIntegrator>
class MultiStageDriverGeneric : public EvolutionDriver {
 public:
  MultiStageDriverGeneric(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm)
      : EvolutionDriver(pin, app_in, pm), integrator(std::make_unique<Integrator>(pin)) {}
  // An application driver that derives from this class must define this
  // function, which defines the application specific list of tasks and
  // the dependencies that must be executed.
  virtual TaskCollection MakeTaskCollection(BlockList_t &blocks, int stage) = 0;
  virtual TaskListStatus Step() {
    PARTHENON_INSTRUMENT
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
    return status;
  }

 protected:
  std::unique_ptr<Integrator> integrator;
};
using MultiStageDriver = MultiStageDriverGeneric<LowStorageIntegrator>;

template <typename Integrator = LowStorageIntegrator>
class MultiStageBlockTaskDriverGeneric : public MultiStageDriverGeneric<Integrator> {
 public:
  MultiStageBlockTaskDriverGeneric(ParameterInput *pin, ApplicationInput *app_in,
                                   Mesh *pm)
      : MultiStageDriverGeneric<Integrator>(pin, app_in, pm) {}
  virtual TaskList MakeTaskList(MeshBlock *pmb, int stage) = 0;
  virtual TaskListStatus Step() {
    PARTHENON_INSTRUMENT
    using DriverUtils::ConstructAndExecuteBlockTasks;
    TaskListStatus status;
    Integrator *integrator = (this->integrator).get();
    SimTime tm = this->tm;
    integrator->dt = tm.dt;
    for (int stage = 1; stage <= integrator->nstages; stage++) {
      status = ConstructAndExecuteBlockTasks<>(this, stage);
      if (status != TaskListStatus::complete) break;
    }
    return status;
  }

 protected:
  std::unique_ptr<Integrator> integrator;
};
using MultiStageBlockTaskDriver = MultiStageBlockTaskDriverGeneric<LowStorageIntegrator>;

} // namespace parthenon

#endif // DRIVER_MULTISTAGE_HPP_
