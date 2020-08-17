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

#ifndef DRIVER_DRIVER_HPP_
#define DRIVER_DRIVER_HPP_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "basic_types.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "outputs/outputs.hpp"
#include "parameter_input.hpp"
#include "tasks/task_list.hpp"

namespace parthenon {

class Outputs;

enum class DriverStatus { complete, timeout, failed };

class Driver {
 public:
  Driver(ParameterInput *pin, Mesh *pm) : pinput(pin), pmesh(pm) {}
  virtual DriverStatus Execute() = 0;
  void InitializeOutputs() { pouts = std::make_unique<Outputs>(pmesh, pinput); }

  ParameterInput *pinput;
  Mesh *pmesh;
  std::unique_ptr<Outputs> pouts;

 protected:
  clock_t tstart_;
#ifdef OPENMP_PARALLEL
  double omp_start_time_;
#endif
  virtual void PreExecute();
  virtual void PostExecute();

 private:
};

class EvolutionDriver : public Driver {
 public:
  EvolutionDriver(ParameterInput *pin, Mesh *pm) : Driver(pin, pm) {
    Real start_time = pinput->GetOrAddReal("parthenon/time", "start_time", 0.0);
    Real tstop = pinput->GetReal("parthenon/time", "tlim");
    int nmax = pinput->GetOrAddInteger("parthenon/time", "nlim", -1);
    int nout = pinput->GetOrAddInteger("parthenon/time", "ncycle_out", 1);
    // TODO(jcd): the 0 below should be the current cycle number, not necessarily 0
    tm = SimTime(start_time, tstop, nmax, 0, nout);
    pouts = std::make_unique<Outputs>(pmesh, pinput, &tm);
  }
  DriverStatus Execute() override;
  void SetGlobalTimeStep();
  void OutputCycleDiagnostics();

  virtual TaskListStatus Step() = 0;
  SimTime tm;

 protected:
  virtual void PostExecute(DriverStatus status);

 private:
  void InitializeBlockTimeSteps();
};

namespace DriverUtils {

template <typename T, class... Args>
TaskListStatus ConstructAndExecuteBlockTasks(T *driver, Args... args) {
  int nmb = driver->pmesh->GetNumMeshBlocksThisRank(Globals::my_rank);
  std::vector<TaskList> task_lists;
  MeshBlock *pmb = driver->pmesh->pblock;
  while (pmb != nullptr) {
    task_lists.push_back(driver->MakeTaskList(pmb, std::forward<Args>(args)...));
    pmb = pmb->next;
  }
  int complete_cnt = 0;
  while (complete_cnt != nmb) {
    // TODO(pgrete): need to let Kokkos::PartitionManager handle this
    for (auto i = 0; i < nmb; ++i) {
      if (!task_lists[i].IsComplete()) {
        auto status = task_lists[i].DoAvailable();
        if (status == TaskListStatus::complete) {
          complete_cnt++;
        }
      }
    }
  }
  return TaskListStatus::complete;
}

template <typename T, class... Args>
TaskListStatus ConstructAndExecuteTaskLists(T *driver, Args... args) {
  int nmb = driver->pmesh->GetNumMeshBlocksThisRank(Globals::my_rank);
  MeshBlock *pmb = driver->pmesh->pblock;
  std::vector<MeshBlock *> blocks(nmb);
  while (pmb != nullptr) {
    // task_lists.push_back(driver->MakeTaskList(pmb, std::forward<Args>(args)...));
    blocks.emplace_back(pmb);
    pmb = pmb->next;
  }
  std::vector<std::shared_ptr<TaskList>> task_lists =
      driver->MakeTaskLists(blocks, std::forward<Args>(args)...);
  int complete_cnt = 0;
  while (complete_cnt != nmb) {
    // TODO(pgrete): need to let Kokkos::PartitionManager handle this
    for (auto i = 0; i < nmb; ++i) {
      if (!task_lists[i]->IsComplete()) {
        auto status = task_lists[i]->DoAvailable();
        if (status == TaskListStatus::complete) {
          complete_cnt++;
        }
      }
    }
  }
  return TaskListStatus::complete;
}

} // namespace DriverUtils

} // namespace parthenon

#endif // DRIVER_DRIVER_HPP_
