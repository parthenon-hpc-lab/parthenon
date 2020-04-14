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

#ifndef DRIVER_HPP_PK
#define DRIVER_HPP_PK

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "task_list/tasks.hpp"
#include <string>
#include <vector>

namespace parthenon {

class Mesh;
class ParameterInput;
class Outputs;

enum class DriverStatus { complete, timeout, failed };

class Driver {
 public:
  Driver(ParameterInput *pin, Mesh *pm, Outputs *pout)
      : pinput(pin), pmesh(pm), pouts(pout) {}
  virtual DriverStatus Execute() = 0;
  ParameterInput *pinput;
  Mesh *pmesh;
  Outputs *pouts;
};

class SimpleDriver : public Driver {
 public:
  SimpleDriver(ParameterInput *pin, Mesh *pm, Outputs *pout) : Driver(pin, pm, pout) {}
  DriverStatus Execute() { return DriverStatus::complete; }
};

class EvolutionDriver : public Driver {
 public:
  EvolutionDriver(ParameterInput *pin, Mesh *pm, Outputs *pout) : Driver(pin, pm, pout) {}
  DriverStatus Execute();
  virtual TaskListStatus Step() = 0;
};

namespace DriverUtils {
template <typename T, class... Args>
TaskListStatus ConstructAndExecuteBlockTasks(T *driver, Args... args) {
#ifdef OPENMP_PARALLEL
  int nthreads = driver->pmesh->GetNumMeshThreads();
#endif
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
    //#pragma omp parallel for reduction(+ : complete_cnt) num_threads(nthreads)
    //schedule(dynamic,1)
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
} // namespace DriverUtils

} // namespace parthenon
#endif
