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
#include "kokkos_abstraction.hpp"
#include "mesh/mesh.hpp"
#include "outputs/outputs.hpp"
#include "task_list/tasks.hpp"

namespace parthenon {

class Mesh;
class ParameterInput;
class Outputs;

enum class DriverStatus { complete, timeout, failed };

class Driver {
 public:
  Driver(ParameterInput *pin, Mesh *pm) : pinput(pin), pmesh(pm), nmb_cycle() {}
  virtual DriverStatus Execute() = 0;
  void InitializeOutputs() { pouts = std::make_unique<Outputs>(pmesh, pinput); }

  ParameterInput *pinput;
  Mesh *pmesh;
  std::unique_ptr<Outputs> pouts;

 protected:
  Kokkos::Timer timer_cycle, timer_main;
  std::uint64_t nmb_cycle;
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
  Kokkos::View<int *, HostMemSpace> mb_locks("mb_locks", nmb);
  auto f = [&](int, int) {
    while (complete_cnt != nmb) {
      for (auto i = 0; i < nmb; ++i) {
        // try to obtain lock by changing val from 0 to 1
        if (Kokkos::atomic_compare_exchange_strong(&mb_locks(i), 0, 1)) {
          if (!task_lists[i].IsComplete()) {
            auto status = task_lists[i].DoAvailable();
            if (status == TaskListStatus::complete) {
              // no reset of the lock here so that no other thread may increment the cnt
              Kokkos::atomic_increment(&complete_cnt);
            }
          }
          Kokkos::atomic_decrement(&mb_locks(i));
        }
      }
    }
  };

#ifdef KOKKOS_ENABLE_OPENMP
  // using a fixed number of partitions (= nthreads) with each partition of size 1,
  // i.e., one thread per partition and this thread is the master thread
  Kokkos::OpenMP::partition_master(f, nthreads, 1);
#else
  f(0, 1);
#endif

  return TaskListStatus::complete;
}

} // namespace DriverUtils

} // namespace parthenon

#endif // DRIVER_DRIVER_HPP_
