//========================================================================================
// (C) (or copyright) 2020-2025. Triad National Security, LLC. All rights reserved.
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

#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "application_input.hpp"
#include "basic_types.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "outputs/outputs.hpp"
#include "parameter_input.hpp"
#include "tasks/tasks.hpp"

namespace parthenon {

class Outputs;

enum class DriverStatus { complete, timeout, failed };

class Driver {
 public:
  Driver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm)
      : pinput(pin), app_input(app_in), pmesh(pm), mbcnt_prev(), time_LBandAMR() {}
  virtual DriverStatus Execute() = 0;
  void InitializeOutputs() { pouts = std::make_unique<Outputs>(pmesh, pinput); }
  void DumpInputParameters();

  ParameterInput *pinput;
  ApplicationInput *app_input;
  Mesh *pmesh;
  std::unique_ptr<Outputs> pouts;
  static double elapsed_main() { return timer_main.seconds(); }
  static double elapsed_cycle() { return timer_cycle.seconds(); }
  static double elapsed_LBandAMR() { return timer_LBandAMR.seconds(); }

 protected:
  static Kokkos::Timer timer_cycle, timer_main, timer_LBandAMR;
  double time_LBandAMR;
  std::uint64_t mbcnt_prev;
  virtual void PreExecute();
  virtual void PostExecute(DriverStatus status);

 private:
};

class EvolutionDriver : public Driver {
 public:
  EvolutionDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm)
      : Driver(pin, app_in, pm) {
    Real start_time =
        pinput->GetOrAddReal("parthenon/time", "start_time", 0.0,
                             "physical time at which to start the simulation");
    Real tstop = pinput->GetOrAddReal("parthenon/time", "tlim",
                                      std::numeric_limits<Real>::infinity(),
                                      "physical time at which to end the simulation");
    Real dt =
        pinput->GetOrAddReal("parthenon/time", "dt", std::numeric_limits<Real>::max(),
                             "initial value of time step before constraining via cfl");
    dt_min =
        pinput->GetOrAddReal("parthenon/time", "dt_min", std::numeric_limits<Real>::min(),
                             "if timestep falls below this level for dt_min_count_max "
                             "iterations, parthenon throws an error");
    dt_max =
        pinput->GetOrAddReal("parthenon/time", "dt_max", std::numeric_limits<Real>::max(),
                             "if timestep is above this level for dt_max_count_max "
                             "iterations, parthenon throws an error");
    dt_init = pinput->GetOrAddReal(
        "parthenon/time", "dt_init", std::numeric_limits<Real>::max(),
        "the first time step will be at least as small as dt_init");
    dt_init_force = pinput->GetOrAddBoolean(
        "parthenon/time", "dt_init_force", false,
        "if set to true, the first time step will be exactly dt_init");

    dt_force = pinput->GetOrAddReal("parthenon/time", "dt_force",
                                    std::numeric_limits<Real>::lowest(),
                                    "if set manually enforces this time step exactly");
    dt_floor = pinput->GetOrAddReal("parthenon/time", "dt_floor",
                                    std::numeric_limits<Real>::min(),
                                    "minimum allowed timestep");
    dt_ceil = pinput->GetOrAddReal("parthenon/time", "dt_ceil",
                                   std::numeric_limits<Real>::max(),
                                   "maximum allowed timestep");
    dt_min_count_max =
        pinput->GetOrAddInteger("parthenon/time", "dt_min_cycle_limit", 10,
                                "number of cycles where dt < dt_min before error");
    dt_max_count_max =
        pinput->GetOrAddInteger("parthenon/time", "dt_max_cycle_limit", 1,
                                "number of cycles where dt > dt_max before error");
    dt_min_count = 0;
    dt_max_count = 0;

    dt_factor = pinput->GetOrAddReal("parthenon/time", "dt_factor", 2.0,
                                     "maximum relative change in dt per timestep");

    const auto ncycle =
        pinput->GetOrAddInteger("parthenon/time", "ncycle", 0, "initial iteration count");
    const auto nmax = pinput->GetOrAddInteger(
        "parthenon/time", "nlim", -1,
        "maximum number of iterations, only limiting if non-negative");
    const auto nout = pinput->GetOrAddInteger("parthenon/time", "ncycle_out", 1,
                                              "cadence of outputs to stdout");
    // disable mesh output by default
    const auto nout_mesh = pinput->GetOrAddInteger("parthenon/time", "ncycle_out_mesh", 0,
                                                   "cadence of outputs describing mesh");
    tm = SimTime(start_time, tstop, nmax, ncycle, nout, nout_mesh, dt);
    pouts = std::make_unique<Outputs>(pmesh, pinput, &tm);

    output_before_amr = pinput->GetOrAddBoolean(
        "parthenon/time", "output_before_amr", false,
        "Set to true to generate outputs in a step BEFORE modifying the mesh at the end "
        "of the step. By default outputs happen AFTER remeshing if remeshing happens. "
        "WARNING: this will make restarts not bitwise-exact.");
  }
  DriverStatus Execute() override;
  virtual void SetGlobalTimeStep();
  virtual void OutputCycleDiagnostics();

  virtual TaskListStatus Step() = 0;
  SimTime tm;

 protected:
  void PostExecute(DriverStatus status) override;
  Real dt_force, dt_init, dt_min, dt_max, dt_floor, dt_ceil;
  Real dt_factor;
  bool dt_init_force;
  int dt_min_count, dt_max_count;
  int dt_min_count_max, dt_max_count_max;
  bool output_before_amr;

 private:
  void InitializeBlockTimeSteps();
};

namespace DriverUtils {

template <typename T, class... Args>
TaskListStatus ConstructAndExecuteBlockTasks(T *driver, Args... args) {
  int nmb = driver->pmesh->GetNumMeshBlocksThisRank(Globals::my_rank);
  TaskCollection tc;
  TaskRegion &tr = tc.AddRegion(nmb);

  int i = 0;
  for (auto &pmb : driver->pmesh->block_list) {
    tr[i++] = driver->MakeTaskList(pmb.get(), std::forward<Args>(args)...);
  }
  TaskListStatus status = tc.Execute(driver->pmesh->task_collection_timeout_in_seconds);
  return status;
}

template <typename T, class... Args>
TaskListStatus ConstructAndExecuteTaskLists(T *driver, Args... args) {
  TaskCollection tc =
      driver->MakeTaskCollection(driver->pmesh->block_list, std::forward<Args>(args)...);
  TaskListStatus status = tc.Execute(driver->pmesh->task_collection_timeout_in_seconds);
  return status;
}

} // namespace DriverUtils

} // namespace parthenon

#endif // DRIVER_DRIVER_HPP_
