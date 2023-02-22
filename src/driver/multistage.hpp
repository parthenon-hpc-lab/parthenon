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

#ifndef DRIVER_MULTISTAGE_HPP_
#define DRIVER_MULTISTAGE_HPP_

#include <memory>
#include <string>
#include <vector>

#include "application_input.hpp"
#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "tasks/task_list.hpp"

namespace parthenon {

struct StagedIntegrator {
  StagedIntegrator() = default;
  explicit StagedIntegrator(ParameterInput *pin);
  int nstages;
  std::vector<Real> delta;
  std::vector<Real> beta;
  std::vector<Real> gam0;
  std::vector<Real> gam1;
  std::vector<std::string> stage_name;
  Real dt;
};

class MultiStageDriver : public EvolutionDriver {
 public:
  MultiStageDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm)
      : EvolutionDriver(pin, app_in, pm),
        integrator(std::make_unique<StagedIntegrator>(pin)) {}
  // An application driver that derives from this class must define this
  // function, which defines the application specific list of tasks and
  // the dependencies that must be executed.
  virtual TaskCollection MakeTaskCollection(BlockList_t &blocks, int stage) = 0;
  virtual TaskListStatus Step();

 protected:
  std::unique_ptr<StagedIntegrator> integrator;
};

class MultiStageBlockTaskDriver : public MultiStageDriver {
 public:
  MultiStageBlockTaskDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm)
      : MultiStageDriver(pin, app_in, pm) {}
  virtual TaskList MakeTaskList(MeshBlock *pmb, int stage) = 0;
  virtual TaskListStatus Step();
};

} // namespace parthenon

#endif // DRIVER_MULTISTAGE_HPP_
