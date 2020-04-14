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
#ifndef EXAMPLE_ADVECTION_ADVECTION_HPP_
#define EXAMPLE_ADVECTION_ADVECTION_HPP_

#include <memory>

#include "driver/driver.hpp"
#include "driver/multistage.hpp"
#include "interface/container.hpp"
#include "interface/state_descriptor.hpp"
#include "mesh/mesh.hpp"
#include "task_list/tasks.hpp"

using parthenon::AmrTag;
using parthenon::BaseTask;
using parthenon::Container;
using parthenon::Mesh;
using parthenon::MeshBlock;
using parthenon::MultiStageBlockTaskDriver;
using parthenon::Outputs;
using parthenon::ParameterInput;
using parthenon::Real;
using parthenon::StateDescriptor;
using parthenon::TaskID;
using parthenon::TaskList;
using parthenon::TaskStatus;

namespace advection_example {

class AdvectionDriver : public MultiStageBlockTaskDriver {
 public:
  AdvectionDriver(ParameterInput *pin, Mesh *pm, Outputs *pout);
  // This next function essentially defines the driver.
  // Call graph looks like
  // main()
  //   EvolutionDriver::Execute (driver.cpp)
  //     MultiStageBlockTaskDriver::Step (multistage.cpp)
  //       DriverUtils::ConstructAndExecuteBlockTasks (driver.hpp)
  //         AdvectionDriver::MakeTaskList (advection.cpp)
  TaskList MakeTaskList(MeshBlock *pmb, int stage);
};

// demonstrate making a custom Task type
using ContainerTaskFunc = std::function<TaskStatus(Container<Real> &)>;
class ContainerTask : public BaseTask {
 public:
  ContainerTask(TaskID id, ContainerTaskFunc func, TaskID dep, Container<Real> rc)
      : BaseTask(id, dep), _func(func), _cont(rc) {}
  TaskStatus operator()() { return _func(_cont); }

 private:
  ContainerTaskFunc _func;
  Container<Real> _cont;
};
using TwoContainerTaskFunc =
    std::function<TaskStatus(Container<Real> &, Container<Real> &)>;
class TwoContainerTask : public BaseTask {
 public:
  TwoContainerTask(TaskID id, TwoContainerTaskFunc func, TaskID dep, Container<Real> rc1,
                   Container<Real> rc2)
      : BaseTask(id, dep), _func(func), _cont1(rc1), _cont2(rc2) {}
  TaskStatus operator()() { return _func(_cont1, _cont2); }

 private:
  TwoContainerTaskFunc _func;
  Container<Real> _cont1;
  Container<Real> _cont2;
};

namespace Advection {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
AmrTag CheckRefinement(Container<Real> &rc);
void PreFill(Container<Real> &rc);
void SquareIt(Container<Real> &rc);
void PostFill(Container<Real> &rc);
Real EstimateTimestep(Container<Real> &rc);
TaskStatus CalculateFluxes(Container<Real> &rc);

} // namespace Advection

} // namespace advection_example

#endif // EXAMPLE_ADVECTION_ADVECTION_HPP_
