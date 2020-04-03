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
#ifndef EXAMPLE_ADVECTION_HPP_
#define EXAMPLE_ADVECTION_HPP_

#include <memory>

#include "driver/driver.hpp"
#include "driver/multistage.hpp"
#include "interface/Container.hpp"
#include "interface/StateDescriptor.hpp"
#include "mesh/mesh.hpp"
#include "task_list/tasks.hpp"

using parthenon::MultiStageBlockTaskDriver;
using parthenon::MeshBlock;
using parthenon::Mesh;
using parthenon::Outputs;
using parthenon::BaseTask;
using parthenon::Real;
using parthenon::CellVariable;
using parthenon::StateDescriptor;
using parthenon::TaskStatus;
using parthenon::TaskList;
using parthenon::TaskListStatus;
using parthenon::TaskID;
using parthenon::Metadata;
using parthenon::Params;
using parthenon::Container;
using parthenon::ParameterInput;
using parthenon::ParArrayND;
using parthenon::Integrator;
using parthenon::BlockStageNamesIntegratorTaskFunc;
using parthenon::BlockStageNamesIntegratorTask;
using parthenon::BlockTaskFunc;
using parthenon::BlockTask;
using parthenon::BoundaryValues;

class AdvectionDriver : public MultiStageBlockTaskDriver {
 public:
  AdvectionDriver(ParameterInput *pin, Mesh *pm, Outputs *pout)
    : MultiStageBlockTaskDriver(pin, pm, pout) {}
  TaskList MakeTaskList(MeshBlock *pmb, int stage);
};

// demonstrate making a custom Task type
using ContainerTaskFunc = std::function<TaskStatus(Container<Real>&)>;
class ContainerTask : public BaseTask {
 public:
  ContainerTask(TaskID id, ContainerTaskFunc func, 
                TaskID dep, Container<Real> rc)
    : BaseTask(id,dep), _func(func), _cont(rc) {}
  TaskStatus operator () () { return _func(_cont); }
 private:
  ContainerTaskFunc _func;
  Container<Real> _cont;
};
using TwoContainerTaskFunc = std::function<TaskStatus(Container<Real>&, Container<Real>&)>;
class TwoContainerTask : public BaseTask {
 public:
  TwoContainerTask(TaskID id, TwoContainerTaskFunc func, 
                   TaskID dep, Container<Real> rc1, Container<Real> rc2)
    : BaseTask(id,dep), _func(func), _cont1(rc1), _cont2(rc2) {}
  TaskStatus operator () () { return _func(_cont1, _cont2); }
 private:
  TwoContainerTaskFunc _func;
  Container<Real> _cont1;
  Container<Real> _cont2;
};

namespace Advection {
  std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
  int CheckRefinement(Container<Real>& rc);
  void PreFill(Container<Real>& rc);
  void SquareIt(Container<Real>& rc);
  void PostFill(Container<Real>& rc);
  Real EstimateTimestep(Container<Real>& rc);
  TaskStatus CalculateFluxes(Container<Real>& rc);
}


#endif // EXAMPLE_ADVECTION_HPP_
