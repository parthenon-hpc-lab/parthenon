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
#include "interface/Container.hpp"
#include "interface/StateDescriptor.hpp"
#include "mesh/mesh.hpp"
#include "task_list/tasks.hpp"

class AdvectionDriver : public parthenon::MultiStageBlockTaskDriver {
 public:
  AdvectionDriver(parthenon::ParameterInput *pin, parthenon::Mesh *pm, parthenon::Outputs *pout)
    : parthenon::MultiStageBlockTaskDriver(pin, pm, pout) {}
  parthenon::TaskList MakeTaskList(parthenon::MeshBlock *pmb, int stage);
};

// demonstrate making a custom Task type
using ContainerTaskFunc = std::function<parthenon::TaskStatus(parthenon::Container<parthenon::Real>&)>;
class ContainerTask : public parthenon::BaseTask {
 public:
  ContainerTask(parthenon::TaskID id, ContainerTaskFunc func,
                parthenon::TaskID dep, parthenon::Container<parthenon::Real> rc)
    : parthenon::BaseTask(id,dep), _func(func), _cont(rc) {}
  parthenon::TaskStatus operator () () { return _func(_cont); }
 private:
  ContainerTaskFunc _func;
  parthenon::Container<parthenon::Real> _cont;
};
using TwoContainerTaskFunc =
  std::function<parthenon::TaskStatus(parthenon::Container<parthenon::Real>&, parthenon::Container<parthenon::Real>&)>;
class TwoContainerTask : public parthenon::BaseTask {
 public:
  TwoContainerTask(parthenon::TaskID id, TwoContainerTaskFunc func,
                   parthenon::TaskID dep, parthenon::Container<parthenon::Real> rc1, parthenon::Container<parthenon::Real> rc2)
    : parthenon::BaseTask(id,dep), _func(func), _cont1(rc1), _cont2(rc2) {}
  parthenon::TaskStatus operator () () { return _func(_cont1, _cont2); }
 private:
  TwoContainerTaskFunc _func;
  parthenon::Container<parthenon::Real> _cont1;
  parthenon::Container<parthenon::Real> _cont2;
};

namespace Advection {
  std::shared_ptr<parthenon::StateDescriptor> Initialize(parthenon::ParameterInput *pin);
  parthenon::AmrTag CheckRefinement(parthenon::Container<parthenon::Real>& rc);
  void PreFill(parthenon::Container<parthenon::Real>& rc);
  void SquareIt(parthenon::Container<parthenon::Real>& rc);
  void PostFill(parthenon::Container<parthenon::Real>& rc);
  parthenon::Real EstimateTimestep(parthenon::Container<parthenon::Real>& rc);
  parthenon::TaskStatus CalculateFluxes(parthenon::Container<parthenon::Real>& rc);
}


#endif // EXAMPLE_ADVECTION_ADVECTION_HPP_
