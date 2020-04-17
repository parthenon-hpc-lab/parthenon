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
#ifndef EXAMPLE_CALCULATE_PI_PI_HPP_
#define EXAMPLE_CALCULATE_PI_PI_HPP_

#include <memory>

#include "driver/driver.hpp"
#include "globals.hpp"
#include "interface/state_descriptor.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "parthenon_manager.hpp"
#include "task_list/tasks.hpp"

using parthenon::AmrTag;
using parthenon::BlockTask;
using parthenon::BlockTaskFunc;
using parthenon::CellVariable;
using parthenon::Container;
using parthenon::Coordinates;
using parthenon::DerivedOwnership;
using parthenon::Driver;
using parthenon::DriverStatus;
using parthenon::Mesh;
using parthenon::MeshBlock;
using parthenon::Metadata;
using parthenon::Outputs;
using parthenon::ParameterInput;
using parthenon::Params;
using parthenon::Real;
using parthenon::StateDescriptor;
using parthenon::TaskID;
using parthenon::TaskList;
using parthenon::TaskListStatus;
using parthenon::TaskStatus;
using parthenon::DriverUtils::ConstructAndExecuteBlockTasks;
using parthenon::Globals::my_rank;
using parthenon::Globals::nranks;

class CalculatePi : public Driver {
 public:
  CalculatePi(ParameterInput *pin, Mesh *pm) : Driver(pin, pm) {
    InitializeOutputs();
    pin->CheckDesired("Pi","radius");
    pin->CheckDesired("graphics","variables");
  }
  TaskList MakeTaskList(MeshBlock *pmb);
  DriverStatus Execute();
};

// putting a "physics" package in a namespace
namespace PiCalculator {

void SetInOrOut(Container<Real> &rc);
AmrTag CheckRefinement(Container<Real> &rc);
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
TaskStatus ComputeArea(MeshBlock *pmb);

} // namespace PiCalculator

#endif // EXAMPLE_CALCULATE_PI_PI_HPP_
