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
#ifndef EXAMPLE_ADVECTION_ADVECTION_PACKAGE_HPP_
#define EXAMPLE_ADVECTION_ADVECTION_PACKAGE_HPP_

#include <memory>

#include "basic_types.hpp"
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
using parthenon::SimTime;
using parthenon::StateDescriptor;
using parthenon::TaskID;
using parthenon::TaskList;
using parthenon::TaskStatus;

namespace advection_package {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
AmrTag CheckRefinement(Container<Real> &rc);
void PreFill(Container<Real> &rc);
void SquareIt(Container<Real> &rc);
void PostFill(Container<Real> &rc);
Real EstimateTimestep(Container<Real> &rc);
TaskStatus CalculateFluxes(Container<Real> &rc);

} // namespace advection_package

#endif // EXAMPLE_ADVECTION_ADVECTION_PACKAGE_HPP_
