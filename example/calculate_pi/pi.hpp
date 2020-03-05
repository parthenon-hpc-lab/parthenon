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


#ifndef CALCULATE_PI_HPP
#define CALCULATE_PI_HPP

#include <memory>

#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "interface/StateDescriptor.hpp"
#include "task_list/tasks.hpp"

class CalculatePi : public Driver {
  public:
   CalculatePi(ParameterInput *pin, Mesh *pm, Outputs *pout) : Driver(pin, pm, pout) {}
   TaskList MakeTaskList(MeshBlock *pmb);
   DriverStatus Execute();
};

void ProcessProperties(std::vector<std::shared_ptr<PropertiesInterface>>& properties, ParameterInput *pin);
void InitializePhysics(std::map<std::string, std::shared_ptr<StateDescriptor>>& physics, ParameterInput *pin); 

// putting a "physics" package in a namespace
namespace PiCalculator {
  void SetInOrOut(Container<Real>& rc);
  int CheckRefinement(Container<Real>& rc);
  std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
  TaskStatus ComputeArea(MeshBlock *pmb);
}

#endif