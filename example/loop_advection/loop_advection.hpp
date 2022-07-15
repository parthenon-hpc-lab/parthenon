//========================================================================================
// (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
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

#ifndef LOOP_ADVECTION_HPP_
#define LOOP_ADVECTION_HPP_

#include <memory>

//#include "driver/multistage.hpp"
//#include "globals.hpp"
//#include "interface/state_descriptor.hpp"
//#include "mesh/mesh.hpp"

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

namespace loop_advection_example {

parthenon::Packages_t ProcessPackages(std::unique_ptr<parthenon::ParameterInput> &pin);

void ProblemGenerator(parthenon::MeshBlock *pmb, parthenon::ParameterInput *pin);
void UserWorkAfterLoop(parthenon::Mesh *mesh, parthenon::ParameterInput *pin,
                       parthenon::SimTime &tm);

parthenon::TaskStatus calc_dBdt(parthenon::MeshBlock *pmb);
parthenon::TaskStatus calc_E_div(parthenon::MeshBlock *pmb);
parthenon::TaskStatus calc_B_div(parthenon::MeshBlock *pmb);


class LoopAdvectionDriver : public parthenon::MultiStageDriver {
 public:
  LoopAdvectionDriver(parthenon::ParameterInput *pin, parthenon::ApplicationInput *app_in,
                   parthenon::Mesh *pm);
  // This next function essentially defines the driver.
  // Call graph looks like
  // main()
  //   EvolutionDriver::Execute (driver.cpp)
  //     MultiStageBlockTaskDriver::Step (multistage.cpp)
  //       DriverUtils::ConstructAndExecuteTaskLists (driver.hpp)
  //         LoopAdvectionDriver::MakeTaskCollection (advection_driver.cpp)
  parthenon::TaskCollection MakeTaskCollection(parthenon::BlockList_t &blocks, int stage);
};

} // namespace LoopAdvection

#endif // LOOP_ADVECTION_HPP_
