//========================================================================================
// (C) (or copyright) 2020-2025. Triad National Security, LLC. All rights
// reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001
// for Los Alamos National Laboratory (LANL), which is operated by Triad
// National Security, LLC for the U.S. Department of Energy/National Nuclear
// Security Administration. All rights in the program are reserved by Triad
// National Security, LLC, and the U.S. Department of Energy/National Nuclear
// Security Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
// in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do
// so.
//========================================================================================

#ifndef SRC_XMHD_DRIVER_HPP_
#define SRC_XMHD_DRIVER_HPP_

#include <memory>
#include <vector>

#include <driver/multistage.hpp>
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

#include "utilities/imexrk_integrator.hpp"

namespace scalar_imex {
using namespace parthenon::driver::prelude;

class ScalarIMEXDriver : public parthenon::MultiStageDriverGeneric<IMEXRKIntegrator> {
  bool do_hydro;
  bool do_advection;
  bool do_em;
  bool do_scalar_imex;

 public:
  ScalarIMEXDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm);
  // This next function essentially defines the driver.
  // Call graph looks like
  // main()
  //   EvolutionDriver::Execute (driver.cpp)
  //     MultiStageBlockTaskDriver::Step (multistage.cpp)
  //       DriverUtils::ConstructAndExecuteTaskLists (driver.hpp)
  //         ScalarIMEXDriver::MakeTaskCollection (advection_driver.cpp)
  TaskCollection MakeTaskCollection(BlockList_t &blocks, int stage) final;
};

parthenon::Packages_t ProcessPackages(std::unique_ptr<parthenon::ParameterInput> &pin);

} // namespace scalar_imex
#endif // SRC_XMHD_DRIVER_HPP_
