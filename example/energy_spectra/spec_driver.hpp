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

#ifndef EXAMPLE_CALCULATE_PI_PI_DRIVER_HPP_
#define EXAMPLE_CALCULATE_PI_PI_DRIVER_HPP_

#include <memory>
#include <vector>

#include <parthenon/driver.hpp>

namespace pi {
using namespace parthenon::driver::prelude;

/**
 * @brief Constructs a driver which estimates PI using AMR.
 */
class PiDriver : public Driver {
 public:
  PiDriver(ParameterInput *pin, ApplicationInput *fin, Mesh *pm) : Driver(pin, fin, pm) {
    InitializeOutputs();
    pin->CheckDesired("Pi", "radius");
  }

  /// MakeTaskList and MakeTasks aren't virtual routines on `Driver`,
  // but each driver is expected to implement at least one of them.
  /// TaskList MakeTaskList(MeshBlock *pmb);
  template <typename T>
  TaskCollection MakeTaskCollection(T &blocks);

  /// `Execute` cylces until simulation completion.
  DriverStatus Execute() override;

 protected:
  void PiPostExecute(Real pi_val);
};

} // namespace pi

#endif // EXAMPLE_CALCULATE_PI_PI_DRIVER_HPP_
