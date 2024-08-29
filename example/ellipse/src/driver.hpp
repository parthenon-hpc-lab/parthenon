//========================================================================================
// (C) (or copyright) 2024. Triad National Security, LLC. All rights reserved.
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

#ifndef _DRIVER_HPP_
#define _DRIVER_HPP_

#include <parthenon/driver.hpp>

class ToyDriver : parthenon::Driver {
public:
  ToyDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm)
    : parthenon::Driver(pin, app_in, pm) {
    InitializeOutputs();
  }
  parthenon::DriverStatus Execute() override {
    pouts->MakeOutputs(pmesh, pinput);
    return parthenon::DriverStatus::complete;
  }
};

#endif // _DRIVER_HPP_
