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

#ifndef DRIVER_HPP_PK
#define DRIVER_HPP_PK

#include <vector>
#include <string>
#include "athena.hpp"
#include "task_list/tasks.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "outputs/outputs.hpp"
#include "interface/Update.hpp"

namespace parthenon {

enum class DriverStatus {complete, failed};

class Driver {
  public:
    Driver(ParameterInput *pin, Mesh *pm, Outputs *pout) : pinput(pin), pmesh(pm), pouts(pout) { }
    virtual DriverStatus Execute() = 0;
    ParameterInput *pinput;
    Mesh *pmesh;
    Outputs *pouts;
};

class SimpleDriver : public Driver {
  public:
    SimpleDriver(ParameterInput *pin, Mesh *pm, Outputs *pout) : Driver(pin,pm,pout) {}
    DriverStatus Execute() { return DriverStatus::complete; }
};

class EvolutionDriver : public Driver {
  public:
    EvolutionDriver(ParameterInput *pin, Mesh *pm, Outputs *pout) : Driver(pin,pm,pout) {}
    DriverStatus Execute();
    virtual TaskListStatus Step() = 0;
};
} // namespace parthenon
#endif
