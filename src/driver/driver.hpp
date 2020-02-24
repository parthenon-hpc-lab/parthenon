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

#endif
