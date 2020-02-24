#ifndef MULTISTAGE_HPP
#define MULTISTAGE_HPP

#include <vector>
#include <string>

#include "driver/driver.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"

struct Integrator {
    Integrator() = default;
    Integrator(int nstages, std::vector<Real> beta) : _nstages(nstages), _beta(beta) {}
    int _nstages;
    std::vector<Real> _beta;
};

class MultiStageDriver : public EvolutionDriver {
  public:
    MultiStageDriver(ParameterInput *pin, Mesh *pm, Outputs *pout);
    static std::vector<std::string> stage_name;
    static Integrator integrator;
  private:
};

class MultiStageBlockTaskDriver : public MultiStageDriver {
  public:
    MultiStageBlockTaskDriver(ParameterInput *pin, Mesh *pm, Outputs *pout) : MultiStageDriver(pin,pm,pout) {}
    TaskListStatus Step();
    virtual TaskList MakeTaskList(MeshBlock *pmb, int stage) = 0;
};

#endif