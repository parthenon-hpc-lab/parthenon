//========================================================================================
// (C) (or copyright) 2020-2023. Triad National Security, LLC. All rights reserved.
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

#ifndef TIME_INTEGRATION_STAGED_INTEGRATOR_HPP_
#define TIME_INTEGRATION_STAGED_INTEGRATOR_HPP_

#include <string>
#include <vector>

#include "basic_types.hpp"
#include "parameter_input.hpp"

namespace parthenon {
class StagedIntegrator {
public:
  StagedIntegrator() = default;
  explicit StagedIntegrator(const std::string &name)
    : name_(name) {}

  // JMM: These might be better as private with accessors, but I will
  // leave them public for backwards compatibility.
  Real dt;
  int nstages;
  std::vector<std::string> stage_name;
  
  const std::string &GetName() const {
    return name_;
  }
protected:
  std::string name_;
  void InitStages_(int nstages);
};

class LowStorageIntegrator : public StagedIntegrator {
public:
  LowStorageIntegrator() = default;
  explicit LowStorageIntegrator(ParameterInput *pin);
  std::vector<Real> delta;
  std::vector<Real> beta;
  std::vector<Real> gam0;
  std::vector<Real> gam1;
};
} // namespace parthenon

#endif // TIME_INTEGRATION_STAGED_INTEGRATOR_HPP_
