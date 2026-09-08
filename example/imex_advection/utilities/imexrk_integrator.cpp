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

#include <memory>
#include <string>
#include <vector>

#include <parthenon/parthenon.hpp>

#include "../utilities/imexrk_integrator.hpp"

namespace scalar_imex {

using namespace parthenon::package::prelude;

IMEXRKIntegrator::IMEXRKIntegrator(const std::string &name) {
  if (name == "SSP2-(2,2,2)") {
    const Real gam = 1.0 - 1.0 / sqrt(2.0);
    nstages = 2;
    a_ = {{gam, 0.0}, {1.0 - 2.0 * gam, gam}};
    b_ = {0.5, 0.5};

    at_ = {{0, 0}, {1, 0}};
    bt_ = {0.5, 0.5};
  } else if (name == "SSP3-(3,3,2)") {
    const Real gam = 1.0 - 1.0 / sqrt(2.0);
    nstages = 3;
    a_ = {{gam, 0, 0}, {1 - 2 * gam, gam, 0}, {0.5 - gam, 0, gam}};
    b_ = {1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0};

    at_ = {{0, 0, 0}, {1.0, 0, 0}, {0.25, 0.25, 0}};
    bt_ = b_;
  } else {
    PARTHENON_FAIL("Unknown IMEX-RK integrator.");
  }
  for (int stage = 1; stage <= nstages; ++stage)
    stage_names_.push_back("stage" + std::to_string(stage));
}

} // namespace scalar_imex
