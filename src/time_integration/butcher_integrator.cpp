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

#include <string>
#include <vector>

#include "basic_types.hpp"
#include "parameter_input.hpp"
#include "staged_integrator.hpp"

namespace parthenon {

// classic Butcher Tableau integrators
// TODO(JMM): Should the names for butcher-tableau-based be the same?
ButcherIntegrator::ButcherIntegrator(ParameterInput *pin)
  : StagedIntegrator(pin->GetOrAddString("parthenon/time", "integrator", "rk2")) {
  if (name_ == "rk1") {
    nstages = nbuffers = 1;
    Resize_(nstages);

    a[0][0] = 0;
    b[0] = 1;
    c[0] = 0;
  } else if (name_ == "rk2") {
    // Heun's method. Should match minimal storage solution
    nstages = nbuffers = 2;
    Resize_(nstages);

    a[0] = {0, 0};
    a[1] = {1, 0};
    b = {0, 1};
    c = {0, 1./3., 2./3.};
  } else if (name_ == "rk4") {
    // Classic RK4 because why not
    nstages = nbuffers = 4;
    Resize_(nstages);

    /* clang-format off */
    a[0] = {0,   0,   0, 0};
    a[1] = {0.5, 0,   0, 0};
    a[2] = {0,   0.5, 0, 0};
    a[3] = {0,   0,   1, 0};
    /* clang-format on */
    b = {1./6., 1./3., 1./3., 1./6.};
    c = {0, 0.5, 0.5, 1};
  }
}

void ButcherIntegrator::Resize_(int nstages) {
  a.resize(nstages);
  for (int i = 0; i < a.size(); ++i) {
    a[i].resize(nstages);
  }
  b.resize(nstages);
  c.resize(nstages);
}

} // namespace parthenon
