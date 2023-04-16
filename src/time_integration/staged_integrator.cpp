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

void StagedIntegrator::MakePeriodicNames_(std::vector<std::string> &names, int n) {
  names.resize(n + 1);
  names[0] = "base";
  for (int i = 1; i < n; i++) {
    names[i] = std::to_string(i);
  }
  names[n] = names[0];
}

} // namespace parthenon
