//========================================================================================
// (C) (or copyright) 2023. Triad National Security, LLC. All rights reserved.
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
#include <math.h>
#include <sstream>
#include <string>

#include <parthenon/package.hpp>

#include "config.hpp"
#include "defs.hpp"
#include "helmholtz_package.hpp"
#include "poisson_cell_package.hpp"
#include "poisson_nodal_package.hpp"
#include "utils/error_checking.hpp"

using namespace parthenon::package::prelude;
using namespace parthenon;

// *************************************************//
// redefine some weakly linked parthenon functions *//
// *************************************************//

namespace linear_solver_example {

Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;
  packages.Add(poisson_cell_package::Initialize(pin.get()));
  packages.Add(poisson_nodal_package::Initialize(pin.get()));
  packages.Add(helmholtz_package::Initialize(pin.get()));
  return packages;
}

} // namespace linear_solver_example
