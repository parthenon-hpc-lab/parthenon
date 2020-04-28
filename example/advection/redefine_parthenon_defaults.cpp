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

#include <parthenon/package.hpp>

#include "advection_package.hpp"

using namespace parthenon::package::prelude;

// *************************************************//
// redefine some weakly linked parthenon functions *//
// *************************************************//

namespace parthenon {

Packages_t ParthenonManager::ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;
  auto pkg = advection_package::Initialize(pin.get());
  packages[pkg->label()] = pkg;
  return packages;
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Container<Real> &rc = real_containers.Get();
  CellVariable<Real> &q = rc.Get("advected");

  for (int k = 0; k < ncells3; k++) {
    for (int j = 0; j < ncells2; j++) {
      for (int i = 0; i < ncells1; i++) {
        Real rsq = std::pow(pcoord->x1v(i), 2) + std::pow(pcoord->x2v(j), 2);
        q(k, j, i) = (rsq < 0.15 * 0.15 ? 1.0 : 0.0);
      }
    }
  }
}

void ParthenonManager::SetFillDerivedFunctions() {
  FillDerivedVariables::SetFillDerivedFunctions(advection_package::PreFill,
                                                advection_package::PostFill);
}

} // namespace parthenon
