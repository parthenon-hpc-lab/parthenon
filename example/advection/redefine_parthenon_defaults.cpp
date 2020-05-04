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
  auto q = rc.Get("advected").data;

  // TODO(jcd): can't seem to call par_for here.  why?
  /*par_for(
      "init problem", 0, ncells3 - 1, 0, ncells2 - 1, 0, ncells1 - 1,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x = coords.x1v(i);
        const Real y = coords.x2v(j);
        Real rsq = x * x + y * y;
        q(k, j, i) = (rsq < 0.15 * 0.15 ? 1.0 : 0);
      });*/
  for (int k = 0; k <= ncells3 - 1; k++) {
    for (int j = 0; j <= ncells2 - 1; j++) {
      for (int i = 0; i <= ncells1 - 1; i++) {
        Real x = coords.x1v(i);
        Real y = coords.x2v(j);
        Real rsq = x * x + y * y;
        q(k, j, i) = (rsq < 0.15 * 0.15 ? 1.0 : 0);
      }
    }
  }
}

void ParthenonManager::SetFillDerivedFunctions() {
  FillDerivedVariables::SetFillDerivedFunctions(advection_package::PreFill,
                                                advection_package::PostFill);
}

} // namespace parthenon
