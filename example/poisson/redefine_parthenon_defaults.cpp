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

#include "config.hpp"
#include "poisson_package.hpp"

#include <parthenon/package.hpp>

using namespace parthenon::package::prelude;

namespace parthenon {

Packages_t ParthenonManager::ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;
  auto pkg = poisson::Initialize(pin.get());
  packages[pkg->label()] = pkg;
  return packages;
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Container<Real> &rc = real_containers.Get();
  auto &phi = rc.Get("field").data;
  auto &rho = rc.Get("potential").data;

  auto pkg = packages["poisson_package"];
  auto profile = pkg->Param<std::string>("potential");

  auto phi_h = phi.GetHostMirror();
  auto rho_h = rho.GetHostMirror();

  IndexRange ib = cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = cellbounds.GetBoundsK(IndexDomain::entire);

  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e; j++) {
      for (int i = ib.s; i <= ib.e; i++) {
        phi_h(k, j, i) = 0; // initial guess. TODO(JMM) set it to a random number?
        Real rsq = (coords.x1v(i) * coords.x1v(i) + coords.x2v(j) * coords.x2v(j) +
                    coords.x3v(k) * coords.x3v(k));
        rho_h(k, j, i) = (rsq < 0.15 * 0.15 ? 1.0 : 0.0);
      }
    }
  }
  phi.DeepCopy(phi_h);
  rho.DeepCopy(rho_h);
}

} // namespace parthenon
