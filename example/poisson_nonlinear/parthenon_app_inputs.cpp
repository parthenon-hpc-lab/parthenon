//========================================================================================
// (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
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

#include <sstream>
#include <string>

#include <parthenon/package.hpp>

#include "config.hpp"
#include "defs.hpp"
#include "poisson_nonlinear_package.hpp"
#include "utils/error_checking.hpp"

using namespace parthenon::package::prelude;
using namespace parthenon;

// *************************************************//
// redefine some weakly linked parthenon functions *//
// *************************************************//

namespace poisson_example {

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  auto &data = pmb->meshblock_data.Get();

  auto cellbounds = pmb->cellbounds;
  IndexRange ib = cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = cellbounds.GetBoundsK(IndexDomain::entire);

  auto coords = pmb->coords;
  PackIndexMap imap;
  const std::vector<std::string> vars({"potential"});
  const auto &q = data->PackVariables(vars, imap);
  const int iphi = imap["potential"].first;

  pmb->par_for(
      "Poisson::ProblemGenerator", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) { q(iphi, k, j, i) = 0.0; });
}

Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;
  auto pkg = poisson_package::Initialize(pin.get());
  packages.Add(pkg);

  return packages;
}

} // namespace poisson_example
