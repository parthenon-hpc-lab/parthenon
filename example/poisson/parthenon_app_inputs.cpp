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
#include "poisson_package.hpp"
#include "utils/error_checking.hpp"

using namespace parthenon::package::prelude;
using namespace parthenon;

// *************************************************//
// redefine some weakly linked parthenon functions *//
// *************************************************//

namespace poisson_example {

void ProblemGenerator(Mesh *pm, ParameterInput *pin, MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();

  Real x0 = pin->GetOrAddReal("poisson", "x0", 0.0);
  Real y0 = pin->GetOrAddReal("poisson", "y0", 0.0);
  Real z0 = pin->GetOrAddReal("poisson", "z0", 0.0);
  Real radius = pin->GetOrAddReal("poisson", "radius", 0.1);

  auto cellbounds = pmb->cellbounds;
  IndexRange ib = cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = cellbounds.GetBoundsK(IndexDomain::entire);

  PackIndexMap imap;
  const std::vector<std::string> vars({"density", "potential"});
  const auto &q_bpack = md->PackVariables(vars, imap);
  const int irho = imap["density"].first;
  const int iphi = imap["potential"].first;

  pmb->par_for(
      "Poisson::ProblemGenerator", 0, q_bpack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = q_bpack.GetCoords(b);
        auto &q = q_bpack(b);
        Real dist2 = std::pow(coords.x1v(i) - x0, 2) + std::pow(coords.x2v(j) - y0, 2) +
                     std::pow(coords.x3v(k) - z0, 2);
        if (dist2 < radius * radius) {
          q(irho, k, j, i) = 1.0 / (4.0 / 3.0 * M_PI * std::pow(radius, 3));
        } else {
          q(irho, k, j, i) = 0.0;
        }
        q(iphi, k, j, i) = 0.0;
      });
}

Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;
  auto pkg = poisson_package::Initialize(pin.get());
  packages.Add(pkg);

  return packages;
}

} // namespace poisson_example
