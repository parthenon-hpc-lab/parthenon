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

  auto desc = parthenon::MakePackDescriptor<poisson_package::res_err>(md);
  auto pack = desc.GetPack(md);

  auto &cellbounds = pmb->cellbounds;
  auto ib = cellbounds.GetBoundsI(IndexDomain::entire, TopologicalElement::NN);
  auto jb = cellbounds.GetBoundsJ(IndexDomain::entire, TopologicalElement::NN);
  auto kb = cellbounds.GetBoundsK(IndexDomain::entire, TopologicalElement::NN);
  pmb->par_for(
      "Poisson::ProblemGenerator", 0, pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = pack.GetCoordinates(b);
        pack(b, TopologicalElement::NN, poisson_package::res_err(), k, j, i) =
            coords.X<1, TopologicalElement::NN>(i);
      });
}

Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;
  auto pkg = poisson_package::Initialize(pin.get());
  packages.Add(pkg);

  return packages;
}

} // namespace poisson_example
