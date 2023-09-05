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
  const int ndim = md->GetMeshPointer()->ndim;

  Real x0 = pin->GetOrAddReal("poisson", "x0", 0.0);
  Real y0 = pin->GetOrAddReal("poisson", "y0", 0.0);
  Real z0 = pin->GetOrAddReal("poisson", "z0", 0.0);
  Real radius0 = pin->GetOrAddReal("poisson", "radius", 0.1);

  auto desc = parthenon::MakePackDescriptor<poisson_package::rhs, poisson_package::res_err>(md);
  auto pack = desc.GetPack(md);

  constexpr auto te = poisson_package::te;

  auto &cellbounds = pmb->cellbounds;
  auto ib = cellbounds.GetBoundsI(IndexDomain::entire, te);
  auto jb = cellbounds.GetBoundsJ(IndexDomain::entire, te);
  auto kb = cellbounds.GetBoundsK(IndexDomain::entire, te);
  pmb->par_for(
      "Poisson::ProblemGenerator", 0, pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = pack.GetCoordinates(b);
        Real x1 = coords.X<1, te>(i);
        Real x2 = coords.X<2, te>(j);
        Real x3 = coords.X<2, te>(j);
        Real dx1 = coords.Dxc<1>(k, j, i);
        Real dx2 = coords.Dxc<2>(k, j, i);
        Real dx3 = coords.Dxc<3>(k, j, i);
        Real rad = (x1 - x0) * (x1 - x0);
        if (ndim > 1) rad += (x2 - y0) * (x2 - y0);
        if (ndim > 2) rad += (x3 - z0) * (x3 - z0);
        rad = std::sqrt(rad);
        Real val = 0.0;
        if (rad < radius0) {
          val = 1.0 / (4.0 / 3.0 * M_PI * std::pow(rad, 3));
        }
        // val = 1.0 * exp(-rad * 10.0 * rad * 10.0);
        //val = std::sin(2.0 * M_PI * x1);
        //if (ndim > 1) val *= std::sin(2.0 * M_PI * x2);
        //if (ndim > 2) val *= std::sin(2.0 * M_PI * x3);
        val = 2.0 * (1.0 - 6.0 * x1 * x1) * x2 * x2 * (1.0 - x2 * x2) + 2.0 * (1.0
        - 6.0 * x2 * x2) * x1 * x1 * (1.0 - x1 * x1);
        pack(b, te, poisson_package::rhs(), k, j, i) = val;
        pack(b, te, poisson_package::res_err(), k, j, i) = x1;// + x2;
      });
}

Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;
  auto pkg = poisson_package::Initialize(pin.get());
  packages.Add(pkg);

  return packages;
}

} // namespace poisson_example
