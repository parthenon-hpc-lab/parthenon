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
  Real interior_D = pin->GetOrAddReal("poisson", "interior_D", 1.0);
  Real exterior_D = pin->GetOrAddReal("poisson", "exterior_D", 1.0);

  auto desc =
      parthenon::MakePackDescriptor<poisson_package::rhs, poisson_package::u,
                                    poisson_package::D, poisson_package::exact>(md);
  auto pack = desc.GetPack(md);

  constexpr auto te = poisson_package::te;
  using TE = parthenon::TopologicalElement;
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
        Real x3 = coords.X<2, te>(k);
        Real x1f = coords.X<1, TE::F1>(i);
        Real x2f = coords.X<2, TE::F2>(j);
        Real x3f = coords.X<2, TE::F3>(k);
        Real dx1 = coords.Dxc<1>(k, j, i);
        Real dx2 = coords.Dxc<2>(k, j, i);
        Real dx3 = coords.Dxc<3>(k, j, i);
        Real rad = (x1 - x0) * (x1 - x0);
        if (ndim > 1) rad += (x2 - y0) * (x2 - y0);
        if (ndim > 2) rad += (x3 - z0) * (x3 - z0);
        rad = std::sqrt(rad);
        Real val = 0.0;
        if (rad < radius0) {
          val = 1.0;
        }

        pack(b, te, poisson_package::rhs(), k, j, i) = val;
        pack(b, te, poisson_package::u(), k, j, i) = 0.0;

        // This may be used as the exact solution u to A.u = rhs, by replacing the
        // above rhs with A.exact
        pack(b, te, poisson_package::exact(), k, j, i) = -exp(-10.0 * rad * rad);

        auto inside_region = [ndim](Real x, Real y, Real z) {
          bool inside1 = (x < -0.25) && (x > -0.75);
          if (ndim > 1) inside1 = inside1 && (y < 0.5) && (y > -0.5);
          if (ndim > 2) inside1 = inside1 && (z < 0.25) && (z > -0.25);

          bool inside2 = (x < 0.5) && (x > -0.75);
          if (ndim > 1) inside2 = inside2 && (y < -0.25) && (y > -0.75);
          if (ndim > 2) inside2 = inside2 && (z < 0.25) && (z > -0.25);

          return inside1 || inside2;
        };
        pack(b, TE::F1, poisson_package::D(), k, j, i) =
            inside_region(x1f, x2, x3) ? interior_D : exterior_D;
        pack(b, TE::F2, poisson_package::D(), k, j, i) =
            inside_region(x1, x2f, x3) ? interior_D : exterior_D;
        pack(b, TE::F3, poisson_package::D(), k, j, i) =
            inside_region(x1, x2, x3f) ? interior_D : exterior_D;
      });
}

Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;
  auto pkg = poisson_package::Initialize(pin.get());
  packages.Add(pkg);

  return packages;
}

} // namespace poisson_example
