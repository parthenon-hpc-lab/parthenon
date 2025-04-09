//========================================================================================
// (C) (or copyright) 2023-2025. Triad National Security, LLC. All rights reserved.
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
#include "diffusion_package.hpp"
#include "utils/error_checking.hpp"

using namespace parthenon::package::prelude;
using namespace parthenon;

// *************************************************//
// redefine some weakly linked parthenon functions *//
// *************************************************//

namespace diffusion_example {

void ProblemGenerator(Mesh *pm, ParameterInput *pin, MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  const int ndim = md->GetMeshPointer()->ndim;

  Real x0 = pin->GetOrAddReal("diffusion", "x0", 0.0);
  Real y0 = pin->GetOrAddReal("diffusion", "y0", 0.0);
  Real z0 = pin->GetOrAddReal("diffusion", "z0", 0.0);
  const Real t0 = pin->GetOrAddReal("diffusion", "t0", 0.001);
  const Real dt = pin->GetOrAddReal("diffusion", "dt", 1.0);
  const bool constant_coeff =
      pin->GetOrAddBoolean("diffusion", "constant_coefficient", true);

  auto desc =
      parthenon::MakePackDescriptor<diffusion_package::u, diffusion_package::D>(md);
  auto pack = desc.GetPack(md);

  using TE = parthenon::TopologicalElement;
  auto &cellbounds = pmb->cellbounds;
  auto ib = cellbounds.GetBoundsI(IndexDomain::entire);
  auto jb = cellbounds.GetBoundsJ(IndexDomain::entire);
  auto kb = cellbounds.GetBoundsK(IndexDomain::entire);
  pmb->par_for(
      "Diffusion::ProblemGenerator", 0, pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e,
      ib.s, ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = pack.GetCoordinates(b);
        Real x1 = coords.Xc<1>(i);
        Real x2 = coords.Xc<2>(j);
        Real x3 = coords.Xc<2>(k);
        Real x1f = coords.X<1, TE::F1>(k, j, i);
        Real x2f = coords.X<2, TE::F2>(k, j, i);
        Real x3f = coords.X<2, TE::F3>(k, j, i);
        Real dx1 = coords.Dxc<1>(k, j, i);
        Real dx2 = coords.Dxc<2>(k, j, i);
        Real dx3 = coords.Dxc<3>(k, j, i);
        Real rad = (x1 - x0) * (x1 - x0);
        if (ndim > 1) rad += (x2 - y0) * (x2 - y0);
        if (ndim > 2) rad += (x3 - z0) * (x3 - z0);
        rad = std::sqrt(rad);

        auto profile = [rad, t0](Real x, Real y, Real z) {
          Real D = 1.0; // initial profile uses constant coefficient
          Real exponent = -rad * rad / (4.0 * D * t0);
          return std::exp(exponent);
        };
        const Real val = profile(x1, x2, x3);
        pack(b, diffusion_package::u(), k, j, i) = val;

        if (constant_coeff) {
          pack(b, TE::F1, diffusion_package::D(), k, j, i) = 1.0 * dt;
          pack(b, TE::F2, diffusion_package::D(), k, j, i) = 1.0 * dt;
          pack(b, TE::F3, diffusion_package::D(), k, j, i) = 1.0 * dt;
        } else {
          pack(b, TE::F1, diffusion_package::D(), k, j, i) = profile(x1f, x2, x3) * dt;
          pack(b, TE::F2, diffusion_package::D(), k, j, i) = profile(x1, x2f, x3) * dt;
          pack(b, TE::F3, diffusion_package::D(), k, j, i) = profile(x1, x2, x3f) * dt;
        }
      });
}

Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;
  auto pkg = diffusion_package::Initialize(pin.get());
  packages.Add(pkg);

  return packages;
}

} // namespace diffusion_example
