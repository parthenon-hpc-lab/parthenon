//========================================================================================
// (C) (or copyright) 2024. Triad National Security, LLC. All rights reserved.
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
#include <utils/error_checking.hpp>
using namespace parthenon::package::prelude;

#include "indicator/indicator.hpp"

void SetupEllipse(MeshBlock *pmb, ParameterInput *pin) {

  // get meshblock data object
  auto &data = pmb->meshblock_data.Get();

  // pull out ellipse data
  auto pkg = pmb->packages.Get("ellipse");
  const auto major_axis = pkg->Param<Real>("major_axis");
  const auto minor_axis = pkg->Param<Real>("minor_axis");
  const Real a2 = major_axis*major_axis;
  const Real b2 = minor_axis*minor_axis;

  // loop bounds for interior of meshblock
  auto cellbounds = pmb->cellbounds;
  IndexRange ib = cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = cellbounds.GetBoundsK(IndexDomain::interior);

  // coordinates object
  auto coords = pmb->coords;

  // build a type-based variable pack
  static auto desc = parthenon::MakePackDescriptor<Indicator::phi>(data.get());
  auto pack = desc.GetPack(data.get());

  const int ndim = pmb->pmy_mesh->ndim;
  PARTHENON_REQUIRE_THROWS(ndim >= 2, "Calculate area must be at least 2d");

  const int b = 0;
  parthenon::par_for(
      PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        // are we on the ellipse?
        Real x = coords.Xc<X1DIR>(k, j, i);
        Real y = coords.Xc<X2DIR>(k, j, i);
        Real condition = ((x*x)/(a2 + 1e-20) + (y*y)/(b2 + 1e-20)) <= 1;
        // set indicator function appropriately
        pack(b, Indicator::phi(), k, j, i) = condition;
      });
  return;
}
