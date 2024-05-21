//========================================================================================
// (C) (or copyright) 2020-2024. Triad National Security, LLC. All rights reserved.
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
#ifndef EXAMPLE_ADVECTION_NEW_ADVECTION_PACKAGE_HPP_
#define EXAMPLE_ADVECTION_NEW_ADVECTION_PACKAGE_HPP_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <parthenon/package.hpp>
#include <utils/robust.hpp>

#define VARIABLE(ns, varname)                                                            \
  struct varname : public parthenon::variable_names::base_t<false> {                     \
    template <class... Ts>                                                               \
    KOKKOS_INLINE_FUNCTION varname(Ts &&...args)                                         \
        : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}         \
    static std::string name() { return #ns "." #varname; }                               \
  }

namespace advection_package {
using namespace parthenon::package::prelude;

namespace Conserved {
VARIABLE(advection, scalar);
VARIABLE(advection, scalar_fine);
VARIABLE(advection, scalar_fine_restricted);
} // namespace Conserved
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
AmrTag CheckRefinement(MeshBlockData<Real> *rc);
Real EstimateTimestep(MeshData<Real> *md);
TaskStatus RestrictScalarFine(MeshData<Real> *md);

template <class pack_desc_t>
TaskStatus CalculateFluxes(pack_desc_t &desc, parthenon::TopologicalElement FACE,
                           parthenon::CellLevel cl, MeshData<Real> *md) {
  using TE = parthenon::TopologicalElement;

  std::shared_ptr<StateDescriptor> pkg =
      md->GetMeshPointer()->packages.Get("advection_package");

  // Pull out velocity and piecewise constant reconstruction offsets
  // for the given direction
  Real v;
  int ioff{0}, joff{0}, koff{0};
  if (FACE == TE::F1) {
    v = pkg->Param<Real>("vx");
    if (v > 0) ioff = -1;
  } else if (FACE == TE::F2) {
    v = pkg->Param<Real>("vy");
    if (v > 0) joff = -1;
  } else if (FACE == TE::F3) {
    v = pkg->Param<Real>("vz");
    if (v > 0) koff = -1;
  }

  auto pack = desc.GetPack(md);

  IndexRange ib = md->GetBoundsI(cl, IndexDomain::interior, FACE);
  IndexRange jb = md->GetBoundsJ(cl, IndexDomain::interior, FACE);
  IndexRange kb = md->GetBoundsK(cl, IndexDomain::interior, FACE);
  parthenon::par_for(
      PARTHENON_AUTO_LABEL, 0, pack.GetNBlocks() - 1, pack.GetLowerBoundHost(0),
      pack.GetUpperBoundHost(0), // Warning: only works for dense variables
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int l, const int k, const int j, const int i) {
        // Calculate the flux using upwind donor cell reconstruction
        pack.flux(b, FACE, l, k, j, i) = v * pack(b, l, k + koff, j + joff, i + ioff);
      });
  return TaskStatus::complete;
}

} // namespace advection_package

#endif // EXAMPLE_ADVECTION_NEW_ADVECTION_PACKAGE_HPP_
