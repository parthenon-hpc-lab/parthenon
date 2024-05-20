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
#ifndef EXAMPLE_ADVECTION_ADVECTION_PACKAGE_HPP_
#define EXAMPLE_ADVECTION_ADVECTION_PACKAGE_HPP_

#include <memory>
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
}
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
AmrTag CheckRefinement(MeshBlockData<Real> *rc);
Real EstimateTimestep(MeshData<Real> *md);

template <parthenon::TopologicalElement FACE> 
TaskStatus CalculateFluxes(MeshData<Real> *md) { 
  using TE = parthenon::TopologicalElement;

  std::shared_ptr<StateDescriptor> pkg = md->GetMeshPointer()->packages.Get("advection_package"); 
  
  // Pull out velocity and piecewise constant reconstruction offsets
  // for the given direction
  Real v;
  parthenon::CoordinateDirection dir;
  int ioff{0}, joff{0}, koff{0}; 
  if (FACE == TE::F1) {
    dir = X1DIR;
    v = pkg->Param<Real>("vx");
    if (v > 0) ioff = -1;
  } else if (FACE == TE::F2) { 
    dir = X2DIR;
    v = pkg->Param<Real>("vy");
    if (v > 0) joff = -1;
  } else if (FACE == TE::F3) { 
    dir = X3DIR;
    v = pkg->Param<Real>("vz");
    if (v > 0) koff = -1;
  }

  static auto desc = parthenon::MakePackDescriptor<Conserved::scalar>(md, {Metadata::WithFluxes}, {parthenon::PDOpt::WithFluxes}); 
  auto pack = desc.GetPack(md);
  
  IndexRange ib = md->GetBoundsI(IndexDomain::interior, FACE);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior, FACE);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior, FACE);
  parthenon::par_for(
      PARTHENON_AUTO_LABEL, 0, pack.GetNBlocks() - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        // Calculate the flux using upwind donor cell reconstruction
        pack.flux(b, dir, Conserved::scalar(), k, j, i) = v * pack(b, Conserved::scalar(), k + koff, j + joff, i + ioff); 
      }
    );
  return TaskStatus::complete;
}
} // namespace advection_package

#endif // EXAMPLE_ADVECTION_ADVECTION_PACKAGE_HPP_
