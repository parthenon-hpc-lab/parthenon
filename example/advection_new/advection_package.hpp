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
  VARIABLE(advection, scalar_fine);
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

  static auto desc = parthenon::MakePackDescriptor<Conserved::scalar>(md, {Metadata::WithFluxes}, {parthenon::PDOpt::WithFluxes}); 
  auto pack = desc.GetPack(md);
  
  IndexRange ib = md->GetBoundsI(parthenon::CellLevel::same, IndexDomain::interior, FACE);
  IndexRange jb = md->GetBoundsJ(parthenon::CellLevel::same, IndexDomain::interior, FACE);
  IndexRange kb = md->GetBoundsK(parthenon::CellLevel::same, IndexDomain::interior, FACE);
  parthenon::par_for(
      PARTHENON_AUTO_LABEL, 0, pack.GetNBlocks() - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        // Calculate the flux using upwind donor cell reconstruction
        pack.flux(b, FACE, Conserved::scalar(), k, j, i) = v * pack(b, Conserved::scalar(), k + koff, j + joff, i + ioff); 
      }
    );
  return TaskStatus::complete;
}

template <class pack_desc_t>
TaskStatus WeightedSumData(parthenon::CellLevel cl, parthenon::TopologicalElement te, pack_desc_t& pd, 
                           MeshData<Real> *in1, MeshData<Real> *in2,
                           Real w1, Real w2, MeshData<Real> *out) {
  auto pack1 = pd.GetPack(in1);
  auto pack2 = pd.GetPack(in2);
  auto pack_out = pd.GetPack(out);

  IndexRange ib = in1->GetBoundsI(cl, IndexDomain::entire, te);
  IndexRange jb = in1->GetBoundsJ(cl, IndexDomain::entire, te);
  IndexRange kb = in1->GetBoundsK(cl, IndexDomain::entire, te);

  parthenon::par_for(PARTHENON_AUTO_LABEL, 0, pack1.GetNBlocks() - 1, 
      pack1.GetLowerBoundHost(0), pack1.GetUpperBoundHost(0), // This is safe for dense vars
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int l, const int k, const int j, const int i) {
        pack_out(b, l, k, j, i) = w1 * pack1(b, l, k, j, i) + w2 * pack2(b, l, k, j, i);
      }
    );
  return TaskStatus::complete;
}

template < class pack_desc_t>
void StokesZero(parthenon::CellLevel cl, parthenon::TopologicalElement TeVar, pack_desc_t& pd, MeshData<Real> *out) {
  auto pack_out = pd.GetPack(out);

  IndexRange ib = out->GetBoundsI(cl, IndexDomain::interior, TeVar);
  IndexRange jb = out->GetBoundsJ(cl, IndexDomain::interior, TeVar);
  IndexRange kb = out->GetBoundsK(cl, IndexDomain::interior, TeVar);

  parthenon::par_for(PARTHENON_AUTO_LABEL, 0, pack_out.GetNBlocks() - 1, 
      pack_out.GetLowerBoundHost(0), pack_out.GetUpperBoundHost(0), // This is safe for dense vars only
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int l, const int k, const int j, const int i) {
        pack_out(b, TeVar, l, k, j, i) = 0.0; 
      }
    );
}

template <class pack_desc_t>
void StokesComponent(Real fac, parthenon::CellLevel cl, 
                           parthenon::TopologicalElement TeVar, parthenon::TopologicalElement TeFlux, 
                           pack_desc_t& pd, int ndim,
                           MeshData<Real> *in, MeshData<Real> *out) {
  auto pack_in = pd.GetPack(in);
  auto pack_out = pd.GetPack(out);

  IndexRange ib = in->GetBoundsI(cl, IndexDomain::interior, TeVar);
  IndexRange jb = in->GetBoundsJ(cl, IndexDomain::interior, TeVar);
  IndexRange kb = in->GetBoundsK(cl, IndexDomain::interior, TeVar);
  int ioff = TopologicalOffsetI(TeFlux) - TopologicalOffsetI(TeVar);
  int joff = TopologicalOffsetJ(TeFlux) - TopologicalOffsetJ(TeVar);
  int koff = TopologicalOffsetK(TeFlux) - TopologicalOffsetK(TeVar);
  PARTHENON_REQUIRE(ioff == 1 || ioff == 0, "Bad combination of TeVar and TeFlux");
  PARTHENON_REQUIRE(joff == 1 || joff == 0, "Bad combination of TeVar and TeFlux");
  PARTHENON_REQUIRE(koff == 1 || koff == 0, "Bad combination of TeVar and TeFlux");
  PARTHENON_REQUIRE((ioff + joff + koff) == 1, "Bad combination of TeVar and TeFlux");
  koff = ndim > 2 ? koff : 0; 
  joff = ndim > 1 ? joff : 0; 
  parthenon::par_for(PARTHENON_AUTO_LABEL, 0, pack_in.GetNBlocks() - 1, 
      pack_in.GetLowerBoundHost(0), pack_in.GetUpperBoundHost(0), // This is safe for dense vars only
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int l, const int k, const int j, const int i) {
        auto &coords = pack_in.GetCoordinates(b);
        pack_out(b, TeVar, l, k, j, i) += fac * (coords.Volume(TeFlux, k, j, i) * pack_in.flux(b, TeFlux, l, k, j, i) 
                                          - coords.Volume(TeFlux, k + koff, j + joff, i + ioff) 
                                          * pack_in.flux(b, TeFlux, l, k + koff, j + joff, i + ioff)) 
                                          / coords.Volume(TeVar, k, j, i); 
      }
    );
}

template <class pack_desc_t>
TaskStatus Stokes(parthenon::CellLevel cl, parthenon::TopologicalType TtVar, 
                  pack_desc_t& pd, int ndim,
                  MeshData<Real> *in, MeshData<Real> *out) { 
  using TE = parthenon::TopologicalElement;
  using TT = parthenon::TopologicalType; 

  // Get the topological type of the generalized flux associated with the 
  // with variables of topological type TtVar 
  TT TtFlx = [TtVar]{
      if (TtVar == TT::Cell) {
        return TT::Face;
      } else if (TtVar == TT::Face) { 
        return TT::Edge;
      } else if (TtVar == TT::Edge) { 
        return TT::Node;
      } else {
        PARTHENON_FAIL("Stokes does not work for node variables, as they are zero dimensional.");
        return TT::Node;
      }
    }();

  auto VarTes = GetTopologicalElements(TtVar); 
  auto FlxTes = GetTopologicalElements(TtFlx); 
  for (auto vte : VarTes) {  
    StokesZero(cl, vte, pd, out);
    for (auto fte : FlxTes) {
      if (IsSubmanifold(fte, vte)) { 
        Real fac = 1.0;
        if (ndim < 3 && fte == TE::F3) continue;
        if (ndim < 2 && fte == TE::F2) continue;
        if (TtVar == TT::Face) { 
          // TODO(LFR): This is untested, need to test in parthenon-mhd downstream or add a test involving curls
          // Flip the sign if the variable is an X1 face and the edge is an X3 edge, or an X2 face ... X1 edge, or an X3 face ... X2 edge 
          const int indicator = ((static_cast<int>(fte) % 3) - (static_cast<int>(vte) % 3) + 3) % 3;
          fac = (indicator == 2) ? -1.0 : 1.0;
        }
        StokesComponent(fac, cl, vte, fte, pd, ndim, in, out);
      }
    }
  }
  return TaskStatus::complete; 
}

} // namespace advection_package

#endif // EXAMPLE_ADVECTION_ADVECTION_PACKAGE_HPP_
