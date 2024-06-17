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

#ifndef EXAMPLE_FINE_ADVECTION_STOKES_HPP_
#define EXAMPLE_FINE_ADVECTION_STOKES_HPP_

#include <memory>
#include <vector>

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

namespace advection_example {
using namespace parthenon::driver::prelude;

template <class pack_desc_t>
TaskStatus WeightedSumDataElement(parthenon::CellLevel cl,
                                  parthenon::TopologicalElement te, pack_desc_t pd,
                                  MeshData<Real> *in1, MeshData<Real> *in2, Real w1,
                                  Real w2, MeshData<Real> *out) {
  auto pack1 = pd.GetPack(in1);
  auto pack2 = pd.GetPack(in2);
  auto pack_out = pd.GetPack(out);

  IndexRange ib = in1->GetBoundsI(cl, IndexDomain::entire, te);
  IndexRange jb = in1->GetBoundsJ(cl, IndexDomain::entire, te);
  IndexRange kb = in1->GetBoundsK(cl, IndexDomain::entire, te);

  PARTHENON_REQUIRE(pack1.GetLowerBoundHost(0) == pack2.GetLowerBoundHost(0),
                    "Packs are different size.");
  PARTHENON_REQUIRE(pack1.GetLowerBoundHost(0) == pack_out.GetLowerBoundHost(0),
                    "Packs are different size.");
  PARTHENON_REQUIRE(pack1.GetUpperBoundHost(0) == pack2.GetUpperBoundHost(0),
                    "Packs are different size.");
  PARTHENON_REQUIRE(pack1.GetUpperBoundHost(0) == pack_out.GetUpperBoundHost(0),
                    "Packs are different size.");
  parthenon::par_for(
      PARTHENON_AUTO_LABEL, 0, pack1.GetNBlocks() - 1, pack1.GetLowerBoundHost(0),
      pack1.GetUpperBoundHost(0), // This is safe for dense vars
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int l, const int k, const int j, const int i) {
        pack_out(b, te, l, k, j, i) =
            w1 * pack1(b, te, l, k, j, i) + w2 * pack2(b, te, l, k, j, i);
      });
  return TaskStatus::complete;
}

template <class pack_desc_t>
TaskStatus WeightedSumData(parthenon::CellLevel cl, parthenon::TopologicalType tt,
                           pack_desc_t pd, MeshData<Real> *in1, MeshData<Real> *in2,
                           Real w1, Real w2, MeshData<Real> *out) {
  for (auto te : parthenon::GetTopologicalElements(tt))
    WeightedSumDataElement(cl, te, pd, in1, in2, w1, w2, out);
  return TaskStatus::complete;
}

template <class pack_desc_t>
void StokesZero(parthenon::CellLevel cl, parthenon::TopologicalElement TeVar,
                pack_desc_t &pd, MeshData<Real> *out) {
  auto pack_out = pd.GetPack(out);

  IndexRange ib = out->GetBoundsI(cl, IndexDomain::interior, TeVar);
  IndexRange jb = out->GetBoundsJ(cl, IndexDomain::interior, TeVar);
  IndexRange kb = out->GetBoundsK(cl, IndexDomain::interior, TeVar);

  parthenon::par_for(
      PARTHENON_AUTO_LABEL, 0, pack_out.GetNBlocks() - 1, pack_out.GetLowerBoundHost(0),
      pack_out.GetUpperBoundHost(0), // This is safe for dense vars only
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int l, const int k, const int j, const int i) {
        pack_out(b, TeVar, l, k, j, i) = 0.0;
      });
}

template <class pack_desc_t>
void StokesComponent(Real fac, parthenon::CellLevel cl,
                     parthenon::TopologicalElement TeVar,
                     parthenon::TopologicalElement TeFlux, pack_desc_t &pd, int ndim,
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

  PARTHENON_REQUIRE(pack_in.GetLowerBoundHost(0) == pack_out.GetLowerBoundHost(0),
                    "Packs are different size.");
  PARTHENON_REQUIRE(pack_in.GetUpperBoundHost(0) == pack_out.GetUpperBoundHost(0),
                    "Packs are different size.");
  parthenon::par_for(
      PARTHENON_AUTO_LABEL, 0, pack_in.GetNBlocks() - 1, pack_in.GetLowerBoundHost(0),
      pack_in.GetUpperBoundHost(0), // This is safe for dense vars only
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int l, const int k, const int j, const int i) {
        auto &coords = pack_in.GetCoordinates(b);
        pack_out(b, TeVar, l, k, j, i) +=
            fac *
            (coords.Volume(cl, TeFlux, k, j, i) * pack_in.flux(b, TeFlux, l, k, j, i) -
             coords.Volume(cl, TeFlux, k + koff, j + joff, i + ioff) *
                 pack_in.flux(b, TeFlux, l, k + koff, j + joff, i + ioff)) /
            coords.Volume(cl, TeVar, k, j, i);
      });
}

template <class pack_desc_t>
TaskStatus Stokes(parthenon::CellLevel cl, parthenon::TopologicalType TtVar,
                  pack_desc_t &pd, int ndim, MeshData<Real> *in, MeshData<Real> *out) {
  using TE = parthenon::TopologicalElement;
  using TT = parthenon::TopologicalType;

  // Get the topological type of the generalized flux associated with the
  // with variables of topological type TtVar
  TT TtFlx = [TtVar] {
    if (TtVar == TT::Cell) {
      return TT::Face;
    } else if (TtVar == TT::Face) {
      return TT::Edge;
    } else if (TtVar == TT::Edge) {
      return TT::Node;
    } else {
      PARTHENON_FAIL(
          "Stokes does not work for node variables, as they are zero dimensional.");
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
          // TODO(LFR): This is untested, need to test in parthenon-mhd downstream or add
          // a test involving curls Flip the sign if the variable is an X1 face and the
          // edge is an X3 edge, or an X2 face ... X1 edge, or an X3 face ... X2 edge
          const int indicator =
              ((static_cast<int>(fte) % 3) - (static_cast<int>(vte) % 3) + 3) % 3;
          fac = (indicator == 2) ? -1.0 : 1.0;
        }
        StokesComponent(fac, cl, vte, fte, pd, ndim, in, out);
      }
    }
  }
  return TaskStatus::complete;
}

} // namespace advection_example

#endif // EXAMPLE_FINE_ADVECTION_STOKES_HPP_
