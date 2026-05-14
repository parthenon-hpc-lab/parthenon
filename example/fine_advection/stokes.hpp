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
#include <utils/indexer.hpp>

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

  constexpr int scratch_size = 0;
  constexpr int scratch_level = 1;
  parthenon::par_for_outer(
      PARTHENON_AUTO_LABEL, scratch_size, scratch_level, 0, pack1.GetNBlocks() - 1, kb.s,
      kb.e, KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k) {
        parthenon::Indexer2D idxer({jb.s, jb.e}, {ib.s, ib.e});
        PARTHENON_REQUIRE(pack1.GetLowerBound(b) == pack2.GetLowerBound(b),
                          "Packs are different size.");
        PARTHENON_REQUIRE(pack1.GetLowerBound(b) == pack_out.GetLowerBound(b),
                          "Packs are different size.");
        PARTHENON_REQUIRE(pack1.GetUpperBound(b) == pack2.GetUpperBound(b),
                          "Packs are different size.");
        PARTHENON_REQUIRE(pack1.GetUpperBound(b) == pack_out.GetUpperBound(b),
                          "Packs are different size.");
        for (int l = pack1.GetLowerBound(b); l <= pack1.GetUpperBound(b); ++l) {
          parthenon::par_for_inner(member, 0, idxer.size() - 1, [&](const int idx) {
            const auto [j, i] = idxer(idx);
            pack_out(b, te, l, k, j, i) =
                w1 * pack1(b, te, l, k, j, i) + w2 * pack2(b, te, l, k, j, i);
          });
        }
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

  constexpr int scratch_size = 0;
  constexpr int scratch_level = 1;
  parthenon::par_for_outer(
      PARTHENON_AUTO_LABEL, scratch_size, scratch_level, 0, pack_out.GetNBlocks() - 1,
      kb.s, kb.e, KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k) {
        parthenon::Indexer2D idxer({jb.s, jb.e}, {ib.s, ib.e});
        for (int l = pack_out.GetLowerBound(b); l <= pack_out.GetUpperBound(b); ++l) {
          parthenon::par_for_inner(member, 0, idxer.size() - 1, [&](const int idx) {
            const auto [j, i] = idxer(idx);
            pack_out(b, TeVar, l, k, j, i) = 0.0;
          });
        }
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

  constexpr int scratch_size = 0;
  constexpr int scratch_level = 1;
  parthenon::par_for_outer(
      PARTHENON_AUTO_LABEL, scratch_size, scratch_level, 0, pack_out.GetNBlocks() - 1,
      kb.s, kb.e, KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k) {
        auto &coords = pack_in.GetCoordinates(b);
        parthenon::Indexer2D idxer({jb.s, jb.e}, {ib.s, ib.e});
        PARTHENON_REQUIRE(pack_in.GetLowerBound(b) == pack_out.GetLowerBound(b),
                          "Packs are different size.");
        PARTHENON_REQUIRE(pack_in.GetUpperBound(b) == pack_out.GetUpperBound(b),
                          "Packs are different size.");
        for (int l = pack_out.GetLowerBound(b); l <= pack_out.GetUpperBound(b); ++l) {
          parthenon::par_for_inner(member, 0, idxer.size() - 1, [&](const int idx) {
            const auto [j, i] = idxer(idx);
            pack_out(b, TeVar, l, k, j, i) +=
                fac *
                (coords.Volume(cl, TeFlux, k, j, i) *
                     pack_in.flux(b, TeFlux, l, k, j, i) -
                 coords.Volume(cl, TeFlux, k + koff, j + joff, i + ioff) *
                     pack_in.flux(b, TeFlux, l, k + koff, j + joff, i + ioff)) /
                coords.Volume(cl, TeVar, k, j, i);
          });
        }
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
