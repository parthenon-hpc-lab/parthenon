//========================================================================================
// (C) (or copyright) 2020-2024. Triad National Security, LLC. All rights
// reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001
// for Los Alamos National Laboratory (LANL), which is operated by Triad
// National Security, LLC for the U.S. Department of Energy/National Nuclear
// Security Administration. All rights in the program are reserved by Triad
// National Security, LLC, and the U.S. Department of Energy/National Nuclear
// Security Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
// in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do
// so.
//========================================================================================

#ifndef SRC_STOKES_HPP_
#define SRC_STOKES_HPP_

#include <memory>
#include <vector>

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <utils/indexer.hpp>

#include "../utilities/index_permutation.hpp"
#include "../utilities/scratch_pack.hpp"

namespace scalar_imex {
using namespace parthenon::driver::prelude;

template <class pack_desc_t>
TaskStatus WeightedSumDataElement(parthenon::CellLevel cl, parthenon::TopologicalElement te, pack_desc_t pd,
                                  MeshData<Real> *in1, MeshData<Real> *in2, Real w1, Real w2, MeshData<Real> *out) {
  auto pack1 = pd.GetPack(in1);
  auto pack2 = pd.GetPack(in2);
  auto pack_out = pd.GetPack(out);

  IndexRange ib = in1->GetBoundsI(cl, IndexDomain::entire, te);
  IndexRange jb = in1->GetBoundsJ(cl, IndexDomain::entire, te);
  IndexRange kb = in1->GetBoundsK(cl, IndexDomain::entire, te);

  constexpr int scratch_size = 0;
  constexpr int scratch_level = 1;
  parthenon::par_for_outer(
      PARTHENON_AUTO_LABEL, scratch_size, scratch_level, 0, pack1.GetNBlocks() - 1, kb.s, kb.e,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k) {
        parthenon::Indexer2D idxer({jb.s, jb.e}, {ib.s, ib.e});
        PARTHENON_REQUIRE(pack1.GetLowerBound(b) == pack2.GetLowerBound(b), "Packs are different size.");
        PARTHENON_REQUIRE(pack1.GetLowerBound(b) == pack_out.GetLowerBound(b), "Packs are different size.");
        PARTHENON_REQUIRE(pack1.GetUpperBound(b) == pack2.GetUpperBound(b), "Packs are different size.");
        PARTHENON_REQUIRE(pack1.GetUpperBound(b) == pack_out.GetUpperBound(b), "Packs are different size.");
        for (int l = pack1.GetLowerBound(b); l <= pack1.GetUpperBound(b); ++l) {
          const auto [j, i] = idxer(0);
          Real *out = &pack_out(b, te, l, k, j, i);
          Real const *const one = &pack1(b, te, l, k, j, i);
          Real const *const two = &pack2(b, te, l, k, j, i);
          parthenon::par_for_inner(member, 0, idxer.size() - 1,
                                   [&](const int idx) { out[idx] = w1 * one[idx] + w2 * two[idx]; });
        }
      });
  return TaskStatus::complete;
}

template <class pack_desc_t>
TaskStatus WeightedSumData(parthenon::CellLevel cl, parthenon::TopologicalType tt, pack_desc_t pd, MeshData<Real> *in1,
                           MeshData<Real> *in2, Real w1, Real w2, MeshData<Real> *out) {
  for (auto te : parthenon::GetTopologicalElements(tt))
    WeightedSumDataElement(cl, te, pd, in1, in2, w1, w2, out);
  return TaskStatus::complete;
}

TaskStatus WeightedSumDataAll(MeshData<Real> *in1, MeshData<Real> *in2, Real w1, Real w2, MeshData<Real> *out) {
  auto pmesh = in1->GetMeshPointer();
  constexpr auto cl = parthenon::CellLevel::same;
  {
    static auto desc = parthenon::MakePackDescriptor<parthenon::variable_names::any>(
        pmesh->resolved_packages.get(), {parthenon::Metadata::Independent, parthenon::Metadata::Cell}, {});
    if (desc.nvar_tot > 0) WeightedSumData(cl, parthenon::TopologicalType::Cell, desc, in1, in2, w1, w2, out);
  }
  {
    static auto desc = parthenon::MakePackDescriptor<parthenon::variable_names::any>(
        pmesh->resolved_packages.get(), {parthenon::Metadata::Independent, parthenon::Metadata::Face}, {});
    if (desc.nvar_tot > 0) WeightedSumData(cl, parthenon::TopologicalType::Face, desc, in1, in2, w1, w2, out);
  }
  {
    static auto desc = parthenon::MakePackDescriptor<parthenon::variable_names::any>(
        pmesh->resolved_packages.get(), {parthenon::Metadata::Independent, parthenon::Metadata::Edge}, {});
    if (desc.nvar_tot > 0) WeightedSumData(cl, parthenon::TopologicalType::Edge, desc, in1, in2, w1, w2, out);
  }
  return TaskStatus::complete;
}

template <class pack_desc_t>
void StokesZero(parthenon::CellLevel cl, parthenon::TopologicalElement TeVar, pack_desc_t &pd, MeshData<Real> *out) {
  auto pack_out = pd.GetPack(out);

  IndexRange ib = out->GetBoundsI(cl, IndexDomain::interior, TeVar);
  IndexRange jb = out->GetBoundsJ(cl, IndexDomain::interior, TeVar);
  IndexRange kb = out->GetBoundsK(cl, IndexDomain::interior, TeVar);

  constexpr int scratch_size = 0;
  constexpr int scratch_level = 1;
  parthenon::par_for_outer(
      PARTHENON_AUTO_LABEL, scratch_size, scratch_level, 0, pack_out.GetNBlocks() - 1, kb.s, kb.e,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k) {
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
void StokesComponent(Real fac, parthenon::CellLevel cl, parthenon::TopologicalElement TeVar,
                     parthenon::TopologicalElement TeFlux, pack_desc_t &pd, int ndim, MeshData<Real> *in,
                     MeshData<Real> *out) {
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

  constexpr size_t scratch_level = 1;
  using scratch_pack_t = parthenon::utils::ScratchPack<decltype(pack_in)>;
  const parthenon::utils::IndexingData cellbounds(in);
  using TE = parthenon::TopologicalElement;
  // Choose the correct memory size for each topological element
  const int njtot = cellbounds.ncellsj(IndexDomain::entire, TeVar == TE::CC ? TE::CC : TE::NN);
  const int nitot = cellbounds.ncellsi(IndexDomain::entire, TeVar == TE::CC ? TE::CC : TE::NN);
  std::size_t scratch_size_in_bytes = parthenon::ScratchPad2D<Real>::shmem_size(njtot, nitot) * (2 + koff);

  parthenon::par_for_outer(
      PARTHENON_AUTO_LABEL, scratch_size_in_bytes, scratch_level, 0, pack_out.GetNBlocks() - 1, kb.s, kb.e,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k) {
        auto &coords = pack_in.GetCoordinates(b);
        parthenon::Indexer2D idxer_entire({0, njtot - 1}, {0, nitot - 1});
        parthenon::ScratchPad2D<Real> Vflux(member.team_scratch(scratch_level), njtot, nitot);
        parthenon::ScratchPad2D<Real> Vvar(member.team_scratch(scratch_level), njtot, nitot);
        parthenon::par_for_inner(member, 0, idxer_entire.size() - 1, [&](const int idx) {
          const auto [j, i] = idxer_entire(idx);
          Vflux(j, i) = coords.Volume(cl, TeFlux, k, j, i);
          Vvar(j, i) = coords.Volume(cl, TeVar, k, j, i);
        });

        parthenon::ScratchPad2D<Real> Vfluxu(member.team_scratch(scratch_level), koff ? njtot : 0, koff ? nitot : 0);
        if (koff) {
          parthenon::par_for_inner(member, 0, idxer_entire.size() - 1, [&](const int idx) {
            const auto [j, i] = idxer_entire(idx);
            Vfluxu(j, i) = coords.Volume(cl, TeFlux, k + koff, j, i);
          });
        }
        member.team_barrier();

        PARTHENON_REQUIRE(pack_in.GetLowerBound(b) == pack_out.GetLowerBound(b), "Packs are different size.");
        PARTHENON_REQUIRE(pack_in.GetUpperBound(b) == pack_out.GetUpperBound(b), "Packs are different size.");

        const int npoints = idxer_entire.GetFlatIdx(jb.e, ib.e) - idxer_entire.GetFlatIdx(jb.s, ib.s) + 1;
        for (int l = pack_out.GetLowerBound(b); l <= pack_out.GetUpperBound(b); ++l) {
          Real const *const flxl = &pack_in.flux(b, TeFlux, l, k, jb.s, ib.s);
          Real const *const flxu = &pack_in.flux(b, TeFlux, l, k + koff, jb.s + joff, ib.s + ioff);
          Real const *const vfl = &Vflux(jb.s, ib.s);
          Real const *const vfu = koff ? &Vfluxu(jb.s + joff, ib.s + ioff) : &Vflux(jb.s + joff, ib.s + ioff);
          Real const *const vv = &Vvar(jb.s, ib.s);
          Real *out = &pack_out(b, TeVar, l, k, jb.s, ib.s);

          parthenon::par_for_inner(member, 0, npoints - 1, [&](const int idx) {
            out[idx] += fac * (flxl[idx] * vfl[idx] - flxu[idx] * vfu[idx]) / vv[idx];
          });
        }
      });
}

template <class pack_desc_t>
TaskStatus Stokes(parthenon::CellLevel cl, parthenon::TopologicalType TtVar, pack_desc_t &pd, int ndim,
                  MeshData<Real> *in, MeshData<Real> *out) {
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
      PARTHENON_FAIL("Stokes does not work for node variables, as they are "
                     "zero dimensional.");
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
          // TODO(LFR): This is untested, need to test in parthenon-mhd
          // downstream or add a test involving curls Flip the sign if the
          // variable is an X1 face and the edge is an X3 edge, or an X2 face
          // ... X1 edge, or an X3 face ... X2 edge
          const int indicator = ((static_cast<int>(fte) % 3) - (static_cast<int>(vte) % 3) + 3) % 3;
          fac = (indicator == 2) ? -1.0 : 1.0;
        }
        StokesComponent(fac, cl, vte, fte, pd, ndim, in, out);
      }
    }
  }
  return TaskStatus::complete;
}

TaskStatus StokesAll(MeshData<Real> *in, MeshData<Real> *out) {
  auto pmesh = in->GetMeshPointer();
  const int ndim = pmesh->ndim;
  constexpr auto cl = parthenon::CellLevel::same;
  {
    static auto desc = parthenon::MakePackDescriptor<parthenon::variable_names::any>(
        pmesh->resolved_packages.get(), {parthenon::Metadata::WithFluxes, parthenon::Metadata::Cell},
        {parthenon::PDOpt::WithFluxes});
    if (desc.nvar_tot > 0) Stokes(cl, parthenon::TopologicalType::Cell, desc, ndim, in, out);
  }
  {
    static auto desc = parthenon::MakePackDescriptor<parthenon::variable_names::any>(
        pmesh->resolved_packages.get(), {parthenon::Metadata::WithFluxes, parthenon::Metadata::Face},
        {parthenon::PDOpt::WithFluxes});
    if (desc.nvar_tot > 0) Stokes(cl, parthenon::TopologicalType::Face, desc, ndim, in, out);
  }
  {
    static auto desc = parthenon::MakePackDescriptor<parthenon::variable_names::any>(
        pmesh->resolved_packages.get(), {parthenon::Metadata::WithFluxes, parthenon::Metadata::Edge},
        {parthenon::PDOpt::WithFluxes});
    if (desc.nvar_tot > 0) Stokes(cl, parthenon::TopologicalType::Edge, desc, ndim, in, out);
  }
  return TaskStatus::complete;
}

} // namespace scalar_imex

#endif // SRC_STOKES_HPP_
