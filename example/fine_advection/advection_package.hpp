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
#ifndef EXAMPLE_FINE_ADVECTION_ADVECTION_PACKAGE_HPP_
#define EXAMPLE_FINE_ADVECTION_ADVECTION_PACKAGE_HPP_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <parthenon/package.hpp>
#include <utils/indexer.hpp>
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
VARIABLE(advection, phi);
VARIABLE(advection, phi_fine);
VARIABLE(advection, phi_fine_restricted);
VARIABLE(advection, C);
VARIABLE(advection, D);
VARIABLE(advection, recon);
VARIABLE(advection, recon_f);
VARIABLE(advection, C_cc);
VARIABLE(advection, D_cc);
VARIABLE(advection, divC);
VARIABLE(advection, divD);
} // namespace Conserved

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
AmrTag CheckRefinement(MeshBlockData<Real> *rc);
void CheckRefinementMesh(MeshData<Real> *md, parthenon::ParArray1D<AmrTag> &amr_tags);
Real EstimateTimestep(MeshData<Real> *md);
TaskStatus FillDerived(MeshData<Real> *md);

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
  constexpr int unused_scratch_size = 0;
  constexpr int unused_scratch_level = 1;
  parthenon::par_for_outer(
      PARTHENON_AUTO_LABEL, unused_scratch_size, unused_scratch_level, 0,
      pack.GetNBlocks() - 1, kb.s, kb.e,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k) {
        parthenon::Indexer2D idxer({jb.s, jb.e}, {ib.s, ib.e});
        for (int l = pack.GetLowerBound(b); l <= pack.GetUpperBound(b); ++l) {
          parthenon::par_for_inner(member, 0, idxer.size() - 1, [&](const int idx) {
            const auto [j, i] = idxer(idx);
            // Calculate the flux using upwind donor cell reconstruction
            pack.flux(b, FACE, l, k, j, i) = v * pack(b, l, k + koff, j + joff, i + ioff);
          });
        }
      });
  return TaskStatus::complete;
}

template <class var, class flux_var>
TaskStatus CalculateVectorFluxes(parthenon::TopologicalElement edge,
                                 parthenon::CellLevel cl, Real fac, MeshData<Real> *md) {
  using TE = parthenon::TopologicalElement;
  using recon = Conserved::recon;
  using recon_f = Conserved::recon_f;

  int ndim = md->GetParentPointer()->ndim;
  static auto desc =
      parthenon::MakePackDescriptor<var, flux_var, advection_package::Conserved::recon,
                                    advection_package::Conserved::recon_f>(
          md, {}, {parthenon::PDOpt::WithFluxes});
  auto pack = desc.GetPack(md);

  // 1. Reconstruct the component of the flux field pointing in the direction of edge in
  // the quartants of the chosen edge
  TE fe;
  if (edge == TE::E1) fe = TE::F1;
  if (edge == TE::E2) fe = TE::F2;
  if (edge == TE::E3) fe = TE::F3;
  IndexRange ib = md->GetBoundsI(cl, IndexDomain::interior, TE::CC);
  IndexRange jb = md->GetBoundsJ(cl, IndexDomain::interior, TE::CC);
  IndexRange kb = md->GetBoundsK(cl, IndexDomain::interior, TE::CC);
  int koff = edge == TE::E3 ? ndim > 2 : 0;
  int joff = edge == TE::E2 ? ndim > 1 : 0;
  int ioff = edge == TE::E1 ? ndim > 0 : 0;
  constexpr int scratch_size = 0;
  constexpr int scratch_level = 1;
  parthenon::par_for_outer(
      PARTHENON_AUTO_LABEL, scratch_size, scratch_level, 0, pack.GetNBlocks() - 1,
      kb.s - (ndim > 2), kb.e + (ndim > 2),
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k) {
        parthenon::Indexer2D idxer({jb.s - (ndim > 1), jb.e + (ndim > 1)},
                                   {ib.s - (ndim > 0), ib.e + (ndim > 0)});
        if (pack.GetLowerBound(b, flux_var()) <= pack.GetUpperBound(b, flux_var())) {
          parthenon::par_for_inner(member, 0, idxer.size() - 1, [&](const int idx) {
            const auto [j, i] = idxer(idx);
            // Piecewise linear in the orthogonal directions, could do something better
            // here
            pack(b, TE::CC, recon(0), k, j, i) =
                0.5 * (pack(b, fe, flux_var(), k, j, i) +
                       pack(b, fe, flux_var(), k + koff, j + joff, i + ioff));
            pack(b, TE::CC, recon(1), k, j, i) =
                0.5 * (pack(b, fe, flux_var(), k, j, i) +
                       pack(b, fe, flux_var(), k + koff, j + joff, i + ioff));
            pack(b, TE::CC, recon(2), k, j, i) =
                0.5 * (pack(b, fe, flux_var(), k, j, i) +
                       pack(b, fe, flux_var(), k + koff, j + joff, i + ioff));
            pack(b, TE::CC, recon(3), k, j, i) =
                0.5 * (pack(b, fe, flux_var(), k, j, i) +
                       pack(b, fe, flux_var(), k + koff, j + joff, i + ioff));
          });
        }
      });

  // 2. Calculate the quartant averaged flux
  koff = edge != TE::E3 ? ndim > 2 : 0;
  joff = edge != TE::E2 ? ndim > 1 : 0;
  ioff = edge != TE::E1 ? ndim > 0 : 0;
  Real npoints = (koff + 1) * (joff + 1) * (ioff + 1);
  ib = md->GetBoundsI(cl, IndexDomain::interior, edge);
  jb = md->GetBoundsJ(cl, IndexDomain::interior, edge);
  kb = md->GetBoundsK(cl, IndexDomain::interior, edge);
  parthenon::par_for_outer(
      PARTHENON_AUTO_LABEL, scratch_size, scratch_level, 0, pack.GetNBlocks() - 1, kb.s,
      kb.e, KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k) {
        parthenon::Indexer2D idxer({jb.s, jb.e}, {ib.s, ib.e});
        if (pack.GetLowerBound(b, var()) <= pack.GetUpperBound(b, var())) {
          parthenon::par_for_inner(member, 0, idxer.size() - 1, [&](const int idx) {
            const auto [j, i] = idxer(idx);
            pack.flux(b, edge, var(), k, j, i) = 0.0;
            for (int dk = -koff; dk < 1; ++dk)
              for (int dj = -joff; dj < 1; ++dj)
                for (int di = -ioff; di < 1; ++di) {
                  // TODO(LFR): Pick out the correct component of the reconstructed flux,
                  // current version is not an issue for piecewise constant flux though.
                  pack.flux(b, edge, var(), k, j, i) +=
                      pack(b, TE::CC, recon(0), k + dk, j + dj, i + di);
                }
            pack.flux(b, edge, var(), k, j, i) /= npoints;
            pack.flux(b, edge, var(), k, j, i) *= fac;
          });
        }
      });

  // 3. Reconstruct the transverse components of the advected field at the edge
  std::vector<TE> faces{TE::F2, TE::F3};
  if (edge == TE::E2) faces = {TE::F3, TE::F1};
  if (edge == TE::E3) faces = {TE::F1, TE::F2};
  for (auto f : faces) {
    ib = md->GetBoundsI(cl, IndexDomain::interior, f);
    jb = md->GetBoundsJ(cl, IndexDomain::interior, f);
    kb = md->GetBoundsK(cl, IndexDomain::interior, f);
    parthenon::par_for_outer(
        PARTHENON_AUTO_LABEL, scratch_size, scratch_level, 0, pack.GetNBlocks() - 1,
        kb.s - (ndim > 2), kb.e + (ndim > 2),
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k) {
          parthenon::Indexer2D idxer({jb.s - (ndim > 1), jb.e + (ndim > 1)},
                                     {ib.s - (ndim > 0), ib.e + (ndim > 0)});
          if (pack.GetLowerBound(b, var()) <= pack.GetUpperBound(b, var())) {
            parthenon::par_for_inner(member, 0, idxer.size() - 1, [&](const int idx) {
              const auto [j, i] = idxer(idx);
              // Piecewise linear in the orthogonal directions, could do something better
              // here
              pack(b, f, recon_f(0), k, j, i) = pack(b, f, var(), k, j, i);
              pack(b, f, recon_f(1), k, j, i) = pack(b, f, var(), k, j, i);
            });
          }
        });
  }

  // 4. Add the diffusive piece to the numerical flux, which is proportional to the curl
  ib = md->GetBoundsI(cl, IndexDomain::interior, edge);
  jb = md->GetBoundsJ(cl, IndexDomain::interior, edge);
  kb = md->GetBoundsK(cl, IndexDomain::interior, edge);
  TE f1 = faces[0];
  TE f2 = faces[1];
  std::array<int, 3> d1{ndim > 0, ndim > 1, ndim > 2};
  std::array<int, 3> d2 = d1;
  d1[static_cast<int>(edge) % 3] = 0;
  d1[static_cast<int>(f1) % 3] = 0;
  d2[static_cast<int>(edge) % 3] = 0;
  d2[static_cast<int>(f2) % 3] = 0;
  parthenon::par_for_outer(
      PARTHENON_AUTO_LABEL, scratch_size, scratch_level, 0, pack.GetNBlocks() - 1, kb.s,
      kb.e, KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k) {
        parthenon::Indexer2D idxer({jb.s, jb.e}, {ib.s, ib.e});
        if (pack.GetLowerBound(b, var()) <= pack.GetUpperBound(b, var())) {
          parthenon::par_for_inner(member, 0, idxer.size() - 1, [&](const int idx) {
            const auto [j, i] = idxer(idx);
            pack.flux(b, edge, var(), k, j, i) +=
                0.5 * (pack(b, f1, recon_f(0), k, j, i) -
                       pack(b, f1, recon_f(1), k - d1[2], j - d1[1], i - d1[0]));
            pack.flux(b, edge, var(), k, j, i) -=
                0.5 * (pack(b, f2, recon_f(0), k, j, i) -
                       pack(b, f2, recon_f(1), k - d2[2], j - d2[1], i - d2[0]));
          });
        }
      });

  // Add in the diffusive component
  return TaskStatus::complete;
}

} // namespace advection_package

#endif // EXAMPLE_FINE_ADVECTION_ADVECTION_PACKAGE_HPP_
