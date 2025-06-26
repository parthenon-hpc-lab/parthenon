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

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <coordinates/coordinates.hpp>
#include <globals.hpp>
#include <parthenon/package.hpp>

#include "advection_package.hpp"
#include "defs.hpp"
#include "kokkos_abstraction.hpp"
#include "reconstruct/dc_inline.hpp"
#include "utils/error_checking.hpp"

using namespace parthenon::package::prelude;

// *************************************************//
// define the "physics" package Advect, which      *//
// includes defining various functions that control*//
// how parthenon functions and any tasks needed to *//
// implement the "physics"                         *//
// *************************************************//

namespace advection_package {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("advection_package");

  Real cfl = pin->GetOrAddReal("Advection", "cfl", 0.45);
  pkg->AddParam<>("cfl", cfl);

  Real vx = pin->GetOrAddReal("Advection", "vx", 1.0);
  Real vy = pin->GetOrAddReal("Advection", "vy", 0.0);
  Real vz = pin->GetOrAddReal("Advection", "vz", 0.0);
  pkg->AddParam<>("vx", vx);
  pkg->AddParam<>("vy", vy);
  pkg->AddParam<>("vz", vz);

  Real refine_tol = pin->GetOrAddReal("Advection", "refine_tol", 0.3);
  pkg->AddParam<>("refine_tol", refine_tol);
  Real derefine_tol = pin->GetOrAddReal("Advection", "derefine_tol", 0.03);
  pkg->AddParam<>("derefine_tol", derefine_tol);

  auto profile_str = pin->GetOrAddString("Advection", "profile", "wave");
  if (!((profile_str == "wave") || (profile_str == "smooth_gaussian") ||
        (profile_str == "hard_sphere") || (profile_str == "block"))) {
    PARTHENON_FAIL(("Unknown profile in advection example: " + profile_str).c_str());
  }
  pkg->AddParam<>("profile", profile_str);

  bool do_regular_advection =
      pin->GetOrAddBoolean("Advection", "do_regular_advection", true);
  pkg->AddParam<>("do_regular_advection", do_regular_advection);
  if (do_regular_advection) {
    int shape_size = pin->GetOrAddInteger("Advection", "shape_size", 1);
    int sparse_size = pin->GetOrAddInteger("Advection", "sparse_size", 1);
    pkg->AddParam<>("sparse_size", sparse_size);
    Real alloc_threshold = pin->GetOrAddReal("Advection", "alloc_threshold", 1.e-6);
    Real dealloc_threshold = pin->GetOrAddReal("Advection", "dealloc_threshold", 5.e-7);
    Metadata m({Metadata::Cell, Metadata::Independent, Metadata::WithFluxes,
                Metadata::FillGhost, Metadata::Sparse},
               std::vector<int>{shape_size});
    m.SetSparseThresholds(alloc_threshold, dealloc_threshold, 0.0);
    std::vector<int> sparse_idxs(sparse_size);
    std::iota(sparse_idxs.begin(), sparse_idxs.end(), 0);
    pkg->AddSparsePool<Conserved::phi>(m, sparse_idxs);
  }

  bool do_fine_advection = pin->GetOrAddBoolean("Advection", "do_fine_advection", true);
  pkg->AddParam<>("do_fine_advection", do_fine_advection);
  if (do_fine_advection) {
    pkg->AddField<Conserved::phi_fine>(
        Metadata({Metadata::Cell, Metadata::Fine, Metadata::Independent,
                  Metadata::WithFluxes, Metadata::FillGhost}));

    pkg->AddField<Conserved::phi_fine_restricted>(
        Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy}));
  }

  bool do_CT_advection = pin->GetOrAddBoolean("Advection", "do_CT_advection", true);
  pkg->AddParam<>("do_CT_advection", do_CT_advection);
  if (do_CT_advection) {
    auto m = Metadata({Metadata::Face, Metadata::Independent, Metadata::WithFluxes,
                       Metadata::FillGhost});
    m.RegisterRefinementOps<parthenon::refinement_ops::ProlongateSharedMinMod,
                            parthenon::refinement_ops::RestrictAverage,
                            parthenon::refinement_ops::ProlongateInternalTothAndRoe>();
    pkg->AddField<Conserved::C>(m);
    pkg->AddField<Conserved::D>(m);
    pkg->AddField<Conserved::recon>(Metadata(
        {Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, std::vector<int>{4}));
    pkg->AddField<Conserved::recon_f>(Metadata(
        {Metadata::Face, Metadata::Derived, Metadata::OneCopy}, std::vector<int>{2}));
    pkg->AddField<Conserved::C_cc>(Metadata(
        {Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, std::vector<int>{3}));
    pkg->AddField<Conserved::D_cc>(Metadata(
        {Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, std::vector<int>{3}));
    pkg->AddField<Conserved::divC>(
        Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy}));
    pkg->AddField<Conserved::divD>(
        Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy}));
  }

  pkg->CheckRefinementMesh = CheckRefinementMesh;
  pkg->EstimateTimestepMesh = EstimateTimestep;
  pkg->FillDerivedMesh = FillDerived;
  return pkg;
}

void CheckRefinementMesh(MeshData<Real> *md, parthenon::ParArray1D<AmrTag> &amr_tags) {
  std::shared_ptr<StateDescriptor> pkg =
      md->GetMeshPointer()->packages.Get("advection_package");
  auto do_regular_advection = pkg->Param<bool>("do_regular_advection");
  if (do_regular_advection) {
    // refine on advected, for example.  could also be a derived quantity
    auto desc = parthenon::MakePackDescriptor<Conserved::phi>(md);
    auto pack = desc.GetPack(md);

    const auto &refine_tol = pkg->Param<Real>("refine_tol");
    const auto &derefine_tol = pkg->Param<Real>("derefine_tol");

    auto ib = md->GetBoundsI(IndexDomain::entire);
    auto jb = md->GetBoundsJ(IndexDomain::entire);
    auto kb = md->GetBoundsK(IndexDomain::entire);
    auto scatter_tags = amr_tags.ToScatterView<Kokkos::Experimental::ScatterMax>();
    parthenon::par_for_outer(
        PARTHENON_AUTO_LABEL, 0, 0, 0, pack.GetNBlocks() - 1, 0,
        pack.GetMaxNumberOfVars() - 1, kb.s, kb.e,
        KOKKOS_LAMBDA(parthenon::team_mbr_t team_member, const int b, const int n,
                      const int k) {
          if (n > pack.GetUpperBound(b)) return;
          typename Kokkos::MinMax<Real>::value_type minmax;
          par_reduce_inner(
              parthenon::inner_loop_pattern_ttr_tag, team_member, jb.s, jb.e, ib.s, ib.e,
              [&](const int j, const int i,
                  typename Kokkos::MinMax<Real>::value_type &lminmax) {
                lminmax.min_val =
                    (pack(b, n, k, j, i) < lminmax.min_val ? pack(b, n, k, j, i)
                                                           : lminmax.min_val);
                lminmax.max_val =
                    (pack(b, n, k, j, i) > lminmax.max_val ? pack(b, n, k, j, i)
                                                           : lminmax.max_val);
              },
              Kokkos::MinMax<Real>(minmax));

          auto tags_access = scatter_tags.access();
          auto flag = AmrTag::same;
          if (minmax.max_val > refine_tol && minmax.min_val < derefine_tol)
            flag = AmrTag::refine;
          if (minmax.max_val < derefine_tol) flag = AmrTag::derefine;
          tags_access(b).update(flag);
        });
    amr_tags.ContributeScatter(scatter_tags);
  }
}

Real EstimateTimestep(MeshData<Real> *md) {
  std::shared_ptr<StateDescriptor> pkg =
      md->GetMeshPointer()->packages.Get("advection_package");
  const auto &cfl = pkg->Param<Real>("cfl");
  const auto &vx = pkg->Param<Real>("vx");
  const auto &vy = pkg->Param<Real>("vy");
  const auto &vz = pkg->Param<Real>("vz");

  auto desc = parthenon::MakePackDescriptor<Conserved::phi>(md);
  auto pack = desc.GetPack(md);

  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);

  // This is obviously overkill for this constant velocity problem
  Real min_dt;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, PARTHENON_AUTO_LABEL, DevExecSpace(), 0,
      pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lmin_dt) {
        auto &coords = pack.GetCoordinates(b);
        lmin_dt = std::min(
            lmin_dt, parthenon::robust::ratio(coords.Dxc<X1DIR>(k, j, i), std::abs(vx)));
        lmin_dt = std::min(
            lmin_dt, parthenon::robust::ratio(coords.Dxc<X2DIR>(k, j, i), std::abs(vy)));
        lmin_dt = std::min(
            lmin_dt, parthenon::robust::ratio(coords.Dxc<X3DIR>(k, j, i), std::abs(vz)));
      },
      Kokkos::Min<Real>(min_dt));

  return cfl * min_dt / 2.0;
}

TaskStatus FillDerived(MeshData<Real> *md) {
  auto desc =
      parthenon::MakePackDescriptor<Conserved::phi_fine, Conserved::phi_fine_restricted,
                                    Conserved::C, Conserved::C_cc, Conserved::D,
                                    Conserved::D_cc, Conserved::divC, Conserved::divD>(
          md);
  auto pack = desc.GetPack(md);

  std::shared_ptr<StateDescriptor> pkg =
      md->GetMeshPointer()->packages.Get("advection_package");

  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);
  const int ndim = md->GetMeshPointer()->ndim;
  const int nghost = parthenon::Globals::nghost;

  auto do_fine_advection = pkg->Param<bool>("do_fine_advection");
  if (do_fine_advection) {
    parthenon::par_for(
        PARTHENON_AUTO_LABEL, 0, pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
        ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const int kf = (ndim > 2) ? (k - nghost) * 2 + nghost : k;
          const int jf = (ndim > 1) ? (j - nghost) * 2 + nghost : j;
          const int fi = (ndim > 0) ? (i - nghost) * 2 + nghost : i;
          pack(b, Conserved::phi_fine_restricted(), k, j, i) = 0.0;
          Real ntot = 0.0;
          for (int koff = 0; koff <= (ndim > 2); ++koff)
            for (int joff = 0; joff <= (ndim > 1); ++joff)
              for (int ioff = 0; ioff <= (ndim > 0); ++ioff) {
                ntot += 1.0;
                pack(b, Conserved::phi_fine_restricted(), k, j, i) +=
                    pack(b, Conserved::phi_fine(), kf + koff, jf + joff, fi + ioff);
              }
          pack(b, Conserved::phi_fine_restricted(), k, j, i) /= ntot;
        });
  }

  auto do_CT_advection = pkg->Param<bool>("do_CT_advection");
  if (do_CT_advection) {
    using TE = parthenon::TopologicalElement;
    parthenon::par_for(
        PARTHENON_AUTO_LABEL, 0, pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
        ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          pack(b, Conserved::C_cc(0), k, j, i) =
              0.5 * (pack(b, TE::F1, Conserved::C(), k, j, i) +
                     pack(b, TE::F1, Conserved::C(), k, j, i + (ndim > 0)));
          pack(b, Conserved::C_cc(1), k, j, i) =
              0.5 * (pack(b, TE::F2, Conserved::C(), k, j, i) +
                     pack(b, TE::F2, Conserved::C(), k, j + (ndim > 1), i));
          pack(b, Conserved::C_cc(2), k, j, i) =
              0.5 * (pack(b, TE::F3, Conserved::C(), k, j, i) +
                     pack(b, TE::F3, Conserved::C(), k + (ndim > 2), j, i));
          auto &coords = pack.GetCoordinates(b);
          pack(b, Conserved::divC(), k, j, i) =
              (pack(b, TE::F1, Conserved::C(), k, j, i + (ndim > 0)) -
               pack(b, TE::F1, Conserved::C(), k, j, i)) /
                  coords.Dxc<X1DIR>(k, j, i) +
              (pack(b, TE::F2, Conserved::C(), k, j + (ndim > 1), i) -
               pack(b, TE::F2, Conserved::C(), k, j, i)) /
                  coords.Dxc<X2DIR>(k, j, i) +
              (pack(b, TE::F3, Conserved::C(), k + (ndim > 2), j, i) -
               pack(b, TE::F3, Conserved::C(), k, j, i)) /
                  coords.Dxc<X3DIR>(k, j, i);

          pack(b, Conserved::D_cc(0), k, j, i) =
              0.5 * (pack(b, TE::F1, Conserved::D(), k, j, i) +
                     pack(b, TE::F1, Conserved::D(), k, j, i + (ndim > 0)));
          pack(b, Conserved::D_cc(1), k, j, i) =
              0.5 * (pack(b, TE::F2, Conserved::D(), k, j, i) +
                     pack(b, TE::F2, Conserved::D(), k, j + (ndim > 1), i));
          pack(b, Conserved::D_cc(2), k, j, i) =
              0.5 * (pack(b, TE::F3, Conserved::D(), k, j, i) +
                     pack(b, TE::F3, Conserved::D(), k + (ndim > 2), j, i));
          pack(b, Conserved::divD(), k, j, i) =
              (pack(b, TE::F1, Conserved::D(), k, j, i + (ndim > 0)) -
               pack(b, TE::F1, Conserved::D(), k, j, i)) /
                  coords.Dxc<X1DIR>(k, j, i) +
              (pack(b, TE::F2, Conserved::D(), k, j + (ndim > 1), i) -
               pack(b, TE::F2, Conserved::D(), k, j, i)) /
                  coords.Dxc<X2DIR>(k, j, i) +
              (pack(b, TE::F3, Conserved::D(), k + (ndim > 2), j, i) -
               pack(b, TE::F3, Conserved::D(), k, j, i)) /
                  coords.Dxc<X3DIR>(k, j, i);
        });
  }
  return TaskStatus::complete;
}
} // namespace advection_package
