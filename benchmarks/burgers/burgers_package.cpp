//========================================================================================
// (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
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
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "coordinates/coordinates.hpp"
#include "defs.hpp"
#include "kokkos_abstraction.hpp"
#include "parthenon/package.hpp"
#include "reconstruct/dc_inline.hpp"
#include "utils/error_checking.hpp"

#include "burgers_package.hpp"
#include "recon.hpp"

using namespace parthenon::package::prelude;

// *************************************************//
// define the "physics" package Advect, which      *//
// includes defining various functions that control*//
// how parthenon functions and any tasks needed to *//
// implement the "physics"                         *//
// *************************************************//

namespace burgers_package {
using parthenon::UserHistoryOperation;

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("burgers_package");

  Real cfl = pin->GetOrAddReal("burgers", "cfl", 0.8);
  pkg->AddParam("cfl", cfl);

  std::string recon_string = pin->GetOrAddString("burgers", "recon", "weno5");
  recon::ReconType recon_type;
  if (recon_string == "weno5") {
    recon_type = recon::ReconType::WENO5;
    int nghost = pin->GetInteger("parthenon/mesh", "nghost");
    PARTHENON_REQUIRE_THROWS(nghost >= 4, "weno5 reconstruction requires 4 or more ghost "
                                          "cells.  Set <parthenon/mesh>/nghost = 4");
  } else if (recon_string == "linear") {
    recon_type = recon::ReconType::Linear;
    int nghost = pin->GetInteger("parthenon/mesh", "nghost");
    if (nghost > 2)
      PARTHENON_WARN("Using more ghost cells than required.  Consider setting "
                     "<parthenon/mesh>/nghost = 2");
  } else {
    std::string msg =
        recon_string +
        " is an invalid option for <burgers>/recon.  Valid options are weno5 and linear.";
    PARTHENON_THROW(msg);
  }
  pkg->AddParam("recon_type", recon_type);

  // number of variable in variable vector
  const auto num_scalars = pin->GetOrAddInteger("burgers", "num_scalars", 1);
  pkg->AddParam("num_scalars", num_scalars);
  PARTHENON_REQUIRE_THROWS(num_scalars > 0,
                           "Burgers benchmark requires num_scalars >= 1");

  // always a three dimensional state vector
  std::vector<int> vec_components(1, 3 + num_scalars);
  Metadata m({Metadata::Cell, Metadata::Independent, Metadata::Intensive,
              Metadata::Conserved, Metadata::FillGhost, Metadata::WithFluxes},
             vec_components);
  pkg->AddField("U", m);

  // reconstructed state on faces
  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, vec_components);
  pkg->AddField("Ulx", m);
  pkg->AddField("Urx", m);
  pkg->AddField("Uly", m);
  pkg->AddField("Ury", m);
  pkg->AddField("Ulz", m);
  pkg->AddField("Urz", m);

  // a derived variable
  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
  pkg->AddField("derived", m);

  pkg->EstimateTimestepMesh = EstimateTimestepMesh;
  pkg->FillDerivedMesh = CalculateDerived;

  return pkg;
}

void CalculateDerived(MeshData<Real> *md) {
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);

  std::vector<std::string> vars({"derived", "U"});
  auto &v = md->PackVariables(vars);
  const int nblocks = md->NumBlocks();
  size_t scratch_size = 0;
  constexpr int scratch_level = 0;
  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "CalculateDerived", DevExecSpace(), scratch_size,
      scratch_level, 0, nblocks - 1, kb.s, kb.e, jb.s, jb.e,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k, const int j) {
        Real *out = &v(b, 0, k, j, 0);
        Real *u1 = &v(b, 1, k, j, 0);
        Real *u2 = &v(b, 2, k, j, 0);
        Real *u3 = &v(b, 3, k, j, 0);
        Real *d = &v(b, 4, k, j, 0);
        parthenon::par_for_inner(
            DEFAULT_INNER_LOOP_PATTERN, member, ib.s, ib.e, [=](const int i) {
              out[i] = 0.5 * d[i] * (u1[i] * u1[i] + u2[i] * u2[i] + u3[i] * u3[i]);
            });
      });
}

// provide the routine that estimates a stable timestep for this package
Real EstimateTimestepMesh(MeshData<Real> *md) {
  Kokkos::Profiling::pushRegion("Task_burgers_EstimateTimestepMesh");
  auto pm = md->GetParentPointer();
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);

  auto &params = pm->packages.Get("burgers_package")->AllParams();
  const auto &cfl = params.Get<Real>("cfl");

  std::vector<std::string> vars({"U"});
  auto &v = md->PackVariables(vars);
  const int ndim = pm->ndim;

  Real min_dt;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "burgers::EstimateTimestep", DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &ldt) {
        auto &coords = v.GetCoords(b);
        ldt = std::min(
            ldt,
            1.0 /
                ((std::abs(v(b, 0, k, j, i))) / coords.Dx(X1DIR, k, j, i) +
                 (ndim > 1) * (std::abs(v(b, 1, k, j, i))) / coords.Dx(X2DIR, k, j, i) +
                 (ndim > 2) * (std::abs(v(b, 2, k, j, i))) / coords.Dx(X3DIR, k, j, i)));
      },
      Kokkos::Min<Real>(min_dt));

  Kokkos::Profiling::popRegion(); // Task_burgers_EstimateTimestepMesh
  return cfl * min_dt;
}

TaskStatus CalculateFluxes(MeshData<Real> *md) {
  using parthenon::ScratchPad1D;
  using parthenon::team_mbr_t;
  Kokkos::Profiling::pushRegion("Task_burgers_CalculateFluxes");

  auto pm = md->GetParentPointer();
  const int ndim = pm->ndim;
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);

  std::vector<std::string> vars({"U", "Ulx", "Urx", "Uly", "Ury", "Ulz", "Urz"});
  std::vector<std::string> flxs({"U"});
  PackIndexMap imap;
  auto v = md->PackVariablesAndFluxes(vars, flxs, imap);
  const int iu_lo = imap["U"].first;
  const int iu_hi = imap["U"].second;
  const int iulx_lo = imap["Ulx"].first;
  const int iurx_lo = imap["Urx"].first;
  const int iuly_lo = imap["Uly"].first;
  const int iury_lo = imap["Ury"].first;
  const int iulz_lo = imap["Ulz"].first;
  const int iurz_lo = imap["Urz"].first;

  auto &params = pm->packages.Get("burgers_package")->AllParams();
  const auto recon_type = params.Get<recon::ReconType>("recon_type");

  const int nblocks = md->NumBlocks();
  const int dk = (ndim > 2 ? 1 : 0);
  const int dj = (ndim > 1 ? 1 : 0);

  // first we'll reconstruct the state to faces
  size_t scratch_size = 0;
  constexpr int scratch_level = 0;
  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "burgers::reconstruction", DevExecSpace(), scratch_size,
      scratch_level, 0, nblocks - 1, kb.s - dk, kb.e + dk, jb.s - dj, jb.e + dj,
      KOKKOS_LAMBDA(team_mbr_t member, const int b, const int k, const int j) {
        bool xrec = (k >= kb.s && k <= kb.e) && (j >= jb.s && j <= jb.e);
        bool yrec = (k >= kb.s && k <= kb.e) && (ndim > 1);
        bool zrec = (j >= jb.s && j <= jb.e) && (ndim > 2);

        if (recon_type == recon::ReconType::WENO5) {
          auto recon_loop = [&](const int s, const int e, Real *m2, Real *m1, Real *c,
                                Real *p1, Real *p2, Real *l, Real *r) {
            parthenon::par_for_inner(
                DEFAULT_INNER_LOOP_PATTERN, member, s, e, [=](const int i) {
                  recon::WENO5Z(m2[i], m1[i], c[i], p1[i], p2[i], l[i], r[i]);
                });
          };

          for (int n = iu_lo; n <= iu_hi; n++) {
            Real *pq = &v(b, n, k, j, 0);
            if (xrec) {
              Real *pql = &v(b, iulx_lo + n, k, j, 1);
              Real *pqr = &v(b, iurx_lo + n, k, j, 0);
              recon_loop(ib.s - 1, ib.e + 1, pq - 2, pq - 1, pq, pq + 1, pq + 2, pql,
                         pqr);
            }
            if (yrec) {
              Real *pql = &v(b, iuly_lo + n, k, j + 1, 0);
              Real *pqr = &v(b, iury_lo + n, k, j, 0);
              recon_loop(ib.s, ib.e, &v(b, n, k, j - 2, 0), &v(b, n, k, j - 1, 0), pq,
                         &v(b, n, k, j + 1, 0), &v(b, n, k, j + 2, 0), pql, pqr);
            }
            if (zrec) {
              Real *pql = &v(b, iulz_lo + n, k + 1, j, 0);
              Real *pqr = &v(b, iurz_lo + n, k, j, 0);
              recon_loop(ib.s, ib.e, &v(b, n, k - 2, j, 0), &v(b, n, k - 1, j, 0), pq,
                         &v(b, n, k + 1, j, 0), &v(b, n, k + 2, j, 0), pql, pqr);
            }
          }
        } else {
          auto recon_loop = [&](const int s, const int e, Real *m1, Real *c, Real *p1,
                                Real *l, Real *r) {
            parthenon::par_for_inner(
                DEFAULT_INNER_LOOP_PATTERN, member, s, e,
                [=](const int i) { recon::Linear(m1[i], c[i], p1[i], l[i], r[i]); });
          };

          for (int n = iu_lo; n <= iu_hi; n++) {
            Real *pq = &v(b, n, k, j, 0);
            if (xrec) {
              Real *pql = &v(b, iulx_lo + n, k, j, 1);
              Real *pqr = &v(b, iurx_lo + n, k, j, 0);
              recon_loop(ib.s - 1, ib.e + 1, pq - 1, pq, pq + 1, pql, pqr);
            }
            if (yrec) {
              Real *pql = &v(b, iuly_lo + n, k, j + 1, 0);
              Real *pqr = &v(b, iury_lo + n, k, j, 0);
              recon_loop(ib.s, ib.e, &v(b, n, k, j - 1, 0), pq, &v(b, n, k, j + 1, 0),
                         pql, pqr);
            }
            if (zrec) {
              Real *pql = &v(b, iulz_lo + n, k + 1, j, 0);
              Real *pqr = &v(b, iurz_lo + n, k, j, 0);
              recon_loop(ib.s, ib.e, &v(b, n, k - 1, j, 0), pq, &v(b, n, k + 1, j, 0),
                         pql, pqr);
            }
          }
        }
      });

  // now we'll solve the Riemann problems to get fluxes
  scratch_size = 2 * ScratchPad1D<Real>::shmem_size(ib.e + 1);
  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "burgers::reconstruction", DevExecSpace(), scratch_size,
      scratch_level, 0, nblocks - 1, kb.s, kb.e + dk, jb.s, jb.e + dj,
      KOKKOS_LAMBDA(team_mbr_t member, const int b, const int k, const int j) {
        bool xflux = (k <= kb.e && j <= jb.e);
        bool yflux = (ndim > 1 && k <= kb.e);
        bool zflux = (ndim > 2 && j <= jb.e);
        ScratchPad1D<Real> sls(member.team_scratch(scratch_level), ib.e + 1);
        ScratchPad1D<Real> srs(member.team_scratch(scratch_level), ib.e + 1);

        auto uflux_loop = [&](const int s, const int e, Real *uxl, Real *uxr, Real *uyl,
                              Real *uyr, Real *uzl, Real *uzr, Real *upl, Real *upr,
                              Real *sl, Real *sr, Real *fux, Real *fuy, Real *fuz) {
          parthenon::par_for_inner(
              DEFAULT_INNER_LOOP_PATTERN, member, s, e, [=](const int i) {
                lr_to_flux(uxl[i], uxr[i], uyl[i], uyr[i], uzl[i], uzr[i], upl[i], upr[i],
                           sl[i], sr[i], fux[i], fuy[i], fuz[i]);
              });
        };
        auto qflux_loop = [&](const int s, const int e, Real *upl, Real *upr, Real *ql,
                              Real *qr, Real *sl, Real *sr, Real *flx) {
          parthenon::par_for_inner(
              DEFAULT_INNER_LOOP_PATTERN, member, s, e, [=](const int i) {
                flx[i] = (sr[i] * upl[i] * ql[i] - sl[i] * upr[i] * qr[i] +
                          sl[i] * sr[i] * (qr[i] - ql[i])) /
                         (sr[i] - sl[i] + (sl[i] * sr[i] == 0.0));
              });
        };

        Real *sl = &sls(0);
        Real *sr = &srs(0);
        if (xflux) {
          Real *uxl = &v(b, iulx_lo, k, j, 0);
          Real *uyl = &v(b, iulx_lo + 1, k, j, 0);
          Real *uzl = &v(b, iulx_lo + 2, k, j, 0);
          Real *uxr = &v(b, iurx_lo, k, j, 0);
          Real *uyr = &v(b, iurx_lo + 1, k, j, 0);
          Real *uzr = &v(b, iurx_lo + 2, k, j, 0);
          Real *fxux = &v(b).flux(X1DIR, 0, k, j, 0);
          Real *fxuy = &v(b).flux(X1DIR, 1, k, j, 0);
          Real *fxuz = &v(b).flux(X1DIR, 2, k, j, 0);
          uflux_loop(ib.s, ib.e + 1, uxl, uxr, uyl, uyr, uzl, uzr, uxl, uxr, sl, sr, fxux,
                     fxuy, fxuz);
          member.team_barrier();
          for (int n = 3; n <= iu_hi; n++) {
            Real *ql = &v(b, iulx_lo + n, k, j, 0);
            Real *qr = &v(b, iurx_lo + n, k, j, 0);
            Real *fxq = &v(b).flux(X1DIR, n, k, j, 0);
            qflux_loop(ib.s, ib.e + 1, uxl, uxr, ql, qr, sl, sr, fxq);
          }
          member.team_barrier();
        }
        if (yflux) {
          Real *uxl = &v(b, iuly_lo, k, j, 0);
          Real *uyl = &v(b, iuly_lo + 1, k, j, 0);
          Real *uzl = &v(b, iuly_lo + 2, k, j, 0);
          Real *uxr = &v(b, iury_lo, k, j, 0);
          Real *uyr = &v(b, iury_lo + 1, k, j, 0);
          Real *uzr = &v(b, iury_lo + 2, k, j, 0);
          Real *fyux = &v(b).flux(X2DIR, 0, k, j, 0);
          Real *fyuy = &v(b).flux(X2DIR, 1, k, j, 0);
          Real *fyuz = &v(b).flux(X2DIR, 2, k, j, 0);
          uflux_loop(ib.s, ib.e, uxl, uxr, uyl, uyr, uzl, uzr, uyl, uyr, sl, sr, fyux,
                     fyuy, fyuz);
          member.team_barrier();
          for (int n = 3; n <= iu_hi; n++) {
            Real *ql = &v(b, iuly_lo + n, k, j, 0);
            Real *qr = &v(b, iury_lo + n, k, j, 0);
            Real *fyq = &v(b).flux(X2DIR, n, k, j, 0);
            qflux_loop(ib.s, ib.e, uyl, uyr, ql, qr, sl, sr, fyq);
          }
          member.team_barrier();
        }
        if (zflux) {
          Real *uxl = &v(b, iulz_lo, k, j, 0);
          Real *uyl = &v(b, iulz_lo + 1, k, j, 0);
          Real *uzl = &v(b, iulz_lo + 2, k, j, 0);
          Real *uxr = &v(b, iurz_lo, k, j, 0);
          Real *uyr = &v(b, iurz_lo + 1, k, j, 0);
          Real *uzr = &v(b, iurz_lo + 2, k, j, 0);
          Real *fzux = &v(b).flux(X3DIR, 0, k, j, 0);
          Real *fzuy = &v(b).flux(X3DIR, 1, k, j, 0);
          Real *fzuz = &v(b).flux(X3DIR, 2, k, j, 0);
          uflux_loop(ib.s, ib.e, uxl, uxr, uyl, uyr, uzl, uzr, uzl, uzr, sl, sr, fzux,
                     fzuy, fzuz);
          member.team_barrier();
          for (int n = 3; n <= iu_hi; n++) {
            Real *ql = &v(b, iulz_lo + n, k, j, 0);
            Real *qr = &v(b, iurz_lo + n, k, j, 0);
            Real *fzq = &v(b).flux(X3DIR, n, k, j, 0);
            qflux_loop(ib.s, ib.e, uzl, uzr, ql, qr, sl, sr, fzq);
          }
          member.team_barrier();
        }
      });

  Kokkos::Profiling::popRegion(); // Task_burgers_CalculateFluxes
  return TaskStatus::complete;
}

} // namespace burgers_package
