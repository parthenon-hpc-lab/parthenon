//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
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
#include <limits>
#include <string>
#include <vector>

#include <coordinates/coordinates.hpp>
#include <parthenon/package.hpp>

#include "advection_package.hpp"

using namespace parthenon::package::prelude;
using parthenon::IndexDomain;
using parthenon::IndexRange;

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
  pkg->AddParam<>("vx", vx);
  Real vy = pin->GetOrAddReal("Advection", "vy", 1.0);
  pkg->AddParam<>("vy", vy);
  Real vz = pin->GetOrAddReal("Advection", "vz", 0.5);
  pkg->AddParam<>("vz", vz);
  Real refine_tol = pin->GetOrAddReal("Advection", "refine_tol", 0.3);
  pkg->AddParam<>("refine_tol", refine_tol);
  Real derefine_tol = pin->GetOrAddReal("Advection", "derefine_tol", 0.03);
  pkg->AddParam<>("derefine_tol", derefine_tol);

  std::string field_name = "advected";
  Metadata m({Metadata::Cell, Metadata::Independent, Metadata::FillGhost});
  pkg->AddField(field_name, m);

  field_name = "one_minus_advected";
  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
  pkg->AddField(field_name, m);

  field_name = "one_minus_advected_sq";
  pkg->AddField(field_name, m);

  // for fun make this last one a multi-component field using SparseVariable
  field_name = "one_minus_sqrt_one_minus_advected_sq";
  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy, Metadata::Sparse},
               12 // just picking a sparse_id out of a hat for demonstration
  );
  pkg->AddField(field_name, m);
  // add another component
  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy, Metadata::Sparse},
               37 // just picking a sparse_id out of a hat for demonstration
  );
  pkg->AddField(field_name, m);

  pkg->FillDerived = SquareIt;
  pkg->CheckRefinement = CheckRefinement;
  pkg->EstimateTimestep = EstimateTimestep;

  return pkg;
}

AmrTag CheckRefinement(Container<Real> &rc) {
  MeshBlock *pmb = rc.pmy_block;
  // refine on advected, for example.  could also be a derived quantity
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  CellVariable<Real> &v = rc.Get("advected");
  Real vmin = 1.0;
  Real vmax = 0.0;
  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e; j++) {
      for (int i = ib.s; i <= ib.e; i++) {
        vmin = (v(k, j, i) < vmin ? v(k, j, i) : vmin);
        vmax = (v(k, j, i) > vmax ? v(k, j, i) : vmax);
      }
    }
  }
  auto pkg = pmb->packages["advection_package"];
  const auto &refine_tol = pkg->Param<Real>("refine_tol");
  const auto &derefine_tol = pkg->Param<Real>("derefine_tol");

  if (vmax > refine_tol && vmin < derefine_tol) return AmrTag::refine;
  if (vmax < derefine_tol) return AmrTag::derefine;
  return AmrTag::same;
}

// demonstrate usage of a "pre" fill derived routine
void PreFill(Container<Real> &rc) {
  MeshBlock *pmb = rc.pmy_block;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  PackIndexMap imap;
  std::vector<std::string> vars({"advected", "one_minus_advected"});
  auto v = rc.PackVariables(vars, imap);
  const int in = imap["advected"].first;
  const int out = imap["one_minus_advected"].first;
  pmb->par_for(
      "advection_package::PreFill", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        v(out, k, j, i) = 1.0 - v(in, k, j, i);
      });
}

// this is the package registered function to fill derived
void SquareIt(Container<Real> &rc) {
  MeshBlock *pmb = rc.pmy_block;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  PackIndexMap imap;
  std::vector<std::string> vars({"one_minus_advected", "one_minus_advected_sq"});
  auto v = rc.PackVariables(vars, imap);
  const int in = imap["one_minus_advected"].first;
  const int out = imap["one_minus_advected_sq"].first;
  pmb->par_for(
      "advection_package::PreFill", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        v(out, k, j, i) = v(in, k, j, i) * v(in, k, j, i);
      });
}

// demonstrate usage of a "post" fill derived routine
void PostFill(Container<Real> &rc) {
  MeshBlock *pmb = rc.pmy_block;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  PackIndexMap imap;
  std::vector<std::string> vars(
      {"one_minus_advected_sq", "one_minus_sqrt_one_minus_advected_sq"});
  auto v = rc.PackVariables(vars, {12, 37}, imap);
  const int in = imap["one_minus_advected_sq"].first;
  const int out12 = imap["one_minus_sqrt_one_minus_advected_sq_12"].first;
  const int out37 = imap["one_minus_sqrt_one_minus_advected_sq_37"].first;
  pmb->par_for(
      "advection_package::PreFill", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        v(out12, k, j, i) = 1.0 - sqrt(v(in, k, j, i));
        v(out37, k, j, i) = 1.0 - v(out12, k, j, i);
      });
}

// provide the routine that estimates a stable timestep for this package
Real EstimateTimestep(Container<Real> &rc) {
  MeshBlock *pmb = rc.pmy_block;
  auto pkg = pmb->packages["advection_package"];
  const auto &cfl = pkg->Param<Real>("cfl");
  const auto &vx = pkg->Param<Real>("vx");
  const auto &vy = pkg->Param<Real>("vy");
  const auto &vz = pkg->Param<Real>("vz");

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  Real min_dt = std::numeric_limits<Real>::max();
  auto &coords = pmb->coords;

  // this is obviously overkill for this constant velocity problem
  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e; j++) {
      for (int i = ib.s; i <= ib.e; i++) {
        min_dt = std::min(min_dt, coords.Dx(X1DIR, k, j, i) / std::abs(vx));
        min_dt = std::min(min_dt, coords.Dx(X2DIR, k, j, i) / std::abs(vy));
        min_dt = std::min(min_dt, coords.Dx(X3DIR, k, j, i) / std::abs(vz));
      }
    }
  }

  return cfl * min_dt;
}

// Compute fluxes at faces given the constant velocity field and
// some field "advected" that we are pushing around.
// This routine implements all the "physics" in this example
TaskStatus CalculateFluxes(Container<Real> &rc) {
  MeshBlock *pmb = rc.pmy_block;
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  int ncellsi = pmb->cellbounds.ncellsi(IndexDomain::entire);

  CellVariable<Real> &q = rc.Get("advected");
  auto pkg = pmb->packages["advection_package"];
  const auto &vx = pkg->Param<Real>("vx");
  const auto &vy = pkg->Param<Real>("vy");
  const auto &vz = pkg->Param<Real>("vz");

  ParArrayND<Real> ql("ql", q.GetDim(4), ncellsi);
  ParArrayND<Real> qr("qr", q.GetDim(4), ncellsi);
  ParArrayND<Real> qltemp("qltemp", q.GetDim(4), ncellsi);

  // get x-fluxes
  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e; j++) {
      // get reconstructed state on faces
      pmb->precon->DonorCellX1(k, j, ib.s - 1, ib.e + 1, q.data, ql, qr);
      if (vx > 0.0) {
        for (int i = ib.s; i <= ib.e + 1; i++) {
          q.flux[X1DIR](k, j, i) = ql(i) * vx;
        }
      } else {
        for (int i = ib.s; i <= ib.e + 1; i++) {
          q.flux[X1DIR](k, j, i) = qr(i) * vx;
        }
      }
    }
  }
  // get y-fluxes
  if (pmb->pmy_mesh->ndim >= 2) {
    for (int k = kb.s; k <= kb.e; k++) {
      pmb->precon->DonorCellX2(k, jb.s - 1, ib.s, ib.e, q.data, ql, qr);
      for (int j = jb.s; j <= jb.e + 1; j++) {
        pmb->precon->DonorCellX2(k, j, ib.s, ib.e, q.data, qltemp, qr);
        if (vy > 0.0) {
          for (int i = ib.s; i <= ib.e; i++) {
            q.flux[X2DIR](k, j, i) = ql(i) * vy;
          }
        } else {
          for (int i = ib.s; i <= ib.e; i++) {
            q.flux[X2DIR](k, j, i) = qr(i) * vy;
          }
        }
        auto temp = ql;
        ql = qltemp;
        qltemp = temp;
      }
    }
  }

  // get z-fluxes
  if (pmb->pmy_mesh->ndim == 3) {
    for (int j = jb.s; j <= jb.e; j++) { // loop ordering is intentional
      pmb->precon->DonorCellX3(kb.s - 1, j, ib.s, ib.e, q.data, ql, qr);
      for (int k = kb.s; k <= kb.e + 1; k++) {
        pmb->precon->DonorCellX3(k, j, ib.s, ib.e, q.data, qltemp, qr);
        if (vz > 0.0) {
          for (int i = ib.s; i <= ib.e; i++) {
            q.flux[X3DIR](k, j, i) = ql(i) * vz;
          }
        } else {
          for (int i = ib.s; i <= ib.e; i++) {
            q.flux[X3DIR](k, j, i) = qr(i) * vz;
          }
        }
        auto temp = ql;
        ql = qltemp;
        qltemp = temp;
      }
    }
  }

  return TaskStatus::complete;
}

} // namespace advection_package
