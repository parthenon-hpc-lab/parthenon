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

#include <parthenon/package.hpp>

#include "advection_package.hpp"

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
  CellVariable<Real> &v = rc.Get("advected");
  Real vmin = 1.0;
  Real vmax = 0.0;
  for (int k = 0; k < pmb->ncells3; k++) {
    for (int j = 0; j < pmb->ncells2; j++) {
      for (int i = 0; i < pmb->ncells1; i++) {
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
  int is = 0;
  int js = 0;
  int ks = 0;
  int ie = pmb->ncells1 - 1;
  int je = pmb->ncells2 - 1;
  int ke = pmb->ncells3 - 1;
  CellVariable<Real> &qin = rc.Get("advected");
  CellVariable<Real> &qout = rc.Get("one_minus_advected");
  for (int k = ks; k <= ke; k++) {
    for (int j = js; j <= je; j++) {
      for (int i = is; i <= ie; i++) {
        qout(k, j, i) = 1.0 - qin(k, j, i);
      }
    }
  }
}

// this is the package registered function to fill derived
void SquareIt(Container<Real> &rc) {
  MeshBlock *pmb = rc.pmy_block;
  int is = 0;
  int js = 0;
  int ks = 0;
  int ie = pmb->ncells1 - 1;
  int je = pmb->ncells2 - 1;
  int ke = pmb->ncells3 - 1;
  CellVariable<Real> &qin = rc.Get("one_minus_advected");
  CellVariable<Real> &qout = rc.Get("one_minus_advected_sq");
  for (int k = ks; k <= ke; k++) {
    for (int j = js; j <= je; j++) {
      for (int i = is; i <= ie; i++) {
        qout(k, j, i) = qin(k, j, i) * qin(k, j, i);
      }
    }
  }
}

// demonstrate usage of a "post" fill derived routine
void PostFill(Container<Real> &rc) {
  MeshBlock *pmb = rc.pmy_block;
  int is = 0;
  int js = 0;
  int ks = 0;
  int ie = pmb->ncells1 - 1;
  int je = pmb->ncells2 - 1;
  int ke = pmb->ncells3 - 1;
  CellVariable<Real> &qin = rc.Get("one_minus_advected_sq");
  // get component 12
  CellVariable<Real> &q0 = rc.Get("one_minus_sqrt_one_minus_advected_sq", 12);
  // and component 37
  CellVariable<Real> &q1 = rc.Get("one_minus_sqrt_one_minus_advected_sq", 37);
  for (int k = ks; k <= ke; k++) {
    for (int j = js; j <= je; j++) {
      for (int i = is; i <= ie; i++) {
        // this will make component 12 = advected
        q0(k, j, i) = 1.0 - sqrt(qin(k, j, i));
        // and this will make component 37 = 1 - advected
        q1(k, j, i) = 1.0 - q0(k, j, i);
      }
    }
  }
}

// provide the routine that estimates a stable timestep for this package
Real EstimateTimestep(Container<Real> &rc) {
  MeshBlock *pmb = rc.pmy_block;
  auto pkg = pmb->packages["advection_package"];
  const auto &cfl = pkg->Param<Real>("cfl");
  const auto &vx = pkg->Param<Real>("vx");
  const auto &vy = pkg->Param<Real>("vy");
  const auto &vz = pkg->Param<Real>("vz");

  int is = pmb->is;
  int js = pmb->js;
  int ks = pmb->ks;
  int ie = pmb->ie;
  int je = pmb->je;
  int ke = pmb->ke;

  Real min_dt = std::numeric_limits<Real>::max();
  ParArrayND<Real> dx1("dx1", pmb->ncells1);
  ParArrayND<Real> dx2("dx2", pmb->ncells1);
  ParArrayND<Real> dx3("dx3", pmb->ncells1);

  // this is obviously overkill for this constant velocity problem
  for (int k = ks; k <= ke; k++) {
    for (int j = js; j <= je; j++) {
      pmb->pcoord->CenterWidth1(k, j, is, ie, dx1);
      pmb->pcoord->CenterWidth2(k, j, is, ie, dx2);
      pmb->pcoord->CenterWidth3(k, j, is, ie, dx3);
      for (int i = is; i <= ie; i++) {
        min_dt = std::min(min_dt, dx1(i) / std::abs(vx));
        min_dt = std::min(min_dt, dx2(i) / std::abs(vy));
        min_dt = std::min(min_dt, dx3(i) / std::abs(vz));
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
  int is = pmb->is;
  int js = pmb->js;
  int ks = pmb->ks;
  int ie = pmb->ie;
  int je = pmb->je;
  int ke = pmb->ke;

  CellVariable<Real> &q = rc.Get("advected");
  auto pkg = pmb->packages["advection_package"];
  const auto &vx = pkg->Param<Real>("vx");
  const auto &vy = pkg->Param<Real>("vy");
  const auto &vz = pkg->Param<Real>("vz");

  ParArrayND<Real> ql("ql", q.GetDim(4), pmb->ncells1);
  ParArrayND<Real> qr("qr", q.GetDim(4), pmb->ncells1);
  ParArrayND<Real> qltemp("qltemp", q.GetDim(4), pmb->ncells1);

  // get x-fluxes
  for (int k = ks; k <= ke; k++) {
    for (int j = js; j <= je; j++) {
      // get reconstructed state on faces
      pmb->precon->DonorCellX1(k, j, is - 1, ie + 1, q.data, ql, qr);
      if (vx > 0.0) {
        for (int i = is; i <= ie + 1; i++) {
          q.flux[0](k, j, i) = ql(i) * vx;
        }
      } else {
        for (int i = is; i <= ie + 1; i++) {
          q.flux[0](k, j, i) = qr(i) * vx;
        }
      }
    }
  }
  // get y-fluxes
  if (pmb->pmy_mesh->ndim >= 2) {
    for (int k = ks; k <= ke; k++) {
      pmb->precon->DonorCellX2(k, js - 1, is, ie, q.data, ql, qr);
      for (int j = js; j <= je + 1; j++) {
        pmb->precon->DonorCellX2(k, j, is, ie, q.data, qltemp, qr);
        if (vy > 0.0) {
          for (int i = is; i <= ie; i++) {
            q.flux[1](k, j, i) = ql(i) * vy;
          }
        } else {
          for (int i = is; i <= ie; i++) {
            q.flux[1](k, j, i) = qr(i) * vy;
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
    for (int j = js; j <= je; j++) { // loop ordering is intentional
      pmb->precon->DonorCellX3(ks - 1, j, is, ie, q.data, ql, qr);
      for (int k = ks; k <= ke + 1; k++) {
        pmb->precon->DonorCellX3(k, j, is, ie, q.data, qltemp, qr);
        if (vz > 0.0) {
          for (int i = is; i <= ie; i++) {
            q.flux[2](k, j, i) = ql(i) * vz;
          }
        } else {
          for (int i = is; i <= ie; i++) {
            q.flux[2](k, j, i) = qr(i) * vz;
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
