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
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <coordinates/coordinates.hpp>
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
  Real vy = pin->GetOrAddReal("Advection", "vy", 1.0);
  Real vz = pin->GetOrAddReal("Advection", "vz", 1.0);
  Real refine_tol = pin->GetOrAddReal("Advection", "refine_tol", 0.3);
  pkg->AddParam<>("refine_tol", refine_tol);
  Real derefine_tol = pin->GetOrAddReal("Advection", "derefine_tol", 0.03);
  pkg->AddParam<>("derefine_tol", derefine_tol);

  auto profile_str = pin->GetOrAddString("Advection", "profile", "wave");
  int profile = -1; // unspecified profile
  if (profile_str.compare("wave") == 0) {
    profile = 0;
  } else if (profile_str.compare("smooth_gaussian") == 0) {
    profile = 1;
  } else if (profile_str.compare("hard_sphere") == 0) {
    profile = 2;
  } else {
    PARTHENON_FAIL("Unknown profile in advection example: " + profile_str);
  }
  pkg->AddParam<>("profile", profile);

  Real amp = pin->GetOrAddReal("Advection", "amp", 1e-6);
  Real vel = std::sqrt(vx * vx + vy * vy + vz * vz);
  Real ang_2 = pin->GetOrAddReal("Advection", "ang_2", -999.9);
  Real ang_3 = pin->GetOrAddReal("Advection", "ang_3", -999.9);

  Real ang_2_vert = pin->GetOrAddBoolean("Advection", "ang_2_vert", false);
  Real ang_3_vert = pin->GetOrAddBoolean("Advection", "ang_3_vert", false);

  // For wavevector along coordinate axes, set desired values of ang_2/ang_3.
  //    For example, for 1D problem use ang_2 = ang_3 = 0.0
  //    For wavevector along grid diagonal, do not input values for ang_2/ang_3.
  // Code below will automatically calculate these imposing periodicity and exactly one
  // wavelength along each grid direction
  Real x1size = pin->GetOrAddReal("parthenon/mesh", "x1max", 1.5) -
                pin->GetOrAddReal("parthenon/mesh", "x1min", -1.5);
  Real x2size = pin->GetOrAddReal("parthenon/mesh", "x2max", 1.0) -
                pin->GetOrAddReal("parthenon/mesh", "x2min", -1.0);
  Real x3size = pin->GetOrAddReal("parthenon/mesh", "x3max", 1.0) -
                pin->GetOrAddReal("parthenon/mesh", "x3min", -1.0);

  // User should never input -999.9 in angles
  if (ang_3 == -999.9) ang_3 = std::atan(x1size / x2size);
  Real sin_a3 = std::sin(ang_3);
  Real cos_a3 = std::cos(ang_3);

  // Override ang_3 input and hardcode vertical (along x2 axis) wavevector
  if (ang_3_vert) {
    sin_a3 = 1.0;
    cos_a3 = 0.0;
    ang_3 = 0.5 * M_PI;
  }

  if (ang_2 == -999.9)
    ang_2 = std::atan(0.5 * (x1size * cos_a3 + x2size * sin_a3) / x3size);
  Real sin_a2 = std::sin(ang_2);
  Real cos_a2 = std::cos(ang_2);

  // Override ang_2 input and hardcode vertical (along x3 axis) wavevector
  if (ang_2_vert) {
    sin_a2 = 1.0;
    cos_a2 = 0.0;
    ang_2 = 0.5 * M_PI;
  }

  Real x1 = x1size * cos_a2 * cos_a3;
  Real x2 = x2size * cos_a2 * sin_a3;
  Real x3 = x3size * sin_a2;

  // For lambda choose the smaller of the 3
  Real lambda = x1;
  if ((pin->GetOrAddInteger("parthenon/mesh", "nx2", 1) > 1) && ang_3 != 0.0)
    lambda = std::min(lambda, x2);
  if ((pin->GetOrAddInteger("parthenon/mesh", "nx3", 1) > 1) && ang_2 != 0.0)
    lambda = std::min(lambda, x3);

  // If cos_a2 or cos_a3 = 0, need to override lambda
  if (ang_3_vert) lambda = x2;
  if (ang_2_vert) lambda = x3;

  // Initialize k_parallel
  Real k_par = 2.0 * (PI) / lambda;

  pkg->AddParam<>("amp", amp);
  pkg->AddParam<>("vel", vel);
  pkg->AddParam<>("vx", vx);
  pkg->AddParam<>("vy", vy);
  pkg->AddParam<>("vz", vz);
  pkg->AddParam<>("k_par", k_par);
  pkg->AddParam<>("cos_a2", cos_a2);
  pkg->AddParam<>("cos_a3", cos_a3);
  pkg->AddParam<>("sin_a2", sin_a2);
  pkg->AddParam<>("sin_a3", sin_a3);

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
  PackIndexMap imap;
  std::vector<std::string> vars({"advected", "one_minus_advected"});
  auto v = rc.PackVariables(vars, imap);
  const int in = imap["advected"].first;
  const int out = imap["one_minus_advected"].first;
  pmb->par_for(
      "advection_package::PreFill", ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        v(out, k, j, i) = 1.0 - v(in, k, j, i);
      });
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
  PackIndexMap imap;
  std::vector<std::string> vars({"one_minus_advected", "one_minus_advected_sq"});
  auto v = rc.PackVariables(vars, imap);
  const int in = imap["one_minus_advected"].first;
  const int out = imap["one_minus_advected_sq"].first;
  pmb->par_for(
      "advection_package::PreFill", ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        v(out, k, j, i) = v(in, k, j, i) * v(in, k, j, i);
      });
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
  PackIndexMap imap;
  std::vector<std::string> vars(
      {"one_minus_advected_sq", "one_minus_sqrt_one_minus_advected_sq"});
  auto v = rc.PackVariables(vars, {12, 37}, imap);
  const int in = imap["one_minus_advected_sq"].first;
  const int out12 = imap["one_minus_sqrt_one_minus_advected_sq_12"].first;
  const int out37 = imap["one_minus_sqrt_one_minus_advected_sq_37"].first;
  pmb->par_for(
      "advection_package::PreFill", ks, ke, js, je, is, ie,
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

  int is = pmb->is;
  int js = pmb->js;
  int ks = pmb->ks;
  int ie = pmb->ie;
  int je = pmb->je;
  int ke = pmb->ke;

  Real min_dt = std::numeric_limits<Real>::max();
  auto &coords = pmb->coords;

  // this is obviously overkill for this constant velocity problem
  for (int k = ks; k <= ke; k++) {
    for (int j = js; j <= je; j++) {
      for (int i = is; i <= ie; i++) {
        if (vx != 0.0)
          min_dt = std::min(min_dt, coords.Dx(X1DIR, k, j, i) / std::abs(vx));
        if (vy != 0.0)
          min_dt = std::min(min_dt, coords.Dx(X2DIR, k, j, i) / std::abs(vy));
        if (vz != 0.0)
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
          q.flux[X1DIR](k, j, i) = ql(i) * vx;
        }
      } else {
        for (int i = is; i <= ie + 1; i++) {
          q.flux[X1DIR](k, j, i) = qr(i) * vx;
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
            q.flux[X2DIR](k, j, i) = ql(i) * vy;
          }
        } else {
          for (int i = is; i <= ie; i++) {
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
    for (int j = js; j <= je; j++) { // loop ordering is intentional
      pmb->precon->DonorCellX3(ks - 1, j, is, ie, q.data, ql, qr);
      for (int k = ks; k <= ke + 1; k++) {
        pmb->precon->DonorCellX3(k, j, is, ie, q.data, qltemp, qr);
        if (vz > 0.0) {
          for (int i = is; i <= ie; i++) {
            q.flux[X3DIR](k, j, i) = ql(i) * vz;
          }
        } else {
          for (int i = is; i <= ie; i++) {
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
