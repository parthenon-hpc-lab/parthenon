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
#include "kokkos_abstraction.hpp"
#include "reconstruct/dc.hpp"

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
  auto v = rc.Get("advected").data;

  auto nx1 = pmb->ncells1;
  auto nx2 = pmb->ncells2;
  auto nx3 = pmb->ncells3;

  Kokkos::complex<Real> minmax(1.0, 0.0); // using real as vmin and imag as vmax
  Kokkos::Sum<Kokkos::complex<Real>> minmax_reducer(minmax);
  // using a "cheap" (read dirty) reduction on a complex variable instead of
  // two scalars to prevent the implementation of a custom reduction on an array
  // Alternatively, we may want to change the reducer type to a view at some point
  // to precent an implicit fence at the end of the kernel. However, then we also need
  // to refactor more code so that this function can return immediate and we collect
  // the result later (similar to calculating timesteps on individual meshblocks)
  Kokkos::parallel_reduce(
      "advection check refinement",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nx3, nx2, nx1}, {1, 1, nx1}),
      KOKKOS_LAMBDA(int k, int j, int i, Kokkos::complex<Real> &minmax) {
        Real vmin = (v(k, j, i) < minmax.real() ? v(k, j, i) : minmax.real());
        Real vmax = (v(k, j, i) > minmax.imag() ? v(k, j, i) : minmax.imag());
        minmax = Kokkos::complex<Real>(vmin, vmax);
      },
      minmax_reducer);

  auto pkg = pmb->packages["advection_package"];
  const auto &refine_tol = pkg->Param<Real>("refine_tol");
  const auto &derefine_tol = pkg->Param<Real>("derefine_tol");

  if (minmax.imag() > refine_tol && minmax.real() < derefine_tol) return AmrTag::refine;
  if (minmax.imag() < derefine_tol) return AmrTag::derefine;
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

  CellVariable<Real> &advected = rc.Get("advected");
  auto pkg = pmb->packages["advection_package"];
  const auto &vx = pkg->Param<Real>("vx");
  const auto &vy = pkg->Param<Real>("vy");
  const auto &vz = pkg->Param<Real>("vz");

  const int scratch_level = 1; // 0 is actual scratch (tiny); 1 is HBM
  const int nx1 = pmb->ncells1;
  const int nvar = advected.GetDim(4);
  size_t scratch_size_in_bytes = parthenon::ScratchPad2D<Real>::shmem_size(nvar, nx1);
  parthenon::ParArray4D<Real> x1flux = advected.flux[X1DIR].Get<4>();
  // get x-fluxes
  pmb->par_for_outer(
      "x1 flux", 2 * scratch_size_in_bytes, scratch_level, ks, ke, js, je,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int k, const int j) {
        parthenon::ScratchPad2D<Real> ql(member.team_scratch(scratch_level), nvar, nx1);
        parthenon::ScratchPad2D<Real> qr(member.team_scratch(scratch_level), nvar, nx1);
        // get reconstructed state on faces
        parthenon::DonorCellX1(member, k, j, is - 1, ie + 1, advected.data, ql, qr);
        // Sync all threads in the team so that scratch memory is consistent
        member.team_barrier();

        for (int n = 0; n < nvar; n++) {
          if (vx > 0.0) {
            parthenon::par_for_inner(member, is, ie + 1, [&](const int i) {
              x1flux(n, k, j, i) = ql(n, i) * vx;
            });
          } else {
            parthenon::par_for_inner(member, is, ie + 1, [&](const int i) {
              x1flux(n, k, j, i) = qr(n, i) * vx;
            });
          }
        }
      });

  // get y-fluxes
  if (pmb->pmy_mesh->ndim >= 2) {
    parthenon::ParArray4D<Real> x2flux = advected.flux[X2DIR].Get<4>();
    pmb->par_for_outer(
        "x2 flux", 3 * scratch_size_in_bytes, scratch_level, ks, ke, js, je + 1,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int k, const int j) {
          // the overall algorithm/use of scratch pad here is clear inefficient and kept
          // just for demonstrating purposes. The key point is that we cannot reuse
          // reconstructed arrays for different `j` with `j` being part of the outer
          // loop given that this loop can be handled by multiple threads simultaneously.

          parthenon::ScratchPad2D<Real> ql(member.team_scratch(scratch_level), nvar, nx1);
          parthenon::ScratchPad2D<Real> qr(member.team_scratch(scratch_level), nvar, nx1);
          parthenon::ScratchPad2D<Real> q_unused(member.team_scratch(scratch_level), nvar,
                                                 nx1);
          // get reconstructed state on faces
          parthenon::DonorCellX2(member, k, j - 1, is, ie, advected.data, ql, q_unused);
          parthenon::DonorCellX2(member, k, j, is, ie, advected.data, q_unused, qr);
          // Sync all threads in the team so that scratch memory is consistent
          member.team_barrier();
          for (int n = 0; n < nvar; n++) {
            if (vy > 0.0) {
              parthenon::par_for_inner(member, is, ie, [&](const int i) {
                x2flux(n, k, j, i) = ql(n, i) * vy;
              });
            } else {
              parthenon::par_for_inner(member, is, ie, [&](const int i) {
                x2flux(n, k, j, i) = qr(n, i) * vy;
              });
            }
          }
        });
  }

  // get z-fluxes
  if (pmb->pmy_mesh->ndim == 3) {
    parthenon::ParArray4D<Real> x3flux = advected.flux[X3DIR].Get<4>();
    pmb->par_for_outer(
        "x3 flux", 3 * scratch_size_in_bytes, scratch_level, ks, ke + 1, js, je,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int k, const int j) {
          // the overall algorithm/use of scratch pad here is clear inefficient and kept
          // just for demonstrating purposes. The key point is that we cannot reuse
          // reconstructed arrays for different `j` with `j` being part of the outer
          // loop given that this loop can be handled by multiple threads simultaneously.

          parthenon::ScratchPad2D<Real> ql(member.team_scratch(scratch_level), nvar, nx1);
          parthenon::ScratchPad2D<Real> qr(member.team_scratch(scratch_level), nvar, nx1);
          parthenon::ScratchPad2D<Real> q_unused(member.team_scratch(scratch_level), nvar,
                                                 nx1);
          // get reconstructed state on faces
          parthenon::DonorCellX3(member, k - 1, j, is, ie, advected.data, ql, q_unused);
          parthenon::DonorCellX3(member, k, j, is, ie, advected.data, q_unused, qr);
          // Sync all threads in the team so that scratch memory is consistent
          member.team_barrier();
          for (int n = 0; n < nvar; n++) {
            if (vz > 0.0) {
              parthenon::par_for_inner(member, is, ie, [&](const int i) {
                x3flux(n, k, j, i) = ql(n, i) * vz;
              });
            } else {
              parthenon::par_for_inner(member, is, ie, [&](const int i) {
                x3flux(n, k, j, i) = qr(n, i) * vz;
              });
            }
          }
        });
  }

  return TaskStatus::complete;
}

} // namespace advection_package
