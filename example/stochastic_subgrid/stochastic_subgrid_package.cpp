//========================================================================================
// (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
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

#include "stochastic_subgrid_package.hpp"

#include <algorithm>
#include <chrono> // NOLINT [build/c++11]
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <Kokkos_Random.hpp>

#include <coordinates/coordinates.hpp>
#include <parthenon/package.hpp>

#include "kokkos_abstraction.hpp"
#include "reconstruct/dc_inline.hpp"
#include "utils/alias_method.hpp"

using namespace parthenon::package::prelude;
using namespace parthenon::AliasMethod;

// *************************************************//
// define the "physics" package Advect, which      *//
// includes defining various functions that control*//
// how parthenon functions and any tasks needed to *//
// implement the "physics"                         *//
// *************************************************//

namespace stochastic_subgrid_package {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("stochastic_subgrid_package");

  // read some fixed input parameters
  {
    const Real cfl = pin->GetOrAddReal("Advection", "cfl", 0.45);
    pkg->AddParam<>("cfl", cfl);

    const Real vx = pin->GetOrAddReal("Advection", "vx", 1.0);
    const Real vy = pin->GetOrAddReal("Advection", "vy", 1.0);
    const Real vz = pin->GetOrAddReal("Advection", "vz", 1.0);
    const Real vel = std::sqrt(vx * vx + vy * vy + vz * vz);
    pkg->AddParam<>("vx", vx);
    pkg->AddParam<>("vy", vy);
    pkg->AddParam<>("vz", vz);
    pkg->AddParam<>("vel", vel);

    const Real amp = pin->GetOrAddReal("Advection", "amp", 1e-6);
    pkg->AddParam<>("amp", amp);

    const Real refine_tol = pin->GetOrAddReal("Advection", "refine_tol", 0.3);
    pkg->AddParam<>("refine_tol", refine_tol);

    const Real derefine_tol = pin->GetOrAddReal("Advection", "derefine_tol", 0.03);
    pkg->AddParam<>("derefine_tol", derefine_tol);

    const auto profile_str = pin->GetOrAddString("Advection", "profile", "wave");
    if (!((profile_str == "wave") || (profile_str == "smooth_gaussian") ||
          (profile_str == "hard_sphere"))) {
      PARTHENON_FAIL(("Unknown profile in advection example: " + profile_str).c_str());
    }
    pkg->AddParam<>("profile", profile_str);
  }

  // compute wavevector
  {
    // For wavevector along coordinate axes, set desired values of ang_2/ang_3.
    //    For example, for 1D problem use ang_2 = ang_3 = 0.0
    //    For wavevector along grid diagonal, do not input values for ang_2/ang_3.
    // Code below will automatically calculate these imposing periodicity and exactly one
    // wavelength along each grid direction
    const Real x1size = pin->GetOrAddReal("parthenon/mesh", "x1max", 1.5) -
                        pin->GetOrAddReal("parthenon/mesh", "x1min", -1.5);
    const Real x2size = pin->GetOrAddReal("parthenon/mesh", "x2max", 1.0) -
                        pin->GetOrAddReal("parthenon/mesh", "x2min", -1.0);
    const Real x3size = pin->GetOrAddReal("parthenon/mesh", "x3max", 1.0) -
                        pin->GetOrAddReal("parthenon/mesh", "x3min", -1.0);

    Real ang_2 = pin->GetOrAddReal("Advection", "ang_2", -999.9);
    Real ang_3 = pin->GetOrAddReal("Advection", "ang_3", -999.9);

    // User should never input -999.9 in angles
    if (ang_3 == -999.9) ang_3 = std::atan(x1size / x2size);
    Real sin_a3 = std::sin(ang_3);
    Real cos_a3 = std::cos(ang_3);

    // Override ang_3 input and hardcode vertical (along x2 axis) wavevector
    const Real ang_3_vert = pin->GetOrAddBoolean("Advection", "ang_3_vert", false);
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
    const Real ang_2_vert = pin->GetOrAddBoolean("Advection", "ang_2_vert", false);
    if (ang_2_vert) {
      sin_a2 = 1.0;
      cos_a2 = 0.0;
      ang_2 = 0.5 * M_PI;
    }

    const Real x1 = x1size * cos_a2 * cos_a3;
    const Real x2 = x2size * cos_a2 * sin_a3;
    const Real x3 = x3size * sin_a2;

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
    const Real k_par = 2.0 * (M_PI) / lambda;

    pkg->AddParam<>("k_par", k_par);
    pkg->AddParam<>("cos_a2", cos_a2);
    pkg->AddParam<>("cos_a3", cos_a3);
    pkg->AddParam<>("sin_a2", sin_a2);
    pkg->AddParam<>("sin_a3", sin_a3);
  }

  // set up power law distribution
  {
    int N_min = pin->GetOrAddInteger("Random", "num_iter_min", 1);
    int N_max = pin->GetOrAddInteger("Random", "num_iter_max", 100);
    Real alpha = pin->GetOrAddReal("Random", "power_law_coeff", -3.0);

    if (N_min <= 0) PARTHENON_FAIL("Random/num_iter_min must be > 0");
    if (N_max < N_min)
      PARTHENON_FAIL("Random/num_iter_max must be >= Random/num_iter_min");

    int N = N_max - N_min + 1;

    Kokkos::View<int *> num_iter_hist("num_iter_histogram", N);
    auto num_iter_hist_host = Kokkos::create_mirror_view(num_iter_hist);

    pkg->AddParam("num_iter_histogram", num_iter_hist);
    pkg->AddParam("N_min", N_min);

    // compute non-normalized probabilities
    std::vector<Real> prob(N);
    for (int i = 0; i < N; ++i) {
      prob[i] = pow(i + N_min, alpha);
      num_iter_hist_host(i) = 0;
    }

    AliasMethod alias(prob);
    pkg->AddParam("alias_method", alias);

    Kokkos::deep_copy(num_iter_hist, num_iter_hist_host);

    // create random pool
    uint64_t seed = pin->GetOrAddInteger("Random", "seed", 0);
    // if we don't have a seed, use the time
    if (seed == 0)
      seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    Kokkos::Random_XorShift64_Pool<parthenon::DevExecSpace> rand_pool(seed);
    pkg->AddParam("random_pool", rand_pool);
  }

  // add fields
  {
    // number of variable in variable vector
    const auto num_vars = pin->GetOrAddInteger("Advection", "num_vars", 1);

    std::string field_name = "advected";
    Metadata m({Metadata::Cell, Metadata::Independent, Metadata::WithFluxes,
                Metadata::FillGhost},
               std::vector<int>({num_vars}));
    pkg->AddField(field_name, m);

    field_name = "dummy_result";
    m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
    pkg->AddField(field_name, m);

    field_name = "num_iter";
    m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
    pkg->AddField(field_name, m);
  }

  pkg->FillDerivedBlock = DoLotsOfWork;
  pkg->CheckRefinementBlock = CheckRefinement;
  pkg->EstimateTimestepBlock = EstimateTimestepBlock;

  return pkg;
}

AmrTag CheckRefinement(MeshBlockData<Real> *rc) {
  auto pmb = rc->GetBlockPointer();
  // refine on advected, for example.  could also be a derived quantity
  auto v = rc->Get("advected").data;

  const IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  const IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  const IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  typename Kokkos::MinMax<Real>::value_type minmax;
  pmb->par_reduce(
      "advection check refinement", 0, v.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e,
      KOKKOS_LAMBDA(const int n, const int k, const int j, const int i,
                    typename Kokkos::MinMax<Real>::value_type &lminmax) {
        lminmax.min_val =
            (v(n, k, j, i) < lminmax.min_val ? v(n, k, j, i) : lminmax.min_val);
        lminmax.max_val =
            (v(n, k, j, i) > lminmax.max_val ? v(n, k, j, i) : lminmax.max_val);
      },
      Kokkos::MinMax<Real>(minmax));

  auto pkg = pmb->packages.Get("stochastic_subgrid_package");
  const auto &refine_tol = pkg->Param<Real>("refine_tol");
  const auto &derefine_tol = pkg->Param<Real>("derefine_tol");

  if (minmax.max_val > refine_tol && minmax.min_val < derefine_tol) return AmrTag::refine;
  if (minmax.max_val < derefine_tol) return AmrTag::derefine;
  return AmrTag::same;
}

// randomly sample an interation number for each cell from the discrete power-law
// distribution
TaskStatus ComputeNumIter(std::shared_ptr<MeshData<Real>> &md, Packages_t &packages) {
  Kokkos::Profiling::pushRegion("Task_ComputeNumIter");

  Kokkos::Profiling::pushRegion("Task_ComputeNumIter_pack");
  auto pack = md->PackVariables(std::vector<std::string>({"num_iter"}));
  Kokkos::Profiling::popRegion();

  auto pkg = packages.Get("stochastic_subgrid_package");
  const auto &pool =
      pkg->Param<Kokkos::Random_XorShift64_Pool<parthenon::DevExecSpace>>("random_pool");

  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  const IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto alias = pkg->Param<AliasMethod>("alias_method");
  int N_min = pkg->Param<int>("N_min");

  par_for(
      parthenon::loop_pattern_mdrange_tag, "ComputeNumIter", parthenon::DevExecSpace(), 0,
      pack.GetDim(5) - 1, 0, pack.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(int b, int v, int k, int j, int i) {
        auto rng = pool.get_state();
        double rand1 = rng.drand();
        double rand2 = rng.drand();
        pool.free_state(rng);

        int num_iter = N_min + alias.Sample(rand1, rand2);
        pack(b, v, k, j, i) = num_iter;
      });

  Kokkos::Profiling::popRegion(); // Task_ComputeNumIter
  return TaskStatus::complete;
}

// this is the package registered function to fill derived
void DoLotsOfWork(MeshBlockData<Real> *rc) {
  auto pmb = rc->GetBlockPointer();
  auto pkg = pmb->packages.Get("stochastic_subgrid_package");

  const IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  const IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  const IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  // packing in principle unnecessary/convoluted here and just done for demonstration
  std::vector<std::string> vars({"num_iter", "advected", "dummy_result"});
  PackIndexMap imap;
  const auto &v = rc->PackVariables(vars, imap);

  const int niter = imap.get("num_iter").first;
  const int in = imap.get("advected").first;
  const int out = imap.get("dummy_result").first;
  const auto num_vars = rc->Get("advected").data.GetDim(4);
  int N_min = pkg->Param<int>("N_min");

  auto hist = pkg->Param<Kokkos::View<int *>>("num_iter_histogram");

  const Real ilog10 = 1.0 / log(10.0);

  pmb->par_for(
      "stochastic_subgrid_package::DoLotsOfWork", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        int num_iter = v(niter, k, j, i);

        // surprisingly, this seems to be almost free
        if (num_iter > 0) {
          Kokkos::atomic_increment(&hist(num_iter - N_min));
        }

        for (int r = 0; r < num_iter; ++r) {
          Real odd = 0.0;
          Real even = 0.0;

          for (int n = 0; n < num_vars; ++n)
            (n % 2 == 0 ? even : odd) += sqrt(n + 1) * v(in + n, k, j, i);

          Real a = pow(10.0, (odd + even) / (fmax(1.0, fabs(odd * even))));
          Real b = pow(10.0, (odd - even) / (fmax(1.0, fabs(odd * even))));
          v(out, k, j, i) += log(a * b) * ilog10 / (log(a) * ilog10 + log(b) * ilog10);
        }
      });
}

// provide the routine that estimates a stable timestep for this package
Real EstimateTimestepBlock(MeshBlockData<Real> *rc) {
  auto pmb = rc->GetBlockPointer();
  auto pkg = pmb->packages.Get("stochastic_subgrid_package");
  const auto &cfl = pkg->Param<Real>("cfl");
  const auto &vx = pkg->Param<Real>("vx");
  const auto &vy = pkg->Param<Real>("vy");
  const auto &vz = pkg->Param<Real>("vz");

  const IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  const auto &coords = pmb->coords;

  // this is obviously overkill for this constant velocity problem
  Real min_dt;
  pmb->par_reduce(
      "stochastic_subgrid_package::EstimateTimestep", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i, Real &lmin_dt) {
        if (vx != 0.0)
          lmin_dt = std::min(lmin_dt, coords.Dx(X1DIR, k, j, i) / std::abs(vx));
        if (vy != 0.0)
          lmin_dt = std::min(lmin_dt, coords.Dx(X2DIR, k, j, i) / std::abs(vy));
        if (vz != 0.0)
          lmin_dt = std::min(lmin_dt, coords.Dx(X3DIR, k, j, i) / std::abs(vz));
      },
      Kokkos::Min<Real>(min_dt));

  return cfl * min_dt;
}

// Compute fluxes at faces given the constant velocity field and
// some field "advected" that we are pushing around.
// This routine implements all the "physics" in this example
TaskStatus CalculateFluxes(std::shared_ptr<MeshBlockData<Real>> &rc) {
  Kokkos::Profiling::pushRegion("Task_Advection_CalculateFluxes");
  auto pmb = rc->GetBlockPointer();
  const IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  const auto &advected = rc->PackVariables(std::vector<std::string>{"advected"});
  auto pkg = pmb->packages.Get("stochastic_subgrid_package");
  const auto &vx = pkg->Param<Real>("vx");
  const auto &vy = pkg->Param<Real>("vy");
  const auto &vz = pkg->Param<Real>("vz");

  const int scratch_level = 1; // 0 is actual scratch (tiny); 1 is HBM
  const int nx1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
  const int nvar = advected.GetDim(4);
  size_t scratch_size_in_bytes = parthenon::ScratchPad2D<Real>::shmem_size(nvar, nx1);
  parthenon::ParArray4D<Real> x1flux = rc->Get("advected").flux[X1DIR].Get<4>();
  // get x-fluxes
  pmb->par_for_outer(
      "x1 flux", 2 * scratch_size_in_bytes, scratch_level, kb.s, kb.e, jb.s, jb.e,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int k, const int j) {
        parthenon::ScratchPad2D<Real> ql(member.team_scratch(scratch_level), nvar, nx1);
        parthenon::ScratchPad2D<Real> qr(member.team_scratch(scratch_level), nvar, nx1);
        // get reconstructed state on faces
        parthenon::DonorCellX1(member, k, j, ib.s - 1, ib.e + 1, advected, ql, qr);
        // Sync all threads in the team so that scratch memory is consistent
        member.team_barrier();

        for (int n = 0; n < nvar; n++) {
          if (vx > 0.0) {
            par_for_inner(member, ib.s, ib.e + 1,
                          [&](const int i) { x1flux(n, k, j, i) = ql(n, i) * vx; });
          } else {
            par_for_inner(member, ib.s, ib.e + 1,
                          [&](const int i) { x1flux(n, k, j, i) = qr(n, i) * vx; });
          }
        }
      });

  // get y-fluxes
  if (pmb->pmy_mesh->ndim >= 2) {
    parthenon::ParArray4D<Real> x2flux = rc->Get("advected").flux[X2DIR].Get<4>();
    pmb->par_for_outer(
        "x2 flux", 3 * scratch_size_in_bytes, scratch_level, kb.s, kb.e, jb.s, jb.e + 1,
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
          parthenon::DonorCellX2(member, k, j - 1, ib.s, ib.e, advected, ql, q_unused);
          parthenon::DonorCellX2(member, k, j, ib.s, ib.e, advected, q_unused, qr);
          // Sync all threads in the team so that scratch memory is consistent
          member.team_barrier();
          for (int n = 0; n < nvar; n++) {
            if (vy > 0.0) {
              par_for_inner(member, ib.s, ib.e,
                            [&](const int i) { x2flux(n, k, j, i) = ql(n, i) * vy; });
            } else {
              par_for_inner(member, ib.s, ib.e,
                            [&](const int i) { x2flux(n, k, j, i) = qr(n, i) * vy; });
            }
          }
        });
  }

  // get z-fluxes
  if (pmb->pmy_mesh->ndim == 3) {
    parthenon::ParArray4D<Real> x3flux = rc->Get("advected").flux[X3DIR].Get<4>();
    pmb->par_for_outer(
        "x3 flux", 3 * scratch_size_in_bytes, scratch_level, kb.s, kb.e + 1, jb.s, jb.e,
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
          parthenon::DonorCellX3(member, k - 1, j, ib.s, ib.e, advected, ql, q_unused);
          parthenon::DonorCellX3(member, k, j, ib.s, ib.e, advected, q_unused, qr);
          // Sync all threads in the team so that scratch memory is consistent
          member.team_barrier();
          for (int n = 0; n < nvar; n++) {
            if (vz > 0.0) {
              par_for_inner(member, ib.s, ib.e,
                            [&](const int i) { x3flux(n, k, j, i) = ql(n, i) * vz; });
            } else {
              par_for_inner(member, ib.s, ib.e,
                            [&](const int i) { x3flux(n, k, j, i) = qr(n, i) * vz; });
            }
          }
        });
  }

  Kokkos::Profiling::popRegion(); // Task_Advection_CalculateFluxes
  return TaskStatus::complete;
}

} // namespace stochastic_subgrid_package
