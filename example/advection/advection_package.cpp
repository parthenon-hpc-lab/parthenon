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

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <coordinates/coordinates.hpp>
#include <parthenon/package.hpp>

#include "advection_package.hpp"
#include "defs.hpp"
#include "kokkos_abstraction.hpp"
#include "reconstruct/dc_inline.hpp"

using namespace parthenon::package::prelude;

// *************************************************//
// define the "physics" package Advect, which      *//
// includes defining various functions that control*//
// how parthenon functions and any tasks needed to *//
// implement the "physics"                         *//
// *************************************************//

namespace advection_package {
using parthenon::UserHistoryOperation;

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
  if (!((profile_str.compare("wave") == 0) ||
        (profile_str.compare("smooth_gaussian") == 0) ||
        (profile_str.compare("hard_sphere") == 0))) {
    PARTHENON_FAIL(("Unknown profile in advection example: " + profile_str).c_str());
  }
  pkg->AddParam<>("profile", profile_str);

  auto buffer_send_pack = pin->GetOrAddBoolean("Advection", "buffer_send_pack", false);
  auto buffer_recv_pack = pin->GetOrAddBoolean("Advection", "buffer_recv_pack", false);
  auto buffer_set_pack = pin->GetOrAddBoolean("Advection", "buffer_set_pack", false);
  pkg->AddParam<>("buffer_send_pack", buffer_send_pack);
  pkg->AddParam<>("buffer_recv_pack", buffer_recv_pack);
  pkg->AddParam<>("buffer_set_pack", buffer_set_pack);

  Real amp = pin->GetOrAddReal("Advection", "amp", 1e-6);
  Real vel = std::sqrt(vx * vx + vy * vy + vz * vz);
  Real ang_2 = pin->GetOrAddReal("Advection", "ang_2", -999.9);
  Real ang_3 = pin->GetOrAddReal("Advection", "ang_3", -999.9);

  Real ang_2_vert = pin->GetOrAddBoolean("Advection", "ang_2_vert", false);
  Real ang_3_vert = pin->GetOrAddBoolean("Advection", "ang_3_vert", false);

  auto fill_derived = pin->GetOrAddBoolean("Advection", "fill_derived", true);
  pkg->AddParam<>("fill_derived", fill_derived);

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
  Real k_par = 2.0 * (M_PI) / lambda;

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

  // number of variable in variable vector
  const auto vec_size = pin->GetOrAddInteger("Advection", "vec_size", 1);
  const auto num_vars = pin->GetOrAddInteger("Advection", "num_vars", 1);
  pkg->AddParam<>("vec_size", vec_size);
  pkg->AddParam<>("num_vars", num_vars);

  // Give a custom labels to advected in the data output
  std::string field_name_base = "advected";
  std::string field_name;
  Metadata m;
  for (int var = 0; var < num_vars; ++var) {
    std::vector<std::string> advected_labels;
    advected_labels.reserve(vec_size);
    for (int j = 0; j < vec_size; ++j) {
      advected_labels.push_back("Advected_" + std::to_string(var) + " _" +
                                std::to_string(j));
    }
    if (var == 0) { // first var is always called just "advected"
      field_name = field_name_base;
    } else {
      field_name = field_name_base + "_" + std::to_string(var);
    }
    m = Metadata({Metadata::Cell, Metadata::Independent, Metadata::FillGhost},
                 std::vector<int>({vec_size}), advected_labels);
    pkg->AddField(field_name, m);
  }
  if (fill_derived) {
    field_name = "one_minus_advected";
    m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
                 std::vector<int>({num_vars}));
    pkg->AddField(field_name, m);

    field_name = "one_minus_advected_sq";
    m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
                 std::vector<int>({num_vars}));
    pkg->AddField(field_name, m);

    // for fun make this last one a multi-component field using SparseVariable
    field_name = "one_minus_sqrt_one_minus_advected_sq";
    m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy, Metadata::Sparse,
                  Metadata::Restart},
                 12, // just picking a sparse_id out of a hat for demonstration
                 std::vector<int>({num_vars}));
    pkg->AddField(field_name, m);
    // add another component
    m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy, Metadata::Sparse,
                  Metadata::Restart},
                 37, // just picking a sparse_id out of a hat for demonstration
                 std::vector<int>({num_vars}));
    pkg->AddField(field_name, m);
  }

  // List (vector) of HistoryOutputVar that will all be enrolled as output variables
  parthenon::HstVar_list hst_vars = {};
  // Now we add a couple of callback functions
  // Note that the specialization of AdvectionHst is completely artificial here and used
  // in the function to differentiate between different behavior.
  // In other words, it's independent of the history machinery itself and just controls
  // the behavior of the AdvectionHst example.
  hst_vars.emplace_back(parthenon::HistoryOutputVar(
      UserHistoryOperation::sum, AdvectionHst<Kokkos::Sum<Real, HostExecSpace>>,
      "total_advected"));
  hst_vars.emplace_back(parthenon::HistoryOutputVar(
      UserHistoryOperation::max, AdvectionHst<Kokkos::Max<Real, HostExecSpace>>,
      "max_advected"));
  hst_vars.emplace_back(parthenon::HistoryOutputVar(
      UserHistoryOperation::min, AdvectionHst<Kokkos::Min<Real, HostExecSpace>>,
      "min_advected"));

  // add callbacks for HST output identified by the `hist_param_key`
  pkg->AddParam<>(parthenon::hist_param_key, hst_vars);

  if (fill_derived) {
    pkg->FillDerivedBlock = SquareIt;
  }
  pkg->CheckRefinementBlock = CheckRefinement;
  pkg->EstimateTimestepBlock = EstimateTimestepBlock;

  return pkg;
}

AmrTag CheckRefinement(MeshBlockData<Real> *rc) {
  // refine on advected, for example.  could also be a derived quantity
  auto pmb = rc->GetBlockPointer();
  auto pkg = pmb->packages.Get("advection_package");
  int num_vars = pkg->Param<int>("num_vars");
  std::vector<std::string> vars = {"advected"};
  for (int var = 1; var < num_vars; ++var) {
    vars.push_back("advected_" + std::to_string(var));
  }
  // type is parthenon::VariablePack<CellVariable<Real>>
  auto v = rc->PackVariables(vars);

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

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

  const auto &refine_tol = pkg->Param<Real>("refine_tol");
  const auto &derefine_tol = pkg->Param<Real>("derefine_tol");

  if (minmax.max_val > refine_tol && minmax.min_val < derefine_tol) return AmrTag::refine;
  if (minmax.max_val < derefine_tol) return AmrTag::derefine;
  return AmrTag::same;
}

// demonstrate usage of a "pre" fill derived routine
void PreFill(MeshBlockData<Real> *rc) {
  auto pmb = rc->GetBlockPointer();
  auto pkg = pmb->packages.Get("advection_package");
  bool fill_derived = pkg->Param<bool>("fill_derived");

  if (fill_derived) {
    IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
    IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

    // packing in principle unnecessary/convoluted here and just done for demonstration
    PackIndexMap imap;
    std::vector<std::string> vars({"advected", "one_minus_advected"});
    const auto &v = rc->PackVariables(vars, imap);
    const int in = imap["advected"].first;
    const int out = imap["one_minus_advected"].first;
    const auto num_vars = rc->Get("advected").data.GetDim(4);
    pmb->par_for(
        "advection_package::PreFill", 0, num_vars - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int n, const int k, const int j, const int i) {
          v(out + n, k, j, i) = 1.0 - v(in + n, k, j, i);
        });
  }
}

// this is the package registered function to fill derived
void SquareIt(MeshBlockData<Real> *rc) {
  auto pmb = rc->GetBlockPointer();

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  // packing in principle unnecessary/convoluted here and just done for demonstration
  PackIndexMap imap;
  std::vector<std::string> vars({"one_minus_advected", "one_minus_advected_sq"});
  auto v = rc->PackVariables(vars, imap);
  const int in = imap["one_minus_advected"].first;
  const int out = imap["one_minus_advected_sq"].first;
  const auto num_vars = rc->Get("advected").data.GetDim(4);
  pmb->par_for(
      "advection_package::SquareIt", 0, num_vars - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int n, const int k, const int j, const int i) {
        v(out + n, k, j, i) = v(in + n, k, j, i) * v(in + n, k, j, i);
      });
}

// demonstrate usage of a "post" fill derived routine
void PostFill(MeshBlockData<Real> *rc) {
  auto pmb = rc->GetBlockPointer();
  auto pkg = pmb->packages.Get("advection_package");
  bool fill_derived = pkg->Param<bool>("fill_derived");

  if (fill_derived) {
    IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
    IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

    // packing in principle unnecessary/convoluted here and just done for demonstration
    PackIndexMap imap;
    std::vector<std::string> vars(
        {"one_minus_advected_sq", "one_minus_sqrt_one_minus_advected_sq"});
    auto v = rc->PackVariables(vars, {12, 37}, imap);
    const int in = imap["one_minus_advected_sq"].first;
    const int out12 = imap["one_minus_sqrt_one_minus_advected_sq_12"].first;
    const int out37 = imap["one_minus_sqrt_one_minus_advected_sq_37"].first;
    const auto num_vars = rc->Get("advected").data.GetDim(4);
    pmb->par_for(
        "advection_package::PostFill", 0, num_vars - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
        ib.e, KOKKOS_LAMBDA(const int n, const int k, const int j, const int i) {
          v(out12 + n, k, j, i) = 1.0 - sqrt(v(in + n, k, j, i));
          v(out37 + n, k, j, i) = 1.0 - v(out12 + n, k, j, i);
        });
  }
}

// Example of how to enroll a history function.
// Templating is *NOT* required and just implemented here to reuse this function
// for testing of the UserHistoryOperations curently available in Parthenon (Sum, Min,
// Max), which translate to the MPI reduction being called over all ranks. T should be
// either Kokkos::Sum, Kokkos::Min, or Kokkos::Max.
template <typename T>
Real AdvectionHst(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();

  // Packing variable over MeshBlock as the function is called for MeshData, i.e., a
  // collection of blocks
  const auto &advected_pack = md->PackVariables(std::vector<std::string>{"advected"});

  const auto ib = advected_pack.cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = advected_pack.cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = advected_pack.cellbounds.GetBoundsK(IndexDomain::interior);

  Real result = 0.0;
  T reducer(result);

  // We choose to apply volume weighting when using the sum reduction.
  // Downstream this choice will be done on a variable by variable basis and volume
  // weighting needs to be applied in the reduction region.
  const bool volume_weighting = std::is_same<T, Kokkos::Sum<Real, HostExecSpace>>::value;

  pmb->par_reduce(
      "AdvectionHst", 0, advected_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lresult) {
        const auto &coords = advected_pack.coords(b);
        // `join` is a function of the Kokkos::ReducerConecpt that allows to use the same
        // call for different reductions
        const Real vol = volume_weighting ? coords.Volume(k, j, i) : 1.0;
        reducer.join(lresult, advected_pack(b, 0, k, j, i) * vol);
      },
      reducer);

  return result;
}

// provide the routine that estimates a stable timestep for this package
Real EstimateTimestepBlock(MeshBlockData<Real> *rc) {
  auto pmb = rc->GetBlockPointer();
  auto pkg = pmb->packages.Get("advection_package");
  const auto &cfl = pkg->Param<Real>("cfl");
  const auto &vx = pkg->Param<Real>("vx");
  const auto &vy = pkg->Param<Real>("vy");
  const auto &vz = pkg->Param<Real>("vz");

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto &coords = pmb->coords;

  // this is obviously overkill for this constant velocity problem
  Real min_dt;
  pmb->par_reduce(
      "advection_package::EstimateTimestep", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
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
  using parthenon::MetadataFlag;

  Kokkos::Profiling::pushRegion("Task_Advection_CalculateFluxes");
  auto pmb = rc->GetBlockPointer();

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto pkg = pmb->packages.Get("advection_package");
  const auto &vx = pkg->Param<Real>("vx");
  const auto &vy = pkg->Param<Real>("vy");
  const auto &vz = pkg->Param<Real>("vz");

  auto v = rc->PackVariablesAndFluxes(std::vector<MetadataFlag>{Metadata::Independent});

  const int scratch_level = 1; // 0 is actual scratch (tiny); 1 is HBM
  const int nx1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
  const int nvar = v.GetDim(4);
  size_t scratch_size_in_bytes = parthenon::ScratchPad2D<Real>::shmem_size(nvar, nx1);
  // get x-fluxes
  pmb->par_for_outer(
      "x1 flux", 2 * scratch_size_in_bytes, scratch_level, kb.s, kb.e, jb.s, jb.e,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int k, const int j) {
        parthenon::ScratchPad2D<Real> ql(member.team_scratch(scratch_level), nvar, nx1);
        parthenon::ScratchPad2D<Real> qr(member.team_scratch(scratch_level), nvar, nx1);
        // get reconstructed state on faces
        parthenon::DonorCellX1(member, k, j, ib.s - 1, ib.e + 1, v, ql, qr);
        // Sync all threads in the team so that scratch memory is consistent
        member.team_barrier();

        for (int n = 0; n < nvar; n++) {
          if (vx > 0.0) {
            par_for_inner(member, ib.s, ib.e + 1, [&](const int i) {
              v.flux(X1DIR, n, k, j, i) = ql(n, i) * vx;
            });
          } else {
            par_for_inner(member, ib.s, ib.e + 1, [&](const int i) {
              v.flux(X1DIR, n, k, j, i) = qr(n, i) * vx;
            });
          }
        }
      });

  // get y-fluxes
  if (pmb->pmy_mesh->ndim >= 2) {
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
          parthenon::DonorCellX2(member, k, j - 1, ib.s, ib.e, v, ql, q_unused);
          parthenon::DonorCellX2(member, k, j, ib.s, ib.e, v, q_unused, qr);
          // Sync all threads in the team so that scratch memory is consistent
          member.team_barrier();
          for (int n = 0; n < nvar; n++) {
            if (vy > 0.0) {
              par_for_inner(member, ib.s, ib.e, [&](const int i) {
                v.flux(X2DIR, n, k, j, i) = ql(n, i) * vy;
              });
            } else {
              par_for_inner(member, ib.s, ib.e, [&](const int i) {
                v.flux(X2DIR, n, k, j, i) = qr(n, i) * vy;
              });
            }
          }
        });
  }

  // get z-fluxes
  if (pmb->pmy_mesh->ndim == 3) {
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
          parthenon::DonorCellX3(member, k - 1, j, ib.s, ib.e, v, ql, q_unused);
          parthenon::DonorCellX3(member, k, j, ib.s, ib.e, v, q_unused, qr);
          // Sync all threads in the team so that scratch memory is consistent
          member.team_barrier();
          for (int n = 0; n < nvar; n++) {
            if (vz > 0.0) {
              par_for_inner(member, ib.s, ib.e, [&](const int i) {
                v.flux(X3DIR, n, k, j, i) = ql(n, i) * vz;
              });
            } else {
              par_for_inner(member, ib.s, ib.e, [&](const int i) {
                v.flux(X3DIR, n, k, j, i) = qr(n, i) * vz;
              });
            }
          }
        });
  }

  Kokkos::Profiling::popRegion(); // Task_Advection_CalculateFluxes
  return TaskStatus::complete;
}

} // namespace advection_package
