//========================================================================================
// (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
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

#include "defs.hpp"
#include "interface/metadata.hpp"
#include "interface/sparse_pool.hpp"
#include "kokkos_abstraction.hpp"
#include "reconstruct/dc_inline.hpp"
#include "sparse_advection_package.hpp"
#include "utils/error_checking.hpp"

using namespace parthenon::package::prelude;

// *************************************************//
// define the "physics" package Advect, which      *//
// includes defining various functions that control*//
// how parthenon functions and any tasks needed to *//
// implement the "physics"                         *//
// *************************************************//

namespace sparse_advection_package {
using parthenon::UserHistoryOperation;

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("sparse_advection_package");

  bool restart_test = pin->GetOrAddBoolean("sparse_advection", "restart_test", false);
  pkg->AddParam("restart_test", restart_test);

  Real cfl = pin->GetOrAddReal("sparse_advection", "cfl", 0.45);
  pkg->AddParam("cfl", cfl);
  Real refine_tol = pin->GetOrAddReal("sparse_advection", "refine_tol", 0.3);
  pkg->AddParam("refine_tol", refine_tol);
  Real derefine_tol = pin->GetOrAddReal("sparse_advection", "derefine_tol", 0.03);
  pkg->AddParam("derefine_tol", derefine_tol);

  Real init_size = pin->GetOrAddReal("sparse_advection", "init_size", 0.1);
  pkg->AddParam("init_size", init_size);

  // set starting positions
  Real pos = 0.8;
  pkg->AddParam("x0", RealArr_t{pos, -pos, -pos, pos});
  pkg->AddParam("y0", RealArr_t{pos, pos, -pos, -pos});

  // add velocities, field 0 moves in (-1,-1) direction, 1 in (1,-1), 2 in (1, 1), and 3
  // in (-1,1)
  Real speed = pin->GetOrAddReal("sparse_advection", "speed", 1.0) / sqrt(2.0);
  pkg->AddParam("vx", RealArr_t{-speed, speed, speed, -speed});
  pkg->AddParam("vy", RealArr_t{-speed, -speed, speed, speed});
  pkg->AddParam("vz", RealArr_t{0.0, 0.0, 0.0, 0.0});

  // add sparse field
  {
    Metadata m({Metadata::Cell, Metadata::Independent, Metadata::WithFluxes,
                Metadata::FillGhost, Metadata::Sparse},
               std::vector<int>({1}));
    SparsePool pool("sparse", m);

    for (int sid = 0; sid < NUM_FIELDS; ++sid) {
      pool.Add(sid);
    }
    pkg->AddSparsePool(pool);
  }

  // add fields for restart test ("Z" prefix so they are after sparse in alphabetical
  // list, helps with reusing velocity vectors)
  if (restart_test) {
    Metadata m_dense({Metadata::Cell, Metadata::Independent, Metadata::WithFluxes,
                      Metadata::FillGhost});
    pkg->AddField("dense_A", m_dense);
    pkg->AddField("dense_B", m_dense);

    Metadata m_sparse({Metadata::Cell, Metadata::Independent, Metadata::WithFluxes,
                       Metadata::FillGhost, Metadata::Sparse});

    SparsePool pool("shape_shift", m_sparse);
    pool.Add(1, std::vector<int>{1}, std::vector<std::string>{"scalar"});
    pool.Add(3, std::vector<int>{3}, Metadata::Vector,
             std::vector<std::string>{"vec_x", "vec_y", "vec_z"});
    pool.Add(4, std::vector<int>{4}, Metadata::Vector);

    pkg->AddSparsePool(pool);
  }

  pkg->CheckRefinementBlock = CheckRefinement;
  pkg->EstimateTimestepBlock = EstimateTimestepBlock;

  return pkg;
}

AmrTag CheckRefinement(MeshBlockData<Real> *rc) {
  // refine on advected, for example.  could also be a derived quantity
  auto pmb = rc->GetBlockPointer();
  auto pkg = pmb->packages.Get("sparse_advection_package");
  std::vector<std::string> vars{"sparse"};
  // type is parthenon::VariablePack<Variable<Real>>
  const auto &v = rc->PackVariables(vars);

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  typename Kokkos::MinMax<Real>::value_type minmax;
  pmb->par_reduce(
      "advection check refinement", 0, v.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e,
      KOKKOS_LAMBDA(const int n, const int k, const int j, const int i,
                    typename Kokkos::MinMax<Real>::value_type &lminmax) {
        if (v.IsAllocated(n)) {
          lminmax.min_val =
              (v(n, k, j, i) < lminmax.min_val ? v(n, k, j, i) : lminmax.min_val);
          lminmax.max_val =
              (v(n, k, j, i) > lminmax.max_val ? v(n, k, j, i) : lminmax.max_val);
        }
      },
      Kokkos::MinMax<Real>(minmax));

  const auto &refine_tol = pkg->Param<Real>("refine_tol");
  const auto &derefine_tol = pkg->Param<Real>("derefine_tol");

  if (minmax.max_val > refine_tol && minmax.min_val < derefine_tol) return AmrTag::refine;
  if (minmax.max_val < derefine_tol) return AmrTag::derefine;
  return AmrTag::same;
}

// provide the routine that estimates a stable timestep for this package
Real EstimateTimestepBlock(MeshBlockData<Real> *rc) {
  auto pmb = rc->GetBlockPointer();
  auto pkg = pmb->packages.Get("sparse_advection_package");
  const auto &cfl = pkg->Param<Real>("cfl");
  const auto &vx = pkg->Param<RealArr_t>("vx");
  const auto &vy = pkg->Param<RealArr_t>("vy");
  const auto &vz = pkg->Param<RealArr_t>("vz");

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto &coords = pmb->coords;

  // this is obviously overkill for this constant velocity problem
  Real min_dt;
  pmb->par_reduce(
      "sparse_advection_package::EstimateTimestep", 0, NUM_FIELDS - 1, kb.s, kb.e, jb.s,
      jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int v, const int k, const int j, const int i, Real &lmin_dt) {
        if (vx[v] != 0.0)
          lmin_dt = std::min(lmin_dt, coords.Dxc<X1DIR>(k, j, i) / std::abs(vx[v]));
        if (vy[v] != 0.0)
          lmin_dt = std::min(lmin_dt, coords.Dxc<X2DIR>(k, j, i) / std::abs(vy[v]));
        if (vz[v] != 0.0)
          lmin_dt = std::min(lmin_dt, coords.Dxc<X3DIR>(k, j, i) / std::abs(vz[v]));
      },
      Kokkos::Min<Real>(min_dt));

  return cfl * min_dt;
}

// Compute fluxes at faces given the constant velocity field and
// some field "advected" that we are pushing around.
// This routine implements all the "physics" in this example
TaskStatus CalculateFluxes(std::shared_ptr<MeshBlockData<Real>> &rc) {
  using parthenon::MetadataFlag;

  PARTHENON_INSTRUMENT
  auto pmb = rc->GetBlockPointer();

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto pkg = pmb->packages.Get("sparse_advection_package");
  const auto &vx = pkg->Param<RealArr_t>("vx");
  const auto &vy = pkg->Param<RealArr_t>("vy");
  const auto &vz = pkg->Param<RealArr_t>("vz");

  const auto &v =
      rc->PackVariablesAndFluxes(std::vector<MetadataFlag>{Metadata::WithFluxes});

  const int scratch_level = 1; // 0 is actual scratch (tiny); 1 is HBM
  const int nx1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
  const int nvar = v.GetDim(4);
  size_t scratch_size_in_bytes = parthenon::ScratchPad2D<Real>::shmem_size(nvar, nx1);
  // get x-fluxes
  pmb->par_for_outer(
      PARTHENON_AUTO_LABEL, 2 * scratch_size_in_bytes, scratch_level, kb.s, kb.e, jb.s,
      jb.e, KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int k, const int j) {
        parthenon::ScratchPad2D<Real> ql(member.team_scratch(scratch_level), nvar, nx1);
        parthenon::ScratchPad2D<Real> qr(member.team_scratch(scratch_level), nvar, nx1);

        // get reconstructed state on faces
        parthenon::DonorCellX1(member, k, j, ib.s - 1, ib.e + 1, v, ql, qr);

        // Sync all threads in the team so that scratch memory is consistent
        member.team_barrier();

        for (int n = 0; n < nvar; n++) {
          if (!v.IsAllocated(n)) continue;
          const auto this_v = vx[v(n).sparse_id % NUM_FIELDS];
          par_for_inner(member, ib.s, ib.e + 1, [&](const int i) {
            v.flux(X1DIR, n, k, j, i) = (this_v > 0.0 ? ql(n, i) : qr(n, i)) * this_v;
          });
        }
      });

  // get y-fluxes
  if (pmb->pmy_mesh->ndim >= 2) {
    pmb->par_for_outer(
        PARTHENON_AUTO_LABEL, 3 * scratch_size_in_bytes, scratch_level, kb.s, kb.e, jb.s,
        jb.e + 1, KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int k, const int j) {
          // the overall algorithm/use of scratch pad here is clearly inefficient and kept
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
            if (!v.IsAllocated(n)) continue;
            const auto this_v = vy[v(n).sparse_id % NUM_FIELDS];
            par_for_inner(member, ib.s, ib.e, [&](const int i) {
              v.flux(X2DIR, n, k, j, i) = (this_v > 0.0 ? ql(n, i) : qr(n, i)) * this_v;
            });
          }
        });
  }

  PARTHENON_REQUIRE_THROWS(pmb->pmy_mesh->ndim == 2,
                           "Sparse Advection example must be 2D");

  return TaskStatus::complete;
}

} // namespace sparse_advection_package
