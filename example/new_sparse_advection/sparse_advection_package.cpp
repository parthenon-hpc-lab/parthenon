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
#include "globals.hpp"
#include "interface/metadata.hpp"
#include "interface/sparse_pack.hpp"
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
  Real vx[4] = {-speed, speed, speed, -speed};
  Real vy[4] = {-speed, -speed, speed, speed};
  pkg->AddParam("vx", RealArr_t{-speed, speed, speed, -speed});
  pkg->AddParam("vy", RealArr_t{-speed, -speed, speed, speed});
  pkg->AddParam("vz", RealArr_t{0.0, 0.0, 0.0, 0.0});

  // add sparse field
  {
    Metadata m({Metadata::Cell, Metadata::Independent, Metadata::WithFluxes,
                Metadata::FillGhost, Metadata::Sparse},
               std::vector<int>({1}));
    SparsePool pool("sparse", m);

    Metadata mv(
        {Metadata::Cell, Metadata::Independent, Metadata::OneCopy, Metadata::Sparse},
        std::vector<int>({1}));
    const std::string control_field_base = "sparse";

    // SparsePool pool_vx("vx", mv, control_field_base);
    // SparsePool pool_vy("vy", mv, control_field_base);

    for (int sid = 0; sid < NUM_FIELDS; ++sid) {
      m.SetSparseThresholds(parthenon::Globals::sparse_config.allocation_threshold,
                            parthenon::Globals::sparse_config.deallocation_threshold,
                            0.0);
      pool.Add(sid, m);


      // mv.SetSparseThresholds(0.0, 0.0, vx[sid]);
      // pool_vx.Add(sid, mv);
      // mv.SetSparseThresholds(0.0, 0.0, vy[sid]);
      // pool_vy.Add(sid, mv);
    }
    pkg->AddSparsePool(pool);

    // pkg->AddSparsePool(pool_vx);
    // pkg->AddSparsePool(pool_vy);
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
  auto pmb = rc->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  // refine on advected, for example.  could also be a derived quantity
  const auto &v = parthenon::SparsePack<sparse_vt>::Get(rc);

  const int b = 0; // Just one block in the pack
  // Get the bounds of the allocated variables
  int lo = v.GetLowerBoundHost(b, sparse_vt());
  int hi = v.GetUpperBoundHost(b, sparse_vt());
  if (hi < lo) return AmrTag::derefine; // Nothing allocated here, safe to derefine

  typename Kokkos::MinMax<Real>::value_type minmax;
  pmb->par_reduce(
      "advection check refinement", lo, hi, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int n, const int k, const int j, const int i,
                    typename Kokkos::MinMax<Real>::value_type &lminmax) {
        lminmax.min_val =
            (v(b, n, k, j, i) < lminmax.min_val ? v(b, n, k, j, i) : lminmax.min_val);
        lminmax.max_val =
            (v(b, n, k, j, i) > lminmax.max_val ? v(b, n, k, j, i) : lminmax.max_val);
      },
      Kokkos::MinMax<Real>(minmax));

  auto pkg = pmb->packages.Get("sparse_advection_package");
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
          lmin_dt = std::min(lmin_dt, coords.Dx(X1DIR, k, j, i) / std::abs(vx[v]));
        if (vy[v] != 0.0)
          lmin_dt = std::min(lmin_dt, coords.Dx(X2DIR, k, j, i) / std::abs(vy[v]));
        if (vz[v] != 0.0)
          lmin_dt = std::min(lmin_dt, coords.Dx(X3DIR, k, j, i) / std::abs(vz[v]));
      },
      Kokkos::Min<Real>(min_dt));

  return cfl * min_dt;
}

TaskStatus CalculateFluxes(std::shared_ptr<MeshData<Real>> &rc) {
  using parthenon::MetadataFlag;

  Kokkos::Profiling::pushRegion("Task_Advection_CalculateFluxes");

  IndexRange ib = rc->GetBoundsI(IndexDomain::interior);
  IndexRange jb = rc->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = rc->GetBoundsK(IndexDomain::interior);
  const int Ni = ib.e + 1 - ib.s + 1;
  const int Nj = jb.e + 1 - jb.s + (rc->GetMeshPointer()->ndim > 1);
  const int Nk = kb.e + 1 - kb.s + (rc->GetMeshPointer()->ndim > 2);
  const int NjNi = Nj * Ni;
  const int NkNjNi = Nk * NjNi;

  auto pkg = rc->GetParentPointer()->packages.Get("sparse_advection_package");
  const auto &vxp = pkg->Param<RealArr_t>("vx");
  const auto &vyp = pkg->Param<RealArr_t>("vy");

  using pack_t = parthenon::SparsePack<sparse_vt>;
  const auto &v = pack_t::GetWithFluxes(rc.get());

  Kokkos::parallel_for(
      "Set newly allocated interior to default",
      Kokkos::TeamPolicy<>(parthenon::DevExecSpace(), v.GetNBlocks(), Kokkos::AUTO),
      KOKKOS_LAMBDA(parthenon::team_mbr_t team_member) {
        const int b = team_member.league_rank();
        int lo = v.GetLowerBound(b, sparse_vt());
        int hi = v.GetUpperBound(b, sparse_vt());

        for (int vidx = 0; vidx <= hi - lo; ++vidx) {
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange<>(team_member, NkNjNi), [&](const int idx) {
                const int k = kb.s + idx / NjNi;
                const int j = jb.s + (idx % NjNi) / Ni;
                const int i = ib.s + idx % Ni;

                const auto spidx = sparse_vt(vidx);
                const int id = v(b, spidx).sparse_id;

                const Real &qp = v(b, spidx, k, j, i);
                const Real &qmx = v(b, spidx, k, j, i - 1);
                const Real &qmy = v(b, spidx, k, j - 1, i);

                v.flux(b, X1DIR, spidx, k, j, i) = (vxp[id] > 0.0 ? qmx : qp) * vxp[id];
                v.flux(b, X2DIR, spidx, k, j, i) = (vyp[id] > 0.0 ? qmy : qp) * vyp[id];
              });
        }
      });

  PARTHENON_REQUIRE_THROWS(rc->GetMeshPointer()->ndim == 2,
                           "Sparse Advection example must be 2D");

  Kokkos::Profiling::popRegion(); // Task_Advection_CalculateFluxes
  return TaskStatus::complete;
}

} // namespace sparse_advection_package
