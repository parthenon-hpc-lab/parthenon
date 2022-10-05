//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2022 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2022. Triad National Security, LLC. All rights reserved.
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
#include <iostream> // debug
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "bvals_cc_in_one.hpp"
#include "bvals_utils.hpp"
#include "config.hpp"
#include "globals.hpp"
#include "interface/variable.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/refinement_cc_in_one.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {
namespace cell_centered_bvars {

using namespace impl;

TaskStatus LoadAndSendFluxCorrections(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::Profiling::pushRegion("Task_LoadAndSendFluxCorrections");

  Mesh *pmesh = md->GetMeshPointer();

  bool all_available = true;
  ForEachBoundary<BoundaryType::flxcor_send>(
      md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb, const sp_cv_t v) -> LoopControl {
        auto &buf = pmesh->boundary_comm_flxcor_map[SendKey(pmb, nb, v)];
        if (!buf.IsAvailableForWrite()) {
          all_available = false;
          return LoopControl::break_out;
        }
        return LoopControl::cont;
      });
  if (!all_available) return TaskStatus::incomplete;

  ForEachBoundary<BoundaryType::flxcor_send>(md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb,
                                                     const sp_cv_t v) {
    PARTHENON_DEBUG_REQUIRE(pmesh->boundary_comm_flxcor_map.count(SendKey(pmb, nb, v)) >
                                0,
                            "Boundary communicator does not exist");
    auto &buf = pmesh->boundary_comm_flxcor_map[SendKey(pmb, nb, v)];

    if (!v->IsAllocated()) {
      buf.Free();
      buf.SendNull();
      return LoopControl::cont; // Cycle to the next boundary
    }

    // This allocate shouldn't do anything, since the buffer should already
    // be allocated if the variable is allocated
    buf.Allocate();

    // Average fluxes over area and load buffer
    IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

    const int ndim = 1 + (jb.e - jb.s > 0 ? 1 : 0) + (kb.e - kb.s > 0 ? 1 : 0);
    auto binfo = BndInfo::GetSendCCFluxCor(pmb, nb, v);

    auto &coords = pmb->coords;
    
    binfo.buf = buf.buffer();
    const int nl = binfo.Nt; 
    const int nm = binfo.Nu; 
    const int nn = binfo.Nv; 
    const int nk = binfo.ek - binfo.sk + 1;
    const int nj = binfo.ej - binfo.sj + 1;
    const int ni = binfo.ei - binfo.si + 1;
 
    int ioff = 1;
    int joff = ndim > 1 ? 1 : 0;
    int koff = ndim > 2 ? 1 : 0;   
    
    if (binfo.dir == X1DIR) ioff = 0; 
    if (binfo.dir == X2DIR) joff = 0; 
    if (binfo.dir == X3DIR) koff = 0; 

    const int NjNi = nj * ni;
    const int NkNjNi = nk * NjNi;
    const int NnNkNjNi = nn * NkNjNi;
    const int NmNnNkNjNi = nm * NnNkNjNi;
    Kokkos::parallel_for(
        "SendFluxCorrection",
        Kokkos::RangePolicy<>(parthenon::DevExecSpace(), 0, nl * NmNnNkNjNi),
        KOKKOS_LAMBDA(const int loop_idx) {
          const int l = loop_idx / NmNnNkNjNi;
          const int m = (loop_idx % NmNnNkNjNi) / NnNkNjNi;
          const int n = (loop_idx % NnNkNjNi) / NkNjNi;
          const int ck = (loop_idx % NkNjNi) / NjNi;
          const int cj = (loop_idx % NjNi) / ni;
          const int ci = loop_idx % ni;

          const int k = binfo.sk + 2 * ck;
          const int j = binfo.sj + 2 * cj;
          const int i = binfo.si + 2 * ci;

          // For the given set of offsets, etc. this should work for any
          // dimensionality since the same flux will be included multiple times
          // in the average
          const Real area00 = coords.Area(binfo.dir, k, j, i);
          const Real area01 = coords.Area(binfo.dir, k, j + joff, i + ioff);
          const Real area10 = coords.Area(binfo.dir, k + koff, j + joff, i);
          const Real area11 = coords.Area(binfo.dir, k + koff, j, i + ioff);

          Real avg_flx = area00 * binfo.var(l, m, n, k, j, i);
          avg_flx += area01 * binfo.var(l, m, n, k + koff, j + joff, i);
          avg_flx += area10 * binfo.var(l, m, n, k, j + joff, i + ioff);
          avg_flx += area11 * binfo.var(l, m, n, k + koff, j, i + ioff);

          avg_flx /= area00 + area01 + area10 + area11;
          const int idx = ci + ni * (cj + nj * (ck + nk * (n + nn * (m + nm * l))));
          binfo.buf(idx) = avg_flx;
        });

    // Send the buffer
    PARTHENON_REQUIRE(buf.GetState() == BufferState::stale, "Not sure how I got here.");
#ifdef MPI_PARALLEL
    Kokkos::fence();
#endif
    buf.Send();
    return LoopControl::cont;
  });

  Kokkos::Profiling::popRegion(); // Task_LoadAndSendFluxCorrections
  return TaskStatus::complete;
}

TaskStatus StartReceiveFluxCorrections(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::Profiling::pushRegion("Task_ReceiveFluxCorrections");
  Mesh *pmesh = md->GetMeshPointer();
  ForEachBoundary<BoundaryType::flxcor_recv>(
      md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb, const sp_cv_t v) {
        PARTHENON_DEBUG_REQUIRE(
            pmesh->boundary_comm_flxcor_map.count(ReceiveKey(pmb, nb, v)) > 0,
            "Boundary communicator does not exist");
        auto &buf = pmesh->boundary_comm_flxcor_map[ReceiveKey(pmb, nb, v)];
        buf.TryStartReceive();
      });
  Kokkos::Profiling::popRegion(); // Task_ReceiveFluxCorrections
  return TaskStatus::complete;
}

TaskStatus ReceiveFluxCorrections(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::Profiling::pushRegion("Task_ReceiveFluxCorrections");
  bool all_received = true;
  Mesh *pmesh = md->GetMeshPointer();
  ForEachBoundary<BoundaryType::flxcor_recv>(
      md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb, const sp_cv_t v) {
        PARTHENON_DEBUG_REQUIRE(
            pmesh->boundary_comm_flxcor_map.count(ReceiveKey(pmb, nb, v)) > 0,
            "Boundary communicator does not exist");
        auto &buf = pmesh->boundary_comm_flxcor_map[ReceiveKey(pmb, nb, v)];
        all_received = buf.TryReceive() && all_received;
      });

  Kokkos::Profiling::popRegion(); // Task_ReceiveFluxCorrections

  if (all_received) return TaskStatus::complete;
  return TaskStatus::incomplete;
}

TaskStatus SetFluxCorrections(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::Profiling::pushRegion("Task_SetFluxCorrections");

  Mesh *pmesh = md->GetMeshPointer();

  ForEachBoundary<BoundaryType::flxcor_recv>(
      md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb, const sp_cv_t v) {
        PARTHENON_DEBUG_REQUIRE(
            pmesh->boundary_comm_flxcor_map.count(ReceiveKey(pmb, nb, v)) > 0,
            "Boundary communicator does not exist");
        auto &buf = pmesh->boundary_comm_flxcor_map[ReceiveKey(pmb, nb, v)];

        // Check if this boundary requires flux correction
        if ((!v->IsAllocated()) || buf.GetState() == BufferState::received_null) {
          buf.Stale();
          return LoopControl::cont;
        }
        
        auto binfo = BndInfo::GetSetCCFluxCor(pmb, nb, v);

        binfo.buf = buf.buffer();
        const int nl = binfo.Nt; 
        const int nm = binfo.Nu; 
        const int nn = binfo.Nv; 
        const int nk = binfo.ek - binfo.sk + 1;
        const int nj = binfo.ej - binfo.sj + 1;
        const int ni = binfo.ei - binfo.si + 1;
   
        const int NjNi = nj * ni;
        const int NkNjNi = nk * NjNi;
        const int NnNkNjNi = nn * NkNjNi;
        const int NmNnNkNjNi = nm * NnNkNjNi;

        Kokkos::parallel_for(
            "SetFluxCorrections",
            Kokkos::RangePolicy<>(parthenon::DevExecSpace(), 0, nl * NmNnNkNjNi),
            KOKKOS_LAMBDA(const int loop_idx) {
              const int l = loop_idx / NmNnNkNjNi;
              const int m = (loop_idx % NmNnNkNjNi) / NnNkNjNi;
              const int n = (loop_idx % NnNkNjNi) / NkNjNi;
              const int k = (loop_idx % NkNjNi) / NjNi + binfo.sk;
              const int j = (loop_idx % NjNi) / ni + binfo.sj;
              const int i = loop_idx % ni + binfo.si;

              const int idx =
                  i - binfo.si + ni * (j - binfo.sj + nj * (k - binfo.sk + nk * (n + nn * (m + nm * l))));
              binfo.var(l, m, n, k, j, i) = binfo.buf(idx);
            });
        buf.Stale();
        return LoopControl::cont;
      });

  Kokkos::Profiling::popRegion(); // Task_SetFluxCorrections
  return TaskStatus::complete;
}

} // namespace cell_centered_bvars
} // namespace parthenon
