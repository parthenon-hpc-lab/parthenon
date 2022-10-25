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

    CoordinateDirection dir;
    int ni = std::max((ib.e - ib.s + 1) / 2, 1);
    int nj = std::max((jb.e - jb.s + 1) / 2, 1);
    int nk = std::max((kb.e - kb.s + 1) / 2, 1);
    const int ndim = 1 + (jb.e - jb.s > 0 ? 1 : 0) + (kb.e - kb.s > 0 ? 1 : 0);

    int ks = kb.s;
    int js = jb.s;
    int is = ib.s;

    int ioff = 1;
    int joff = ndim > 1 ? 1 : 0;
    int koff = ndim > 2 ? 1 : 0;

    if (nb.fid == BoundaryFace::inner_x1 || nb.fid == BoundaryFace::outer_x1) {
      dir = X1DIR;
      ni = 1;
      ioff = 0;
      if (nb.fid == BoundaryFace::inner_x1)
        is = ib.s;
      else
        is = ib.e + 1;
    } else if (nb.fid == BoundaryFace::inner_x2 || nb.fid == BoundaryFace::outer_x2) {
      dir = X2DIR;
      nj = 1;
      joff = 0;
      if (nb.fid == BoundaryFace::inner_x2)
        js = jb.s;
      else
        js = jb.e + 1;
    } else if (nb.fid == BoundaryFace::inner_x3 || nb.fid == BoundaryFace::outer_x3) {
      dir = X3DIR;
      nk = 1;
      koff = 0;
      if (nb.fid == BoundaryFace::inner_x3)
        ks = kb.s;
      else
        ks = kb.e + 1;
    } else {
      PARTHENON_FAIL("Flux corrections only occur on faces for CC variables.");
    }

    auto &flx = v->flux[dir];
    auto &coords = pmb->coords;
    buf_pool_t<Real>::weak_t &buf_arr = buf.buffer();

    const int nl = flx.GetDim(6);
    const int nm = flx.GetDim(5);
    const int nn = flx.GetDim(4);

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

          const int k = ks + 2 * ck;
          const int j = js + 2 * cj;
          const int i = is + 2 * ci;

          // For the given set of offsets, etc. this should work for any
          // dimensionality since the same flux will be included multiple times
          // in the average
          const Real area00 = coords.da(dir, k, j, i);
          const Real area01 = coords.da(dir, k, j + joff, i + ioff);
          const Real area10 = coords.da(dir, k + koff, j + joff, i);
          const Real area11 = coords.da(dir, k + koff, j, i + ioff);

          Real avg_flx = area00 * flx(l, m, n, k, j, i);
          avg_flx += area01 * flx(l, m, n, k + koff, j + joff, i);
          avg_flx += area10 * flx(l, m, n, k, j + joff, i + ioff);
          avg_flx += area11 * flx(l, m, n, k + koff, j, i + ioff);

          avg_flx /= area00 + area01 + area10 + area11;
          const int idx = ci + ni * (cj + nj * (ck + nk * (n + nn * (m + nm * l))));
          buf_arr(idx) = avg_flx;
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

        // Need to caculate these bounds based on mesh position
        // Average fluxes over area and load buffer
        IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
        IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
        IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

        int ks = kb.s;
        int js = jb.s;
        int is = ib.s;
        int ke = kb.e;
        int je = jb.e;
        int ie = ib.e;
        CoordinateDirection dir;
        if (nb.fid == BoundaryFace::inner_x1 || nb.fid == BoundaryFace::outer_x1) {
          dir = X1DIR;
          if (nb.fid == BoundaryFace::inner_x1)
            ie = is;
          else
            is = ++ie;
          if (nb.ni.fi1 == 0)
            je -= pmb->block_size.nx2 / 2;
          else
            js += pmb->block_size.nx2 / 2;
          if (nb.ni.fi2 == 0)
            ke -= pmb->block_size.nx3 / 2;
          else
            ks += pmb->block_size.nx3 / 2;
        } else if (nb.fid == BoundaryFace::inner_x2 || nb.fid == BoundaryFace::outer_x2) {
          dir = X2DIR;
          if (nb.fid == BoundaryFace::inner_x2)
            je = js;
          else
            js = ++je;
          if (nb.ni.fi1 == 0)
            ie -= pmb->block_size.nx1 / 2;
          else
            is += pmb->block_size.nx1 / 2;
          if (nb.ni.fi2 == 0)
            ke -= pmb->block_size.nx3 / 2;
          else
            ks += pmb->block_size.nx3 / 2;
        } else if (nb.fid == BoundaryFace::inner_x3 || nb.fid == BoundaryFace::outer_x3) {
          dir = X3DIR;
          if (nb.fid == BoundaryFace::inner_x3)
            ke = ks;
          else
            ks = ++ke;
          if (nb.ni.fi1 == 0)
            ie -= pmb->block_size.nx1 / 2;
          else
            is += pmb->block_size.nx1 / 2;
          if (nb.ni.fi2 == 0)
            je -= pmb->block_size.nx2 / 2;
          else
            js += pmb->block_size.nx2 / 2;
        } else {
          PARTHENON_FAIL("Flux corrections only occur on faces for CC variables.");
        }

        auto &flx = v->flux[dir];
        buf_pool_t<Real>::weak_t &buf_arr = buf.buffer();
        const int nl = flx.GetDim(6);
        const int nm = flx.GetDim(5);
        const int nn = flx.GetDim(4);
        const int nk = ke - ks + 1;
        const int nj = je - js + 1;
        const int ni = ie - is + 1;
        const int NjNi = nj * ni;
        const int NkNjNi = nk * NjNi;
        const int NnNkNjNi = nn * NkNjNi;
        const int NmNnNkNjNi = nm * NnNkNjNi;
        if (nl * NmNnNkNjNi > buf_arr.size()) {
          PARTHENON_FAIL("Buffer to small")
        }

        Kokkos::parallel_for(
            "SetFluxCorrections",
            Kokkos::RangePolicy<>(parthenon::DevExecSpace(), 0, nl * NmNnNkNjNi),
            KOKKOS_LAMBDA(const int loop_idx) {
              const int l = loop_idx / NmNnNkNjNi;
              const int m = (loop_idx % NmNnNkNjNi) / NnNkNjNi;
              const int n = (loop_idx % NnNkNjNi) / NkNjNi;
              const int k = (loop_idx % NkNjNi) / NjNi + ks;
              const int j = (loop_idx % NjNi) / ni + js;
              const int i = loop_idx % ni + is;

              const int idx =
                  i - is + ni * (j - js + nj * (k - ks + nk * (n + nn * (m + nm * l))));
              flx(l, m, n, k, j, i) = buf_arr(idx);
            });
        buf.Stale();
        return LoopControl::cont;
      });

  Kokkos::Profiling::popRegion(); // Task_SetFluxCorrections
  return TaskStatus::complete;
}

} // namespace cell_centered_bvars
} // namespace parthenon
