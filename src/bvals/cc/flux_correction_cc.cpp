//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
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
//! \file flux_correction_cc.cpp
//  \brief functions that perform flux correction for CELL_CENTERED variables

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "parthenon_mpi.hpp"

#include "bvals/cc/bvals_cc.hpp"
#include "coordinates/coordinates.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "parameter_input.hpp"
#include "utils/buffer_utils.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

//----------------------------------------------------------------------------------------
//! \fn void CellCenteredBoundaryVariable::SendFluxCorrection(bool is_allocated)
//  \brief Restrict, pack and send the surface flux to the coarse neighbor(s)

void CellCenteredBoundaryVariable::SendFluxCorrection(bool is_allocated) {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  auto &coords = pmb->coords;
  const IndexDomain interior = IndexDomain::interior;

  for (int n = 0; n < pmb->pbval->nneighbor; n++) {
    NeighborBlock &nb = pmb->pbval->neighbor[n];
    // flux corrections are only exchanged between blocks that both have the variable
    // allocated (receiving non-zero flux corrections wouldn't trigger an allocation).
    // Therefore, if the neighbor is on the same rank and either the neighbor doesn't have
    // this variable allocated or this block doesn't have this variable allocated, there
    // is nothing to do. If the neighbor is on a different rank, we don't know the
    // allocation status of the neighbor, so we send the MPI message (MPI_Start below)
    // regardless whether this block has the variable allocated or not. The neighbor only
    // uses the flux corrections if it has the variable allocated
    if ((nb.snb.rank == Globals::my_rank) &&
        (!IsLocalNeighborAllocated(n) || !is_allocated)) {
      continue;
    }

    if (nb.ni.type != NeighborConnect::face) break;
    if (bd_var_flcor_.sflag[nb.bufid] == BoundaryStatus::completed) continue;

    if (nb.snb.level != pmb->loc.level - 1) {
      bd_var_flcor_.sflag[nb.bufid] = BoundaryStatus::completed;
      continue;
    }

    BufArray1D<Real> &sbuf = bd_var_flcor_.send[nb.bufid];

    // if sparse is enabled, we set the first value in the buffer to 1/0 to indicate if
    // the sending block has this variable allocated (1) or not (0)
    if (Globals::sparse_config.enabled) {
      Kokkos::deep_copy(Kokkos::subview(sbuf, 0), is_allocated ? 1.0 : 0.0);
    }

    // if this variable is allocated, fill the send buffer
    if (is_allocated) {
      IndexRange ib = pmb->cellbounds.GetBoundsI(interior);
      IndexRange jb = pmb->cellbounds.GetBoundsJ(interior);
      IndexRange kb = pmb->cellbounds.GetBoundsK(interior);
      int nx1 = pmb->cellbounds.ncellsi(interior);
      int nx2 = pmb->cellbounds.ncellsj(interior);
      int nx3 = pmb->cellbounds.ncellsk(interior);
      int ll = ll_;
      int ml = ml_;
      int nl = nl_;
      int msize = mu_ - ml_ + 1;
      int nsize = nu_ - nl_ + 1;
      // x1 direction
      if (nb.fid == BoundaryFace::inner_x1 || nb.fid == BoundaryFace::outer_x1) {
        int i = ib.s + nx1 * nb.fid;
        int ks = kb.s;
        int js = jb.s;
        int ksize = (kb.e - kb.s + 1) / 2;
        int jsize = (jb.e - jb.s + 1) / 2;
        auto &x1flx = x1flux;
        if (pmb->block_size.nx3 > 1) { // 3D
          pmb->par_for(
              "SendFluxCorrection3D_x1", ll_, lu_, ml_, mu_, nl_, nu_, 0, ksize - 1, 0,
              jsize - 1,
              KOKKOS_LAMBDA(const int lll, const int mm, const int nn, const int k,
                            const int j) {
                const int kf = 2 * k + ks;
                const int jf = 2 * j + js;
                const Real amm = coords.Area(X1DIR, kf, jf, i);
                const Real amp = coords.Area(X1DIR, kf, jf + 1, i);
                const Real apm = coords.Area(X1DIR, kf + 1, jf, i);
                const Real app = coords.Area(X1DIR, kf + 1, jf + 1, i);
                const Real tarea = amm + amp + apm + app;
                const int l = lll - ll;
                const int m = mm - ml;
                const int n = nn - nl;
                // add 1 because index 0 is used for allocation flag
                const int p = 1 + j + jsize * (k + ksize * (n + nsize * (m + msize * l)));
                sbuf(p) = (x1flx(lll, mm, nn, kf, jf, i) * amm +
                           x1flx(lll, mm, nn, kf, jf + 1, i) * amp +
                           x1flx(lll, mm, nn, kf + 1, jf, i) * apm +
                           x1flx(lll, mm, nn, kf + 1, jf + 1, i) * app) /
                          tarea;
              });
        } else if (pmb->block_size.nx2 > 1) { // 2D
          int k = kb.s;
          pmb->par_for(
              "SendFluxCorrection2D_x1", ll_, lu_, ml_, mu_, nl_, nu_, 0, jsize - 1,
              KOKKOS_LAMBDA(const int lll, const int mm, const int nn, const int j) {
                const int jf = 2 * j + js;
                const Real am = coords.Area(X1DIR, k, jf, i);
                const Real ap = coords.Area(X1DIR, k, jf + 1, i);
                const Real tarea = am + ap;
                const int l = lll - ll;
                const int m = mm - ml;
                const int n = nn - nl;
                // add 1 because index 0 is used for allocation flag
                const int p = 1 + j + jsize * (n + nsize * (m + msize * l));
                sbuf(p) = (x1flx(lll, mm, nn, k, jf, i) * am +
                           x1flx(lll, mm, nn, k, jf + 1, i) * ap) /
                          tarea;
              });
        } else { // 1D
          int k = kb.s, j = jb.s;
          pmb->par_for(
              "SendFluxCorrection1D_x1", ll_, lu_, ml_, mu_, nl_, nu_,
              KOKKOS_LAMBDA(const int lll, const int mm, const int nn) {
                const int l = lll - ll;
                const int m = mm - ml;
                const int n = nn - nl;
                // add 1 because index 0 is used for allocation flag
                const int p = 1 + n + nsize * (m + msize * l);
                sbuf(p) = x1flx(lll, mm, nn, k, j, i);
              });
        }
        // x2 direction
      } else if (nb.fid == BoundaryFace::inner_x2 || nb.fid == BoundaryFace::outer_x2) {
        int j = jb.s + nx2 * (nb.fid & 1);
        int ks = kb.s;
        int is = ib.s;
        int ksize = (kb.e - kb.s + 1) / 2;
        int isize = (ib.e - ib.s + 1) / 2;
        auto &x2flx = x2flux;
        if (pmb->block_size.nx3 > 1) { // 3D
          pmb->par_for(
              "SendFluxCorrection3D_x2", ll_, lu_, ml_, mu_, nl_, nu_, 0, ksize - 1, 0,
              isize - 1,
              KOKKOS_LAMBDA(const int lll, const int mm, const int nn, const int k,
                            const int i) {
                const int kf = 2 * k + ks;
                const int ii = 2 * i + is;
                const Real area00 = coords.Area(X2DIR, kf, j, ii);
                const Real area01 = coords.Area(X2DIR, kf, j, ii + 1);
                const Real area10 = coords.Area(X2DIR, kf + 1, j, ii);
                const Real area11 = coords.Area(X2DIR, kf + 1, j, ii + 1);
                const Real tarea = area00 + area01 + area10 + area11;
                const int l = lll - ll;
                const int m = mm - ml;
                const int n = nn - nl;
                // add 1 because index 0 is used for allocation flag
                const int p = 1 + i + isize * (k + ksize * (n + nsize * (m + msize * l)));
                sbuf(p) = (x2flx(lll, mm, nn, kf, j, ii) * area00 +
                           x2flx(lll, mm, nn, kf, j, ii + 1) * area01 +
                           x2flx(lll, mm, nn, kf + 1, j, ii) * area10 +
                           x2flx(lll, mm, nn, kf + 1, j, ii + 1) * area11) /
                          tarea;
              });
        } else if (pmb->block_size.nx2 > 1) { // 2D
          int k = kb.s;
          pmb->par_for(
              "SendFluxCorrection2D_x2", ll_, lu_, ml_, mu_, nl_, nu_, 0, isize - 1,
              KOKKOS_LAMBDA(const int lll, const int mm, const int nn, const int i) {
                const int ii = 2 * i + is;
                const Real area0 = coords.Area(X2DIR, k, j, ii);
                const Real area1 = coords.Area(X2DIR, k, j, ii + 1);
                const Real tarea = area0 + area1;
                const int l = lll - ll;
                const int m = mm - ml;
                const int n = nn - nl;
                // add 1 because index 0 is used for allocation flag
                const int p = 1 + i + isize * (n + nsize * (m + msize * l));
                sbuf(p) = (x2flx(lll, mm, nn, k, j, ii) * area0 +
                           x2flx(lll, mm, nn, k, j, ii + 1) * area1) /
                          tarea;
              });
        }
        // x3 direction - 3D only
      } else if (nb.fid == BoundaryFace::inner_x3 || nb.fid == BoundaryFace::outer_x3) {
        int k = kb.s + nx3 * (nb.fid & 1);
        int js = jb.s;
        int is = ib.s;
        int jsize = (jb.e - jb.s + 1) / 2;
        int isize = (ib.e - ib.s + 1) / 2;
        auto &x3flx = x3flux;
        pmb->par_for(
            "SendFluxCorrection3D_x3", ll_, lu_, ml_, mu_, nl_, nu_, 0, jsize - 1, 0,
            isize - 1,
            KOKKOS_LAMBDA(const int lll, const int mm, const int nn, const int j,
                          const int i) {
              const int jf = 2 * j + js;
              const int ii = 2 * i + is;
              const Real area00 = coords.Area(X3DIR, k, jf, ii);
              const Real area01 = coords.Area(X3DIR, k, jf, ii + 1);
              const Real area10 = coords.Area(X3DIR, k, jf + 1, ii);
              const Real area11 = coords.Area(X3DIR, k, jf + 1, ii + 1);
              const Real tarea = area00 + area01 + area10 + area11;
              const int l = lll - ll;
              const int m = mm - ml;
              const int n = nn - nl;
              // add 1 because index 0 is used for allocation flag
              const int p = 1 + i + isize * (j + jsize * (n + nsize * (m + msize * l)));
              sbuf(p) = (x3flx(lll, mm, nn, k, jf, ii) * area00 +
                         x3flx(lll, mm, nn, k, jf, ii + 1) * area01 +
                         x3flx(lll, mm, nn, k, jf + 1, ii) * area10 +
                         x3flx(lll, mm, nn, k, jf + 1, ii + 1) * area11) /
                        tarea;
            });
      }
    }
    pmb->exec_space.fence();
    if (nb.snb.rank == Globals::my_rank) {
      // on the same node, this will only be called if this variable and the neighbor is
      // allocated
      PARTHENON_REQUIRE_THROWS(
          is_allocated && IsLocalNeighborAllocated(n),
          "Trying copy flux corrections from/to unallocated variable");
      CopyFluxCorrectionBufferSameProcess(nb);
    } else {
      // send regardless whether allocated or not
#ifdef MPI_PARALLEL
      PARTHENON_MPI_CHECK(MPI_Start(&(bd_var_flcor_.req_send[nb.bufid])));
#endif
    }

    bd_var_flcor_.sflag[nb.bufid] = BoundaryStatus::completed;
  }
}

//----------------------------------------------------------------------------------------
//! \fn bool CellCenteredBoundaryVariable::ReceiveFluxCorrection()
//  \brief Receive and apply the surface flux from the finer neighbor(s)

bool CellCenteredBoundaryVariable::ReceiveFluxCorrection(bool is_allocated) {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  bool bflag = true;

  for (int n = 0; n < pmb->pbval->nneighbor; n++) {
    NeighborBlock &nb = pmb->pbval->neighbor[n];
    if (nb.ni.type != NeighborConnect::face) break;
    if (nb.snb.level == pmb->loc.level + 1) {
      if (bd_var_flcor_.flag[nb.bufid] == BoundaryStatus::completed) continue;
      if (bd_var_flcor_.flag[nb.bufid] == BoundaryStatus::waiting) {
        if (nb.snb.rank == Globals::my_rank) { // on the same process
          // if this variable and the neighbor is allcoated, we wait until we get the flux
          // corrections, otherwise the neighbor won't send anything
          if (is_allocated && IsLocalNeighborAllocated(n)) {
            bflag = false;
            continue;
          }

        } else {
#ifdef MPI_PARALLEL
          // receive regardless whether allocated or not

          int test;
          // Comment from original Athena++ code about the MPI_Iprobe call:
          //
          // Although MPI_Iprobe does nothing for us (it checks arrival of any message but
          // we do not use the result), this is ABSOLUTELY NECESSARY for the performance
          // of Athena++. Although non-blocking MPI communications look like multi-tasking
          // running behind our code, actually they are not. The network interface card
          // can run autonomously from the CPU, but to move the data between the memory
          // and the network interface and initiate/complete communications, MPI has to do
          // something using CPU. So to process communications, we have to allow MPI to
          // use CPU. Theoretically MPI can use multi-thread for this (OpenMPI can be
          // configured so) but it is not common because of performance and compatibility
          // issues. Instead, MPI processes communications whenever any MPI function is
          // called. MPI_Iprobe is one of the cheapest function in MPI and by calling this
          // occasionally MPI can process communications "as if it is in the background".
          // Using only MPI_Test, the communications were very slow. I suspect that
          // MPI_Test changes the ordering of the messages internally (I guess it tries to
          // promote the message it is Testing), and if we call MPI_Test for different
          // messages, they are left half done. So if we remove them, I am sure we will
          // see significant performance drop. I could not dig it up right now, Collela or
          // Woodward mentioned this in a paper.
          PARTHENON_MPI_CHECK(MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,
                                         &test, MPI_STATUS_IGNORE));
          PARTHENON_MPI_CHECK(
              MPI_Test(&(bd_var_flcor_.req_recv[nb.bufid]), &test, MPI_STATUS_IGNORE));
          if (!static_cast<bool>(test)) {
            bflag = false;
            continue;
          }
          bd_var_flcor_.flag[nb.bufid] = BoundaryStatus::arrived;
#endif
        }
      }

      if (!is_allocated) {
        // we discard the flux corrections since we don't have the variable allocated on
        // the block
        bd_var_flcor_.flag[nb.bufid] = BoundaryStatus::completed;
        continue;
      }

      // if we get here, we have variable allocated, if sparse is enabled, check if the
      // sending block has it allocated
      BufArray1D<Real> &rbuf = bd_var_flcor_.recv[nb.bufid];

#ifdef ENABLE_SPARSE
      bool source_allocated = true;
      if (Globals::sparse_config.enabled) {
        if (nb.snb.rank == Globals::my_rank) {
          source_allocated = IsLocalNeighborAllocated(n);
        } else {
          Real flag;
          Kokkos::deep_copy(flag, Kokkos::subview(rbuf, 0));
          source_allocated = (flag == 1.0);
        }
      }
#else
      constexpr bool source_allocated = true;
#endif

      // boundary arrived; apply flux correction
      PARTHENON_REQUIRE_THROWS(is_allocated,
                               "CellCenteredBoundaryVariable::ReceiveFluxCorrection: "
                               "Unexpected unallocated variable");

      int ll = ll_;
      int ml = ml_;
      int nl = nl_;
      int msize = mu_ - ml_ + 1;
      int nsize = nu_ - nl_ + 1;
      const IndexDomain interior = IndexDomain::interior;
      IndexRange ib = pmb->cellbounds.GetBoundsI(interior);
      IndexRange jb = pmb->cellbounds.GetBoundsJ(interior);
      IndexRange kb = pmb->cellbounds.GetBoundsK(interior);
      if (nb.fid == BoundaryFace::inner_x1 || nb.fid == BoundaryFace::outer_x1) {
        int il = ib.s + (ib.e - ib.s) * nb.fid + nb.fid;
        int jl = jb.s, ju = jb.e, kl = kb.s, ku = kb.e;
        if (nb.ni.fi1 == 0)
          ju -= pmb->block_size.nx2 / 2;
        else
          jl += pmb->block_size.nx2 / 2;
        if (nb.ni.fi2 == 0)
          ku -= pmb->block_size.nx3 / 2;
        else
          kl += pmb->block_size.nx3 / 2;
        int jsize = ju - jl + 1;
        int ksize = ku - kl + 1;
        auto &x1flx = x1flux;
        pmb->par_for(
            "ReceiveFluxCorrection_x1", ll_, lu_, ml_, mu_, nl_, nu_, kl, ku, jl, ju,
            KOKKOS_LAMBDA(const int lll, const int mm, const int nn, const int k,
                          const int j) {
              const int l = lll - ll;
              const int m = mm - ml;
              const int n = nn - nl;
              // add 1 because index 0 is used for allocation flag
              const int p =
                  1 + j - jl + jsize * ((k - kl) + ksize * (n + nsize * (m + msize * l)));
              x1flx(lll, mm, nn, k, j, il) = source_allocated ? rbuf(p) : 0.0;
            });
      } else if (nb.fid == BoundaryFace::inner_x2 || nb.fid == BoundaryFace::outer_x2) {
        int jl = jb.s + (jb.e - jb.s) * (nb.fid & 1) + (nb.fid & 1);
        int il = ib.s, iu = ib.e, kl = kb.s, ku = kb.e;
        if (nb.ni.fi1 == 0)
          iu -= pmb->block_size.nx1 / 2;
        else
          il += pmb->block_size.nx1 / 2;
        if (nb.ni.fi2 == 0)
          ku -= pmb->block_size.nx3 / 2;
        else
          kl += pmb->block_size.nx3 / 2;
        int ksize = ku - kl + 1;
        int isize = iu - il + 1;
        auto &x2flx = x2flux;
        pmb->par_for(
            "ReceiveFluxCorrection_x2", ll_, lu_, ml_, mu_, nl_, nu_, kl, ku, il, iu,
            KOKKOS_LAMBDA(const int lll, const int mm, const int nn, const int k,
                          const int i) {
              const int l = lll - ll;
              const int m = mm - ml;
              const int n = nn - nl;
              // add 1 because index 0 is used for allocation flag
              const int p =
                  1 + i - il + isize * ((k - kl) + ksize * (n + nsize * (m + msize * l)));
              x2flx(lll, mm, nn, k, jl, i) = source_allocated ? rbuf(p) : 0.0;
            });
      } else if (nb.fid == BoundaryFace::inner_x3 || nb.fid == BoundaryFace::outer_x3) {
        int kl = kb.s + (kb.e - kb.s) * (nb.fid & 1) + (nb.fid & 1);
        int il = ib.s, iu = ib.e, jl = jb.s, ju = jb.e;
        if (nb.ni.fi1 == 0)
          iu -= pmb->block_size.nx1 / 2;
        else
          il += pmb->block_size.nx1 / 2;
        if (nb.ni.fi2 == 0)
          ju -= pmb->block_size.nx2 / 2;
        else
          jl += pmb->block_size.nx2 / 2;
        int jsize = ju - jl + 1;
        int isize = iu - il + 1;
        auto &x3flx = x3flux;
        pmb->par_for(
            "ReceiveFluxCorrection_x3", ll_, lu_, ml_, mu_, nl_, nu_, jl, ju, il, iu,
            KOKKOS_LAMBDA(const int lll, const int mm, const int nn, const int j,
                          const int i) {
              const int l = lll - ll;
              const int m = mm - ml;
              const int n = nn - nl;
              // add 1 because index 0 is used for allocation flag
              const int p =
                  1 + i - il + isize * ((j - jl) + jsize * (n + nsize * (m + msize * l)));
              x3flx(lll, mm, nn, kl, j, i) = source_allocated ? rbuf(p) : 0.0;
            });
      }
      bd_var_flcor_.flag[nb.bufid] = BoundaryStatus::completed;
    }
  }
  return bflag;
}

} // namespace parthenon
