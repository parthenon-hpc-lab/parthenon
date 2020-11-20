//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
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

#include <memory>
#include <string>
#include <vector>

#include "bvals_cc_in_one.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"

namespace parthenon {
namespace cell_centered_bvars {

struct BndInfo {
  bool is_used = false;
  int si = 0;
  int ei = 0;
  int sj = 0;
  int ej = 0;
  int sk = 0;
  int ek = 0;
  parthenon::ParArray1D<Real> buf; // comm buffer
  parthenon::ParArray4D<Real> var; // data variable (could also be coarse array)
};

// send boundary buffers with MeshBlockPack support
// TODO(pgrete) should probaly be moved to the bvals or interface folders
TaskStatus SendBoundaryBuffers(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::Profiling::pushRegion("SendBoundaryBuffersInOne");

  Kokkos::Profiling::pushRegion("Create bndinfo array");
  // TODO(?) talk about whether the number of buffers should be a compile time const
  const int num_buffers = 56;
  parthenon::ParArray2D<BndInfo> boundary_info("boundary_info", md->NumBlocks(),
                                               num_buffers);
  auto boundary_info_h = Kokkos::create_mirror_view(boundary_info);

  for (int b = 0; b < md->NumBlocks(); b++) {
    auto &rc = md->GetBlockData(b);
    auto pmb = rc->GetBlockPointer();

    for (auto &v : rc->GetCellVariableVector()) {
      if (v->IsSet(parthenon::Metadata::FillGhost)) {
        v->resetBoundary();
      }
    }

    int mylevel = pmb->loc.level;
    for (int n = 0; n < pmb->pbval->nneighbor; n++) {
      parthenon::NeighborBlock &nb = pmb->pbval->neighbor[n];
      // TODO(?) currently this only works for a single "Variable" per container.
      // Need to update the buffer sizes so that it matches the packed Variables.
      assert(rc->GetCellVariableVector().size() == 1);
      auto *bd_var_ = rc->GetCellVariableVector()[0]->vbvar->GetPBdVar();
      if (bd_var_->sflag[nb.bufid] == parthenon::BoundaryStatus::completed) continue;
      boundary_info_h(b, n).is_used = true;

      if (nb.snb.level == mylevel) {
        IndexDomain interior = IndexDomain::interior;
        const parthenon::IndexShape &cellbounds = pmb->cellbounds;
        boundary_info_h(b, n).si = (nb.ni.ox1 > 0)
                                       ? (cellbounds.ie(interior) - NGHOST + 1)
                                       : cellbounds.is(interior);
        boundary_info_h(b, n).ei = (nb.ni.ox1 < 0)
                                       ? (cellbounds.is(interior) + NGHOST - 1)
                                       : cellbounds.ie(interior);
        boundary_info_h(b, n).sj = (nb.ni.ox2 > 0)
                                       ? (cellbounds.je(interior) - NGHOST + 1)
                                       : cellbounds.js(interior);
        boundary_info_h(b, n).ej = (nb.ni.ox2 < 0)
                                       ? (cellbounds.js(interior) + NGHOST - 1)
                                       : cellbounds.je(interior);
        boundary_info_h(b, n).sk = (nb.ni.ox3 > 0)
                                       ? (cellbounds.ke(interior) - NGHOST + 1)
                                       : cellbounds.ks(interior);
        boundary_info_h(b, n).ek = (nb.ni.ox3 < 0)
                                       ? (cellbounds.ks(interior) + NGHOST - 1)
                                       : cellbounds.ke(interior);
      } else if (nb.snb.level < mylevel) {
        // ssize = LoadBoundaryBufferToCoarser(bd_var_.send[nb.bufid], nb);
      } else {
        // ssize = LoadBoundaryBufferToFiner(bd_var_.send[nb.bufid], nb);
      }
      // on the same process fill the target buffer directly
      if (nb.snb.rank == parthenon::Globals::my_rank) {
        auto target_block = pmb->pmy_mesh->FindMeshBlock(nb.snb.gid);
        // TODO(?) again hardcoded 0 index for single Variable
        boundary_info_h(b, n).buf =
            target_block->pbval->bvars[0]->GetPBdVar()->recv[nb.targetid];
      } else {
        boundary_info_h(b, n).buf = bd_var_->send[nb.bufid];
      }
    }
  }

  // TODO(?) track which buffers are actually used, extract subview, and only
  // copy/loop over that
  Kokkos::deep_copy(boundary_info, boundary_info_h);

  Kokkos::Profiling::popRegion(); // Create bndinfo array

  const auto NbNb = md->NumBlocks() * num_buffers;
  auto var_pack = md->PackVariables(std::vector<MetadataFlag>({Metadata::FillGhost}));
  const auto Nv = var_pack.GetDim(4);

  Kokkos::parallel_for(
      "SendBoundaryBuffers",
      Kokkos::TeamPolicy<>(parthenon::DevExecSpace(), NbNb, Kokkos::AUTO),
      KOKKOS_LAMBDA(parthenon::team_mbr_t team_member) {
        const int b = team_member.league_rank() / num_buffers;
        const int n = team_member.league_rank() - b * num_buffers;
        if (boundary_info(b, n).is_used) {
          const int si = boundary_info(b, n).si;
          const int ei = boundary_info(b, n).ei;
          const int sj = boundary_info(b, n).sj;
          const int ej = boundary_info(b, n).ej;
          const int sk = boundary_info(b, n).sk;
          const int ek = boundary_info(b, n).ek;
          const int Ni = ei + 1 - si;
          const int Nj = ej + 1 - sj;
          const int Nk = ek + 1 - sk;
          const int NvNkNj = Nv * Nk * Nj;
          const int NkNj = Nk * Nj;
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange<>(team_member, NvNkNj), [&](const int idx) {
                const int v = idx / NkNj;
                int k = (idx - v * NkNj) / Nj;
                int j = idx - v * NkNj - k * Nj;
                k += sk;
                j += sj;

                Kokkos::parallel_for(
                    Kokkos::ThreadVectorRange(team_member, si, ei + 1), [&](const int i) {
                      boundary_info(b, n).buf(i - si +
                                              Ni * (j - sj + Nj * (k - sk + Nk * v))) =
                          var_pack(b, v, k, j, i);
                    });
              });
        }
      });

  Kokkos::fence();
  Kokkos::Profiling::pushRegion("Set complete and/or start sending via MPI");
  for (int b = 0; b < md->NumBlocks(); b++) {
    auto &rc = md->GetBlockData(b);
    auto pmb = rc->GetBlockPointer();

    int mylevel = pmb->loc.level;
    for (int n = 0; n < pmb->pbval->nneighbor; n++) {
      parthenon::NeighborBlock &nb = pmb->pbval->neighbor[n];
      // TODO(?) currently this only works for a single "Variable" per container.
      // Need to update the buffer sizes so that it matches the packed Variables.
      auto *bd_var_ = rc->GetCellVariableVector()[0]->vbvar->GetPBdVar();
      if (bd_var_->sflag[nb.bufid] == parthenon::BoundaryStatus::completed) continue;

      // on the same rank the data has been directly copied to the target buffer
      if (nb.snb.rank == parthenon::Globals::my_rank) {
        // TODO(?) check performance of FindMeshBlock. Could be caching from call above.
        auto target_block = pmb->pmy_mesh->FindMeshBlock(nb.snb.gid);
        target_block->pbval->bvars[0]->GetPBdVar()->flag[nb.targetid] =
            parthenon::BoundaryStatus::arrived;
      } else {
#ifdef MPI_PARALLEL
        MPI_Start(&(bd_var_->req_send[nb.bufid]));
#endif
      }

      bd_var_->sflag[nb.bufid] = parthenon::BoundaryStatus::completed;
    }
  }
  Kokkos::Profiling::popRegion(); // Set complete and/or start sending via MPI

  Kokkos::Profiling::popRegion(); // SendBoundaryBuffersInOne
  // TODO(?) reintroduce sparse logic (or merge with above)
  return TaskStatus::complete;
}

TaskStatus ReceiveBoundaryBuffers(std::shared_ptr<MeshData<Real>> &md) {
  bool ret = true;
  for (int i = 0; i < md->NumBlocks(); i++) {
    auto &rc = md->GetBlockData(i);
    // receives the boundary
    for (auto &v : rc->GetCellVariableVector()) {
      if (!v->mpiStatus) {
        if (v->IsSet(parthenon::Metadata::FillGhost)) {
          // ret = ret & v->vbvar->ReceiveBoundaryBuffers();
          // In case we have trouble with multiple arrays causing
          // problems with task status, we should comment one line
          // above and uncomment the if block below
          v->resetBoundary();
          v->mpiStatus = v->vbvar->ReceiveBoundaryBuffers();
          ret = (ret & v->mpiStatus);
        }
      }
    }
  }

  // TODO(?) reintroduce sparse logic (or merge with above)
  if (ret) return TaskStatus::complete;
  return TaskStatus::incomplete;
}

// set boundaries from buffers with MeshBlockPack support
// TODO(pgrete) should probaly be moved to the bvals or interface folders
TaskStatus SetBoundaries(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::Profiling::pushRegion("SetBoundariesInOne");

  Kokkos::Profiling::pushRegion("Create bndinfo array");
  // TODO(?) talk about whether the number of buffers should be a compile time const
  const int num_buffers = 56;
  parthenon::ParArray2D<BndInfo> boundary_info("boundary_info", md->NumBlocks(),
                                               num_buffers);
  auto boundary_info_h = Kokkos::create_mirror_view(boundary_info);

  auto CalcIndicesSame = [](int ox, int &s, int &e, const IndexRange &bounds) {
    if (ox == 0) {
      s = bounds.s;
      e = bounds.e;
    } else if (ox > 0) {
      s = bounds.e + 1;
      e = bounds.e + NGHOST;
    } else {
      s = bounds.s - NGHOST;
      e = bounds.s - 1;
    }
  };

  auto CalcIndicesFromCoarser = [](const int &ox, int &s, int &e,
                                   const IndexRange &bounds, const std::int64_t &lx,
                                   const int &cng, const bool include_dim) {
    if (ox == 0) {
      s = bounds.s;
      e = bounds.e;
      if (include_dim) {
        if ((lx & 1LL) == 0LL) {
          e += cng;
        } else {
          s -= cng;
        }
      }
    } else if (ox > 0) {
      s = bounds.e + 1;
      e = bounds.e + cng;
    } else {
      s = bounds.s - cng;
      e = bounds.s - 1;
    }
  };

  IndexDomain interior = IndexDomain::interior;

  for (int b = 0; b < md->NumBlocks(); b++) {
    auto &rc = md->GetBlockData(b);
    auto pmb = rc->GetBlockPointer();

    int mylevel = pmb->loc.level;
    for (int n = 0; n < pmb->pbval->nneighbor; n++) {
      parthenon::NeighborBlock &nb = pmb->pbval->neighbor[n];
      // TODO(?) currently this only works for a single "Variable" per container.
      // Need to update the buffer sizes so that it matches the packed Variables.
      assert(rc->GetCellVariableVector().size() == 1);
      auto *bd_var_ = rc->GetCellVariableVector()[0]->vbvar->GetPBdVar();

      auto &si = boundary_info_h(b, n).si;
      auto &ei = boundary_info_h(b, n).ei;
      auto &sj = boundary_info_h(b, n).sj;
      auto &ej = boundary_info_h(b, n).ej;
      auto &sk = boundary_info_h(b, n).sk;
      auto &ek = boundary_info_h(b, n).ek;

      if (nb.snb.level == mylevel) {
        const parthenon::IndexShape &cellbounds = pmb->cellbounds;
        CalcIndicesSame(nb.ni.ox1, si, ei, cellbounds.GetBoundsI(interior));
        CalcIndicesSame(nb.ni.ox2, sj, ej, cellbounds.GetBoundsJ(interior));
        CalcIndicesSame(nb.ni.ox3, sk, ek, cellbounds.GetBoundsK(interior));
        boundary_info_h(b, n).var = rc->GetCellVariableVector()[0]->data.Get<4>();
      } else if (nb.snb.level < mylevel) {
        const IndexShape &c_cellbounds = pmb->c_cellbounds;
        const auto &cng = pmb->cnghost;
        CalcIndicesFromCoarser(nb.ni.ox1, si, ei, c_cellbounds.GetBoundsI(interior),
                               pmb->loc.lx1, cng, true);
        CalcIndicesFromCoarser(nb.ni.ox2, sj, ej, c_cellbounds.GetBoundsJ(interior),
                               pmb->loc.lx2, cng, pmb->block_size.nx2 > 1);
        CalcIndicesFromCoarser(nb.ni.ox3, sk, ek, c_cellbounds.GetBoundsK(interior),
                               pmb->loc.lx3, cng, pmb->block_size.nx3 > 1);

        // coarse_buf?
        boundary_info_h(b, n).var =
            rc->GetCellVariableVector()[0]->vbvar->coarse_buf.Get<4>();
      } else {
        const IndexShape &cellbounds = pmb->cellbounds;

        if (nb.ni.ox1 == 0) {
          si = cellbounds.is(interior);
          ei = cellbounds.ie(interior);
          if (nb.ni.fi1 == 1)
            si += pmb->block_size.nx1 / 2;
          else
            ei -= pmb->block_size.nx1 / 2;
        } else if (nb.ni.ox1 > 0) {
          si = cellbounds.ie(interior) + 1;
          ei = cellbounds.ie(interior) + NGHOST;
        } else {
          si = cellbounds.is(interior) - NGHOST;
          ei = cellbounds.is(interior) - 1;
        }

        if (nb.ni.ox2 == 0) {
          sj = cellbounds.js(interior);
          ej = cellbounds.je(interior);
          if (pmb->block_size.nx2 > 1) {
            if (nb.ni.ox1 != 0) {
              if (nb.ni.fi1 == 1)
                sj += pmb->block_size.nx2 / 2;
              else
                ej -= pmb->block_size.nx2 / 2;
            } else {
              if (nb.ni.fi2 == 1)
                sj += pmb->block_size.nx2 / 2;
              else
                ej -= pmb->block_size.nx2 / 2;
            }
          }
        } else if (nb.ni.ox2 > 0) {
          sj = cellbounds.je(interior) + 1;
          ej = cellbounds.je(interior) + NGHOST;
        } else {
          sj = cellbounds.js(interior) - NGHOST;
          ej = cellbounds.js(interior) - 1;
        }

        if (nb.ni.ox3 == 0) {
          sk = cellbounds.ks(interior);
          ek = cellbounds.ke(interior);
          if (pmb->block_size.nx3 > 1) {
            if (nb.ni.ox1 != 0 && nb.ni.ox2 != 0) {
              if (nb.ni.fi1 == 1)
                sk += pmb->block_size.nx3 / 2;
              else
                ek -= pmb->block_size.nx3 / 2;
            } else {
              if (nb.ni.fi2 == 1)
                sk += pmb->block_size.nx3 / 2;
              else
                ek -= pmb->block_size.nx3 / 2;
            }
          }
        } else if (nb.ni.ox3 > 0) {
          sk = cellbounds.ke(interior) + 1;
          ek = cellbounds.ke(interior) + NGHOST;
        } else {
          sk = cellbounds.ks(interior) - NGHOST;
          ek = cellbounds.ks(interior) - 1;
        }
        boundary_info_h(b, n).var = rc->GetCellVariableVector()[0]->data.Get<4>();
      }
      boundary_info_h(b, n).buf = bd_var_->recv[nb.bufid];
      assert(rc->GetCellVariableVector().size() == 1);
      boundary_info_h(b, n).is_used = true;
      // safe to set completed here as the kernel updating all buffers is
      // called immediately afterwards
      bd_var_->flag[nb.bufid] = parthenon::BoundaryStatus::completed;
    }
  }
  Kokkos::deep_copy(boundary_info, boundary_info_h);

  Kokkos::Profiling::popRegion(); // Create bndinfo array

  const auto NbNb = md->NumBlocks() * num_buffers;

  // TODO(pgrete) this var pack can probably go. Currently only needed for 4th dim.
  auto var_pack = md->PackVariables(std::vector<MetadataFlag>({Metadata::FillGhost}));
  const auto Nv = var_pack.GetDim(4);

  Kokkos::parallel_for(
      "SetBoundaries",
      Kokkos::TeamPolicy<>(parthenon::DevExecSpace(), NbNb, Kokkos::AUTO),
      KOKKOS_LAMBDA(parthenon::team_mbr_t team_member) {
        const int b = team_member.league_rank() / num_buffers;
        const int n = team_member.league_rank() - b * num_buffers;
        if (boundary_info(b, n).is_used) {
          const int si = boundary_info(b, n).si;
          const int ei = boundary_info(b, n).ei;
          const int sj = boundary_info(b, n).sj;
          const int ej = boundary_info(b, n).ej;
          const int sk = boundary_info(b, n).sk;
          const int ek = boundary_info(b, n).ek;
          const int Ni = ei + 1 - si;
          const int Nj = ej + 1 - sj;
          const int Nk = ek + 1 - sk;
          const int NvNkNj = Nv * Nk * Nj;
          const int NkNj = Nk * Nj;
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange<>(team_member, NvNkNj), [&](const int idx) {
                const int v = idx / NkNj;
                int k = (idx - v * NkNj) / Nj;
                int j = idx - v * NkNj - k * Nj;
                k += sk;
                j += sj;

                Kokkos::parallel_for(
                    Kokkos::ThreadVectorRange(team_member, si, ei + 1), [&](const int i) {
                      boundary_info(b, n).var(v, k, j, i) = boundary_info(b, n).buf(
                          i - si + Ni * (j - sj + Nj * (k - sk + Nk * v)));
                    });
              });
        }
      });

  Kokkos::Profiling::popRegion(); // SetBoundariesInOne
  // TODO(?) reintroduce sparse logic (or merge with above)
  return TaskStatus::complete;
}

} // namespace cell_centered_bvars
} // namespace parthenon
