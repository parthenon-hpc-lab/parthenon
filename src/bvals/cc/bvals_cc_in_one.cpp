//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
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
#include <iostream> // debug
#include <memory>
#include <string>
#include <vector>

#include "basic_types.hpp"
#include "bvals/bvals_interfaces.hpp"
#include "bvals_cc_in_one.hpp"
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

//----------------------------------------------------------------------------------------
//! \fn void cell_centered_bvars::CalcIndicesSetSame(int ox, int &s, int &e,
//                                                   const IndexRange &bounds)
//  \brief Calculate indices for SetBoundary routines for buffers on the same level

void CalcIndicesSetSame(int ox, int &s, int &e, const IndexRange &bounds) {
  if (ox == 0) {
    s = bounds.s;
    e = bounds.e;
  } else if (ox > 0) {
    s = bounds.e + 1;
    e = bounds.e + Globals::nghost;
  } else {
    s = bounds.s - Globals::nghost;
    e = bounds.s - 1;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void cell_centered_bvars::CalcIndicesSetFomCoarser(const int &ox, int &s, int &e,
//                                                         const IndexRange &bounds,
//                                                         const std::int64_t &lx,
//                                                         const int &cng,
//                                                         const bool include_dim)
//  \brief Calculate indices for SetBoundary routines for buffers from coarser levels

void CalcIndicesSetFromCoarser(const int &ox, int &s, int &e, const IndexRange &bounds,
                               const std::int64_t &lx, const int &cng,
                               const bool include_dim) {
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
}

//----------------------------------------------------------------------------------------
//! \fn void cell_centered_bvars::CalcIndicesSetFromFiner(int &si, int &ei, int &sj,
//                                                        int &ej, int &sk, int &ek,
//                                                        const NeighborBlock &nb,
//                                                        MeshBlock *pmb)
//  \brief Calculate indices for SetBoundary routines for buffers from finer levels

void CalcIndicesSetFromFiner(int &si, int &ei, int &sj, int &ej, int &sk, int &ek,
                             const NeighborBlock &nb, MeshBlock *pmb) {
  IndexDomain interior = IndexDomain::interior;
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
    ei = cellbounds.ie(interior) + Globals::nghost;
  } else {
    si = cellbounds.is(interior) - Globals::nghost;
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
    ej = cellbounds.je(interior) + Globals::nghost;
  } else {
    sj = cellbounds.js(interior) - Globals::nghost;
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
    ek = cellbounds.ke(interior) + Globals::nghost;
  } else {
    sk = cellbounds.ks(interior) - Globals::nghost;
    ek = cellbounds.ks(interior) - 1;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void cell_centered_bvars::CalcIndicesLoadSame(int ox, int &s, int &e,
//                                                    const IndexRange &bounds)
//  \brief Calculate indices for LoadBoundary routines for buffers on the same level
//         and to coarser.

void CalcIndicesLoadSame(int ox, int &s, int &e, const IndexRange &bounds) {
  if (ox == 0) {
    s = bounds.s;
    e = bounds.e;
  } else if (ox > 0) {
    s = bounds.e - Globals::nghost + 1;
    e = bounds.e;
  } else {
    s = bounds.s;
    e = bounds.s + Globals::nghost - 1;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void cell_centered_bvars::CalcIndicesLoadToFiner(int &si, int &ei, int &sj,
//                                                       int &ej, int &sk, int &ek,
//                                                       const NeighborBlock &nb,
//                                                       MeshBlock *pmb)
//  \brief Calculate indices for LoadBoundary routines for buffers to finer levels

void CalcIndicesLoadToFiner(int &si, int &ei, int &sj, int &ej, int &sk, int &ek,
                            const NeighborBlock &nb, MeshBlock *pmb) {
  int cn = pmb->cnghost - 1;

  IndexDomain interior = IndexDomain::interior;
  const IndexShape &cellbounds = pmb->cellbounds;
  si = (nb.ni.ox1 > 0) ? (cellbounds.ie(interior) - cn) : cellbounds.is(interior);
  ei = (nb.ni.ox1 < 0) ? (cellbounds.is(interior) + cn) : cellbounds.ie(interior);
  sj = (nb.ni.ox2 > 0) ? (cellbounds.je(interior) - cn) : cellbounds.js(interior);
  ej = (nb.ni.ox2 < 0) ? (cellbounds.js(interior) + cn) : cellbounds.je(interior);
  sk = (nb.ni.ox3 > 0) ? (cellbounds.ke(interior) - cn) : cellbounds.ks(interior);
  ek = (nb.ni.ox3 < 0) ? (cellbounds.ks(interior) + cn) : cellbounds.ke(interior);

  // send the data first and later prolongate on the target block
  // need to add edges for faces, add corners for edges
  if (nb.ni.ox1 == 0) {
    if (nb.ni.fi1 == 1)
      si += pmb->block_size.nx1 / 2 - pmb->cnghost;
    else
      ei -= pmb->block_size.nx1 / 2 - pmb->cnghost;
  }
  if (nb.ni.ox2 == 0 && pmb->block_size.nx2 > 1) {
    if (nb.ni.ox1 != 0) {
      if (nb.ni.fi1 == 1)
        sj += pmb->block_size.nx2 / 2 - pmb->cnghost;
      else
        ej -= pmb->block_size.nx2 / 2 - pmb->cnghost;
    } else {
      if (nb.ni.fi2 == 1)
        sj += pmb->block_size.nx2 / 2 - pmb->cnghost;
      else
        ej -= pmb->block_size.nx2 / 2 - pmb->cnghost;
    }
  }
  if (nb.ni.ox3 == 0 && pmb->block_size.nx3 > 1) {
    if (nb.ni.ox1 != 0 && nb.ni.ox2 != 0) {
      if (nb.ni.fi1 == 1)
        sk += pmb->block_size.nx3 / 2 - pmb->cnghost;
      else
        ek -= pmb->block_size.nx3 / 2 - pmb->cnghost;
    } else {
      if (nb.ni.fi2 == 1)
        sk += pmb->block_size.nx3 / 2 - pmb->cnghost;
      else
        ek -= pmb->block_size.nx3 / 2 - pmb->cnghost;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn size_t ResetSendBuffers(MeshData<Real> *md, bool cache_is_valid)
//  \brief Resets boundary variable pointer (tbd if still required) and restricts
//         cell centered variables if a cached version of boundary_info is used.
//  \return The total number of buffers used in boundary_info

size_t ResetSendBuffers(MeshData<Real> *md, bool cache_is_valid) {
  Kokkos::Profiling::pushRegion("Reset boundaries");

  size_t buffers_used = 0;
  // reset buffer and count required buffers
  for (int block = 0; block < md->NumBlocks(); block++) {
    auto &rc = md->GetBlockData(block);
    auto pmb = rc->GetBlockPointer();

    int mylevel = pmb->loc.level;
    for (auto &v : rc->GetCellVariableVector()) {
      if (v->IsAllocated() && v->IsSet(Metadata::FillGhost)) {
        v->resetBoundary();
        for (int n = 0; n < pmb->pbval->nneighbor; n++) {
          parthenon::NeighborBlock &nb = pmb->pbval->neighbor[n];
          auto *pbd_var_ = v->vbvar->GetPBdVar();
          if (pbd_var_->sflag[nb.bufid] == parthenon::BoundaryStatus::completed) continue;
          buffers_used += 1;
        }
      }
    }
  }

  Kokkos::Profiling::popRegion(); // Reset boundaries

  return buffers_used;
}

//----------------------------------------------------------------------------------------
//! \fn void ResetSendBufferBoundaryInfo(MeshData<Real> *md, size_t buffers_used)
//  \brief Reset/recreates boundary_info for send buffers from cell centered vars.
//         The new boundary_info is directly stored in the MeshData object.
//         Also handles restriction in case of AMR.

void ResetSendBufferBoundaryInfo(MeshData<Real> *md, size_t buffers_used) {
  Kokkos::Profiling::pushRegion("Create send_boundary_info");

  auto boundary_info = BufferCache_t("send_boundary_info", buffers_used);
  auto boundary_info_h = Kokkos::create_mirror_view(boundary_info);

  // we only allocate this array here, no need to initialize its values, since they will
  // be set on the device
  ParArray1D<bool> sending_nonzero_flags("sending_nonzero_flags", buffers_used);
  auto sending_nonzero_flags_h = Kokkos::create_mirror_view(sending_nonzero_flags);

  // TODO(JMM): The current method relies on an if statement in the par_for_outer.
  // Revisit later?

  // Get coarse and fine bounds. Same for all blocks.
  auto &rc = md->GetBlockData(0);
  auto pmb = rc->GetBlockPointer();
  IndexShape cellbounds = pmb->cellbounds;
  IndexShape c_cellbounds = pmb->c_cellbounds;

  auto pmesh = md->GetMeshPointer();
  bool multilevel = pmesh->multilevel;

  // now fill the buffer information
  // TODO(JL): This is not robust, we rely on the 3 nested for-loops to happen exactly the
  // same way in multiple functions
  int b = 0; // buffer index
  for (auto block = 0; block < md->NumBlocks(); block++) {
    auto &rc = md->GetBlockData(block);
    auto pmb = rc->GetBlockPointer();

    int mylevel = pmb->loc.level;
    for (auto &v : rc->GetCellVariableVector()) {
      if (v->IsAllocated() && v->IsSet(Metadata::FillGhost)) {
        for (int n = 0; n < pmb->pbval->nneighbor; n++) {
          sending_nonzero_flags_h(b) = false;

          parthenon::NeighborBlock &nb = pmb->pbval->neighbor[n];
          auto *pbd_var_ = v->vbvar->GetPBdVar();
          if (pbd_var_->sflag[nb.bufid] == parthenon::BoundaryStatus::completed) continue;

          auto &si = boundary_info_h(b).si;
          auto &ei = boundary_info_h(b).ei;
          auto &sj = boundary_info_h(b).sj;
          auto &ej = boundary_info_h(b).ej;
          auto &sk = boundary_info_h(b).sk;
          auto &ek = boundary_info_h(b).ek;
          auto &Nv = boundary_info_h(b).Nv;
          Nv = v->GetDim(4);

          boundary_info_h(b).coords = pmb->coords;
          if (multilevel) {
            boundary_info_h(b).coarse_coords = pmb->pmr->GetCoarseCoords();
          }

          IndexDomain interior = IndexDomain::interior;
          auto &var_cc = v->data;
          boundary_info_h(b).fine =
              var_cc.Get<4>(); // TODO(JMM) in general should be a loop
          if (multilevel) {
            boundary_info_h(b).coarse = v->vbvar->coarse_buf.Get<4>();
          }
          if (nb.snb.level == mylevel) {
            const parthenon::IndexShape &cellbounds = pmb->cellbounds;
            CalcIndicesLoadSame(nb.ni.ox1, si, ei, cellbounds.GetBoundsI(interior));
            CalcIndicesLoadSame(nb.ni.ox2, sj, ej, cellbounds.GetBoundsJ(interior));
            CalcIndicesLoadSame(nb.ni.ox3, sk, ek, cellbounds.GetBoundsK(interior));
            boundary_info_h(b).var = var_cc.Get<4>();

          } else if (nb.snb.level < mylevel) {
            const IndexShape &c_cellbounds = pmb->c_cellbounds;
            // "Same" logic is the same for loading to a coarse buffer, just using
            // c_cellbounds
            CalcIndicesLoadSame(nb.ni.ox1, si, ei, c_cellbounds.GetBoundsI(interior));
            CalcIndicesLoadSame(nb.ni.ox2, sj, ej, c_cellbounds.GetBoundsJ(interior));
            CalcIndicesLoadSame(nb.ni.ox3, sk, ek, c_cellbounds.GetBoundsK(interior));

            auto &coarse_buf = v->vbvar->coarse_buf;
            boundary_info_h(b).var = coarse_buf.Get<4>();
            boundary_info_h(b).restriction = true;

          } else {
            CalcIndicesLoadToFiner(si, ei, sj, ej, sk, ek, nb, pmb.get());
            boundary_info_h(b).var = var_cc.Get<4>();
          }
          // if on the same process and neighbor has this var allocated, then  fill the
          // target buffer directly
          if ((nb.snb.rank == parthenon::Globals::my_rank) &&
              v->vbvar->local_neighbor_allocated[n]) {
            auto target_block = pmb->pmy_mesh->FindMeshBlock(nb.snb.gid);
            boundary_info_h(b).buf =
                target_block->pbval->bvars.at(v->label())->GetPBdVar()->recv[nb.targetid];
          } else {
            boundary_info_h(b).buf = pbd_var_->send[nb.bufid];
          }
          b++;
        }
      }
    }
  }
  Kokkos::deep_copy(boundary_info, boundary_info_h);
  Kokkos::deep_copy(sending_nonzero_flags, sending_nonzero_flags_h);
  md->SetSendBuffers(boundary_info, sending_nonzero_flags);

  // Restrict whichever buffers need restriction.
  cell_centered_refinement::Restrict(boundary_info, cellbounds, c_cellbounds);

  Kokkos::Profiling::popRegion(); // Create send_boundary_info
}

//----------------------------------------------------------------------------------------
//! \fn void SendAndNotify(MeshData<Real> *md)
//  \brief Starts async MPI communication for neighbor MeshBlocks on different ranks and
//         sets flag to arrived for buffers on MeshBlocks on the same rank as data between
//         those has already been copied directly.

void SendAndNotify(MeshData<Real> *md) {
  Kokkos::Profiling::pushRegion("Set complete and/or start sending via MPI");

  // copy sending_nonzero_flags to host
  const auto sending_nonzero_flags = md->GetSendingNonzeroFlags();
  const auto sending_nonzero_flags_h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sending_nonzero_flags);

  struct NeighborAllocTask {
    std::string var_label;
    std::shared_ptr<MeshBlock> this_block, target_block;
    int n;
    parthenon::NeighborBlock nb;
  };

  std::vector<NeighborAllocTask> neighbor_alloc_tasks;

  // TODO(JL): This is not robust, we rely on the 3 nested for-loops to happen exactly the
  // same way in multiple functions
  int b = 0; // buffer index
  for (int block = 0; block < md->NumBlocks(); block++) {
    auto &rc = md->GetBlockData(block);
    auto pmb = rc->GetBlockPointer();

    int mylevel = pmb->loc.level;
    for (auto &v : rc->GetCellVariableVector()) {
      if (v->IsAllocated() && v->IsSet(Metadata::FillGhost)) {
        for (int n = 0; n < pmb->pbval->nneighbor; n++) {
          parthenon::NeighborBlock &nb = pmb->pbval->neighbor[n];
          auto *pbd_var_ = v->vbvar->GetPBdVar();
          if (pbd_var_->sflag[nb.bufid] == parthenon::BoundaryStatus::completed) continue;

          // if the neighbor does not have this variable allocated and we're sending
          // non-zero values, then the neighbor needs to newly allocate this variable
          bool new_neighbor_alloc =
              !v->vbvar->local_neighbor_allocated[n] &&
              sending_nonzero_flags_h( // b
                  std::min(b, static_cast<int>(sending_nonzero_flags_h.extent(0)) - 1));

          // on the same rank the data has been directly copied to the target buffer
          if (nb.snb.rank == parthenon::Globals::my_rank) {
            // TODO(?) check performance of FindMeshBlock. Could be caching from call
            // above.
            auto target_block = pmb->pmy_mesh->FindMeshBlock(nb.snb.gid);

            if (new_neighbor_alloc) {
              // printf("Block %4i is telling block %4i to allocate %s (local buf id %2i,
              // "
              //        "nb.bufid = %2i, nb.targetid = %2i)\n",
              //        pmb->gid, target_block->gid, v->label().c_str(), b, nb.bufid,
              //        nb.targetid);

              NeighborAllocTask task;
              task.var_label = v->label();
              task.this_block = pmb;
              task.target_block = target_block;
              task.n = n;
              task.nb = nb;

              neighbor_alloc_tasks.push_back(task);
            }

            if (v->vbvar->local_neighbor_allocated[n]) {
              target_block->pbval->bvars.at(v->label())->GetPBdVar()->flag[nb.targetid] =
                  parthenon::BoundaryStatus::arrived;
            }
          } else {
#ifdef MPI_PARALLEL
            PARTHENON_REQUIRE_THROWS(
                !new_neighbor_alloc,
                "New neighbor allocation with MPI is not implemented yet");
            PARTHENON_MPI_CHECK(MPI_Start(&(pbd_var_->req_send[nb.bufid])));
#endif
          }

          pbd_var_->sflag[nb.bufid] = parthenon::BoundaryStatus::completed;
          ++b;
        }
      }
    }
  }

  // process neighbor alloc tasks
  for (const auto &t : neighbor_alloc_tasks) {
    // printf("Neighbor alloc task, block %i -> %i, variable %s, neighbor idx %i\n",
    //        t.this_block->gid, t.target_block->gid, t.var_label.c_str(), t.n);

    // pointers to the variable in question for this stage on the sending and target
    // blocks
    BoundaryData<> *this_bd, *target_bd;

    // update all stages of sending block
    for (auto stage : t.this_block->meshblock_data.Stages()) {
      if (stage.first == md->StageName()) {
        // this is the current stage
        auto v = stage.second->GetCellVarPtr(t.var_label);
        this_bd = v->vbvar->GetPBdVar();

        // update boundary flags for current stage
        this_bd->sflag[t.nb.bufid] = parthenon::BoundaryStatus::completed;

        // since this is a new allocation on the neighbor block, it won't send us
        // anything, this time, so set boundary flag to arrived on sending boundary
        // variable
        this_bd->flag[t.nb.bufid] = BoundaryStatus::arrived;

        break;
      }
    }

    // allocate variable on target block
    t.target_block->AllocateSparse(t.var_label);

    // set boundary flags and update local_neighbor_allocated
    for (auto stage : t.target_block->meshblock_data.Stages()) {
      if (stage.first == md->StageName()) {
        // this is the current stage
        auto v = stage.second->GetCellVarPtr(t.var_label);
        target_bd = v->vbvar->GetPBdVar();

        for (int n = 0; n < t.target_block->pbval->nneighbor; n++) {
          // only for current stage: set send flag to completed on target block, since it
          // won't be sending anything out this cycle, and set receive flags to arrived
          // since the other neighbors don't know yet that this block allocated the
          // variable and thus won't be sending anything

          const parthenon::NeighborBlock &nb = t.target_block->pbval->neighbor[n];
          target_bd->sflag[nb.bufid] = parthenon::BoundaryStatus::completed;
          target_bd->flag[nb.bufid] = parthenon::BoundaryStatus::arrived;
        }

        break;
      }
    }

    // move copy data directly to neighbor's receiving buffer (only for current stage)
    Kokkos::deep_copy(target_bd->recv[t.nb.targetid], this_bd->send[t.nb.bufid]);
  }

  Kokkos::Profiling::popRegion(); // Set complete and/or start sending via MPI
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus SendBoundaryBuffers(std::shared_ptr<MeshData<Real>> &md)
//  \brief Fills and starts sending boundary buffers for cell centered variables for
//         all MeshBlocks contained the the MeshData object.
//  \return Complete when buffer filling is done and MPI communication started.
//          Guarantees that buffers for MeshBlocks on the same rank are done, but MPI
//          communication between ranks may still be in process.

// TODO(pgrete) should probaly be moved to the bvals or interface folders
TaskStatus SendBoundaryBuffers(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::Profiling::pushRegion("Task_SendBoundaryBuffers_MeshData");

  for (int b = 0; b < md->NumBlocks(); ++b) {
    md->GetBlockData(b)->SetLocalNeighborAllcoated();
  }

  auto boundary_info = md->GetSendBuffers();
  auto sending_nonzero_flags = md->GetSendingNonzeroFlags();
  bool have_cache = boundary_info.is_allocated();
  auto buffers_used = ResetSendBuffers(md.get(), have_cache);

  if (!have_cache || (buffers_used != boundary_info.size())) {
    ResetSendBufferBoundaryInfo(md.get(), buffers_used);
    boundary_info = md->GetSendBuffers();
    sending_nonzero_flags = md->GetSendingNonzeroFlags();
  } else {
    Kokkos::Profiling::pushRegion("Restrict boundaries");
    // Get coarse and fine bounds. Same for all blocks.
    auto &rc = md->GetBlockData(0);
    auto pmb = rc->GetBlockPointer();
    IndexShape cellbounds = pmb->cellbounds;
    IndexShape c_cellbounds = pmb->c_cellbounds;

    // Need to restrict here only if cached boundary_info is reused
    // Otherwise restriction happens when the new boundary_info is created
    cell_centered_refinement::Restrict(boundary_info, cellbounds, c_cellbounds);
    Kokkos::Profiling::popRegion(); // Reset boundaries
  }

  const Real threshold = Globals::sparse_config.allocation_threshold;

  Kokkos::parallel_for(
      "SendBoundaryBuffers",
      Kokkos::TeamPolicy<>(parthenon::DevExecSpace(), buffers_used, Kokkos::AUTO),
      KOKKOS_LAMBDA(parthenon::team_mbr_t team_member) {
        const int b = team_member.league_rank();
        const int &si = boundary_info(b).si;
        const int &ei = boundary_info(b).ei;
        const int &sj = boundary_info(b).sj;
        const int &ej = boundary_info(b).ej;
        const int &sk = boundary_info(b).sk;
        const int &ek = boundary_info(b).ek;
        const int Ni = ei + 1 - si;
        const int Nj = ej + 1 - sj;
        const int Nk = ek + 1 - sk;
        const int &Nv = boundary_info(b).Nv;
        const int NvNkNj = Nv * Nk * Nj;
        const int NkNj = Nk * Nj;

        sending_nonzero_flags(b) = false;

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange<>(team_member, NvNkNj), [&](const int idx) {
              const int v = idx / NkNj;
              int k = (idx - v * NkNj) / Nj;
              int j = idx - v * NkNj - k * Nj;
              k += sk;
              j += sj;

              Kokkos::parallel_for(
                  Kokkos::ThreadVectorRange(team_member, si, ei + 1), [&](const int i) {
                    const Real &val = boundary_info(b).var(v, k, j, i);
                    boundary_info(b).buf(i - si +
                                         Ni * (j - sj + Nj * (k - sk + Nk * v))) = val;
                    if (std::abs(val) > threshold) {
                      sending_nonzero_flags(b) = true;
                    }
                  });
            });
      });

#ifdef MPI_PARALLEL
  // Ensure buffer filling kernel finished before MPI_Start is called in the following
  Kokkos::fence();
#endif

  SendAndNotify(md.get());

  Kokkos::Profiling::popRegion(); // Task_SendBoundaryBuffers_MeshData
  // TODO(?) reintroduce sparse logic (or merge with above)
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus ReceiveBoundaryBuffers(std::shared_ptr<MeshData<Real>> &md)
//  \brief Checks for completion of communication TO receiving buffers for
//         all MeshBlocks contained the the MeshData object.
//  \return Complete when all buffers arrived or otherwise incomplete

TaskStatus ReceiveBoundaryBuffers(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::Profiling::pushRegion("Task_ReceiveBoundaryBuffers_MeshData");
  bool ret = true;
  for (int i = 0; i < md->NumBlocks(); i++) {
    auto &rc = md->GetBlockData(i);
    auto task_status = rc->ReceiveBoundaryBuffers();
    if (task_status == TaskStatus::incomplete) {
      ret = false;
    }
  }

  // TODO(?) reintroduce sparse logic (or merge with above)
  Kokkos::Profiling::popRegion(); // Task_ReceiveBoundaryBuffers_MeshData
  if (ret) return TaskStatus::complete;
  return TaskStatus::incomplete;
}

//----------------------------------------------------------------------------------------
//! \fn size_t GetNumSetFromBufers(MeshData<Real> *md)
//  \brief Returns number of receiving buffers.
auto GetNumSetFromBufers(MeshData<Real> *md) {
  Kokkos::Profiling::pushRegion("Create set_boundary_info");

  // first calculate the number of active buffers
  size_t buffers_used = 0;
  for (int block = 0; block < md->NumBlocks(); block++) {
    auto &rc = md->GetBlockData(block);
    auto pmb = rc->GetBlockPointer();
    for (auto &v : rc->GetCellVariableVector()) {
      if (v->IsAllocated() && v->IsSet(Metadata::FillGhost)) {
        for (int n = 0; n < pmb->pbval->nneighbor; n++) {
          buffers_used += 1;
        }
      }
    }
  }

  return buffers_used;
}

//----------------------------------------------------------------------------------------
//! \fn void ResetSetFromBufferBoundaryInfo(MeshData<Real> *md, size_t buffers_used)
//  \brief Reset/recreates boundary_info to fill cell centered vars from the receiving
//         buffers. The new boundary_info is directly stored in the MeshData object.

void ResetSetFromBufferBoundaryInfo(MeshData<Real> *md, size_t buffers_used) {
  Kokkos::Profiling::pushRegion("Create set_boundary_info");

  IndexDomain interior = IndexDomain::interior;

  auto boundary_info = BufferCache_t("set_boundary_info", buffers_used);
  auto boundary_info_h = Kokkos::create_mirror_view(boundary_info);
  // now fill the buffer info
  int b = 0;
  for (int block = 0; block < md->NumBlocks(); block++) {
    auto &rc = md->GetBlockData(block);
    auto pmb = rc->GetBlockPointer();

    int mylevel = pmb->loc.level;
    for (auto &v : rc->GetCellVariableVector()) {
      if (v->IsAllocated() && v->IsSet(Metadata::FillGhost)) {
        for (int n = 0; n < pmb->pbval->nneighbor; n++) {
          parthenon::NeighborBlock &nb = pmb->pbval->neighbor[n];
          auto *pbd_var_ = v->vbvar->GetPBdVar();

          auto &si = boundary_info_h(b).si;
          auto &ei = boundary_info_h(b).ei;
          auto &sj = boundary_info_h(b).sj;
          auto &ej = boundary_info_h(b).ej;
          auto &sk = boundary_info_h(b).sk;
          auto &ek = boundary_info_h(b).ek;
          auto &Nv = boundary_info_h(b).Nv;
          Nv = v->GetDim(4);

          if (nb.snb.level == mylevel) {
            const parthenon::IndexShape &cellbounds = pmb->cellbounds;
            CalcIndicesSetSame(nb.ni.ox1, si, ei, cellbounds.GetBoundsI(interior));
            CalcIndicesSetSame(nb.ni.ox2, sj, ej, cellbounds.GetBoundsJ(interior));
            CalcIndicesSetSame(nb.ni.ox3, sk, ek, cellbounds.GetBoundsK(interior));
            boundary_info_h(b).var = v->data.Get<4>();
          } else if (nb.snb.level < mylevel) {
            const IndexShape &c_cellbounds = pmb->c_cellbounds;
            const auto &cng = pmb->cnghost;
            CalcIndicesSetFromCoarser(nb.ni.ox1, si, ei,
                                      c_cellbounds.GetBoundsI(interior), pmb->loc.lx1,
                                      cng, true);
            CalcIndicesSetFromCoarser(nb.ni.ox2, sj, ej,
                                      c_cellbounds.GetBoundsJ(interior), pmb->loc.lx2,
                                      cng, pmb->block_size.nx2 > 1);
            CalcIndicesSetFromCoarser(nb.ni.ox3, sk, ek,
                                      c_cellbounds.GetBoundsK(interior), pmb->loc.lx3,
                                      cng, pmb->block_size.nx3 > 1);

            boundary_info_h(b).var = v->vbvar->coarse_buf.Get<4>();
          } else {
            CalcIndicesSetFromFiner(si, ei, sj, ej, sk, ek, nb, pmb.get());
            boundary_info_h(b).var = v->data.Get<4>();
          }
          boundary_info_h(b).buf = pbd_var_->recv[nb.bufid];
          // safe to set completed here as the kernel updating all buffers is
          // called immediately afterwards
          pbd_var_->flag[nb.bufid] = parthenon::BoundaryStatus::completed;
          b++;
        }
      }
    }
  }
  Kokkos::deep_copy(boundary_info, boundary_info_h);
  md->SetSetBuffers(boundary_info);

  Kokkos::Profiling::popRegion(); // Create set_boundary_info
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus SetBoundaries(std::shared_ptr<MeshData<Real>> &md)
//  \brief Set ghost zone data from receiving buffers for
//         all MeshBlocks contained the the MeshData object.
//  \return Complete when kernel is launched (though kernel may not be done yet)

// TODO(pgrete) should probaly be moved to the bvals or interface folders
TaskStatus SetBoundaries(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::Profiling::pushRegion("Task_SetBoundaries_MeshData");

  const auto buffers_used = GetNumSetFromBufers(md.get());

  auto boundary_info = md->GetSetBuffers();
  if (!boundary_info.is_allocated() || (buffers_used != boundary_info.size())) {
    ResetSetFromBufferBoundaryInfo(md.get(), buffers_used);
    boundary_info = md->GetSetBuffers();
  }

  Kokkos::parallel_for(
      "SetBoundaries",
      Kokkos::TeamPolicy<>(parthenon::DevExecSpace(), boundary_info.extent(0),
                           Kokkos::AUTO),
      KOKKOS_LAMBDA(parthenon::team_mbr_t team_member) {
        const int b = team_member.league_rank();
        // TODO(pgrete) profile perf implication of using reference.
        // Test in two jobs indicted a 10% difference, but were also run on diff. nodes
        const int &si = boundary_info(b).si;
        const int &ei = boundary_info(b).ei;
        const int &sj = boundary_info(b).sj;
        const int &ej = boundary_info(b).ej;
        const int &sk = boundary_info(b).sk;
        const int &ek = boundary_info(b).ek;

        const int Ni = ei + 1 - si;
        const int Nj = ej + 1 - sj;
        const int Nk = ek + 1 - sk;
        const int &Nv = boundary_info(b).Nv;

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
                    boundary_info(b).var(v, k, j, i) = boundary_info(b).buf(
                        i - si + Ni * (j - sj + Nj * (k - sk + Nk * v)));
                  });
            });
      });

  Kokkos::Profiling::popRegion(); // Task_SetBoundaries_MeshData
  // TODO(?) reintroduce sparse logic (or merge with above)
  return TaskStatus::complete;
}

} // namespace cell_centered_bvars
} // namespace parthenon
