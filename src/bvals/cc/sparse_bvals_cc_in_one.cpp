//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020 The Parthenon collaboration
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
#include <vector>

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

namespace impl {
template <class F>
void IterateBoundaries(std::shared_ptr<MeshData<Real>> &md, F func) {
  for (int block = 0; block < md->NumBlocks(); ++block) {
    auto &rc = md->GetBlockData(block);
    auto pmb = rc->GetBlockPointer();
    for (auto &v : rc->GetCellVariableVector()) {
      if (v->IsSet(Metadata::FillGhost)) {
        for (int n = 0; n < pmb->pbval->nneighbor; ++n) {
          auto &nb = pmb->pbval->neighbor[n];
          func(pmb, rc, nb, v);
        }
      }
    }
  }
}

using sp_mb_t = std::shared_ptr<MeshBlock>;
using sp_mbd_t = std::shared_ptr<MeshBlockData<Real>>;
using sp_cv_t = std::shared_ptr<CellVariable<Real>>;
using nb_t = NeighborBlock;

class WriteRegion {
 public:
  WriteRegion(std::string region_name) {
    // std::cout << "Running " << region_name << "... " << std::flush;
  }
  ~WriteRegion() {
    // std::cout << "done." << std::endl;
  }
};

} // namespace impl

using namespace impl;

// pmesh->boundary_comm_map.clear() after every remesh 
// in InitializeBlockTimeStepsAndBoundaries()
TaskStatus BuildSparseBoundaryBuffers(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::Profiling::pushRegion("Task_BuildSendBoundaryBuffers");
  Mesh *pmesh = md->GetMeshPointer();

  int isize, jsize, ksize;
  {
    WriteRegion region("domain size");
    auto &rc = md->GetBlockData(0);
    auto pmb = rc->GetBlockPointer();
    auto &cb = pmb->cellbounds;
    IndexDomain in = IndexDomain::interior;
    isize = cb.ie(in) - cb.is(in) + 1;
    jsize = cb.je(in) - cb.js(in) + 1;
    ksize = cb.ke(in) - cb.ks(in) + 1;
  }

  {
    WriteRegion region("sparse buffers");
    // Build buffers
    IterateBoundaries(md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb, const sp_cv_t v) {
      // Calculate the buffer size, this should be safe even for unsame levels
      int buf_size = (nb.ni.ox1 == 0 ? isize : Globals::nghost) *
                         (nb.ni.ox2 == 0 ? jsize : Globals::nghost) *
                         (nb.ni.ox3 == 0 ? ksize : Globals::nghost) +
                     1;

      // Add a buffer pool if one does not exist for this size
      if (pmesh->pool_map.count(buf_size) == 0) {
        pmesh->pool_map.emplace(std::make_pair(buf_size, buf_pool_t<Real>([buf_size]() {
                                                 return buf_pool_t<Real>::base_t(
                                                     "pool buffer", buf_size);
                                               })));
      }

      const int sender_id = pmb->gid;
      const int receiver_id = nb.snb.gid;
      // This tag is still pretty limiting, since 2^7 = 128
      // int tag = (nb.snb.lid << 10) | pmb->lid;
      int tag = 512 * nb.snb.lid + pmb->lid;

      const int receiver_rank = nb.snb.rank;
      const int sender_rank = Globals::my_rank;

#ifdef MPI_PARALLEL
      // TODO: Fix this to use Philipp's communicators and deal with mpi or no mpi
      const comm_t comm = pmesh->GetMPIComm(v->label() + "_sparse_comm");
#else 
     // Setting to zero is fine here since this doesn't actually get used when everything
     // is on the same rank 
     const comm_t comm = 0;
#endif
      // Build sending buffers
      pmesh->boundary_comm_map[{sender_id, receiver_id, v->label()}] =
          CommBuffer<buf_pool_t<Real>::owner_t>(
              tag, sender_rank, receiver_rank, comm,
              [pmesh, buf_size]() { return pmesh->pool_map.at(buf_size).Get(); });

      // Also build the non-local receive buffers here
      if (sender_rank != receiver_rank) {
        // int tag = (pmb->lid << 10) | nb.snb.lid;
        int tag_r = 512 * pmb->lid + nb.snb.lid;
        pmesh->boundary_comm_map[{receiver_id, sender_id, v->label()}] =
            CommBuffer<buf_pool_t<Real>::owner_t>(
                tag_r, receiver_rank, sender_rank, comm,
                [pmesh, buf_size]() { return pmesh->pool_map.at(buf_size).Get(); });
      }
    });
  }

  Kokkos::Profiling::popRegion();
  return TaskStatus::complete;
}

TaskStatus LoadAndSendSparseBoundaryBuffers(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::Profiling::pushRegion("Task_LoadAndSendBoundaryBuffers");

  Mesh *pmesh = md->GetMeshPointer();

  // Allocate channels sending from active data and then check to see if
  // if buffers have changed
  bool rebuild = false;
  int nbound = 0;
  {
    WriteRegion region("allocate send resource");
    IterateBoundaries(md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb, const sp_cv_t v) {
      PARTHENON_DEBUG_REQUIRE(
          pmesh->boundary_comm_map.count({pmb->gid, nb.snb.gid, v->label()}) > 0,
          "Boundary communicator does not exist");
      auto &buf = pmesh->boundary_comm_map[{pmb->gid, nb.snb.gid, v->label()}];

      if (v->IsAllocated()) {
        buf.Allocate();
      } else {
        buf.Free();
      }

      if (nbound < md->send_bnd_info_h.size()) {
        rebuild =
            rebuild || !UsingSameResource(
                           md->send_bnd_info_h(nbound).buf,
                           buf.buffer());
      } else {
        rebuild = true;
      }

      ++nbound;
    });
  }

  if (rebuild) {
    WriteRegion region("rebuild send_bnd_info");
    md->send_bnd_info = BufferCache_t("send_boundary_info", nbound);
    md->send_bnd_info_h = Kokkos::create_mirror_view(md->send_bnd_info);

    int iarr = 0;
    IterateBoundaries(md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb, const sp_cv_t v) {
      md->send_bnd_info_h(iarr).allocated = v->IsAllocated();
      if (v->IsAllocated()) {
        md->send_bnd_info_h(iarr) = BndInfo::Sender(pmb, nb, v);
        auto &buf = pmesh->boundary_comm_map[{pmb->gid, nb.snb.gid, v->label()}];
        md->send_bnd_info_h(iarr).buf = buf.buffer();
      }

      ++iarr;
    });
    Kokkos::deep_copy(md->send_bnd_info, md->send_bnd_info_h);
  }

  // Restrict
  {
    WriteRegion region("restrict");
    auto &rc = md->GetBlockData(0);
    auto pmb = rc->GetBlockPointer();
    IndexShape cellbounds = pmb->cellbounds;
    IndexShape c_cellbounds = pmb->c_cellbounds;
    cell_centered_refinement::Restrict(md->send_bnd_info, cellbounds, c_cellbounds);
  }

  // Load buffer data
  const Real threshold = Globals::sparse_config.allocation_threshold;
  auto &bnd_info = md->send_bnd_info;
  PARTHENON_DEBUG_REQUIRE(bnd_info.size() == nbound, "Need same size for boundary info");
  ParArray1D<bool> sending_nonzero_flags("sending_nonzero_flags", nbound);
  auto sending_nonzero_flags_h = Kokkos::create_mirror_view(sending_nonzero_flags);
  {
    WriteRegion region("launch buffer load");
    Kokkos::parallel_for(
        "SendBoundaryBuffers",
        Kokkos::TeamPolicy<>(parthenon::DevExecSpace(), nbound, Kokkos::AUTO),
        KOKKOS_LAMBDA(parthenon::team_mbr_t team_member) {
          const int b = team_member.league_rank();

          sending_nonzero_flags(b) = false;
          if (!bnd_info(b).allocated) return;

          const int &si = bnd_info(b).si;
          const int &ei = bnd_info(b).ei;
          const int &sj = bnd_info(b).sj;
          const int &ej = bnd_info(b).ej;
          const int &sk = bnd_info(b).sk;
          const int &ek = bnd_info(b).ek;
          const int Ni = ei + 1 - si;
          const int Nj = ej + 1 - sj;
          const int Nk = ek + 1 - sk;

          const int &Nv = bnd_info(b).Nv;
          const int &Nt = bnd_info(b).Nt;
          const int &Nu = bnd_info(b).Nu;

          const int NjNi = Nj * Ni;
          const int NkNjNi = Nk * NjNi;
          const int NvNkNjNi = Nv * NkNjNi;
          const int NtNvNkNjNi = Nt * NvNkNjNi;
          const int NuNtNvNkNjNi = Nu * NtNvNkNjNi;

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange<>(team_member, NuNtNvNkNjNi), [&](const int idx) {
                const int u = idx / NtNvNkNjNi;
                const int t = (idx % NtNvNkNjNi) / NvNkNjNi;
                const int v = (idx % NvNkNjNi) / NkNjNi;
                const int k = (idx % NkNjNi) / NjNi + sk;
                const int j = (idx % NjNi) / Ni + sj;
                const int i = idx % Ni + si;

                const Real val = bnd_info(b).var(u, t, v, k, j, i);
                bnd_info(b).buf(
                    i - si +
                    Ni * (j - sj + Nj * (k - sk + Nk * (v + Nv * (t + Nt * u))))) = val;
                if (std::abs(val) > threshold) sending_nonzero_flags(b) = true;
              });
        });

    // Send buffers
    Kokkos::deep_copy(sending_nonzero_flags_h, sending_nonzero_flags);
    Kokkos::fence();
  }

  {
    WriteRegion region("sending boundaries");
    int iarr = 0;
    IterateBoundaries(md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb, const sp_cv_t v) {
      auto &buf = pmesh->boundary_comm_map[{pmb->gid, nb.snb.gid, v->label()}];
      buf.IsSentGuard();
      if (sending_nonzero_flags_h(iarr))
        buf.Send();
      else
        buf.SendNull();
      ++iarr;
    });
  }

  Kokkos::Profiling::popRegion();
  return TaskStatus::complete;
}

TaskStatus ReceiveSparseBoundaryBuffers(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::Profiling::pushRegion("Task_ReceiveBoundaryBuffers");
  // WriteRegion region("Receive sparse boundary");
  bool all_received = true;
  Mesh *pmesh = md->GetMeshPointer();
  IterateBoundaries(md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb, const sp_cv_t v) {
    auto &buf = pmesh->boundary_comm_map[{nb.snb.gid, pmb->gid, v->label()}];
    all_received = all_received && buf.TryReceive();

    // Allocate variable if it is receiving actual data in any boundary
    // (the state could also be BufferState::received_null, which corresponds to no data)
    if (buf.GetState() == BufferState::received && !v->IsAllocated()) {
      pmb->AllocateSparse(v->label());
      // TODO: Need to flag this so that the array gets filled with something sensible
    }
  });

  Kokkos::Profiling::popRegion();

  if (all_received) return TaskStatus::complete;
  return TaskStatus::incomplete;
}

TaskStatus SetInternalSparseBoundaryBuffers(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::Profiling::pushRegion("Task_SetInternalBoundaries");

  Mesh *pmesh = md->GetMeshPointer();

  // Check for rebuild
  bool rebuild = false;
  int nbound = 0;
  {
    WriteRegion region("check for rebuild receive");
    IterateBoundaries(md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb, const sp_cv_t v) {
      auto &buf = pmesh->boundary_comm_map[{nb.snb.gid, pmb->gid, v->label()}];
      if (nbound < md->recv_bnd_info_h.size()) {
        rebuild =
            rebuild || !UsingSameResource(
                           md->recv_bnd_info_h(nbound).buf,
                           buf.buffer());
        if ((buf.GetState() == BufferState::received) &&
            !md->recv_bnd_info_h(nbound).allocated) {
          rebuild = true;
        }
        if ((buf.GetState() == BufferState::received_null) &&
            md->recv_bnd_info_h(nbound).allocated) {
          rebuild = true;
        }
      } else {
        rebuild = true;
      }
      ++nbound;
    });
  }

  if (rebuild) {
    WriteRegion region("rebuild receive");
    md->recv_bnd_info = BufferCache_t("recv_boundary_info", nbound);
    md->recv_bnd_info_h = Kokkos::create_mirror_view(md->recv_bnd_info);
    int iarr = 0;
    IterateBoundaries(md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb, const sp_cv_t v) {
      auto &buf = pmesh->boundary_comm_map[{nb.snb.gid, pmb->gid, v->label()}];

      if (v->IsAllocated()) md->recv_bnd_info_h(iarr) = BndInfo::Setter(pmb, nb, v);

      md->recv_bnd_info_h(iarr).buf = buf.buffer();
      if (buf.GetState() == BufferState::received) {
        md->recv_bnd_info_h(iarr).allocated = true;
        PARTHENON_DEBUG_REQUIRE(v->IsAllocated(),
                                "Variable must be allocated to receive");
      } else if (buf.GetState() == BufferState::received_null) {
        md->recv_bnd_info_h(iarr).allocated = false;
      } else {
        PARTHENON_FAIL("Buffer should be in a received state.");
      }

      ++iarr;
    });
    Kokkos::deep_copy(md->recv_bnd_info, md->recv_bnd_info_h);
  }

  // const Real threshold = Globals::sparse_config.allocation_threshold;
  auto &bnd_info = md->recv_bnd_info;
  {
    WriteRegion region("set boundary buffers");
    Kokkos::parallel_for(
        "SetBoundaryBuffers",
        Kokkos::TeamPolicy<>(parthenon::DevExecSpace(), nbound, Kokkos::AUTO),
        KOKKOS_LAMBDA(parthenon::team_mbr_t team_member) {
          const int b = team_member.league_rank();

          const int &si = bnd_info(b).si;
          const int &ei = bnd_info(b).ei;
          const int &sj = bnd_info(b).sj;
          const int &ej = bnd_info(b).ej;
          const int &sk = bnd_info(b).sk;
          const int &ek = bnd_info(b).ek;

          const int Ni = ei + 1 - si;
          const int Nj = ej + 1 - sj;
          const int Nk = ek + 1 - sk;
          const int &Nv = bnd_info(b).Nv;
          const int &Nt = bnd_info(b).Nt;
          const int &Nu = bnd_info(b).Nu;

          const int NjNi = Nj * Ni;
          const int NkNjNi = Nk * NjNi;
          const int NvNkNjNi = Nv * NkNjNi;
          const int NtNvNkNjNi = Nt * NvNkNjNi;
          const int NuNtNvNkNjNi = Nu * NtNvNkNjNi;

          // printf("b = %i Nv = %i Nk = %i Nj = %i Ni = %i size = %i allocated = %i \n",
          //     b, Nv, Nk, Nj, Ni, bnd_info(b).buf.size(), bnd_info(b).allocated);
          if (bnd_info(b).allocated) {
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange<>(team_member, NuNtNvNkNjNi), [&](const int idx) {
                  const int u = idx / NtNvNkNjNi;
                  const int t = (idx % NtNvNkNjNi) / NvNkNjNi;
                  const int v = (idx % NvNkNjNi) / NkNjNi;
                  const int k = (idx % NkNjNi) / NjNi + sk;
                  const int j = (idx % NjNi) / Ni + sj;
                  const int i = idx % Ni + si;
                  const Real val = bnd_info(b).buf(
                      i - si +
                      Ni * (j - sj + Nj * (k - sk + Nk * (v + Nv * (t + Nt * u)))));
                  bnd_info(b).var(t, u, v, k, j, i) = val;
                });
          } else if (bnd_info(b).var.size() > 0) {
            Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team_member, NuNtNvNkNjNi),
                                 [&](const int idx) {
                                   const int u = idx / NtNvNkNjNi;
                                   const int t = (idx % NtNvNkNjNi) / NvNkNjNi;
                                   const int v = (idx % NvNkNjNi) / NkNjNi;
                                   const int k = (idx % NkNjNi) / NjNi + sk;
                                   const int j = (idx % NjNi) / Ni + sj;
                                   const int i = idx % Ni + si;
                                   bnd_info(b).var(t, u, v, k, j, i) = 0.0;
                                 });
          }
        });
  }

  {
    WriteRegion region("stale");
    IterateBoundaries(md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb, const sp_cv_t v) {
      pmesh->boundary_comm_map[{nb.snb.gid, pmb->gid, v->label()}].Stale();
    });
  }
  Kokkos::Profiling::popRegion();
  return TaskStatus::complete;
}

} // namespace cell_centered_bvars
} // namespace parthenon
