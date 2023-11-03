//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2023 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2023. Triad National Security, LLC. All rights reserved.
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
//! \file mesh_amr.cpp
//  \brief implementation of Mesh::AdaptiveMeshRefinement() and related utilities

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>

#include "parthenon_mpi.hpp"

#include "bvals/boundary_conditions.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "interface/update.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/meshblock_tree.hpp"
#include "parthenon_arrays.hpp"
#include "utils/buffer_utils.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

#ifdef MPI_PARALLEL
//----------------------------------------------------------------------------------------
//! \fn int CreateAMRMPITag(int lid, int ox1, int ox2, int ox3)
//  \brief calculate an MPI tag for AMR block transfer
// tag = local id of destination (remaining bits) + ox1(1 bit) + ox2(1 bit) + ox3(1 bit)
//       + physics(5 bits)

// See comments on BoundaryBase::CreateBvalsMPITag()

int CreateAMRMPITag(int lid, int ox1, int ox2, int ox3) {
  // the trailing zero is used as "id" to indicate an AMR related tag
  return (lid << 8) | (ox1 << 7) | (ox2 << 6) | (ox3 << 5) | 0;
}

MPI_Request SendCoarseToFine(int lid_recv, int dest_rank, const LogicalLocation &fine_loc,
                             Variable<Real> *var, Mesh *pmesh) {
  MPI_Request req;
  MPI_Comm comm = pmesh->GetMPIComm(var->label());

  const int ox1 = ((fine_loc.lx1() & 1LL) == 1LL);
  const int ox2 = ((fine_loc.lx2() & 1LL) == 1LL);
  const int ox3 = ((fine_loc.lx3() & 1LL) == 1LL);

  int tag = CreateAMRMPITag(lid_recv, ox1, ox2, ox3);
  if (var->IsAllocated()) {
    PARTHENON_MPI_CHECK(MPI_Isend(var->data.data(), var->data.size(), MPI_PARTHENON_REAL,
                                  dest_rank, tag, comm, &req));
  } else {
    PARTHENON_MPI_CHECK(
        MPI_Isend(var->data.data(), 0, MPI_PARTHENON_REAL, dest_rank, tag, comm, &req));
  }
  return req;
}
#endif

bool TryRecvCoarseToFine(int lid_recv, int send_rank, const LogicalLocation &fine_loc,
                         Variable<Real> *var_in, Variable<Real> *var, MeshBlock *pmb,
                         Mesh *pmesh) {
  const int ox1 = ((fine_loc.lx1() & 1LL) == 1LL);
  const int ox2 = ((fine_loc.lx2() & 1LL) == 1LL);
  const int ox3 = ((fine_loc.lx3() & 1LL) == 1LL);

  int test = 1;
#ifdef MPI_PARALLEL
  MPI_Comm comm = pmesh->GetMPIComm(var->label());
  int tag = CreateAMRMPITag(lid_recv, ox1, ox2, ox3);
  MPI_Status status;
  PARTHENON_MPI_CHECK(MPI_Iprobe(send_rank, tag, comm, &test, &status));
#endif
  if (test) {
    int size = var_in->IsAllocated();
#ifdef MPI_PARALLEL
    PARTHENON_MPI_CHECK(MPI_Get_count(&status, MPI_PARTHENON_REAL, &size));
#endif
    if (size > 0) {
      if (!pmb->IsAllocated(var->label())) pmb->AllocateSparse(var->label());
      auto fb = var_in->data;
#ifdef MPI_PARALLEL
      PARTHENON_MPI_CHECK(MPI_Recv(var->data.data(), var->data.size(), MPI_PARTHENON_REAL,
                                   send_rank, tag, comm, MPI_STATUS_IGNORE));
      fb = var->data;
#endif
      auto cb = var->coarse_s;
      const int nt = fb.GetDim(6) - 1;
      const int nu = fb.GetDim(5) - 1;
      const int nv = fb.GetDim(4) - 1;

      for (auto te : var->GetTopologicalElements()) {
        IndexRange ib = pmb->c_cellbounds.GetBoundsI(IndexDomain::entire, te);
        IndexRange jb = pmb->c_cellbounds.GetBoundsJ(IndexDomain::entire, te);
        IndexRange kb = pmb->c_cellbounds.GetBoundsK(IndexDomain::entire, te);

        IndexRange ib_int = pmb->cellbounds.GetBoundsI(IndexDomain::interior, te);
        IndexRange jb_int = pmb->cellbounds.GetBoundsJ(IndexDomain::interior, te);
        IndexRange kb_int = pmb->cellbounds.GetBoundsK(IndexDomain::interior, te);

        const int ks = (ox3 == 0) ? 0 : (kb_int.e - kb_int.s + 1) / 2;
        const int js = (ox2 == 0) ? 0 : (jb_int.e - jb_int.s + 1) / 2;
        const int is = (ox1 == 0) ? 0 : (ib_int.e - ib_int.s + 1) / 2;
        const int idx_te = static_cast<int>(te) % 3;
        parthenon::par_for(
            DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), 0, nt, 0, nu, 0,
            nv, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA(const int t, const int u, const int v, const int k, const int j,
                          const int i) {
              cb(idx_te, t, u, v, k, j, i) = fb(idx_te, t, u, v, k + ks, j + js, i + is);
            });
      }
    } else {
      if (pmb->IsAllocated(var->label()) &&
          !var->metadata().IsSet(Metadata::ForceAllocOnNewBlocks))
        pmb->DeallocateSparse(var->label());
#ifdef MPI_PARALLEL
      PARTHENON_MPI_CHECK(MPI_Recv(var->data.data(), 0, MPI_PARTHENON_REAL, send_rank,
                                   tag, comm, MPI_STATUS_IGNORE));
#endif
    }
  }

  return test;
}

#ifdef MPI_PARALLEL
MPI_Request SendFineToCoarse(int lid_recv, int dest_rank, const LogicalLocation &fine_loc,
                             Variable<Real> *var, Mesh *pmesh) {
  MPI_Request req;
  MPI_Comm comm = pmesh->GetMPIComm(var->label());

  const int ox1 = ((fine_loc.lx1() & 1LL) == 1LL);
  const int ox2 = ((fine_loc.lx2() & 1LL) == 1LL);
  const int ox3 = ((fine_loc.lx3() & 1LL) == 1LL);

  int tag = CreateAMRMPITag(lid_recv, ox1, ox2, ox3);
  if (var->IsAllocated()) {
    PARTHENON_MPI_CHECK(MPI_Isend(var->coarse_s.data(), var->coarse_s.size(),
                                  MPI_PARTHENON_REAL, dest_rank, tag, comm, &req));
  } else {
    PARTHENON_MPI_CHECK(MPI_Isend(var->coarse_s.data(), 0, MPI_PARTHENON_REAL, dest_rank,
                                  tag, comm, &req));
  }
  return req;
}
#endif

bool TryRecvFineToCoarse(int lid_recv, int send_rank, const LogicalLocation &fine_loc,
                         Variable<Real> *var_in, Variable<Real> *var, MeshBlock *pmb,
                         Mesh *pmesh) {
  const int ox1 = ((fine_loc.lx1() & 1LL) == 1LL);
  const int ox2 = ((fine_loc.lx2() & 1LL) == 1LL);
  const int ox3 = ((fine_loc.lx3() & 1LL) == 1LL);

  int test = 1;
#ifdef MPI_PARALLEL
  MPI_Comm comm = pmesh->GetMPIComm(var->label());
  int tag = CreateAMRMPITag(lid_recv, ox1, ox2, ox3);
  MPI_Status status;
  PARTHENON_MPI_CHECK(MPI_Iprobe(send_rank, tag, comm, &test, &status));
#endif
  if (test) {
    int size = var_in->IsAllocated();
#ifdef MPI_PARALLEL
    PARTHENON_MPI_CHECK(MPI_Get_count(&status, MPI_PARTHENON_REAL, &size));
#endif
    if (size > 0) {
      if (!pmb->IsAllocated(var->label())) pmb->AllocateSparse(var->label());
      auto cb = var_in->coarse_s;
#ifdef MPI_PARALLEL
      // This has to be an MPI_Recv w/o buffering
      PARTHENON_MPI_CHECK(MPI_Recv(var->coarse_s.data(), var->coarse_s.size(),
                                   MPI_PARTHENON_REAL, send_rank, tag, comm,
                                   MPI_STATUS_IGNORE));
      cb = var->coarse_s;
#endif
      auto fb = var->data;
      const int nt = fb.GetDim(6) - 1;
      const int nu = fb.GetDim(5) - 1;
      const int nv = fb.GetDim(4) - 1;

      for (auto te : var->GetTopologicalElements()) {
        IndexRange ib = pmb->c_cellbounds.GetBoundsI(IndexDomain::interior, te);
        IndexRange jb = pmb->c_cellbounds.GetBoundsJ(IndexDomain::interior, te);
        IndexRange kb = pmb->c_cellbounds.GetBoundsK(IndexDomain::interior, te);
        // Deal with ownership of shared elements by removing right side of index
        // space if fine block is on the left side of a direction. I think this
        // should work fine even if the ownership model is changed elsewhere, since
        // the fine blocks should be consistent in their shared elements at this point
        if (ox3 == 0) kb.e -= TopologicalOffsetK(te);
        if (ox2 == 0) jb.e -= TopologicalOffsetJ(te);
        if (ox1 == 0) ib.e -= TopologicalOffsetI(te);
        const int ks = (ox3 == 0) ? 0 : (kb.e - kb.s + 1 - TopologicalOffsetK(te));
        const int js = (ox2 == 0) ? 0 : (jb.e - jb.s + 1 - TopologicalOffsetJ(te));
        const int is = (ox1 == 0) ? 0 : (ib.e - ib.s + 1 - TopologicalOffsetI(te));
        const int idx_te = static_cast<int>(te) % 3;
        parthenon::par_for(
            DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), 0, nt, 0, nu, 0,
            nv, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA(const int t, const int u, const int v, const int k, const int j,
                          const int i) {
              fb(idx_te, t, u, v, k + ks, j + js, i + is) = cb(idx_te, t, u, v, k, j, i);
            });
      }
      // We have to block here w/o buffering so that the write is guaranteed to be
      // finished before another fine block that is restricted to a sub-region of
      // this coarse block makes an MPI call and overwrites the coarse buffer.
      Kokkos::fence();
    } else {
#ifdef MPI_PARALLEL
      PARTHENON_MPI_CHECK(MPI_Recv(var->data.data(), 0, MPI_PARTHENON_REAL, send_rank,
                                   tag, comm, MPI_STATUS_IGNORE));
#endif
    }
  }

  return test;
}

#ifdef MPI_PARALLEL
MPI_Request SendSameToSame(int lid_recv, int dest_rank, Variable<Real> *var,
                           MeshBlock *pmb, Mesh *pmesh) {
  MPI_Request req;
  MPI_Comm comm = pmesh->GetMPIComm(var->label());
  int tag = CreateAMRMPITag(lid_recv, 0, 0, 0);
  if (var->IsAllocated()) {
    // Metadata about this field also needs to be copied from this rank to the
    // receiving rank (namely the dereference count and the dealloc_count). Not
    // doing this can cause some subtle differences between runs of the same
    // problem on different numbers of ranks. Rather than send individual messages,
    // we just package these in the first two ghost zones of the block. This is ok
    // because the AMR communication is only required to communicate the interior
    // zones. The ghosts are communicated to prevent the need to allocate buffers,
    // but the data in them is not required to be valid. This requires that we have
    // at least two ghost zones.
    PARTHENON_REQUIRE(
        Globals::nghost > 1,
        "AMR SameToSame communication requires blocks to have at least two ghost zones");
    auto counter_subview = Kokkos::subview(var->data, std::make_pair(0, 2));
    auto counter_subview_h = Kokkos::create_mirror_view(HostMemSpace(), counter_subview);
    counter_subview_h(0) = pmb->pmr->DereferenceCount();
    counter_subview_h(1) = var->dealloc_count;
    Kokkos::deep_copy(counter_subview, counter_subview_h);

    PARTHENON_MPI_CHECK(MPI_Isend(var->data.data(), var->data.size(), MPI_PARTHENON_REAL,
                                  dest_rank, tag, comm, &req));
  } else {
    var->com_state[0] = pmb->pmr->DereferenceCount();
    var->com_state[1] = var->dealloc_count;
    PARTHENON_MPI_CHECK(
        MPI_Isend(var->com_state, 2, MPI_INT, dest_rank, tag, comm, &req));
  }
  return req;
}

bool TryRecvSameToSame(int lid_recv, int send_rank, Variable<Real> *var, MeshBlock *pmb,
                       Mesh *pmesh) {
  MPI_Comm comm = pmesh->GetMPIComm(var->label());
  int tag = CreateAMRMPITag(lid_recv, 0, 0, 0);

  int test;
  MPI_Status status;
  PARTHENON_MPI_CHECK(MPI_Iprobe(send_rank, tag, comm, &test, &status));
  if (test) {
    int size;
    PARTHENON_MPI_CHECK(MPI_Get_count(&status, MPI_PARTHENON_REAL, &size));
    if (size > 2) {
      if (!pmb->IsAllocated(var->label())) pmb->AllocateSparse(var->label());
      PARTHENON_MPI_CHECK(MPI_Recv(var->data.data(), var->data.size(), MPI_PARTHENON_REAL,
                                   send_rank, tag, comm, MPI_STATUS_IGNORE));
      auto counter_subview = Kokkos::subview(var->data, std::make_pair(0, 2));
      auto counter_subview_h =
          Kokkos::create_mirror_view_and_copy(HostMemSpace(), counter_subview);
      pmb->pmr->DereferenceCount() = counter_subview_h(0);
      var->dealloc_count = counter_subview_h(1);
    } else {
      if (pmb->IsAllocated(var->label()) &&
          !var->metadata().IsSet(Metadata::ForceAllocOnNewBlocks))
        pmb->DeallocateSparse(var->label());
      PARTHENON_MPI_CHECK(
          MPI_Recv(var->com_state, 2, MPI_INT, send_rank, tag, comm, MPI_STATUS_IGNORE));
      pmb->pmr->DereferenceCount() = var->com_state[0];
      var->dealloc_count = var->com_state[1];
    }
  }
  return test;
}
#endif

//----------------------------------------------------------------------------------------
// \!fn void Mesh::LoadBalancingAndAdaptiveMeshRefinement(ParameterInput *pin)
// \brief Main function for adaptive mesh refinement

void Mesh::LoadBalancingAndAdaptiveMeshRefinement(ParameterInput *pin,
                                                  ApplicationInput *app_in) {
  PARTHENON_INSTRUMENT
  int nnew = 0, ndel = 0;

  if (adaptive) {
    UpdateMeshBlockTree(nnew, ndel);
    nbnew += nnew;
    nbdel += ndel;
  }

  modified = false;
  if (nnew != 0 || ndel != 0) { // at least one (de)refinement happened
    modified = RedistributeAndRefineMeshBlocks(pin, app_in, nbtotal + nnew - ndel, true);
  } else if (step_since_lb >= lb_interval_) {
    modified = RedistributeAndRefineMeshBlocks(pin, app_in, nbtotal, false);
  }
}

// Private routines
namespace {
Real DistributeTrial(std::vector<Real> const &cost, std::vector<int> &start,
                     std::vector<int> &nb, const int nranks, Real target_max) {
  Real trial_max = 0.0;
  const int nblocks = cost.size();
  int nleft = nblocks;
  int last = 0;
  for (int rank = 0; rank < nranks; rank++) {
    const int ranks_left = nranks - rank;
    start[rank] = last;
    Real rank_cost = cost[last];
    nleft--;
    while (nleft > ranks_left && rank_cost + cost[last + 1] < target_max) {
      last += 1;
      rank_cost += cost[last];
      nleft--;
    }
    last++;
    if (rank == nranks - 1) {
      for (int n = last; n < nblocks; n++) {
        rank_cost += cost[n];
      }
      last = nblocks;
    }
    nb[rank] = last - start[rank];
    trial_max = std::max(trial_max, rank_cost);
  }
  return trial_max;
}

double CalculateNewBalance(std::vector<double> const &cost, std::vector<int> &start,
                           std::vector<int> &nb, const double avg_cost,
                           const double max_block_cost) {
  PARTHENON_INSTRUMENT
  const int nblocks = cost.size();
  const int max_rank = std::min(nblocks, Globals::nranks);

  for (int i = max_rank; i < Globals::nranks; i++) {
    start[i] = -1;
    nb[i] = 0;
  }

  // set the bounds for the search
  double a = avg_cost;
  double b = std::max(2.0 * avg_cost, 1.01 * max_block_cost);
  double h = b - a;
  constexpr double invphi = 0.618033988749894848204586834366;
  constexpr double invphi2 = 0.38196601125010515179541316563436188228;
  double c = a + invphi2 * h;
  double d = a + invphi * h;
  double yc = DistributeTrial(cost, start, nb, max_rank, c);
  double yd = DistributeTrial(cost, start, nb, max_rank, d);

  while (yc != yd) {
    if (yc < yd) {
      b = d;
      d = c;
      yd = yc;
      h = invphi * h;
      c = a + invphi2 * h;
      yc = DistributeTrial(cost, start, nb, max_rank, c);
    } else {
      a = c;
      c = d;
      yc = yd;
      h = invphi * h;
      d = a + invphi * h;
      yd = DistributeTrial(cost, start, nb, max_rank, d);
    }
  }

  return yc;
}

void AssignBlocks(const std::vector<int> &start, const std::vector<int> &nb,
                  std::vector<int> &rank) {
  const int nblocks = rank.size();
  for (int i = 0; i < std::min(nblocks, Globals::nranks); i++) {
    for (int n = start[i]; n < start[i] + nb[i]; n++) {
      rank[n] = i;
    }
  }
}

std::tuple<double, double, double> BlockCostInfo(std::vector<double> const &cost,
                                                 std::vector<int> const &start,
                                                 std::vector<int> const &nb) {
  const int nblocks = cost.size();
  const int max_rank = std::min(Globals::nranks, nblocks);
  double avg_cost = 0.0;
  double max_block_cost = 0.0;
  double max_rank_cost = 0.0;
  for (int i = 0; i < max_rank; i++) {
    double rank_cost = 0.0;
    for (int b = start[i]; b < start[i] + nb[i]; b++) {
      rank_cost += cost[b];
      avg_cost += cost[b];
      max_block_cost = std::max(max_block_cost, cost[b]);
    }
    max_rank_cost = std::max(max_rank_cost, rank_cost);
  }
  avg_cost /= max_rank;
  return std::make_tuple(avg_cost, max_block_cost, max_rank_cost);
}

} // namespace

void Mesh::SetSimpleBalance(const int nblocks, std::vector<int> &start,
                            std::vector<int> &nb) {
  const int max_rank = std::min(nblocks, Globals::nranks);
  int nassign = nblocks / max_rank;
  start[0] = 0;
  nb[0] = nassign;
  int nassigned = nassign;
  for (int i = 1; i < max_rank; i++) {
    nassign = (nblocks - nassigned) / (max_rank - i);
    start[i] = start[i - 1] + nb[i - 1];
    nb[i] = nassign;
    nassigned += nassign;
  }
  for (int i = max_rank; i < Globals::nranks; i++) {
    start[i] = 0;
    nb[i] = 0;
  }
}

void Mesh::CalculateLoadBalance(std::vector<double> const &cost, std::vector<int> &rank,
                                std::vector<int> &start, std::vector<int> &nb) {
  PARTHENON_INSTRUMENT
  if ((lb_automatic_ || lb_manual_)) {
    SetSimpleBalance(cost.size(), start, nb);
    auto [avg_cost, max_block_cost, max_rank_cost] = BlockCostInfo(cost, start, nb);
    double new_max = CalculateNewBalance(cost, start, nb, avg_cost, max_block_cost);
  } else {
    // just try to distribute blocks evenly
    SetSimpleBalance(cost.size(), start, nb);
  }
  AssignBlocks(start, nb, rank);
  // now assign blocks to ranks
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::ResetLoadBalanceVariables()
// \brief reset counters and flags for load balancing

void Mesh::ResetLoadBalanceVariables() {
  if (lb_automatic_) {
#ifdef ENABLE_LB_TIMERS
    // make a local copy to make CUDA happy
    auto bcost = block_cost;
    parthenon::par_for(
        loop_pattern_flatrange_tag, "reset cost_d", DevExecSpace(), 0,
        block_list.size() - 1, KOKKOS_LAMBDA(const int b) { bcost(b) = TINY_NUMBER; });
    for (int b = 0; b < block_list.size(); b++)
      block_cost_host[b] = TINY_NUMBER;
#endif
  } else if (lb_manual_) {
    for (int b = 0; b < block_list.size(); b++) {
      block_cost_host[b] = TINY_NUMBER;
    }
  }
  step_since_lb = 0;
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::UpdateMeshBlockTree(int &nnew, int &ndel)
// \brief collect refinement flags and manipulate the MeshBlockTree

void Mesh::UpdateMeshBlockTree(int &nnew, int &ndel) {
  PARTHENON_INSTRUMENT
  // compute nleaf= number of leaf MeshBlocks per refined block
  int nleaf = 2;
  if (!mesh_size.symmetry(X2DIR)) nleaf = 4;
  if (!mesh_size.symmetry(X3DIR)) nleaf = 8;

  // collect refinement flags from all the meshblocks
  // count the number of the blocks to be (de)refined
  nref[Globals::my_rank] = 0;
  nderef[Globals::my_rank] = 0;
  for (auto const &pmb : block_list) {
    if (pmb->pmr->refine_flag_ == 1) nref[Globals::my_rank]++;
    if (pmb->pmr->refine_flag_ == -1) nderef[Globals::my_rank]++;
  }
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(
      MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, nref.data(), 1, MPI_INT, MPI_COMM_WORLD));
  PARTHENON_MPI_CHECK(
      MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, nderef.data(), 1, MPI_INT, MPI_COMM_WORLD));
#endif

  // count the number of the blocks to be (de)refined and displacement
  int tnref = 0, tnderef = 0;
  for (int n = 0; n < Globals::nranks; n++) {
    tnref += nref[n];
    tnderef += nderef[n];
  }
  if (tnref == 0 && tnderef < nleaf) { // nothing to do
    return;
  }

  int rd = 0, dd = 0;
  for (int n = 0; n < Globals::nranks; n++) {
    rdisp[n] = rd;
    ddisp[n] = dd;
    // technically could overflow, since sizeof() operator returns
    // std::size_t = long unsigned int > int
    // on many platforms (LP64). However, these are used below in MPI calls for
    // integer arguments (recvcounts, displs). MPI does not support > 64-bit count ranges
    bnref[n] = static_cast<int>(nref[n] * sizeof(LogicalLocation));
    bnderef[n] = static_cast<int>(nderef[n] * sizeof(LogicalLocation));
    brdisp[n] = static_cast<int>(rd * sizeof(LogicalLocation));
    bddisp[n] = static_cast<int>(dd * sizeof(LogicalLocation));
    rd += nref[n];
    dd += nderef[n];
  }

  // allocate memory for the location arrays
  LogicalLocation *lref{}, *lderef{}, *clderef{};
  if (tnref > 0) lref = new LogicalLocation[tnref];
  if (tnderef >= nleaf) {
    lderef = new LogicalLocation[tnderef];
    clderef = new LogicalLocation[tnderef / nleaf];
  }

  // collect the locations and costs
  int iref = rdisp[Globals::my_rank], ideref = ddisp[Globals::my_rank];
  for (auto const &pmb : block_list) {
    if (pmb->pmr->refine_flag_ == 1) lref[iref++] = pmb->loc;
    if (pmb->pmr->refine_flag_ == -1 && tnderef >= nleaf) lderef[ideref++] = pmb->loc;
  }
#ifdef MPI_PARALLEL
  if (tnref > 0) {
    PARTHENON_MPI_CHECK(MPI_Allgatherv(MPI_IN_PLACE, bnref[Globals::my_rank], MPI_BYTE,
                                       lref, bnref.data(), brdisp.data(), MPI_BYTE,
                                       MPI_COMM_WORLD));
  }
  if (tnderef >= nleaf) {
    PARTHENON_MPI_CHECK(MPI_Allgatherv(MPI_IN_PLACE, bnderef[Globals::my_rank], MPI_BYTE,
                                       lderef, bnderef.data(), bddisp.data(), MPI_BYTE,
                                       MPI_COMM_WORLD));
  }
#endif

  // calculate the list of the newly derefined blocks
  int ctnd = 0;
  if (tnderef >= nleaf) {
    int lk = 0, lj = 0;
    if (!mesh_size.symmetry(X2DIR)) lj = 1;
    if (!mesh_size.symmetry(X3DIR)) lk = 1;
    for (int n = 0; n < tnderef; n++) {
      if ((lderef[n].lx1() & 1LL) == 0LL && (lderef[n].lx2() & 1LL) == 0LL &&
          (lderef[n].lx3() & 1LL) == 0LL) {
        int r = n, rr = 0;
        for (std::int64_t k = 0; k <= lk; k++) {
          for (std::int64_t j = 0; j <= lj; j++) {
            for (std::int64_t i = 0; i <= 1; i++) {
              if (r < tnderef) {
                if ((lderef[n].lx1() + i) == lderef[r].lx1() &&
                    (lderef[n].lx2() + j) == lderef[r].lx2() &&
                    (lderef[n].lx3() + k) == lderef[r].lx3() &&
                    lderef[n].level() == lderef[r].level())
                  rr++;
                r++;
              }
            }
          }
        }
        if (rr == nleaf) {
          clderef[ctnd] = lderef[n].GetParent();
          ctnd++;
        }
      }
    }
  }
  // sort the lists by level
  if (ctnd > 1) {
    std::sort(clderef, &(clderef[ctnd - 1]),
              [](const LogicalLocation &left, const LogicalLocation &right) {
                return left.level() > right.level();
              });
  }
  if (tnderef >= nleaf) delete[] lderef;

  // Now the lists of the blocks to be refined and derefined are completed
  // Start tree manipulation
  // Step 1. perform refinement
  for (int n = 0; n < tnref; n++) {
    MeshBlockTree *bt = tree.FindMeshBlock(lref[n]);
    bt->Refine(nnew);
  }
  if (tnref != 0) delete[] lref;

  // Step 2. perform derefinement
  for (int n = 0; n < ctnd; n++) {
    MeshBlockTree *bt = tree.FindMeshBlock(clderef[n]);
    bt->Derefine(ndel);
  }
  if (tnderef >= nleaf) delete[] clderef;
}

//----------------------------------------------------------------------------------------
// \!fn bool Mesh::GatherCostListAndCheckBalance()
// \brief collect the cost from MeshBlocks and check the load balance

void Mesh::GatherCostList() {
  PARTHENON_INSTRUMENT
  if (lb_automatic_) {
#ifdef ENABLE_LB_TIMERS
    auto cost_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), block_cost);
    for (int b = 0; b < block_cost_host.size(); b++)
      cost_h(b) += block_cost_host[b];
#ifdef MPI_PARALLEL
    PARTHENON_MPI_CHECK(MPI_Allgatherv(cost_h.data(), nblist[Globals::my_rank],
                                       MPI_DOUBLE, costlist.data(), nblist.data(),
                                       nslist.data(), MPI_DOUBLE, MPI_COMM_WORLD));
#endif
#endif
  }
  if (lb_manual_) {
#ifdef MPI_PARALLEL
    PARTHENON_MPI_CHECK(MPI_Allgatherv(block_cost_host.data(), nblist[Globals::my_rank],
                                       MPI_DOUBLE, costlist.data(), nblist.data(),
                                       nslist.data(), MPI_DOUBLE, MPI_COMM_WORLD));
#endif
  }
  return;
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::RedistributeAndRefineMeshBlocks(ParameterInput *pin, int ntot)
// \brief redistribute MeshBlocks according to the new load balance

bool Mesh::RedistributeAndRefineMeshBlocks(ParameterInput *pin, ApplicationInput *app_in,
                                           int ntot, bool modified) {
  PARTHENON_INSTRUMENT

  GatherCostList();
  // store old things
  const int onbs = nslist[Globals::my_rank];
  const int onbe = onbs + nblist[Globals::my_rank] - 1;
  const int nbtold = nbtotal;

  std::vector<int> newrank;
  if (!modified) {
    // The mesh hasn't actually changed.  Let's just check the load balancing
    // and only move things around if needed
    if (lb_automatic_ || lb_manual_) {
      PARTHENON_INSTRUMENT
      auto [avg_cost, max_block_cost, max_rank_cost] =
          BlockCostInfo(costlist, nslist, nblist);
      std::vector<int> start_trial(Globals::nranks);
      std::vector<int> nb_trial(Globals::nranks);
      double new_max =
          CalculateNewBalance(costlist, start_trial, nb_trial, avg_cost, max_block_cost);
      // if the improvement isn't large enough, just return because we're done
      if ((max_rank_cost - new_max) / max_rank_cost < lb_tolerance_) {
        ResetLoadBalanceVariables();
        return false;
      }
      newrank.resize(ntot);
      AssignBlocks(start_trial, nb_trial, newrank);
      nslist = std::move(start_trial);
      nblist = std::move(nb_trial);
    } else {
      // default balancing on number of meshblocks should be good to go since
      // the mesh hasn't changed
      return false;
    }
  }

  // if we got here, we're going to be changing or moving the mesh around

  // kill any cached packs
  mesh_data.PurgeNonBase();
  mesh_data.Get()->ClearCaches();

  // compute nleaf= number of leaf MeshBlocks per refined block
  int nleaf = 2;
  if (!mesh_size.symmetry(X2DIR)) nleaf = 4;
  if (!mesh_size.symmetry(X3DIR)) nleaf = 8;

  // construct new lists
  std::vector<LogicalLocation> newloc(ntot);
  if (newrank.size() == 0) newrank.resize(ntot);
  std::vector<double> newcost(ntot);
  std::vector<int> newtoold(ntot);
  std::vector<int> oldtonew(nbtotal);
  std::unordered_set<LogicalLocation> newly_refined;

  { // Construct new list region
    PARTHENON_INSTRUMENT
    tree.GetMeshBlockList(newloc.data(), newtoold.data(), nbtotal);

    // create a list mapping the previous gid to the current one
    oldtonew[0] = 0;
    int mb_idx = 1;
    for (int n = 1; n < ntot; n++) {
      if (newtoold[n] == newtoold[n - 1] + 1) { // normal
        oldtonew[mb_idx++] = n;
      } else if (newtoold[n] == newtoold[n - 1] + nleaf) { // derefined
        for (int j = 0; j < nleaf - 1; j++)
          oldtonew[mb_idx++] = n - 1;
        oldtonew[mb_idx++] = n;
      }
    }
    // fill the last block
    for (; mb_idx < nbtold; mb_idx++)
      oldtonew[mb_idx] = ntot - 1;

    current_level = 0;
    for (int n = 0; n < ntot; n++) {
      // "on" = "old n" = "old gid" = "old global MeshBlock ID"
      int on = newtoold[n];
      if (newloc[n].level() > current_level) // set the current max level
        current_level = newloc[n].level();
      if (newloc[n].level() >= loclist[on].level()) { // same or refined
        newcost[n] = costlist[on];
        // Keep a list of all blocks refined for below
        if (newloc[n].level() > loclist[on].level()) {
          newly_refined.insert(newloc[n]);
        }
      } else {
        double acost = 0.0;
        for (int l = 0; l < nleaf; l++)
          acost += costlist[on + l];
        newcost[n] = acost / nleaf;
      }
    }
  } // Construct new list region

  // Calculate new load balance
  if (modified) CalculateLoadBalance(newcost, newrank, nslist, nblist);

  int nbs = nslist[Globals::my_rank];
  int nbe = nbs + nblist[Globals::my_rank] - 1;

#ifdef ENABLE_LB_TIMERS
  block_cost.Realloc(nbe - nbs + 1);
#endif
  block_cost_host.resize(nbe - nbs + 1);

  // Restrict fine to coarse buffers
  ProResCache_t restriction_cache;
  int nrestrict = 0;
  for (int on = onbs; on <= onbe; on++) {
    int nn = oldtonew[on];
    auto pmb = FindMeshBlock(on);
    if (newloc[nn].level() < loclist[on].level()) nrestrict += pmb->vars_cc_.size();
  }
  restriction_cache.Initialize(nrestrict, resolved_packages.get());
  int irestrict = 0;
  for (int on = onbs; on <= onbe; on++) {
    int nn = oldtonew[on];
    if (newloc[nn].level() < loclist[on].level()) {
      auto pmb = FindMeshBlock(on);
      for (auto &var : pmb->vars_cc_) {
        restriction_cache.RegisterRegionHost(irestrict++,
                                             ProResInfo::GetInteriorRestrict(pmb, var),
                                             var.get(), resolved_packages.get());
      }
    }
  }
  restriction_cache.CopyToDevice();
  auto [cellbounds, c_cellbounds] = GetCellBounds();
  refinement::Restrict(resolved_packages.get(), restriction_cache, cellbounds,
                       c_cellbounds);

  Kokkos::fence();

#ifdef MPI_PARALLEL
  // Send data from old to new blocks
  std::vector<MPI_Request> send_reqs;
  { // AMR Send region
    PARTHENON_INSTRUMENT
    for (int n = onbs; n <= onbe; n++) {
      int nn = oldtonew[n];
      LogicalLocation &oloc = loclist[n];
      LogicalLocation &nloc = newloc[nn];
      auto pb = FindMeshBlock(n);
      if (nloc.level() == oloc.level() &&
          newrank[nn] != Globals::my_rank) { // same level, different rank
        for (auto &var : pb->vars_cc_)
          send_reqs.emplace_back(SendSameToSame(nn - nslist[newrank[nn]], newrank[nn],
                                                var.get(), pb.get(), this));
      } else if (nloc.level() > oloc.level()) { // c2f
        // c2f must communicate to multiple leaf blocks (unlike f2c, same2same)
        for (int l = 0; l < nleaf; l++) {
          const int nl = nn + l; // Leaf block index in new global block list
          LogicalLocation &nloc = newloc[nl];
          for (auto &var : pb->vars_cc_)
            send_reqs.emplace_back(SendCoarseToFine(nl - nslist[newrank[nl]], newrank[nl],
                                                    nloc, var.get(), this));
        } // end loop over nleaf (unique to c2f branch in this step 6)
      } else if (nloc.level() < oloc.level()) { // f2c: restrict + pack + send
        for (auto &var : pb->vars_cc_)
          send_reqs.emplace_back(SendFineToCoarse(nn - nslist[newrank[nn]], newrank[nn],
                                                  oloc, var.get(), this));
      }
    }
  }    // AMR Send region
#endif // MPI_PARALLEL

  // Construct a new MeshBlock list (moving the data within the MPI rank)
  BlockList_t new_block_list(nbe - nbs + 1);
  { // AMR Construct new MeshBlockList region
    PARTHENON_INSTRUMENT
    RegionSize block_size = GetBlockSize();

    for (int n = nbs; n <= nbe; n++) {
      int on = newtoold[n];
      if ((ranklist[on] == Globals::my_rank) &&
          (loclist[on].level() == newloc[n].level())) {
        // on the same MPI rank and same level -> just move it
        new_block_list[n - nbs] = FindMeshBlock(on);
        if (!new_block_list[n - nbs]) {
          BoundaryFlag block_bcs[6];
          SetBlockSizeAndBoundaries(newloc[n], block_size, block_bcs);
          new_block_list[n - nbs] =
              MeshBlock::Make(n, n - nbs, newloc[n], block_size, block_bcs, this, pin,
                              app_in, packages, resolved_packages, gflag);
        }
      } else {
        // on a different refinement level or MPI rank - create a new block
        BoundaryFlag block_bcs[6];
        SetBlockSizeAndBoundaries(newloc[n], block_size, block_bcs);
        // append new block to list of MeshBlocks
        new_block_list[n - nbs] =
            MeshBlock::Make(n, n - nbs, newloc[n], block_size, block_bcs, this, pin,
                            app_in, packages, resolved_packages, gflag);
      }
    }
  } // AMR Construct new MeshBlockList region

  // Replace the MeshBlock list
  auto old_block_list = std::move(block_list);
  block_list = std::move(new_block_list);

  // Ensure local and global ids are correct
  for (int n = nbs; n <= nbe; n++) {
    block_list[n - nbs]->gid = n;
    block_list[n - nbs]->lid = n - nbs;
  }

  // Receive the data and load into MeshBlocks
  { // AMR Recv and unpack data
    PARTHENON_INSTRUMENT
    bool all_received;
    int niter = 0;
    if (block_list.size() > 0) {
      // Create a vector for holding the status of all communications, it is sized to fit
      // the maximal number of calculations that this rank could receive: the number of
      // blocks on the rank x the number of variables x times the number of fine blocks
      // that would communicate if every block had been coarsened (8 in 3D)
      std::vector<bool> finished(
          std::max((nbe - nbs + 1), 1) * FindMeshBlock(nbs)->vars_cc_.size() * 8, false);
      do {
        all_received = true;
        niter++;
        int idx = 0;
        for (int n = nbs; n <= nbe; n++) {
          int on = newtoold[n];
          LogicalLocation &oloc = loclist[on];
          LogicalLocation &nloc = newloc[n];
          auto pb = FindMeshBlock(n);
          if (oloc.level() == nloc.level() &&
              ranklist[on] != Globals::my_rank) { // same level, different rank
#ifdef MPI_PARALLEL
            for (auto &var : pb->vars_cc_) {
              if (!finished[idx])
                finished[idx] =
                    TryRecvSameToSame(n - nbs, ranklist[on], var.get(), pb.get(), this);
              all_received = finished[idx++] && all_received;
            }
#endif
          } else if (oloc.level() > nloc.level()) { // f2c
            for (int l = 0; l < nleaf; l++) {
              auto pob = pb;
              if (ranklist[on + l] == Globals::my_rank)
                pob = old_block_list[on + l - onbs];
              LogicalLocation &oloc = loclist[on + l];
              for (auto &var : pb->vars_cc_) {
                if (!finished[idx]) {
                  auto var_in = pob->meshblock_data.Get()->GetVarPtr(var->label());
                  finished[idx] =
                      TryRecvFineToCoarse(n - nbs, ranklist[on + l], oloc, var_in.get(),
                                          var.get(), pb.get(), this);
                }
                all_received = finished[idx++] && all_received;
              }
            }
          } else if (oloc.level() < nloc.level()) { // c2f
            for (auto &var : pb->vars_cc_) {
              if (!finished[idx]) {
                auto pob = pb;
                if (ranklist[on] == Globals::my_rank) pob = old_block_list[on - onbs];
                auto var_in = pob->meshblock_data.Get()->GetVarPtr(var->label());
                finished[idx] = TryRecvCoarseToFine(
                    n - nbs, ranklist[on], nloc, var_in.get(), var.get(), pb.get(), this);
              }
              all_received = finished[idx++] && all_received;
            }
          }
        }
        // rb_idx is a running index, so we repeat the loop until all vals are true
      } while (!all_received && niter < 1e7);
      if (!all_received) PARTHENON_FAIL("AMR Receive failed");
    }
    // Fence here to be careful that all communication is finished before moving
    // on to prolongation
    Kokkos::fence();

    // Prolongate blocks that had a coarse buffer filled (i.e. c2f blocks)
    ProResCache_t prolongation_cache;
    int nprolong = 0;
    for (int nn = nbs; nn <= nbe; nn++) {
      int on = newtoold[nn];
      auto pmb = FindMeshBlock(nn);
      if (newloc[nn].level() > loclist[on].level()) nprolong += pmb->vars_cc_.size();
    }
    prolongation_cache.Initialize(nprolong, resolved_packages.get());
    int iprolong = 0;
    for (int nn = nbs; nn <= nbe; nn++) {
      int on = newtoold[nn];
      if (newloc[nn].level() > loclist[on].level()) {
        auto pmb = FindMeshBlock(nn);
        for (auto &var : pmb->vars_cc_) {
          prolongation_cache.RegisterRegionHost(
              iprolong++, ProResInfo::GetInteriorProlongate(pmb, var), var.get(),
              resolved_packages.get());
        }
      }
    }
    prolongation_cache.CopyToDevice();

    refinement::ProlongateShared(resolved_packages.get(), prolongation_cache, cellbounds,
                                 c_cellbounds);

    // update the lists
    loclist = std::move(newloc);
    ranklist = std::move(newrank);
    costlist = std::move(newcost);

    // A block newly refined and prolongated may have neighbors which were
    // already refined to the new level.
    // If so, the prolongated versions of shared elements will not reflect
    // the true, finer versions present in the neighbor block.
    // We must create any new fine buffers and fill them from these neighbors
    // in order to maintain a consistent global state.
    // Thus we rebuild and synchronize the mesh now, but using a unique
    // neighbor precedence favoring the "old" fine blocks over "new" ones
    for (auto &pmb : block_list) {
      pmb->pbval->SearchAndSetNeighbors(tree, ranklist.data(), nslist.data(),
                                        newly_refined);
    }
    // Make sure all old sends/receives are done before we reconfigure the mesh
#ifdef MPI_PARALLEL
    if (send_reqs.size() != 0)
      PARTHENON_MPI_CHECK(
          MPI_Waitall(send_reqs.size(), send_reqs.data(), MPI_STATUSES_IGNORE));
#endif
    // Re-initialize the mesh with our temporary ownership/neighbor configurations.
    // No buffers are different when we switch to the final precedence order.
    Initialize(false, pin, app_in);

    // Internal refinement relies on the fine shared values, which are only consistent
    // after being updated with any previously fine versions
    refinement::ProlongateInternal(resolved_packages.get(), prolongation_cache,
                                   cellbounds, c_cellbounds);

    // Rebuild just the ownership model, this time weighting the "new" fine blocks just
    // like any other blocks at their level.
    for (auto &pmb : block_list) {
      pmb->pbval->SearchAndSetNeighbors(tree, ranklist.data(), nslist.data());
    }
  } // AMR Recv and unpack data

  ResetLoadBalanceVariables();
  return true;
}
} // namespace parthenon
