//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2022 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
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

namespace {

int CreateAMRMPITag(int lid, int ox1, int ox2, int ox3) {
  // the trailing zero is used as "id" to indicate an AMR related tag
  return (lid << 8) | (ox1 << 7) | (ox2 << 6) | (ox3 << 5) | 0;
}

struct SendStatus { 
  MPI_Request req; 
  bool IsComplete() {
    int test; 
    PARTHENON_MPI_CHECK(MPI_Test(&req, &test, MPI_STATUS_IGNORE)) 
    return static_cast<bool>(test); 
  }
}; 

struct RecvStatus { 
  MPI_Request req; 
  bool IsComplete() {
    int test; 
    PARTHENON_MPI_CHECK(MPI_Test(&req, &test, MPI_STATUS_IGNORE)) 
    return static_cast<bool>(test); 
  }
}; 

MPI_Request SendCoarseToFine(int lid_recv, int dest_rank, int ox1, int ox2, int ox3, CellVariable<Real> *var) {
  SendStatus req;
  int idx = 0; 
  MPI_Comm comm = mpi_comm_map_[var->label()];
  int tag = CreateAMRMPITag(lid_recv, ox1, ox2, ox3);
  if (var->IsAllocated()){
    PARTHENON_MPI_CHECK(MPI_Isend(var->data.data(), var->data.size(), MPI_PARTHENON_REAL,
                                  dest_rank, tag, comm, &req)); 
  } else { 
    PARTHENON_MPI_CHECK(MPI_Isend(var->data.data(), 0, MPI_PARTHENON_REAL,
                                  dest_rank, tag, comm, &req));
  }
  return req; 
}

bool TryRecvCoarseToFine(int lid_recv, int send_rank, int ox1, int ox2, int ox3, 
                         CellVariable<Real>* var, MeshBlock *pmb, RecvState *state) {
  static const IndexRange ib = pmb->cellbounds.GetBoundsI(IndexRange::entire_coarse);
  static const IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexRange::entire_coarse);
  static const IndexRange kb = pmb->cellbounds.GetBoundsK(IndexRange::entire_coarse);
  
  static const IndexRange ib_int = pmb->cellbounds.GetBoundsI(IndexRange::interior);
  static const IndexRange jb_int = pmb->cellbounds.GetBoundsJ(IndexRange::interior);
  static const IndexRange kb_int = pmb->cellbounds.GetBoundsK(IndexRange::interior);
  
  MPI_Comm comm = mpi_comm_map_[var->label()];
  int tag = CreateAMRMPITag(lid_recv, ox1, ox2, ox3);

  int test; 
  MPI_Status status; 
  PARTHENON_MPI_CHECK(MPI_Iprobe(send_rank, tag, comm, &test, &status));
  if (test) {
    PARTHENON_MPI_CHECK(MPI_Get_count(&status, MPI_PARTHENON_REAL, &size));
    if (size > 0) {
      if (!pmb->IsAllocated(var->label())) pmb->AllocateSparse(var->label()); 
      PARTHENON_MPI_CHECK(MPI_Recv(var->data.data(), var.data.size(),
                                   MPI_PARTHENON_REAL, send_rank, tag, comm));
      const int ks = (ox3 == 0) ? 0 : (kb_int.e - kb_int.s + 1) / 2;
      const int js = (ox2 == 0) ? 0 : (jb_int.e - jb_int.s + 1) / 2;
      const int is = (ox1 == 0) ? 0 : (ib_int.e - ib_int.s + 1) / 2;
      pmb->par_for("FillSameRankCoarseToFineAMR", 0, nt, nu, nv, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA(const int t, const int u, const int v, const int k, const int j, const int i) {
            var->coarse_s(t, u, v, k, j, i) = var(t, u, v, k + ks, j + js, i + is);
          }); 
    } else { 
      if (pmb->IsAllocated(var->label())) pmb->DeallocateSparse(var->label()); 
      PARTHENON_MPI_CHECK(MPI_Recv(var->data.data(), 0, MPI_PARTHENON_REAL, send_rank, tag, comm)); 
    }
  }
  
  return test; 
} 

MPI_Request SendFineToCoarse(int lid_recv, int ox1, int ox2, int ox3, CellVariable<Real> *var) {
  MPI_Request req;
  int idx = 0; 
  MPI_Comm comm = mpi_comm_map_[var->label()];
  int tag = CreateAMRMPITag(lid_recv, ox1, ox2, ox3);
  if (var->IsAllocated()){
    PARTHENON_MPI_CHECK(MPI_Isend(var->coarse_s.data(), var->coarse_s.size(), MPI_PARTHENON_REAL,
                                  dest_rank, tag, comm, &req)); 
  } else { 
    PARTHENON_MPI_CHECK(MPI_Isend(var->coarse_s.data(), 0, MPI_PARTHENON_REAL,
                                  dest_rank, tag, comm, &req));
  }
  return req; 
}

bool TryRecvFineToCoarse(int lid_recv, int ox1, int ox2, int ox3, CellVariable<Real> *var, MeshBlock *pmb) {
  static const IndexRange ib = pmb->cellbounds.GetBoundsI(IndexRange::interior_coarse);
  static const IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexRange::interior_coarse);
  static const IndexRange kb = pmb->cellbounds.GetBoundsK(IndexRange::interior_coarse);
  
  static const IndexRange ib_int = pmb->cellbounds.GetBoundsI(IndexRange::interior);
  static const IndexRange jb_int = pmb->cellbounds.GetBoundsJ(IndexRange::interior);
  static const IndexRange kb_int = pmb->cellbounds.GetBoundsK(IndexRange::interior);
  
  MPI_Comm comm = mpi_comm_map_[var->label()];
  int tag = CreateAMRMPITag(lid_recv, ox1, ox2, ox3);

  int test; 
  MPI_Status status; 
  PARTHENON_MPI_CHECK(MPI_Iprobe(send_rank, tag, comm, &test, &status));
  if (test) {
    PARTHENON_MPI_CHECK(MPI_Get_count(&status, MPI_PARTHENON_REAL, &size));
    if (size > 0) {
      if (!pmb->IsAllocated(var->label())) pmb->AllocateSparse(var->label()); 
      // This has to be an MPI_Recv w/o buffering 
      PARTHENON_MPI_CHECK(MPI_Recv(var->coarse_s.data(), var.coarse_s.size(),
                                   MPI_PARTHENON_REAL, send_rank, tag, comm));
      const int ks = (ox3 == 0) ? 0 : (kb_int.e - kb.int.s + 1) / 2;
      const int js = (ox2 == 0) ? 0 : (jb_int.e - jb_int.s + 1) / 2;
      const int is = (ox1 == 0) ? 0 : (ib_int.e - ib_int.s + 1) / 2;
      parthenon::par_for("FillSameRankCoarseToFineAMR", 0, nt, nu, nv, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA(const int t, const int u, const int v, const int k, const int j, const int i) {
            var(t, u, v, k + ks, j + js, i + is) = var->coarse_s(t, u, v, k, j, i);
          }); 
      // We have to block here w/o buffering so that the write is guaranteed to be finished 
      // before we get here again 
      Kokkos::fence();
    } else { 
      if (pmb->IsAllocated(var->label())) pmb->DeallocateSparse(var->label()); 
      PARTHENON_MPI_CHECK(MPI_Recv(var->data.data(), 0,
                                   MPI_PARTHENON_REAL, send_rank, tag, comm)); 
    }
  }
  
  return test;  
}

MPI_Request SendSameToSame(int lid_recv, int recv_rank, CellVariable<Real> *var) {
  return SendCoarseToFine(int lid_recv, int dest_rank, 0, 0, 0, var); 
}

bool TryRecvSameToSame(int lid_recv, int send_rank, CellVariable<Real> *var) {
  MPI_Comm comm = mpi_comm_map_[var->label()];
  int tag = CreateAMRMPITag(lid_recv, 0, 0, 0);

  int test; 
  MPI_Status status; 
  PARTHENON_MPI_CHECK(MPI_Iprobe(send_rank, tag, comm, &test, &status));
  if (test) {
    PARTHENON_MPI_CHECK(MPI_Get_count(&status, MPI_PARTHENON_REAL, &size));
    if (size > 0) {
      if (!pmb->IsAllocated(var->label())) pmb->AllocateSparse(var->label()); 
      PARTHENON_MPI_CHECK(MPI_Recv(var->data.data(), var.data.size(),
                                   MPI_PARTHENON_REAL, send_rank, tag, comm));
    } else { 
      if (pmb->IsAllocated(var->label())) pmb->DeallocateSparse(var->label());
      PARTHENON_MPI_CHECK(MPI_Recv(var->data.data(), 0,
                                   MPI_PARTHENON_REAL, send_rank, tag, comm)); 
    }
  }
  
  return test;  
}
} // unnamed namespace

namespace parthenon {

//----------------------------------------------------------------------------------------
// \!fn void Mesh::LoadBalancingAndAdaptiveMeshRefinement(ParameterInput *pin)
// \brief Main function for adaptive mesh refinement

void Mesh::LoadBalancingAndAdaptiveMeshRefinement(ParameterInput *pin,
                                                  ApplicationInput *app_in) {
  Kokkos::Profiling::pushRegion("LoadBalancingAndAdaptiveMeshRefinement");
  int nnew = 0, ndel = 0;

  if (adaptive) {
    UpdateMeshBlockTree(nnew, ndel);
    nbnew += nnew;
    nbdel += ndel;
  }

  lb_flag_ |= lb_automatic_;

  UpdateCostList();

  modified = false;
  if (nnew != 0 || ndel != 0) { // at least one (de)refinement happened
    GatherCostListAndCheckBalance();
    RedistributeAndRefineMeshBlocks(pin, app_in, nbtotal + nnew - ndel);
    modified = true;
  } else if (lb_flag_ && step_since_lb >= lb_interval_) {
    if (!GatherCostListAndCheckBalance()) { // load imbalance detected
      RedistributeAndRefineMeshBlocks(pin, app_in, nbtotal);
      modified = true;
    }
    step_since_lb = 0;
    //lb_flag_ = false;
  }
  Kokkos::Profiling::popRegion(); // LoadBalancingAndAdaptiveMeshRefinement
}

// Private routines
namespace {

struct BlockRankRange {
  BlockRankRange(int rank_start, int rank_stop, int block_start, int block_stop)
    : rstart(rank_start), rstop(rank_stop), bstart(block_start), bstop(block_stop) {}
  const int rstart, rstop, bstart, bstop;
};

void bisect_blocks(std::vector<Real> &sum_cost, BlockRankRange &r, std::vector<int> &start, std::vector<int> &nb) {
  const int nranks = r.rstop - r.rstart + 1;
  const int nleft = nranks / 2;
  const int nright = nranks - nleft;
  int split = (r.bstart + r.bstop + 1) / 2;
  Real cost_left = sum_cost[split-1] - (r.bstart>0 ? sum_cost[r.bstart-1] : 0.0);
  Real cost_right = sum_cost[r.bstop] - sum_cost[split-1];
  const Real total_cost = cost_left + cost_right;
  const Real cost_per_rank = total_cost/nranks;
  const Real target_left = nleft * cost_per_rank;
  const Real target_right = nright * cost_per_rank;

  if (cost_left < target_left) {
    while (r.bstop-split+1 > nright) {
      const Real l = cost_left/nleft;
      const Real r = cost_right/nright;
      const Real om = std::max(l,r);
      const Real delta = sum_cost[split] - sum_cost[split-1];
      const Real nl = (cost_left + delta)/nleft;
      const Real nr = (cost_right - delta)/nright;
      const Real nm = std::max(nl,nr);
      if (nm < om) {
        cost_left += delta;
        cost_right -= delta;
        split++;
      } else {
        break;
      }
    }
  } else if (cost_right < target_right) {
    while (split - r.bstart > nleft) {
      const Real l = cost_left/nleft;
      const Real r = cost_right/nright;
      const Real om = std::max(l,r);
      const Real delta = sum_cost[split-1] - (split>1 ? sum_cost[split-2] : 0.0);
      const Real nl = (cost_left - delta)/nleft;
      const Real nr = (cost_right + delta)/nright;
      const Real nm = std::max(nl,nr);
      if (nm < om) {
        cost_left -= delta;
        cost_right += delta;
        split -= 1;
      } else {
        break;
      }
    }
  }

  if (nleft == 1) {
    start[r.rstart] = r.bstart;
    nb[r.rstart] = split - r.bstart;
  } else {
    BlockRankRange rnew(r.rstart, r.rstart+nleft-1, r.bstart, split-1);
    bisect_blocks(sum_cost, rnew, start, nb);
  }

  if (nright == 1) {
    start[r.rstop] = split;
    nb[r.rstop] = r.bstop - split + 1;
  } else {
    BlockRankRange rnew(r.rstart+nleft, r.rstop, split, r.bstop);
    bisect_blocks(sum_cost, rnew, start, nb);
  }
}

void AssignAndUpdateBlocks(std::vector<Real> const &costlist, std::vector<int> &ranklist,
                           std::vector<int> &start, std::vector<int> &nb) {
  start.resize(Globals::nranks);
  nb.resize(Globals::nranks);
  const int nblocks = costlist.size();
  ranklist.resize(nblocks);
  std::vector<Real> sum_costs(nblocks);
  sum_costs[0] = costlist[0];
  for (int b = 1; b < nblocks; b++) {
    sum_costs[b] = sum_costs[b-1] + costlist[b];
  }
  for (int i = 0; i < Globals::nranks; i++) {
    start[i] = -1;
    nb[i] = 0;
  }
  int max_rank = std::min(Globals::nranks, nblocks) - 1;
  if (max_rank == 0) {
    start[0] = 0;
    nb[0] = nblocks;
    for (int b = 0; b < nblocks; b++) ranklist[b] = 0;
  } else {
    BlockRankRange root(0, max_rank, 0, nblocks-1);
    bisect_blocks(sum_costs, root, start, nb);

    for (int i = 0; i <= max_rank; i++) {
      for (int b = start[i]; b < start[i]+nb[i]; b++) {
        ranklist[b] = i;
      }
    }
  }
}


/**
 * @brief This routine assigns blocks to ranks by attempting to place index-contiguous
 * blocks of equal total cost on each rank.
 *
 * @param costlist (Input) A map of global block ID to a relative weight.
 * @param ranklist (Output) A map of global block ID to ranks.
 */
void AssignBlocks(std::vector<double> const &costlist, std::vector<int> &ranklist) {
  ranklist.resize(costlist.size());

  double const total_cost = std::accumulate(costlist.begin(), costlist.end(), 0.0);

  int rank = std::min(Globals::nranks, (int) ranklist.size())-1;
  double target_cost = total_cost / rank;
  double my_cost = 0.0;
  double remaining_cost = total_cost;
  // create rank list from the end: the master MPI rank should have less load
  for (int block_id = costlist.size() - 1; block_id >= 0; block_id--) {
    // ensure ranks get at least one block
    if (rank == block_id) {
      for (int b = block_id; b >= 0; b--) {
        ranklist[b] = b;
      }
      break;
    }
    if (target_cost == 0.0) {
      std::stringstream msg;
      msg << "### FATAL ERROR in CalculateLoadBalance" << std::endl
          << "There is at least one process which has no MeshBlock" << std::endl
          << "Decrease the number of processes or use smaller MeshBlocks." << std::endl;
      PARTHENON_FAIL(msg);
    }
    my_cost += costlist[block_id];
    ranklist[block_id] = rank;
    if (my_cost >= target_cost && rank > 0) {
      rank--;
      remaining_cost -= my_cost;
      my_cost = 0.0;
      target_cost = remaining_cost / (rank + 1);
    }
  }
}

void UpdateBlockList(std::vector<int> const &ranklist, std::vector<int> &nslist,
                     std::vector<int> &nblist) {
  nslist.resize(Globals::nranks);
  nblist.resize(Globals::nranks);

  int rank = ranklist[0];
  for (int r = 0; r < rank; r++) {
    nslist[r] = -1;
    nblist[r] = -1;
  }
  nslist[rank] = 0;
  for (int block_id = 1; block_id < ranklist.size(); block_id++) {
    if (ranklist[block_id] != ranklist[block_id - 1]) {
      nblist[rank] = block_id - nslist[rank];
      nslist[++rank] = block_id;
    }
  }
  nblist[rank] = ranklist.size() - nslist[rank];
}
} // namespace

//----------------------------------------------------------------------------------------
// \brief Calculate distribution of MeshBlocks based on the cost list
void Mesh::CalculateLoadBalance(std::vector<double> const &costlist,
                                std::vector<int> &ranklist, std::vector<int> &nslist,
                                std::vector<int> &nblist) {
  Kokkos::Profiling::pushRegion("CalculateLoadBalance");
  auto const total_blocks = costlist.size();

  AssignAndUpdateBlocks(costlist, ranklist, nslist, nblist);

  // Assigns blocks to ranks on a rougly cost-equal basis.
  //AssignBlocks(costlist, ranklist);

  // Updates nslist with the ID of the starting block on each rank and the count of blocks
  // on each rank.
  //UpdateBlockList(ranklist, nslist, nblist);

  if (Globals::nranks > total_blocks) {
    if (!adaptive) {
      // mesh is refined statically, treat this an as error (all ranks need to
      // participate)
      std::stringstream msg;
      msg << "### FATAL ERROR in CalculateLoadBalance" << std::endl
          << "There are fewer MeshBlocks than OpenMP threads on each MPI rank"
          << std::endl
          << "Decrease the number of threads or use more MeshBlocks." << std::endl;
      PARTHENON_FAIL(msg);
    } else if (Globals::my_rank == 0) {
      // we have AMR, print warning only on Rank 0
      std::cout << "### WARNING in CalculateLoadBalance" << std::endl
                << "There are fewer MeshBlocks than MPI ranks"
                << std::endl
                << "This is likely fine if the number of meshblocks is expected to grow "
                   "during the "
                   "simulations. Otherwise, it might be worthwhile to decrease the "
                   "number of ranks or "
                   "use more meshblocks."
                << std::endl;
    }
  }
  Kokkos::Profiling::popRegion(); // CalculateLoadBalance
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::ResetLoadBalanceVariables()
// \brief reset counters and flags for load balancing

void Mesh::ResetLoadBalanceVariables() {
  if (lb_automatic_) {
    for (auto &pmb : block_list) {
      costlist[pmb->gid] = TINY_NUMBER;
      pmb->ResetTimeMeasurement();
    }
  }
  //lb_flag_ = false;
  step_since_lb = 0;
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::UpdateCostList()
// \brief update the cost list

void Mesh::UpdateCostList() {
  if (lb_automatic_) {
    double w = static_cast<double>(lb_interval_ - 1) / static_cast<double>(lb_interval_);
    for (auto &pmb : block_list) {
      costlist[pmb->gid] = costlist[pmb->gid] * w + pmb->cost_;
    }
  } else if (lb_flag_) {
    for (auto &pmb : block_list) {
      costlist[pmb->gid] = pmb->cost_;
    }
  }
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::UpdateMeshBlockTree(int &nnew, int &ndel)
// \brief collect refinement flags and manipulate the MeshBlockTree

void Mesh::UpdateMeshBlockTree(int &nnew, int &ndel) {
  Kokkos::Profiling::pushRegion("UpdateMeshBlockTree");
  // compute nleaf= number of leaf MeshBlocks per refined block
  int nleaf = 2;
  if (mesh_size.nx2 > 1) nleaf = 4;
  if (mesh_size.nx3 > 1) nleaf = 8;

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
    Kokkos::Profiling::popRegion();    // UpdateMeshBlockTree
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
    if (mesh_size.nx2 > 1) lj = 1;
    if (mesh_size.nx3 > 1) lk = 1;
    for (int n = 0; n < tnderef; n++) {
      if ((lderef[n].lx1 & 1LL) == 0LL && (lderef[n].lx2 & 1LL) == 0LL &&
          (lderef[n].lx3 & 1LL) == 0LL) {
        int r = n, rr = 0;
        for (std::int64_t k = 0; k <= lk; k++) {
          for (std::int64_t j = 0; j <= lj; j++) {
            for (std::int64_t i = 0; i <= 1; i++) {
              if (r < tnderef) {
                if ((lderef[n].lx1 + i) == lderef[r].lx1 &&
                    (lderef[n].lx2 + j) == lderef[r].lx2 &&
                    (lderef[n].lx3 + k) == lderef[r].lx3 &&
                    lderef[n].level == lderef[r].level)
                  rr++;
                r++;
              }
            }
          }
        }
        if (rr == nleaf) {
          clderef[ctnd].lx1 = lderef[n].lx1 >> 1;
          clderef[ctnd].lx2 = lderef[n].lx2 >> 1;
          clderef[ctnd].lx3 = lderef[n].lx3 >> 1;
          clderef[ctnd].level = lderef[n].level - 1;
          ctnd++;
        }
      }
    }
  }
  // sort the lists by level
  if (ctnd > 1) std::sort(clderef, &(clderef[ctnd - 1]), LogicalLocation::Greater);

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

  Kokkos::Profiling::popRegion(); // UpdateMeshBlockTree
}

//----------------------------------------------------------------------------------------
// \!fn bool Mesh::GatherCostListAndCheckBalance()
// \brief collect the cost from MeshBlocks and check the load balance

bool Mesh::GatherCostListAndCheckBalance() {
  if (lb_manual_ || lb_automatic_) {
#ifdef MPI_PARALLEL
    PARTHENON_MPI_CHECK(MPI_Allgatherv(MPI_IN_PLACE, nblist[Globals::my_rank], MPI_DOUBLE,
                                       costlist.data(), nblist.data(), nslist.data(),
                                       MPI_DOUBLE, MPI_COMM_WORLD));
#endif
    double maxcost = 0.0, avecost = 0.0;
    for (int rank = 0; rank < Globals::nranks; rank++) {
      double rcost = 0.0;
      int ns = nslist[rank];
      int ne = ns + nblist[rank];
      for (int n = ns; n < ne; ++n)
        rcost += costlist[n];
      maxcost = std::max(maxcost, rcost);
      avecost += rcost;
    }
    avecost /= Globals::nranks;

    if (adaptive)
      lb_tolerance_ =
          2.0 * static_cast<double>(Globals::nranks) / static_cast<double>(nbtotal);

    if (maxcost > (1.0 + lb_tolerance_) * avecost) return false;
  }
  return true;
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::RedistributeAndRefineMeshBlocks(ParameterInput *pin, int ntot)
// \brief redistribute MeshBlocks according to the new load balance

void Mesh::RedistributeAndRefineMeshBlocks(ParameterInput *pin, ApplicationInput *app_in,
                                           int ntot) {
  Kokkos::Profiling::pushRegion("RedistributeAndRefineMeshBlocks");
  // kill any cached packs
  mesh_data.PurgeNonBase();
  mesh_data.Get()->ClearCaches();

  // compute nleaf= number of leaf MeshBlocks per refined block
  int nleaf = 2;
  if (mesh_size.nx2 > 1) nleaf = 4;
  if (mesh_size.nx3 > 1) nleaf = 8;

  // Step 1. construct new lists
  Kokkos::Profiling::pushRegion("Step1: Construct new list");
  std::vector<LogicalLocation> newloc(ntot);
  std::vector<int> newrank(ntot);
  std::vector<double> newcost(ntot);
  std::vector<int> newtoold(ntot);
  std::vector<int> oldtonew(nbtotal);

  int nbtold = nbtotal;
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
    if (newloc[n].level > current_level) // set the current max level
      current_level = newloc[n].level;
    if (newloc[n].level >= loclist[on].level) { // same or refined
      newcost[n] = costlist[on];
    } else {
      double acost = 0.0;
      for (int l = 0; l < nleaf; l++)
        acost += costlist[on + l];
      newcost[n] = acost / nleaf;
    }
  }
#ifdef MPI_PARALLEL
  // store old nbstart and nbend before load balancing in Step 2.
  int onbs = nslist[Globals::my_rank];
  int onbe = onbs + nblist[Globals::my_rank] - 1;
#endif
  Kokkos::Profiling::popRegion(); // Step 1

  // Step 2. Calculate new load balance
  CalculateLoadBalance(newcost, newrank, nslist, nblist);

  int nbs = nslist[Globals::my_rank];
  int nbe = nbs + nblist[Globals::my_rank] - 1;

#ifdef MPI_PARALLEL
  int bnx1 = GetBlockSize().nx1;
  int bnx2 = GetBlockSize().nx2;
  int bnx3 = GetBlockSize().nx3;
  // Step 3. count the number of the blocks to be sent / received
  Kokkos::Profiling::pushRegion("Step 3: Count blocks");
  int nsend = 0, nrecv = 0;
  for (int n = nbs; n <= nbe; n++) {
    int on = newtoold[n];
    if (loclist[on].level > newloc[n].level) { // f2c
      for (int k = 0; k < nleaf; k++) {
        if (ranklist[on + k] != Globals::my_rank) nrecv++;
      }
    } else {
      if (ranklist[on] != Globals::my_rank) nrecv++;
    }
  }
  for (int n = onbs; n <= onbe; n++) {
    int nn = oldtonew[n];
    if (loclist[n].level < newloc[nn].level) { // c2f
      for (int k = 0; k < nleaf; k++) {
        if (newrank[nn + k] != Globals::my_rank) nsend++;
      }
    } else {
      if (newrank[nn] != Globals::my_rank) nsend++;
    }
  }

  Kokkos::Profiling::popRegion(); // Step 3
  // Step 4. calculate buffer sizes
  //Kokkos::Profiling::pushRegion("Step 4: Calc buffer sizes");
  //BufArray1D<Real> *sendbuf, *recvbuf;
  //// use the first MeshBlock in the linked list of blocks belonging to this MPI rank as a
  //// representative of all MeshBlocks for counting the "load-balancing registered" and
  //// "SMR/AMR-enrolled" quantities (loop over MeshBlock::vars_cc_, not MeshRefinement)

  //// TODO(felker): add explicit check to ensure that elements of pb->vars_cc/fc_ and
  //// pb->pmr->pvars_cc/fc_ v point to the same objects, if adaptive

  //// TODO(JL) Why are we using all variables for same-level but only the variables in pmr
  //// for c2f and f2c?s
  //int num_cc = pdummy_block->vars_cc_.size();
  //int num_pmr_cc = pdummy_block->pmr->pvars_cc_.size();
  //int num_fc = pdummy_block->vars_fc_.size();
  //int nx4_tot = 0;
  //for (auto &pvar_cc : pdummy_block->vars_cc_) {
  //  nx4_tot += pvar_cc->GetDim(4);
  //}

  //const int f2 = (ndim >= 2) ? 1 : 0; // extra cells/faces from being 2d
  //const int f3 = (ndim >= 3) ? 1 : 0; // extra cells/faces from being 3d

  //// cell-centered quantities enrolled in SMR/AMR
  //int bssame = bnx1 * bnx2 * bnx3 * nx4_tot;
  //int bsf2c = (bnx1 / 2) * ((bnx2 + 1) / 2) * ((bnx3 + 1) / 2) * nx4_tot;
  //int bsc2f =
  //    (bnx1 / 2 + 2) * ((bnx2 + 1) / 2 + 2 * f2) * ((bnx3 + 1) / 2 + 2 * f3) * nx4_tot;
  //// face-centered quantities enrolled in SMR/AMR
  //bssame += num_fc * ((bnx1 + 1) * bnx2 * bnx3 + bnx1 * (bnx2 + f2) * bnx3 +
  //                    bnx1 * bnx2 * (bnx3 + f3));
  //bsf2c += num_fc * (((bnx1 / 2) + 1) * ((bnx2 + 1) / 2) * ((bnx3 + 1) / 2) +
  //                   (bnx1 / 2) * (((bnx2 + 1) / 2) + f2) * ((bnx3 + 1) / 2) +
  //                   (bnx1 / 2) * ((bnx2 + 1) / 2) * (((bnx3 + 1) / 2) + f3));
  //bsc2f +=
  //    num_fc *
  //    (((bnx1 / 2) + 1 + 2) * ((bnx2 + 1) / 2 + 2 * f2) * ((bnx3 + 1) / 2 + 2 * f3) +
  //     (bnx1 / 2 + 2) * (((bnx2 + 1) / 2) + f2 + 2 * f2) * ((bnx3 + 1) / 2 + 2 * f3) +
  //     (bnx1 / 2 + 2) * ((bnx2 + 1) / 2 + 2 * f2) * (((bnx3 + 1) / 2) + f3 + 2 * f3));

  //// add num_cc/num_pmr_cc to all buffer sizes for storing allocation statuses
  //bssame += num_cc;
  //bsc2f += num_pmr_cc;
  //bsf2c += num_pmr_cc;

  //// add one more element to buffer size for storing the derefinement counter
  //bssame++;
  //Kokkos::Profiling::popRegion(); // Step 4

  //MPI_Request *req_send, *req_recv;

  //// Step 5. Allocate space for send and recieve buffers
  //Kokkos::Profiling::pushRegion("Step 5: Allocate send and recv buf");
  //size_t buf_size = 0;
  //if (nrecv != 0) {
  //  recvbuf = new BufArray1D<Real>[nrecv];
  //  for (int n = nbs; n <= nbe; n++) {
  //    int on = newtoold[n];
  //    LogicalLocation &oloc = loclist[on];
  //    LogicalLocation &nloc = newloc[n];
  //    if (oloc.level > nloc.level) { // f2c
  //      for (int l = 0; l < nleaf; l++) {
  //        if (ranklist[on + l] == Globals::my_rank) continue;
  //        buf_size += bsf2c;
  //      }
  //    } else { // same level or c2f
  //      if (ranklist[on] == Globals::my_rank) continue;
  //      int size;
  //      if (oloc.level == nloc.level) {
  //        size = bssame;
  //      } else {
  //        size = bsc2f;
  //      }
  //      buf_size += size;
  //    }
  //  }
  //}
  //if (nsend != 0) {
  //  sendbuf = new BufArray1D<Real>[nsend];
  //  for (int n = onbs; n <= onbe; n++) {
  //    int nn = oldtonew[n];
  //    LogicalLocation &oloc = loclist[n];
  //    LogicalLocation &nloc = newloc[nn];
  //    auto pb = FindMeshBlock(n);
  //    if (nloc.level == oloc.level) { // same level
  //      if (newrank[nn] == Globals::my_rank) continue;
  //      buf_size += bssame;
  //    } else if (nloc.level > oloc.level) { // c2f
  //      // c2f must communicate to multiple leaf blocks (unlike f2c, same2same)
  //      for (int l = 0; l < nleaf; l++) {
  //        if (newrank[nn + l] == Globals::my_rank) continue;
  //        buf_size += bsc2f;
  //      }      // end loop over nleaf (unique to c2f branch in this step 6)
  //    } else { // f2c: restrict + pack + send
  //      if (newrank[nn] == Globals::my_rank) continue;
  //      buf_size += bsf2c;
  //    }
  //  }
  //}
  //BufArray1D<Real> bufs("RedistributeAndRefineMeshBlocks sendrecv bufs", buf_size);
  //Kokkos::Profiling::popRegion(); // Step 5

  // Step 6. allocate and start receiving buffers
  //Kokkos::Profiling::pushRegion("Step 6: Pack buffer and start recv");
  //size_t buf_offset = 0;
  //if (nrecv != 0) {
  //  req_recv = new MPI_Request[nrecv];
  //  int rb_idx = 0; // recv buffer index
  //  for (int n = nbs; n <= nbe; n++) {
  //    int on = newtoold[n];
  //    LogicalLocation &oloc = loclist[on];
  //    LogicalLocation &nloc = newloc[n];
  //    if (oloc.level > nloc.level) { // f2c
  //      for (int l = 0; l < nleaf; l++) {
  //        if (ranklist[on + l] == Globals::my_rank) continue;
  //        LogicalLocation &lloc = loclist[on + l];
  //        int ox1 = ((lloc.lx1 & 1LL) == 1LL), ox2 = ((lloc.lx2 & 1LL) == 1LL),
  //            ox3 = ((lloc.lx3 & 1LL) == 1LL);
  //        recvbuf[rb_idx] =
  //            BufArray1D<Real>(bufs, std::make_pair(buf_offset, buf_offset + bsf2c));
  //        buf_offset += bsf2c;
  //        int tag = CreateAMRMPITag(n - nbs, ox1, ox2, ox3);
  //        PARTHENON_MPI_CHECK(MPI_Irecv(recvbuf[rb_idx].data(), bsf2c, MPI_PARTHENON_REAL,
  //                                      ranklist[on + l], tag, MPI_COMM_WORLD,
  //                                      &(req_recv[rb_idx])));
  //        rb_idx++;
  //      }
  //    } else { // same level or c2f
  //      if (ranklist[on] == Globals::my_rank) continue;
  //      int size;
  //      if (oloc.level == nloc.level) {
  //        size = bssame;
  //      } else {
  //        size = bsc2f;
  //      }
  //      recvbuf[rb_idx] =
  //          BufArray1D<Real>(bufs, std::make_pair(buf_offset, buf_offset + size));
  //      buf_offset += size;
  //      int tag = CreateAMRMPITag(n - nbs, 0, 0, 0);
  //      PARTHENON_MPI_CHECK(MPI_Irecv(recvbuf[rb_idx].data(), size, MPI_PARTHENON_REAL,
  //                                    ranklist[on], tag, MPI_COMM_WORLD,
  //                                    &(req_recv[rb_idx])));
  //      rb_idx++;
  //    }
  //  }
  //}
  //Kokkos::Profiling::popRegion(); // Step 6
  
  // Step 7 - eps: Restrict fine to coarse buffers 
  if (nsend != 0) { 
    for (int on = onbs; on <= onbe; on++) { 
      int nn = oldtonew[on]; 
      LogicalLocation &oloc = loclist[on]; 
      LogicalLocation &nloc = newloc[nn];
      auto pmb = FindMeshBlock(on); 
      if (nloc.level < oloc.level) {
        const IndexDomain interior = IndexDomain::interior;
        IndexRange cib = pb->c_cellbounds.GetBoundsI(interior);
        IndexRange cjb = pb->c_cellbounds.GetBoundsJ(interior);
        IndexRange ckb = pb->c_cellbounds.GetBoundsK(interior);
        
        // Need to restrict this block before doing sends
        for (auto& var : pb->vars_cc_)
          pmb->pmr->RestrictCellCenteredValues(var->data, var->coarse_s, 0, var->GetDim(4) - 1, 
                                               cib.s, cib.e, cjb.s,cjb.e, ckb.s, ckb.e); 
      } 
    }
  }
  Kokkos::fence(); 

  // Step 7. allocate, pack and start sending buffers
  Kokkos::Profiling::pushRegion("Step 7: Pack and send buffers");
  std::vector<MPI_Request> send_reqs; 
  if (nsend != 0) {
    //req_send = new MPI_Request[nsend];
    //std::vector<int> tags(nsend);
    //std::vector<int> dest(nsend);
    //std::vector<int> count(nsend);
    //int sb_idx = 0; // send buffer index
    for (int n = onbs; n <= onbe; n++) {
      int nn = oldtonew[n];
      LogicalLocation &oloc = loclist[n];
      LogicalLocation &nloc = newloc[nn];
      auto pb = FindMeshBlock(n);
      if (nloc.level == oloc.level && newrank[nn] != Globals::my_rank) { // same level, different rank
        for (auto& var : pb->vars_cc_)
          send_reqs.emplace_back(SendSameToSame(nn - nslist[newrank[nn]], 0, 0, 0, var.get())); 
        //sendbuf[sb_idx] =
        //    BufArray1D<Real>(bufs, std::make_pair(buf_offset, buf_offset + bssame));
        //buf_offset += bssame;
        //PrepareSendSameLevel(pb.get(), sendbuf[sb_idx]);
        //tags[sb_idx] = CreateAMRMPITag(nn - nslist[newrank[nn]], 0, 0, 0);
        //dest[sb_idx] = newrank[nn];
        //count[sb_idx] = bssame;
        //sb_idx++;
      } else if (nloc.level > oloc.level) { // c2f
        // c2f must communicate to multiple leaf blocks (unlike f2c, same2same)
        for (int l = 0; l < nleaf; l++) {
          int ox3 = l % 8 / 4; 
          int ox2 = l % 4 / 2; 
          int ox1 = l % 2 / 1;
          for (auto& var : pb->vars_cc_) 
            send_reqs.emplace_back(SendCoarseToFine(nn + l - nslist[newrank[nn + l]], ox1, ox2, ox3, var.get()));
          //if (newrank[nn + l] == Globals::my_rank) continue;
          //sendbuf[sb_idx] =
          //    BufArray1D<Real>(bufs, std::make_pair(buf_offset, buf_offset + bsc2f));
          //buf_offset += bsc2f;
          //PrepareSendCoarseToFineAMR(pb.get(), sendbuf[sb_idx], newloc[nn + l]);
          //tags[sb_idx] = CreateAMRMPITag(nn + l - nslist[newrank[nn + l]], 0, 0, 0);
          //dest[sb_idx] = newrank[nn + l];
          //count[sb_idx] = bsc2f;
          //sb_idx++;
        }      // end loop over nleaf (unique to c2f branch in this step 6)
      } else { // f2c: restrict + pack + send
        const int ox1 = ((oloc.lx1 & 1LL) == 1LL);
        const int ox2 = ((oloc.lx2 & 1LL) == 1LL),
        const int ox3 = ((oloc.lx3 & 1LL) == 1LL);
        for (auto& var : pb->vars_cc_) 
          send_reqs.emplace_back(SendFineToCoarse(nn - nslist[newrank[nn]], ox1, ox2, ox3, var.get()));
        //if (newrank[nn] == Globals::my_rank) continue;
        //sendbuf[sb_idx] =
        //    BufArray1D<Real>(bufs, std::make_pair(buf_offset, buf_offset + bsf2c));
        //buf_offset += bsf2c;
        //PrepareSendFineToCoarseAMR(pb.get(), sendbuf[sb_idx]);
        //int ox1 = ((oloc.lx1 & 1LL) == 1LL), ox2 = ((oloc.lx2 & 1LL) == 1LL),
        //    ox3 = ((oloc.lx3 & 1LL) == 1LL);
        //tags[sb_idx] = CreateAMRMPITag(nn - nslist[newrank[nn]], ox1, ox2, ox3);
        //dest[sb_idx] = newrank[nn];
        //count[sb_idx] = bsf2c;
        //sb_idx++;
      }
    }
    //// wait until all send buffers are filled
    //Kokkos::fence();
    //for (auto idx = 0; idx < sb_idx; idx++) {
    //  PARTHENON_MPI_CHECK(MPI_Isend(sendbuf[idx].data(), count[idx], MPI_PARTHENON_REAL,
    //                                dest[idx], tags[idx], MPI_COMM_WORLD,
    //                                &(req_send[idx])));
    //}
  }                               // if (nsend !=0)
  Kokkos::Profiling::popRegion(); // Step 7
#endif                            // MPI_PARALLEL

  // Step 8. construct a new MeshBlock list (moving the data within the MPI rank)
  Kokkos::Profiling::pushRegion("Step 8: Construct new MeshBlockList");
  {
    RegionSize block_size = GetBlockSize();

    BlockList_t new_block_list(nbe - nbs + 1);
    for (int n = nbs; n <= nbe; n++) {
      int on = newtoold[n];
      if ((ranklist[on] == Globals::my_rank) && (loclist[on].level == newloc[n].level)) {
        // on the same MPI rank and same level -> just move it
        new_block_list[n - nbs] = FindMeshBlock(on);
      } else {
        // on a different refinement level or MPI rank - create a new block
        BoundaryFlag block_bcs[6];
        SetBlockSizeAndBoundaries(newloc[n], block_size, block_bcs);
        // append new block to list of MeshBlocks
        new_block_list[n - nbs] =
            MeshBlock::Make(n, n - nbs, newloc[n], block_size, block_bcs, this, pin,
                            app_in, packages, resolved_packages, gflag);
        // fill the conservative variables
        //if ((loclist[on].level > newloc[n].level)) { // fine to coarse (f2c)
        //  for (int ll = 0; ll < nleaf; ll++) {
        //    if (ranklist[on + ll] != Globals::my_rank) continue;
        //    // fine to coarse on the same MPI rank (different AMR level) - restriction
        //    auto pob = FindMeshBlock(on + ll);

        //    // allocte sparse variables that were allocated on old block
        //    for (auto var : pob->meshblock_data.Get()->GetCellVariableVector()) {
        //      if (var->IsSparse() && var->IsAllocated()) {
        //        new_block_list[n - nbs]->AllocateSparse(var->label());
        //      }
        //    }
        //    FillSameRankFineToCoarseAMR(pob.get(), new_block_list[n - nbs].get(),
        //                                loclist[on + ll]);
        //  }
        //} else if ((loclist[on].level < newloc[n].level) && // coarse to fine (c2f)
        //           (ranklist[on] == Globals::my_rank)) {
        //  // coarse to fine on the same MPI rank (different AMR level) - prolongation
        //  auto pob = FindMeshBlock(on);

        //  // allocte sparse variables that were allocated on old block
        //  for (auto var : pob->meshblock_data.Get()->GetCellVariableVector()) {
        //    if (var->IsSparse() && var->IsAllocated()) {
        //      new_block_list[n - nbs]->AllocateSparse(var->label());
        //    }
        //  }
        //  FillSameRankCoarseToFineAMR(pob.get(), new_block_list[n - nbs].get(),
        //                              newloc[n]);
        //}
      }
    }

    // Replace the MeshBlock list
    block_list = std::move(new_block_list);

    // Ensure local and global ids are correct
    for (int n = nbs; n <= nbe; n++) {
      block_list[n - nbs]->gid = n;
      block_list[n - nbs]->lid = n - nbs;
    }
  }
  Kokkos::Profiling::popRegion(); // Step 8: Construct new MeshBlockList

  // Step 9. Receive the data and load into MeshBlocks
  Kokkos::Profiling::pushRegion("Step 9: Recv data and unpack");
  // This is a test: try MPI_Waitall later.
#ifdef MPI_PARALLEL
  if (nrecv != 0) {
    int test;
    std::vector<bool> received(nrecv, false);
    int rb_idx;
    bool all_received;
    do {
      all_received = true; 
      rb_idx = 0; // recv buffer index
      for (int n = nbs; n <= nbe; n++) {
        int on = newtoold[n];
        LogicalLocation &oloc = loclist[on];
        LogicalLocation &nloc = newloc[n];
        auto pb = FindMeshBlock(n);
        if (oloc.level == nloc.level && ranklist[on] != Globals::my_rank) { // same
          for (auto& var : pb->vars_cc)
            all_received = TryRecvSameToSame(n - nbs, ranklist[on], var.get()) && all_received; 
          //if (ranklist[on] == Globals::my_rank) continue;
          //if (!received[rb_idx]) {
          //  PARTHENON_MPI_CHECK(MPI_Test(&(req_recv[rb_idx]), &test, MPI_STATUS_IGNORE));
          //  if (static_cast<bool>(test)) {
          //    FinishRecvSameLevel(pb.get(), recvbuf[rb_idx]);
          //    received[rb_idx] = true;
          //  }
          //}
          //rb_idx++;
        } else if (oloc.level > nloc.level) { // f2c
          for (int l = 0; l < nleaf; l++) {
            int ox3 = l % 8 / 4; 
            int ox2 = l % 4 / 2; 
            int ox1 = l % 2 / 1;
            for (auto& var : pb->vars_cc)
              all_received = TryReceiveFineToCoarse(n - nbs, ox1, ox2, ox3, var.get(), pb.get()) && all_received;
            //if (ranklist[on + l] == Globals::my_rank) continue;
            //if (!received[rb_idx]) {
            //  PARTHENON_MPI_CHECK(
            //      MPI_Test(&(req_recv[rb_idx]), &test, MPI_STATUS_IGNORE));
            //  if (static_cast<bool>(test)) {
            //    FinishRecvFineToCoarseAMR(pb.get(), recvbuf[rb_idx], loclist[on + l]);
            //    received[rb_idx] = true;
            //  }
            //}
            //rb_idx++;
          }
        } else { // c2f 
          const int ox1 = ((nloc.lx1 & 1LL) == 1LL);
          const int ox2 = ((nloc.lx2 & 1LL) == 1LL),
          const int ox3 = ((nloc.lx3 & 1LL) == 1LL); 
          for (auto& var : pb->vars_cc)
            all_received = TryReceiveCoarseToFine(n - nbs, ox1, ox2, ox3, var.get(), pb.get()) && all_received;
          
          //if (ranklist[on] == Globals::my_rank) continue;
          //if (!received[rb_idx]) {
          //  PARTHENON_MPI_CHECK(MPI_Test(&(req_recv[rb_idx]), &test, MPI_STATUS_IGNORE));
          //  if (static_cast<bool>(test)) {
          //    FinishRecvCoarseToFineAMR(pb.get(), recvbuf[rb_idx]);
          //    received[rb_idx] = true;
          //  }
          //}
          //rb_idx++;
        }
      }
      // rb_idx is a running index, so we repeat the loop until all vals are true
    } while (!all_received);
    //} while (!std::all_of(received.begin(), received.begin() + rb_idx,
    //                      [](bool v) { return v; }));
    Kokkos::fence();
  }
#endif

  // deallocate arrays
  newtoold.clear();
  oldtonew.clear();
#ifdef MPI_PARALLEL
  if (nsend != 0) {
    PARTHENON_MPI_CHECK(MPI_Waitall(send_reqs.size(), send_reqs.data(), MPI_STATUSES_IGNORE));
    //PARTHENON_MPI_CHECK(MPI_Waitall(nsend, req_send, MPI_STATUSES_IGNORE));
    //delete[] sendbuf;
    //delete[] req_send;
  }
  //if (nrecv != 0) {
  //  delete[] recvbuf;
  //  delete[] req_recv;
  //}
#endif
  Kokkos::Profiling::popRegion(); // Step 9
  
  // TODO(LFR): Need to prolongate where necessary 
  // update the lists
  loclist = std::move(newloc);
  ranklist = std::move(newrank);
  costlist = std::move(newcost);

  // re-initialize the MeshBlocks
  for (auto &pmb : block_list) {
    pmb->pbval->SearchAndSetNeighbors(tree, ranklist.data(), nslist.data());
  }
  Initialize(false, pin, app_in);

  ResetLoadBalanceVariables();

  Kokkos::Profiling::popRegion(); // RedistributeAndRefineMeshBlocks
}

// AMR: step 6, branch 1 (same2same: just pack+send)

void Mesh::PrepareSendSameLevel(MeshBlock *pmb, BufArray1D<Real> &sendbuf) {
  // inital offset, data starts after allocation flags
  int p = pmb->vars_cc_.size();

  // subview to set allocation flags
  auto alloc_subview = Kokkos::subview(sendbuf, std::make_pair(0, p));
  auto alloc_subview_h = Kokkos::create_mirror_view(HostMemSpace(), alloc_subview);

  const int f2 = (ndim >= 2) ? 1 : 0; // extra cells/faces from being 2d
  const int f3 = (ndim >= 3) ? 1 : 0; // extra cells/faces from being 3d

  const IndexDomain interior = IndexDomain::interior;
  IndexRange ib = pmb->cellbounds.GetBoundsI(interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(interior);
  // this helper fn is used for AMR and non-refinement load balancing of
  // MeshBlocks. Therefore, unlike PrepareSendCoarseToFineAMR(), etc., it loops over
  // MeshBlock::vars_cc/fc_ containers, not MeshRefinement::pvars_cc/fc_ containers

  // TODO(felker): add explicit check to ensure that elements of pmb->vars_cc/fc_ and
  // pmb->pmr->pvars_cc/fc_ v point to the same objects, if adaptive

  // (C++11) range-based for loop: (automatic type deduction fails when iterating over
  // container with std::reference_wrapper; could use auto var_cc_r = var_cc.get())
  for (int i = 0; i < pmb->vars_cc_.size(); ++i) {
    auto &pvar_cc = pmb->vars_cc_[i];
    alloc_subview_h(i) = pvar_cc->IsAllocated() ? 1.0 : 0.0;
    int nu = pvar_cc->GetDim(4) - 1;
    if (pvar_cc->IsAllocated()) {
      ParArray4D<Real> var_cc = pvar_cc->data.Get<4>();
      BufferUtility::PackData(var_cc, sendbuf, 0, nu, ib.s, ib.e, jb.s, jb.e, kb.s, kb.e,
                              p, pmb);
    } else {
      // increment offset
      p += (nu + 1) * (ib.e + 1 - ib.s) * (jb.e + 1 - jb.s) * (kb.e + 1 - kb.s);
    }
  }
  for (auto &pvar_fc : pmb->vars_fc_) {
    auto &var_fc = *pvar_fc;
    ParArray3D<Real> x1f = var_fc.x1f.Get<3>();
    ParArray3D<Real> x2f = var_fc.x2f.Get<3>();
    ParArray3D<Real> x3f = var_fc.x3f.Get<3>();
    BufferUtility::PackData(x1f, sendbuf, ib.s, ib.e + 1, jb.s, jb.e, kb.s, kb.e, p, pmb);
    BufferUtility::PackData(x2f, sendbuf, ib.s, ib.e, jb.s, jb.e + f2, kb.s, kb.e, p,
                            pmb);
    BufferUtility::PackData(x3f, sendbuf, ib.s, ib.e, jb.s, jb.e, kb.s, kb.e + f3, p,
                            pmb);
  }

  Kokkos::deep_copy(alloc_subview, alloc_subview_h);

  // WARNING(felker): casting from "Real *" to "int *" in order to append single integer
  // to send buffer is slightly unsafe (especially if sizeof(int) > sizeof(Real))
  if (adaptive) {
    Kokkos::deep_copy(pmb->exec_space,
                      Kokkos::View<int, Kokkos::MemoryUnmanaged>(
                          reinterpret_cast<int *>(Kokkos::subview(sendbuf, p).data())),
                      pmb->pmr->deref_count_);
  }
  return;
}

// step 6, branch 2 (c2f: just pack+send)

void Mesh::PrepareSendCoarseToFineAMR(MeshBlock *pb, BufArray1D<Real> &sendbuf,
                                      LogicalLocation &lloc) {
  const int f2 = (ndim >= 2) ? 1 : 0; // extra cells/faces from being 2d
  const int f3 = (ndim >= 3) ? 1 : 0; // extra cells/faces from being 3d
  int ox1 = static_cast<int>((lloc.lx1 & 1LL) == 1LL);
  int ox2 = static_cast<int>((lloc.lx2 & 1LL) == 1LL);
  int ox3 = static_cast<int>((lloc.lx3 & 1LL) == 1LL);
  const IndexDomain interior = IndexDomain::interior;
  // pack
  int il, iu, jl, ju, kl, ku;
  if (ox1 == 0) {
    il = pb->cellbounds.is(interior) - 1;
    iu = pb->cellbounds.is(interior) + pb->block_size.nx1 / 2;
  } else {
    il = pb->cellbounds.is(interior) + pb->block_size.nx1 / 2 - 1;
    iu = pb->cellbounds.ie(interior) + 1;
  }
  if (ox2 == 0) {
    jl = pb->cellbounds.js(interior) - f2;
    ju = pb->cellbounds.js(interior) + pb->block_size.nx2 / 2;
  } else {
    jl = pb->cellbounds.js(interior) + pb->block_size.nx2 / 2 - f2;
    ju = pb->cellbounds.je(interior) + f2;
  }
  if (ox3 == 0) {
    kl = pb->cellbounds.ks(interior) - f3;
    ku = pb->cellbounds.ks(interior) + pb->block_size.nx3 / 2;
  } else {
    kl = pb->cellbounds.ks(interior) + pb->block_size.nx3 / 2 - f3;
    ku = pb->cellbounds.ke(interior) + f3;
  }

  // inital offset, data starts after allocation flags
  int p = pb->pmr->pvars_cc_.size();

  // subview to set allocation flags
  auto alloc_subview = Kokkos::subview(sendbuf, std::make_pair(0, p));
  auto alloc_subview_h = Kokkos::create_mirror_view(HostMemSpace(), alloc_subview);

  for (int i = 0; i < pb->pmr->pvars_cc_.size(); ++i) {
    auto &cc_var = pb->pmr->pvars_cc_[i];
    alloc_subview_h(i) = cc_var->IsAllocated() ? 1.0 : 0.0;
    int nu = cc_var->GetDim(4) - 1;
    if (cc_var->IsAllocated()) {
      ParArray4D<Real> var_cc = cc_var->data.Get<4>();
      BufferUtility::PackData(var_cc, sendbuf, 0, nu, il, iu, jl, ju, kl, ku, p, pb);
    } else {
      BufferUtility::PackZero(sendbuf, 0, nu, il, iu, jl, ju, kl, ku, p, pb);
    }
  }

  for (auto fc_pair : pb->pmr->pvars_fc_) {
    FaceField *var_fc = std::get<0>(fc_pair);
    ParArray3D<Real> x1f = (*var_fc).x1f.Get<3>();
    ParArray3D<Real> x2f = (*var_fc).x2f.Get<3>();
    ParArray3D<Real> x3f = (*var_fc).x3f.Get<3>();
    BufferUtility::PackData(x1f, sendbuf, il, iu + 1, jl, ju, kl, ku, p, pb);
    BufferUtility::PackData(x2f, sendbuf, il, iu, jl, ju + f2, kl, ku, p, pb);
    BufferUtility::PackData(x3f, sendbuf, il, iu, jl, ju, kl, ku + f3, p, pb);
  }

  Kokkos::deep_copy(alloc_subview, alloc_subview_h);

  return;
}

// step 6, branch 3 (f2c: restrict, pack, send)

void Mesh::PrepareSendFineToCoarseAMR(MeshBlock *pb, BufArray1D<Real> &sendbuf) {
  // restrict and pack
  const int f2 = (ndim >= 2) ? 1 : 0; // extra cells/faces from being 2d
  const int f3 = (ndim >= 3) ? 1 : 0; // extra cells/faces from being 3d

  const IndexDomain interior = IndexDomain::interior;
  IndexRange cib = pb->c_cellbounds.GetBoundsI(interior);
  IndexRange cjb = pb->c_cellbounds.GetBoundsJ(interior);
  IndexRange ckb = pb->c_cellbounds.GetBoundsK(interior);

  auto &pmr = pb->pmr;

  // inital offset, data starts after allocation flags
  int p = pmr->pvars_cc_.size();

  // subview to set allocation flags
  auto alloc_subview = Kokkos::subview(sendbuf, std::make_pair(0, p));
  auto alloc_subview_h = Kokkos::create_mirror_view(HostMemSpace(), alloc_subview);

  for (int i = 0; i < pmr->pvars_cc_.size(); ++i) {
    auto &cc_var = pmr->pvars_cc_[i];
    alloc_subview_h(i) = cc_var->IsAllocated() ? 1.0 : 0.0;
    int nu = cc_var->GetDim(4) - 1;
    if (cc_var->IsAllocated()) {
      ParArrayND<Real> var_cc = cc_var->data;
      ParArrayND<Real> coarse_cc = cc_var->coarse_s;
      pmr->RestrictCellCenteredValues(var_cc, coarse_cc, 0, nu, cib.s, cib.e, cjb.s,
                                      cjb.e, ckb.s, ckb.e);
      // TOGO(pgrete) remove temp var once Restrict func interface is updated
      ParArray4D<Real> coarse_cc_ = coarse_cc.Get<4>();
      BufferUtility::PackData(coarse_cc_, sendbuf, 0, nu, cib.s, cib.e, cjb.s, cjb.e,
                              ckb.s, ckb.e, p, pb);
    } else {
      BufferUtility::PackZero(sendbuf, 0, nu, cib.s, cib.e, cjb.s, cjb.e, ckb.s, ckb.e, p,
                              pb);
    }
  }

  for (auto fc_pair : pb->pmr->pvars_fc_) {
    FaceField *var_fc = std::get<0>(fc_pair);
    FaceField *coarse_fc = std::get<1>(fc_pair);
    ParArray3D<Real> x1f = (*coarse_fc).x1f.Get<3>();
    ParArray3D<Real> x2f = (*coarse_fc).x2f.Get<3>();
    ParArray3D<Real> x3f = (*coarse_fc).x3f.Get<3>();
    pmr->RestrictFieldX1((*var_fc).x1f, (*coarse_fc).x1f, cib.s, cib.e + 1, cjb.s, cjb.e,
                         ckb.s, ckb.e);
    BufferUtility::PackData(x1f, sendbuf, cib.s, cib.e + 1, cjb.s, cjb.e, ckb.s, ckb.e, p,
                            pb);
    pmr->RestrictFieldX2((*var_fc).x2f, (*coarse_fc).x2f, cib.s, cib.e, cjb.s, cjb.e + f2,
                         ckb.s, ckb.e);
    BufferUtility::PackData(x2f, sendbuf, cib.s, cib.e, cjb.s, cjb.e + f2, ckb.s, ckb.e,
                            p, pb);
    pmr->RestrictFieldX3((*var_fc).x3f, (*coarse_fc).x3f, cib.s, cib.e, cjb.s, cjb.e,
                         ckb.s, ckb.e + f3);
    BufferUtility::PackData(x3f, sendbuf, cib.s, cib.e, cjb.s, cjb.e, ckb.s, ckb.e + f3,
                            p, pb);
  }

  Kokkos::deep_copy(alloc_subview, alloc_subview_h);

  return;
}

// step 7: f2c, same MPI rank, different level (just restrict+copy, no pack/send)

void Mesh::FillSameRankFineToCoarseAMR(MeshBlock *pob, MeshBlock *pmb,
                                       LogicalLocation &loc) {
  auto &pmr = pob->pmr;
  const IndexDomain interior = IndexDomain::interior;
  int il =
      pmb->cellbounds.is(interior) + ((loc.lx1 & 1LL) == 1LL) * pmb->block_size.nx1 / 2;
  int jl =
      pmb->cellbounds.js(interior) + ((loc.lx2 & 1LL) == 1LL) * pmb->block_size.nx2 / 2;
  int kl =
      pmb->cellbounds.ks(interior) + ((loc.lx3 & 1LL) == 1LL) * pmb->block_size.nx3 / 2;

  IndexRange cib = pob->c_cellbounds.GetBoundsI(interior);
  IndexRange cjb = pob->c_cellbounds.GetBoundsJ(interior);
  IndexRange ckb = pob->c_cellbounds.GetBoundsK(interior);
  // absent a zip() feature for range-based for loops, manually advance the
  // iterator over "SMR/AMR-enrolled" cell-centered quantities on the new
  // MeshBlock in lock-step with pob
  auto pmb_cc_it = pmb->pmr->pvars_cc_.begin();
  // iterate MeshRefinement std::vectors on pob
  for (auto cc_var : pmr->pvars_cc_) {
    const bool fine_allocated = cc_var->IsAllocated();
    if (!(*pmb_cc_it)->IsAllocated()) {
      PARTHENON_REQUIRE_THROWS(!fine_allocated,
                               "Mesh::FillSameRankFineToCoarseAMR: Destination not "
                               "allocated but source allocated");
      pmb_cc_it++;
      continue;
    }
    ParArrayND<Real> var_cc = cc_var->data;
    ParArrayND<Real> coarse_cc = cc_var->coarse_s;
    int nu = cc_var->GetDim(4) - 1;

    if (fine_allocated) {
      pmr->RestrictCellCenteredValues(var_cc, coarse_cc, 0, nu, cib.s, cib.e, cjb.s,
                                      cjb.e, ckb.s, ckb.e);
    }

    // copy from old/original/other MeshBlock (pob) to newly created block (pmb)
    ParArrayND<Real> src = coarse_cc;
    ParArrayND<Real> dst = (*pmb_cc_it)->data;
    int koff = kl - ckb.s;
    int joff = jl - cjb.s;
    int ioff = il - cib.s;
    pmb->par_for(
        "FillSameRankFineToCoarseAMR", 0, nu, ckb.s, ckb.e, cjb.s, cjb.e, cib.s, cib.e,
        KOKKOS_LAMBDA(const int nv, const int k, const int j, const int i) {
          // if the destination (coarse) is allocated, but source (fine) is not allocated,
          // we just fill destination with 0's
          dst(nv, k + koff, j + joff, i + ioff) = fine_allocated ? src(nv, k, j, i) : 0.0;
        });
    pmb_cc_it++;
  }

  const int f2 = (ndim >= 2) ? 1 : 0; // extra cells/faces from being 2d
  const int f3 = (ndim >= 3) ? 1 : 0; // extra cells/faces from being 3d

  auto pmb_fc_it = pmb->pmr->pvars_fc_.begin();
  for (auto fc_pair : pmr->pvars_fc_) {
    FaceField *var_fc = std::get<0>(fc_pair);
    FaceField *coarse_fc = std::get<1>(fc_pair);
    pmr->RestrictFieldX1((*var_fc).x1f, (*coarse_fc).x1f, cib.s, cib.e + 1, cjb.s, cjb.e,
                         ckb.s, ckb.e);
    pmr->RestrictFieldX2((*var_fc).x2f, (*coarse_fc).x2f, cib.s, cib.e, cjb.s, cjb.e + f2,
                         ckb.s, ckb.e);
    pmr->RestrictFieldX3((*var_fc).x3f, (*coarse_fc).x3f, cib.s, cib.e, cjb.s, cjb.e,
                         ckb.s, ckb.e + f3);
    FaceField &src_b = *coarse_fc;
    FaceField &dst_b = *std::get<0>(*pmb_fc_it); // pmb->pfield->b;
    for (int k = kl, fk = ckb.s; fk <= ckb.e; k++, fk++) {
      for (int j = jl, fj = cjb.s; fj <= cjb.e; j++, fj++) {
        for (int i = il, fi = cib.s; fi <= cib.e + 1; i++, fi++)
          dst_b.x1f(k, j, i) = src_b.x1f(fk, fj, fi);
      }
    }
    for (int k = kl, fk = ckb.s; fk <= ckb.e; k++, fk++) {
      for (int j = jl, fj = cjb.s; fj <= cjb.e + f2; j++, fj++) {
        for (int i = il, fi = cib.s; fi <= cib.e; i++, fi++)
          dst_b.x2f(k, j, i) = src_b.x2f(fk, fj, fi);
      }
    }

    int ks = pmb->cellbounds.ks(interior);
    int js = pmb->cellbounds.js(interior);
    if (pmb->block_size.nx2 == 1) {
      int iu = il + pmb->block_size.nx1 / 2 - 1;
      for (int i = il; i <= iu; i++)
        dst_b.x2f(ks, js + 1, i) = dst_b.x2f(ks, js, i);
    }
    for (int k = kl, fk = ckb.s; fk <= ckb.e + f3; k++, fk++) {
      for (int j = jl, fj = cjb.s; fj <= cjb.e; j++, fj++) {
        for (int i = il, fi = cib.s; fi <= cib.e; i++, fi++)
          dst_b.x3f(k, j, i) = src_b.x3f(fk, fj, fi);
      }
    }
    if (pmb->block_size.nx3 == 1) {
      int iu = il + pmb->block_size.nx1 / 2 - 1, ju = jl + pmb->block_size.nx2 / 2 - 1;
      if (pmb->block_size.nx2 == 1) ju = jl;
      for (int j = jl; j <= ju; j++) {
        for (int i = il; i <= iu; i++)
          dst_b.x3f(ks + 1, j, i) = dst_b.x3f(ks, j, i);
      }
    }
    pmb_fc_it++;
  }
  return;
}

// step 7: c2f, same MPI rank, different level (just copy+prolongate, no pack/send)

void Mesh::FillSameRankCoarseToFineAMR(MeshBlock *pob, MeshBlock *pmb,
                                       LogicalLocation &newloc) {
  auto &pmr = pmb->pmr;

  const int f2 = (ndim >= 2) ? 1 : 0; // extra cells/faces from being 2d
  const int f3 = (ndim >= 3) ? 1 : 0; // extra cells/faces from being 3d

  const IndexDomain interior = IndexDomain::interior;
  int il = pob->c_cellbounds.is(interior) - 1;
  int iu = pob->c_cellbounds.ie(interior) + 1;
  int jl = pob->c_cellbounds.js(interior) - f2;
  int ju = pob->c_cellbounds.je(interior) + f2;
  int kl = pob->c_cellbounds.ks(interior) - f3;
  int ku = pob->c_cellbounds.ke(interior) + f3;

  int cis = ((newloc.lx1 & 1LL) == 1LL) * pob->block_size.nx1 / 2 +
            pob->cellbounds.is(interior) - 1;
  int cjs = ((newloc.lx2 & 1LL) == 1LL) * pob->block_size.nx2 / 2 +
            pob->cellbounds.js(interior) - f2;
  int cks = ((newloc.lx3 & 1LL) == 1LL) * pob->block_size.nx3 / 2 +
            pob->cellbounds.ks(interior) - f3;

  auto pob_cc_it = pob->pmr->pvars_cc_.begin();
  // iterate MeshRefinement std::vectors on new pmb
  for (auto cc_var : pmr->pvars_cc_) {
    PARTHENON_REQUIRE_THROWS(cc_var->IsAllocated() == (*pob_cc_it)->IsAllocated(),
                             "Mesh::FillSameRankCoarseToFineAMR: Allocation mismatch");
    if (!cc_var->IsAllocated()) {
      pob_cc_it++;
      continue;
    }

    ParArrayND<Real> var_cc = cc_var->data;
    ParArrayND<Real> coarse_cc = cc_var->coarse_s;
    int nu = var_cc.GetDim(4) - 1;

    ParArrayND<Real> src = (*pob_cc_it)->data;
    ParArrayND<Real> dst = coarse_cc;
    // fill the coarse buffer
    // WARNING: potential Cuda stream pitfall (exec space of coarse and fine MB)
    // Need to make sure that both src and dst are done with all other task up to here
    pob->par_for(
        "FillSameRankCoarseToFineAMR", 0, nu, kl, ku, jl, ju, il, iu,
        KOKKOS_LAMBDA(const int nv, const int k, const int j, const int i) {
          dst(nv, k, j, i) = src(nv, k - kl + cks, j - jl + cjs, i - il + cis);
        });
    // keeping the original, following block for reference to indexing
    // for (int nv = 0; nv <= nu; nv++) {
    //   for (int k = kl, ck = cks; k <= ku; k++, ck++) {
    //     for (int j = jl, cj = cjs; j <= ju; j++, cj++) {
    //       for (int i = il, ci = cis; i <= iu; i++, ci++)
    //         dst(nv, k, j, i) = src(nv, ck, cj, ci);
    //     }
    //   }
    // }
    pmr->ProlongateCellCenteredValues(
        dst, var_cc, 0, nu, pob->c_cellbounds.is(interior),
        pob->c_cellbounds.ie(interior), pob->c_cellbounds.js(interior),
        pob->c_cellbounds.je(interior), pob->c_cellbounds.ks(interior),
        pob->c_cellbounds.ke(interior));
    pob_cc_it++;
  }
  auto pob_fc_it = pob->pmr->pvars_fc_.begin();
  // iterate MeshRefinement std::vectors on new pmb
  for (auto fc_pair : pmr->pvars_fc_) {
    FaceField *var_fc = std::get<0>(fc_pair);
    FaceField *coarse_fc = std::get<1>(fc_pair);

    FaceField &src_b = *std::get<0>(*pob_fc_it);
    FaceField &dst_b = *coarse_fc;
    for (int k = kl, ck = cks; k <= ku; k++, ck++) {
      for (int j = jl, cj = cjs; j <= ju; j++, cj++) {
        for (int i = il, ci = cis; i <= iu + 1; i++, ci++)
          dst_b.x1f(k, j, i) = src_b.x1f(ck, cj, ci);
      }
    }
    for (int k = kl, ck = cks; k <= ku; k++, ck++) {
      for (int j = jl, cj = cjs; j <= ju + f2; j++, cj++) {
        for (int i = il, ci = cis; i <= iu; i++, ci++)
          dst_b.x2f(k, j, i) = src_b.x2f(ck, cj, ci);
      }
    }
    for (int k = kl, ck = cks; k <= ku + f3; k++, ck++) {
      for (int j = jl, cj = cjs; j <= ju; j++, cj++) {
        for (int i = il, ci = cis; i <= iu; i++, ci++)
          dst_b.x3f(k, j, i) = src_b.x3f(ck, cj, ci);
      }
    }
    pmr->ProlongateSharedFieldX1(
        dst_b.x1f, (*var_fc).x1f, pob->c_cellbounds.is(interior),
        pob->c_cellbounds.ie(interior) + 1, pob->c_cellbounds.js(interior),
        pob->c_cellbounds.je(interior), pob->c_cellbounds.ks(interior),
        pob->c_cellbounds.ke(interior));
    pmr->ProlongateSharedFieldX2(
        dst_b.x2f, (*var_fc).x2f, pob->c_cellbounds.is(interior),
        pob->c_cellbounds.ie(interior), pob->c_cellbounds.js(interior),
        pob->c_cellbounds.je(interior) + f2, pob->c_cellbounds.ks(interior),
        pob->c_cellbounds.ke(interior));
    pmr->ProlongateSharedFieldX3(
        dst_b.x3f, (*var_fc).x3f, pob->c_cellbounds.is(interior),
        pob->c_cellbounds.ie(interior), pob->c_cellbounds.js(interior),
        pob->c_cellbounds.je(interior), pob->c_cellbounds.ks(interior),
        pob->c_cellbounds.ke(interior) + f3);
    pmr->ProlongateInternalField(
        *var_fc, pob->c_cellbounds.is(interior), pob->c_cellbounds.ie(interior),
        pob->c_cellbounds.js(interior), pob->c_cellbounds.je(interior),
        pob->c_cellbounds.ks(interior), pob->c_cellbounds.ke(interior));

    pob_fc_it++;
  }
  return;
}

// step 8 (receive and load), branch 1 (same2same: unpack)
void Mesh::FinishRecvSameLevel(MeshBlock *pmb, BufArray1D<Real> &recvbuf) {
  // inital offset, data starts after allocation flags
  int p = pmb->vars_cc_.size();

  // subview to set allocation flags
  auto alloc_subview = Kokkos::subview(recvbuf, std::make_pair(0, p));
  auto alloc_subview_h =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), alloc_subview);

  const int f2 = (ndim >= 2) ? 1 : 0; // extra cells/faces from being 2d
  const int f3 = (ndim >= 3) ? 1 : 0; // extra cells/faces from being 3d

  const IndexDomain interior = IndexDomain::interior;
  IndexRange ib = pmb->cellbounds.GetBoundsI(interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(interior);
  
  for (int i = 0; i < pmb->vars_cc_.size(); ++i) {
    auto &pvar_cc = pmb->vars_cc_[i];
    int nu = pvar_cc->GetDim(4) - 1;

    if (alloc_subview_h(i) == 1.0) {
      // allocated on sending block
      if (!pvar_cc->IsAllocated()) {
        // need to allocate locally
        pmb->AllocateSparse(pvar_cc->label());
      }
      PARTHENON_REQUIRE_THROWS(
          pvar_cc->IsAllocated(),
          "FinishRecvSameLevel: Received variable that was allocated on sending "
          "block but it is not allocated on receiving block");
      ParArray4D<Real> var_cc_ = pvar_cc->data.Get<4>();
      BufferUtility::UnpackData(recvbuf, var_cc_, 0, nu, ib.s, ib.e, jb.s, jb.e, kb.s,
                                kb.e, p, pmb);
    } else {
      // increment offset
      p += (nu + 1) * (ib.e + 1 - ib.s) * (jb.e + 1 - jb.s) * (kb.e + 1 - kb.s);
      PARTHENON_REQUIRE_THROWS(
          !pvar_cc->IsAllocated(),
          "FinishRecvSameLevel: Received variable that was not allocated on sending "
          "block but it is allocated on receiving block");
    }
  }
  for (auto &pvar_fc : pmb->vars_fc_) {
    auto &var_fc = *pvar_fc;
    ParArray3D<Real> x1f = var_fc.x1f.Get<3>();
    ParArray3D<Real> x2f = var_fc.x2f.Get<3>();
    ParArray3D<Real> x3f = var_fc.x3f.Get<3>();
    BufferUtility::UnpackData(recvbuf, x1f, ib.s, ib.e + 1, jb.s, jb.e, kb.s, kb.e, p,
                              pmb);
    BufferUtility::UnpackData(recvbuf, x2f, ib.s, ib.e, jb.s, jb.e + f2, kb.s, kb.e, p,
                              pmb);
    BufferUtility::UnpackData(recvbuf, x3f, ib.s, ib.e, jb.s, jb.e, kb.s, kb.e + f3, p,
                              pmb);
    if (pmb->block_size.nx2 == 1) {
      for (int i = ib.s; i <= ib.e; i++)
        var_fc.x2f(kb.s, jb.s + 1, i) = var_fc.x2f(kb.s, jb.s, i);
    }
    if (pmb->block_size.nx3 == 1) {
      for (int j = jb.s; j <= jb.e; j++) {
        for (int i = ib.s; i <= ib.e; i++)
          var_fc.x3f(kb.s + 1, j, i) = var_fc.x3f(kb.s, j, i);
      }
    }
  }
  // WARNING(felker): casting from "Real *" to "int *" in order to read single
  // appended integer from received buffer is slightly unsafe
  if (adaptive) {
    Kokkos::deep_copy(pmb->exec_space, pmb->pmr->deref_count_,
                      Kokkos::View<int, Kokkos::MemoryUnmanaged>(
                          reinterpret_cast<int *>(Kokkos::subview(recvbuf, p).data())));
  }
  return;
}

// step 8 (receive and load), branch 2 (f2c: unpack)
void Mesh::FinishRecvFineToCoarseAMR(MeshBlock *pb, BufArray1D<Real> &recvbuf,
                                     LogicalLocation &lloc) {
  const int f2 = (ndim >= 2) ? 1 : 0; // extra cells/faces from being 2d
  const int f3 = (ndim >= 3) ? 1 : 0; // extra cells/faces from being 3d

  const IndexDomain interior = IndexDomain::interior;
  IndexRange ib = pb->cellbounds.GetBoundsI(interior);
  IndexRange jb = pb->cellbounds.GetBoundsJ(interior);
  IndexRange kb = pb->cellbounds.GetBoundsK(interior);

  int ox1 = static_cast<int>((lloc.lx1 & 1LL) == 1LL);
  int ox2 = static_cast<int>((lloc.lx2 & 1LL) == 1LL);
  int ox3 = static_cast<int>((lloc.lx3 & 1LL) == 1LL);
  int il, iu, jl, ju, kl, ku;

  if (ox1 == 0)
    il = ib.s, iu = ib.s + pb->block_size.nx1 / 2 - 1;
  else
    il = ib.s + pb->block_size.nx1 / 2, iu = ib.e;
  if (ox2 == 0)
    jl = jb.s, ju = jb.s + pb->block_size.nx2 / 2 - f2;
  else
    jl = jb.s + pb->block_size.nx2 / 2, ju = jb.e;
  if (ox3 == 0)
    kl = kb.s, ku = kb.s + pb->block_size.nx3 / 2 - f3;
  else
    kl = kb.s + pb->block_size.nx3 / 2, ku = kb.e;

  // inital offset, data starts after allocation flags
  int p = pb->pmr->pvars_cc_.size();

  // subview to set allocation flags
  auto alloc_subview = Kokkos::subview(recvbuf, std::make_pair(0, p));
  auto alloc_subview_h =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), alloc_subview);

  for (int i = 0; i < pb->pmr->pvars_cc_.size(); ++i) {
    auto &cc_var = pb->pmr->pvars_cc_[i];
    int nu = cc_var->GetDim(4) - 1;

    if ((alloc_subview_h(i) == 1.0) && !cc_var->IsAllocated()) {
      // need to allocate locally
      pb->AllocateSparse(cc_var->label());
      PARTHENON_REQUIRE_THROWS(
          cc_var->IsAllocated(),
          "Mesh::FinishRecvFineToCoarseAMR: Failed to allocate variable");
    }

    if (cc_var->IsAllocated()) {
      ParArray4D<Real> var_cc = cc_var->data.Get<4>();
      BufferUtility::UnpackData(recvbuf, var_cc, 0, nu, il, iu, jl, ju, kl, ku, p, pb);
    } else {
      // increment offset
      p += (nu + 1) * (iu + 1 - il) * (ju + 1 - jl) * (ku + 1 - kl);
    }
  }
  for (auto fc_pair : pb->pmr->pvars_fc_) {
    FaceField *var_fc = std::get<0>(fc_pair);
    FaceField &dst_b = *var_fc;
    ParArray3D<Real> x1f = dst_b.x1f.Get<3>();
    ParArray3D<Real> x2f = dst_b.x2f.Get<3>();
    ParArray3D<Real> x3f = dst_b.x3f.Get<3>();
    BufferUtility::UnpackData(recvbuf, x1f, il, iu + 1, jl, ju, kl, ku, p, pb);
    BufferUtility::UnpackData(recvbuf, x2f, il, iu, jl, ju + f2, kl, ku, p, pb);
    BufferUtility::UnpackData(recvbuf, x3f, il, iu, jl, ju, kl, ku + f3, p, pb);
    if (pb->block_size.nx2 == 1) {
      for (int i = il; i <= iu; i++)
        dst_b.x2f(kb.s, jb.s + 1, i) = dst_b.x2f(kb.s, jb.s, i);
    }
    if (pb->block_size.nx3 == 1) {
      for (int j = jl; j <= ju; j++) {
        for (int i = il; i <= iu; i++)
          dst_b.x3f(kb.s + 1, j, i) = dst_b.x3f(kb.s, j, i);
      }
    }
  }
  return;
}

// step 8 (receive and load), branch 2 (c2f: unpack+prolongate)
void Mesh::FinishRecvCoarseToFineAMR(MeshBlock *pb, BufArray1D<Real> &recvbuf) {
  const int f2 = (ndim >= 2) ? 1 : 0; // extra cells/faces from being 2d
  const int f3 = (ndim >= 3) ? 1 : 0; // extra cells/faces from being 3d
  auto &pmr = pb->pmr;
  // inital offset, data starts after allocation flags
  int p = pmr->pvars_cc_.size();

  // subview to set allocation flags
  auto alloc_subview = Kokkos::subview(recvbuf, std::make_pair(0, p));
  auto alloc_subview_h = Kokkos::create_mirror_view(HostMemSpace(), alloc_subview);

  const IndexDomain interior = IndexDomain::interior;
  IndexRange cib = pb->c_cellbounds.GetBoundsI(interior);
  IndexRange cjb = pb->c_cellbounds.GetBoundsJ(interior);
  IndexRange ckb = pb->c_cellbounds.GetBoundsK(interior);

  int il = cib.s - 1, iu = cib.e + 1, jl = cjb.s - f2, ju = cjb.e + f2, kl = ckb.s - f3,
      ku = ckb.e + f3;

  for (int i = 0; i < pmr->pvars_cc_.size(); ++i) {
    auto &cc_var = pmr->pvars_cc_[i];
    int nu = cc_var->GetDim(4) - 1;
    if ((alloc_subview_h(i) == 1.0) && !cc_var->IsAllocated()) {
      // need to allocate locally
      pb->AllocateSparse(cc_var->label());
    }
  }

  for (int i = 0; i < pmr->pvars_cc_.size(); ++i) {
    auto &cc_var = pmr->pvars_cc_[i];
    int nu = cc_var->GetDim(4) - 1;
    //PARTHENON_REQUIRE_THROWS(
        //cc_var->IsAllocated(),
        //"Mesh::FinishRecvCoarseToFineAMR: Failed to allocate variable " + cc_var->label());
    if (cc_var->IsAllocated()) {
      ParArrayND<Real> var_cc = cc_var->data;
      PARTHENON_REQUIRE_THROWS(nu == cc_var->GetDim(4) - 1, "nu mismatch");
      ParArrayND<Real> coarse_cc = cc_var->coarse_s;
      ParArray4D<Real> coarse_cc_ = coarse_cc.Get<4>();
      BufferUtility::UnpackData(recvbuf, coarse_cc_, 0, nu, il, iu, jl, ju, kl, ku, p,
                                pb);
      pmr->ProlongateCellCenteredValues(coarse_cc, var_cc, 0, nu, cib.s, cib.e, cjb.s,
                                        cjb.e, ckb.s, ckb.e);
    } else {
      // increment offset
      p += (nu + 1) * (iu + 1 - il) * (ju + 1 - jl) * (ku + 1 - kl);
    }
  }

  for (auto fc_pair : pb->pmr->pvars_fc_) {
    FaceField *var_fc = std::get<0>(fc_pair);
    FaceField *coarse_fc = std::get<1>(fc_pair);

    ParArray3D<Real> x1f = (*coarse_fc).x1f.Get<3>();
    ParArray3D<Real> x2f = (*coarse_fc).x2f.Get<3>();
    ParArray3D<Real> x3f = (*coarse_fc).x3f.Get<3>();
    BufferUtility::UnpackData(recvbuf, x1f, il, iu + 1, jl, ju, kl, ku, p, pb);
    BufferUtility::UnpackData(recvbuf, x2f, il, iu, jl, ju + f2, kl, ku, p, pb);
    BufferUtility::UnpackData(recvbuf, x3f, il, iu, jl, ju, kl, ku + f3, p, pb);
    pmr->ProlongateSharedFieldX1((*coarse_fc).x1f, (*var_fc).x1f, cib.s, cib.e + 1, cjb.s,
                                 cjb.e, ckb.s, ckb.e);
    pmr->ProlongateSharedFieldX2((*coarse_fc).x2f, (*var_fc).x2f, cib.s, cib.e, cjb.s,
                                 cjb.e + f2, ckb.s, ckb.e);
    pmr->ProlongateSharedFieldX3((*coarse_fc).x3f, (*var_fc).x3f, cib.s, cib.e, cjb.s,
                                 cjb.e, ckb.s, ckb.e + f3);
    pmr->ProlongateInternalField(*var_fc, cib.s, cib.e, cjb.s, cjb.e, ckb.s, ckb.e);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn int CreateAMRMPITag(int lid, int ox1, int ox2, int ox3)
//  \brief calculate an MPI tag for AMR block transfer
// tag = local id of destination (remaining bits) + ox1(1 bit) + ox2(1 bit) + ox3(1 bit)
//       + physics(5 bits)

// See comments on BoundaryBase::CreateBvalsMPITag()

} // namespace parthenon
