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

  // Step 7 - eps: Restrict fine to coarse buffers 
  for (int on = onbs; on <= onbe; on++) { 
    int nn = oldtonew[on]; 
    if (newloc[nn].level < loclist[on].level) {
      const IndexDomain interior = IndexDomain::interior;
      auto pmb = FindMeshBlock(on); 
      IndexRange cib = pmb->c_cellbounds.GetBoundsI(interior);
      IndexRange cjb = pmb->c_cellbounds.GetBoundsJ(interior);
      IndexRange ckb = pmb->c_cellbounds.GetBoundsK(interior);
      // Need to restrict this block before doing sends
      for (auto& var : pmb->vars_cc_) {
        if (var->IsAllocated()) {
          ParArrayND<Real> fb = var->data; 
          ParArrayND<Real> cb = var->coarse_s;
          pmb->pmr->RestrictCellCenteredValues(fb, cb, 0, var->GetDim(4) - 1, 
                                               cib.s, cib.e, cjb.s,cjb.e, ckb.s, ckb.e); 
        }
      }
    } 
  }
  Kokkos::fence(); 

#ifdef MPI_PARALLEL
  // Step 7. Send data from old to new blocks
  Kokkos::Profiling::pushRegion("Step 7: Send");
  std::vector<MPI_Request> send_reqs; 
  for (int n = onbs; n <= onbe; n++) {
    int nn = oldtonew[n];
    LogicalLocation &oloc = loclist[n];
    LogicalLocation &nloc = newloc[nn];
    auto pb = FindMeshBlock(n);
    if (nloc.level == oloc.level && newrank[nn] != Globals::my_rank) { // same level, different rank
      for (auto& var : pb->vars_cc_)
        send_reqs.emplace_back(SendSameToSame(nn - nslist[newrank[nn]], newrank[nn], var.get())); 
    } else if (nloc.level > oloc.level) { // c2f
      // c2f must communicate to multiple leaf blocks (unlike f2c, same2same)
      for (int l = 0; l < nleaf; l++) {
        LogicalLocation &nloc = newloc[nn + l]; 
        const int nl = nn + l; // Leaf block index in new global block list 
        for (auto& var : pb->vars_cc_) 
          send_reqs.emplace_back(SendCoarseToFine(nl - nslist[newrank[nl]], newrank[nl], 
                                                  nloc, var.get()));
      }      // end loop over nleaf (unique to c2f branch in this step 6)
    } else if (nloc.level < oloc.level) { // f2c: restrict + pack + send
      for (auto& var : pb->vars_cc_) 
        send_reqs.emplace_back(SendFineToCoarse(nn - nslist[newrank[nn]], newrank[nn], 
                                                oloc, var.get()));
    }
  }                              
  Kokkos::Profiling::popRegion(); // Step 7
#endif                            // MPI_PARALLEL

  // Step 8. construct a new MeshBlock list (moving the data within the MPI rank)
  Kokkos::Profiling::pushRegion("Step 8: Construct new MeshBlockList");
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
    }
  }

  // Replace the MeshBlock list
  auto old_block_list = std::move(block_list);
  block_list = std::move(new_block_list);

  // Ensure local and global ids are correct
  for (int n = nbs; n <= nbe; n++) {
    block_list[n - nbs]->gid = n;
    block_list[n - nbs]->lid = n - nbs;
  }
  Kokkos::Profiling::popRegion(); // Step 8: Construct new MeshBlockList

  // Step 9. Receive the data and load into MeshBlocks
  Kokkos::Profiling::pushRegion("Step 9: Recv data and unpack");
#ifdef MPI_PARALLEL
    int test;
    bool all_received;
    int niter = 0;
    std::vector<bool> finished((nbe - nbs + 1) * FindMeshBlock(nbs)->vars_cc_.size() * 8, false);
    do {
      all_received = true; 
      niter++; 
      int idx = 0; 
      for (int n = nbs; n <= nbe; n++) {
        int on = newtoold[n];
        LogicalLocation &oloc = loclist[on];
        LogicalLocation &nloc = newloc[n];
        auto pb = FindMeshBlock(n);
        if (oloc.level == nloc.level && ranklist[on] != Globals::my_rank) { // same level, different rank
          for (auto& var : pb->vars_cc_) {
            if (!finished[idx]) finished[idx] = TryRecvSameToSame(n - nbs, ranklist[on], var.get(), pb.get());
            all_received = finished[idx++] && all_received; 
          }
        } else if (oloc.level > nloc.level) { // f2c
          for (int l = 0; l < nleaf; l++) {
            LogicalLocation &oloc = loclist[on + l];
            for (auto& var : pb->vars_cc_) {
              if (!finished[idx]) finished[idx] = TryRecvFineToCoarse(n - nbs, ranklist[on + l], oloc, var.get(), pb.get());
              all_received = finished[idx++] && all_received; 
            }
          }
        } else if (oloc.level < nloc.level) { // c2f 
          for (auto& var : pb->vars_cc_) {
            if (!finished[idx]) finished[idx] = TryRecvCoarseToFine(n - nbs, ranklist[on], nloc, var.get(), pb.get());
            all_received = finished[idx++] && all_received;
          }
        }
      }
      // rb_idx is a running index, so we repeat the loop until all vals are true
    } while (!all_received && niter < 1e4);
    if (!all_received) PARTHENON_FAIL("AMR Receive failed");
    Kokkos::fence();

    // Prolongate blocks that had a coarse buffer filled (i.e. c2f blocks) 
    for (int nn = nbs; nn <= nbe; nn++) { 
      int on = newtoold[nn]; 
      LogicalLocation &oloc = loclist[on]; 
      LogicalLocation &nloc = newloc[nn];
      auto pmb = FindMeshBlock(nn); 
      if (nloc.level > oloc.level) {
        const IndexDomain interior = IndexDomain::interior;
        IndexRange cib = pmb->c_cellbounds.GetBoundsI(interior);
        IndexRange cjb = pmb->c_cellbounds.GetBoundsJ(interior);
        IndexRange ckb = pmb->c_cellbounds.GetBoundsK(interior);
        // Need to restrict this block before doing sends
        for (auto& var : pmb->vars_cc_) {
          if (var->IsAllocated()) {
            ParArrayND<Real> fb = var->data; 
            ParArrayND<Real> cb = var->coarse_s;
            pmb->pmr->ProlongateCellCenteredValues(cb, fb, 0, var->GetDim(4) - 1, 
                                                   cib.s, cib.e, cjb.s, cjb.e, ckb.s, ckb.e);                                      
          } 
        }
      } 
    }
#endif
  // deallocate arrays
  newtoold.clear();
  oldtonew.clear();
#ifdef MPI_PARALLEL
  if (send_reqs.size() != 0)
    PARTHENON_MPI_CHECK(MPI_Waitall(send_reqs.size(), send_reqs.data(), MPI_STATUSES_IGNORE));
#endif
  Kokkos::Profiling::popRegion(); // Step 9
  
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

int CreateAMRMPITag(int lid, int ox1, int ox2, int ox3) {
  // the trailing zero is used as "id" to indicate an AMR related tag
  return (lid << 8) | (ox1 << 7) | (ox2 << 6) | (ox3 << 5) | 0;
}

MPI_Request Mesh::SendCoarseToFine(int lid_recv, int dest_rank, const LogicalLocation &fine_loc, CellVariable<Real> *var) {
  MPI_Request req;
  MPI_Comm comm = mpi_comm_map_[var->label()];
  const int ox1 = ((fine_loc.lx1 & 1LL) == 1LL);
  const int ox2 = ((fine_loc.lx2 & 1LL) == 1LL);
  const int ox3 = ((fine_loc.lx3 & 1LL) == 1LL); 
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

bool Mesh::TryRecvCoarseToFine(int lid_recv, int send_rank, const LogicalLocation &fine_loc, 
                         CellVariable<Real>* var, MeshBlock *pmb) {
  static const IndexRange ib = pmb->c_cellbounds.GetBoundsI(IndexDomain::entire);
  static const IndexRange jb = pmb->c_cellbounds.GetBoundsJ(IndexDomain::entire);
  static const IndexRange kb = pmb->c_cellbounds.GetBoundsK(IndexDomain::entire);
  
  static const IndexRange ib_int = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  static const IndexRange jb_int = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  static const IndexRange kb_int = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  
  MPI_Comm comm = mpi_comm_map_[var->label()];
  const int ox1 = ((fine_loc.lx1 & 1LL) == 1LL);
  const int ox2 = ((fine_loc.lx2 & 1LL) == 1LL);
  const int ox3 = ((fine_loc.lx3 & 1LL) == 1LL); 
  int tag = CreateAMRMPITag(lid_recv, ox1, ox2, ox3);
  
  int test; 
  MPI_Status status; 
  PARTHENON_MPI_CHECK(MPI_Iprobe(send_rank, tag, comm, &test, &status));
  if (test) {
    int size; 
    PARTHENON_MPI_CHECK(MPI_Get_count(&status, MPI_PARTHENON_REAL, &size));
    if (size > 0) {
      if (!pmb->IsAllocated(var->label())) pmb->AllocateSparse(var->label()); 
      PARTHENON_MPI_CHECK(MPI_Recv(var->data.data(), var->data.size(),
                                   MPI_PARTHENON_REAL, send_rank, tag, comm, MPI_STATUS_IGNORE));
      const int ks = (ox3 == 0) ? 0 : (kb_int.e - kb_int.s + 1) / 2;
      const int js = (ox2 == 0) ? 0 : (jb_int.e - jb_int.s + 1) / 2;
      const int is = (ox1 == 0) ? 0 : (ib_int.e - ib_int.s + 1) / 2;
      auto cb = var->coarse_s; 
      auto fb = var->data; 
      const int nt = fb.GetDim(6) - 1;
      const int nu = fb.GetDim(5) - 1;
      const int nv = fb.GetDim(4) - 1;
      const int t = 0; 
      const int u = 0; 
      parthenon::par_for(DEFAULT_LOOP_PATTERN, 
            "FillSameRankCoarseToFineAMR", DevExecSpace(), 0, nv, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA(const int v, const int k, const int j, const int i) {
            cb(t, u, v, k, j, i) = fb(t, u, v, k + ks, j + js, i + is);
          }); 
    } else {
      if (pmb->IsAllocated(var->label())) pmb->DeallocateSparse(var->label()); 
      PARTHENON_MPI_CHECK(MPI_Recv(var->data.data(), 0, MPI_PARTHENON_REAL, send_rank, tag, comm, MPI_STATUS_IGNORE)); 
    }
  }
  
  return test; 
} 

MPI_Request Mesh::SendFineToCoarse(int lid_recv, int dest_rank, const LogicalLocation &fine_loc, CellVariable<Real> *var) {
  MPI_Request req;
  MPI_Comm comm = mpi_comm_map_[var->label()];
  const int ox1 = ((fine_loc.lx1 & 1LL) == 1LL);
  const int ox2 = ((fine_loc.lx2 & 1LL) == 1LL);
  const int ox3 = ((fine_loc.lx3 & 1LL) == 1LL); 
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

bool Mesh::TryRecvFineToCoarse(int lid_recv, int send_rank, const LogicalLocation &fine_loc, CellVariable<Real> *var, MeshBlock *pmb) {
  static const IndexRange ib = pmb->c_cellbounds.GetBoundsI(IndexDomain::interior);
  static const IndexRange jb = pmb->c_cellbounds.GetBoundsJ(IndexDomain::interior);
  static const IndexRange kb = pmb->c_cellbounds.GetBoundsK(IndexDomain::interior);
  
  static const IndexRange ib_int = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  static const IndexRange jb_int = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  static const IndexRange kb_int = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  
  MPI_Comm comm = mpi_comm_map_[var->label()];
  const int ox1 = ((fine_loc.lx1 & 1LL) == 1LL);
  const int ox2 = ((fine_loc.lx2 & 1LL) == 1LL);
  const int ox3 = ((fine_loc.lx3 & 1LL) == 1LL); 
  int tag = CreateAMRMPITag(lid_recv, ox1, ox2, ox3);

  const int ks = (ox3 == 0) ? 0 : (kb_int.e - kb_int.s + 1) / 2;
  const int js = (ox2 == 0) ? 0 : (jb_int.e - jb_int.s + 1) / 2;
  const int is = (ox1 == 0) ? 0 : (ib_int.e - ib_int.s + 1) / 2;
  
  int test; 
  MPI_Status status; 
  PARTHENON_MPI_CHECK(MPI_Iprobe(send_rank, tag, comm, &test, &status));
  if (test) {
    int size;
    PARTHENON_MPI_CHECK(MPI_Get_count(&status, MPI_PARTHENON_REAL, &size));
    if (size > 0) {
      if (!pmb->IsAllocated(var->label())) pmb->AllocateSparse(var->label()); 
      // This has to be an MPI_Recv w/o buffering
      PARTHENON_MPI_CHECK(MPI_Recv(var->coarse_s.data(), var->coarse_s.size(),
                                   MPI_PARTHENON_REAL, send_rank, tag, comm, MPI_STATUS_IGNORE));
      auto fb = var->data; 
      auto cb = var->coarse_s;
      const int nt = fb.GetDim(6) - 1;
      const int nu = fb.GetDim(5) - 1;
      const int nv = fb.GetDim(4) - 1;
      const int t = 0; 
      const int u = 0; 
      parthenon::par_for(DEFAULT_LOOP_PATTERN, 
            "FillSameRankCoarseToFineAMR", DevExecSpace(), 0, nv, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA(const int v, const int k, const int j, const int i) {
            fb(t, u, v, k + ks, j + js, i + is) = cb(t, u, v, k, j, i);
          }); 
      // We have to block here w/o buffering so that the write is guaranteed to be finished 
      // before we get here again 
      Kokkos::fence();
    } else { 
      PARTHENON_MPI_CHECK(MPI_Recv(var->data.data(), 0,
                                   MPI_PARTHENON_REAL, send_rank, tag, comm, MPI_STATUS_IGNORE)); 
    }
  }
  
  return test;  
}

MPI_Request Mesh::SendSameToSame(int lid_recv, int dest_rank, CellVariable<Real> *var) {
  return SendCoarseToFine(lid_recv, dest_rank, LogicalLocation(), var); 
}

bool Mesh::TryRecvSameToSame(int lid_recv, int send_rank, CellVariable<Real> *var, MeshBlock *pmb) {
  MPI_Comm comm = mpi_comm_map_[var->label()];
  int tag = CreateAMRMPITag(lid_recv, 0, 0, 0);

  int test; 
  MPI_Status status; 
  PARTHENON_MPI_CHECK(MPI_Iprobe(send_rank, tag, comm, &test, &status));
  if (test) {
    int size;
    PARTHENON_MPI_CHECK(MPI_Get_count(&status, MPI_PARTHENON_REAL, &size));
    if (size > 0) {
      if (!pmb->IsAllocated(var->label())) pmb->AllocateSparse(var->label()); 
      PARTHENON_MPI_CHECK(MPI_Recv(var->data.data(), var->data.size(),
                                   MPI_PARTHENON_REAL, send_rank, tag, comm, MPI_STATUS_IGNORE));
    } else { 
      if (pmb->IsAllocated(var->label())) pmb->DeallocateSparse(var->label());
      PARTHENON_MPI_CHECK(MPI_Recv(var->data.data(), 0,
                                   MPI_PARTHENON_REAL, send_rank, tag, comm, MPI_STATUS_IGNORE)); 
    }
  }
  return test;  
}

} // namespace parthenon
