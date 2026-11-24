//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2025 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2026. Triad National Security, LLC. All rights reserved.
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

// This file was made in part with generative AI.

//! \file mesh-amr_loadbalance.cpp
//  \brief implementation of Mesh::AdaptiveMeshRefinement() and related utilities

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "parthenon_mpi.hpp"

#include "bvals/boundary_conditions.hpp"
#include "bvals/comms/bvals_in_one.hpp"
#include "bvals/comms/coalesced_buffers.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "interface/mesh_data.hpp"
#include "interface/update.hpp"
#include "mesh/forest/block_ownership.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_neighbors.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/swarm_amr_remesh.hpp"
#include "parthenon_arrays.hpp"
#include "tasks/tasks.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

namespace {

// Wait for all outstanding boundary communication to finish, then discard every
// communication buffer, coalesced message structure, and tag. Remesh communication
// builds its own (temporary) channels on top of an empty communication infrastructure,
// and the ordinary halo infrastructure is rebuilt for the new mesh afterwards.
void ResetCommunicationInfrastructure(Mesh *pmesh) {
  bool can_delete;
  std::int64_t test_iters = 0;
  constexpr std::int64_t max_it = 1e10;
  do {
    can_delete = true;
    for (auto &[k, comm] : pmesh->boundary_comm_map) {
      can_delete = comm.IsSafeToDelete() && can_delete;
    }
    test_iters++;
  } while (!can_delete && test_iters < max_it);
  PARTHENON_REQUIRE(test_iters < max_it,
                    "Too many iterations waiting to delete communication buffers.");
  pmesh->boundary_comm_map.clear();
  pmesh->pcoalesced_comms->clear();
  pmesh->tag_map.clear();
}

// Complete once all of the remesh sends from md have finished. Some MPI implementations
// only progress rendezvous sends while the sender is inside MPI, so leaving the remesh
// with sends in flight can stall the receiving ranks while this rank does MPI-free work.
TaskStatus RemeshSendsComplete(std::shared_ptr<MeshData<Real>> &md) {
  Mesh *pmesh = md->GetMeshPointer();
  auto &cache = md->GetBvarsCache().GetSubCache(BoundaryType::remesh_send, true);
  bool done = std::all_of(cache.buf_vec.begin(), cache.buf_vec.end(),
                          [](auto pbuf) { return pbuf->IsAvailableForWrite(); });
  if (pmesh->do_coalesced_comms)
    done = pmesh->pcoalesced_comms->IsAvailableForWrite(md.get(),
                                                        BoundaryType::remesh_send) &&
           done;
  return done ? TaskStatus::complete : TaskStatus::incomplete;
}

// Gids of the blocks in the other mesh (described by gid_of) that overlap loc. For a
// one-level remesh this is the same location, its parent, or all of its daughters.
std::vector<int>
GetOverlappingGids(const LogicalLocation &loc,
                   const std::unordered_map<LogicalLocation, int> &gid_of, int ndim) {
  auto same = gid_of.find(loc);
  if (same != gid_of.end()) return std::vector<int>{same->second};
  auto parent = gid_of.find(loc.GetParent());
  if (parent != gid_of.end()) return std::vector<int>{parent->second};
  std::vector<int> gids;
  for (const auto &d : loc.GetDaughters(ndim)) {
    auto it = gid_of.find(d);
    PARTHENON_REQUIRE(it != gid_of.end(),
                      "Remeshing only supports one level of refinement per block.");
    gids.push_back(it->second);
  }
  std::sort(gids.begin(), gids.end());
  return gids;
}

} // namespace

//----------------------------------------------------------------------------------------
// \brief Build the remesh "neighbor" lists connecting old blocks on this rank to the new
// blocks that overlap them, and new blocks on this rank (gids nbs to nbe) to the old
// blocks that overlap them. The old blocks' lists are set directly; the new blocks'
// lists are returned (indexed by gid - nbs), since the new blocks may not exist yet.
// Blocks that are reused in place (same location, same rank) get no channels. Old and
// new blocks can share a gid, which is unambiguous since a remesh channel always goes
// from an old block to a new block.

std::vector<std::vector<NeighborBlock>>
Mesh::SetRemeshNeighbors_(const BlockList_t &old_blocks, int onbs, int nbs, int nbe,
                          const std::vector<LogicalLocation> &newloc,
                          const std::vector<int> &newrank,
                          const std::unordered_map<LogicalLocation, int> &old_gid_of,
                          const std::unordered_map<LogicalLocation, int> &new_gid_of) {
  const auto make_nb = [this](const LogicalLocation &loc, int rank, int gid) {
    NeighborBlock nb(this, loc, loc, rank, gid, Kokkos::Array<int, 3>{0, 0, 0}, 0, 0, 0,
                     0, 0);
    nb.remesh_transfer = true;
    return nb;
  };
  for (auto &pmb : old_blocks)
    pmb->remesh_neighbors.clear();

  // Senders: old blocks on this rank
  for (int i = 0; i < old_blocks.size(); ++i) {
    auto &pmb = old_blocks[i];
    const int on = onbs + i;
    for (int nn : GetOverlappingGids(loclist[on], new_gid_of, ndim)) {
      // Same location on the same rank means the block object is reused as is
      if (newloc[nn] == loclist[on] && newrank[nn] == Globals::my_rank) continue;
      pmb->remesh_neighbors.push_back(make_nb(newloc[nn], newrank[nn], nn));
    }
  }

  // Receivers: new blocks on this rank that are not reused old blocks
  std::vector<std::vector<NeighborBlock>> new_neighbors(nbe - nbs + 1);
  for (int nn = nbs; nn <= nbe; ++nn) {
    auto &neighbors = new_neighbors[nn - nbs];
    const auto ogids = GetOverlappingGids(newloc[nn], old_gid_of, ndim);
    if (loclist[ogids[0]] == newloc[nn] && ranklist[ogids[0]] == Globals::my_rank)
      continue;
    for (int on : ogids)
      neighbors.push_back(make_nb(loclist[on], ranklist[on], on));
    if (ogids.size() > 1) {
      // A coarsened block receives from all of its daughters, which share elements in
      // the interior of the parent; assign each shared element to a single daughter.
      std::vector<forest::NeighborLocation> daughters;
      for (const auto &nb : neighbors)
        daughters.emplace_back(nb.loc, nb.loc, forest::LogicalCoordinateTransformation());
      for (auto &nb : neighbors)
        nb.ownership = DetermineOwnership(nb.loc, daughters, false);
    }
  }
  return new_neighbors;
}

//----------------------------------------------------------------------------------------
// \brief Same-level moves between ranks also carry per-block and per-variable counters
// (derefinement count, sparse deallocation counters) that are not field data. They are
// exchanged here in one integer message per pair of ranks.

void Mesh::ExchangeRemeshCounters_(const BlockList_t &old_blocks, int onbs,
                                   const BlockList_t &new_blocks, int nbs,
                                   const std::vector<int> &oldtonew,
                                   const std::vector<int> &newtoold,
                                   const std::vector<LogicalLocation> &newloc,
                                   const std::vector<int> &newrank) {
#ifdef MPI_PARALLEL
  const auto sorted_labels = [](const MeshBlock *pmb) {
    std::vector<std::string> labels;
    for (const auto &v : pmb->vars_cc_)
      labels.push_back(v->label());
    std::sort(labels.begin(), labels.end());
    return labels;
  };

  // Pack: for each block, the derefinement count and then the dealloc count of each
  // remeshed variable in label order
  std::map<int, std::vector<int>> send_data;
  for (int i = 0; i < old_blocks.size(); ++i) {
    const int on = onbs + i;
    const int nn = oldtonew[on];
    if (newloc[nn] != loclist[on] || newrank[nn] == Globals::my_rank) continue;
    auto &pmb = old_blocks[i];
    auto &buf = send_data[newrank[nn]];
    buf.push_back(pmb->pmr ? pmb->pmr->DerefinementCount() : 0);
    for (const auto &label : sorted_labels(pmb.get()))
      buf.push_back(pmb->meshblock_data.Get()->GetVarPtr(label)->dealloc_count);
  }

  std::map<int, std::vector<int>> recv_data;
  std::map<int, std::vector<MeshBlock *>> recv_blocks;
  for (int i = 0; i < new_blocks.size(); ++i) {
    const int nn = nbs + i;
    const int on = newtoold[nn];
    if (newloc[nn] != loclist[on] || ranklist[on] == Globals::my_rank) continue;
    recv_blocks[ranklist[on]].push_back(new_blocks[i].get());
  }
  for (auto &[rank, blocks] : recv_blocks)
    recv_data[rank].resize(blocks.size() * (1 + blocks[0]->vars_cc_.size()));

  MPI_Comm comm = GetMPIComm(remesh_counter_comm_label);
  std::vector<MPI_Request> reqs;
  for (auto &[rank, buf] : recv_data) {
    reqs.emplace_back();
    PARTHENON_MPI_CHECK(
        MPI_Irecv(buf.data(), buf.size(), MPI_INT, rank, 0, comm, &reqs.back()));
  }
  for (auto &[rank, buf] : send_data) {
    reqs.emplace_back();
    PARTHENON_MPI_CHECK(
        MPI_Isend(buf.data(), buf.size(), MPI_INT, rank, 0, comm, &reqs.back()));
  }
  PARTHENON_MPI_CHECK(MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE));

  for (auto &[rank, blocks] : recv_blocks) {
    const auto &buf = recv_data[rank];
    std::size_t idx = 0;
    for (auto *pmb : blocks) {
      const int deref = buf[idx++];
      if (pmb->pmr) pmb->pmr->DerefinementCount() = deref;
      for (const auto &label : sorted_labels(pmb))
        pmb->meshblock_data.Get()->GetVarPtr(label)->dealloc_count = buf[idx++];
    }
  }
#endif
}

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
    lb_flag_ = false;
  }
}

// Private routines
namespace {
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

  int rank = (Globals::nranks)-1;
  double target_cost = total_cost / Globals::nranks;
  double my_cost = 0.0;
  double remaining_cost = total_cost;
  // create rank list from the end: the master MPI rank should have less load
  for (int block_id = costlist.size() - 1; block_id >= 0; block_id--) {
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
  // First count the number of blocks on each rank
  nblist = std::vector<int>(Globals::nranks, 0);
  for (int b = ranklist.size() - 1; b >= 0; --b)
    nblist[ranklist[b]]++;

  // Then find the starting gid of the blocks, assuming they
  // are apportioned in increasing order
  nslist = std::vector<int>(Globals::nranks, 0);
  for (int b = 1; b < nslist.size(); ++b)
    nslist[b] = nslist[b - 1] + nblist[b - 1];
}
} // namespace

//----------------------------------------------------------------------------------------
// \brief Calculate distribution of MeshBlocks based on the cost list
void Mesh::CalculateLoadBalance(std::vector<double> const &costlist,
                                std::vector<int> &ranklist, std::vector<int> &nslist,
                                std::vector<int> &nblist) {
  PARTHENON_INSTRUMENT
  auto const total_blocks = costlist.size();

  using it = std::vector<double>::const_iterator;
  std::pair<it, it> const min_max = std::minmax_element(costlist.begin(), costlist.end());

  double const mincost = min_max.first == costlist.begin() ? 0.0 : *min_max.first;
  double const maxcost = min_max.second == costlist.begin() ? 0.0 : *min_max.second;

  // Assigns blocks to ranks on a rougly cost-equal basis.
  AssignBlocks(costlist, ranklist);

  // Updates nslist with the ID of the starting block on each rank and the count of blocks
  // on each rank.
  UpdateBlockList(ranklist, nslist, nblist);

#ifdef MPI_PARALLEL
  if (total_blocks % (Globals::nranks) != 0 && !adaptive && !lb_flag_ &&
      maxcost == mincost && Globals::my_rank == 0) {
    std::cout << "### Warning in CalculateLoadBalance" << std::endl
              << "The number of MeshBlocks cannot be divided evenly. "
              << "This will result in poor load balancing." << std::endl;
  }
#endif
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
                << "There are fewer MeshBlocks than OpenMP threads on each MPI rank"
                << std::endl
                << "This is likely fine if the number of meshblocks is expected to grow "
                   "during the "
                   "simulations. Otherwise, it might be worthwhile to decrease the "
                   "number of threads or "
                   "use more meshblocks."
                << std::endl;
    }
  }
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
  lb_flag_ = false;
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
  PARTHENON_INSTRUMENT
  // compute nleaf= number of leaf MeshBlocks per refined block
  int nleaf = 2;
  if (!mesh_size.symmetry(X2DIR)) nleaf = 4;
  if (!mesh_size.symmetry(X3DIR)) nleaf = 8;

  // collect refinement flags from all the meshblocks
  // count the number of the blocks to be (de)refined
  nref[Globals::my_rank] = 0;   // n to refine. Local per rank.
  nderef[Globals::my_rank] = 0; // n to derefine. Local per rank.
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
  int tnref = 0, tnderef = 0; // totals to refine/de-refine. Not local per rank.
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
  std::vector<LogicalLocation> lref, lderef, clderef;
  if (tnref > 0) lref.resize(tnref);
  if (tnderef >= nleaf) {
    lderef.resize(tnderef);
    clderef.resize(tnderef / nleaf);
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
                                       lref.data(), bnref.data(), brdisp.data(), MPI_BYTE,
                                       MPI_COMM_WORLD));
  }
  if (tnderef >= nleaf) {
    PARTHENON_MPI_CHECK(MPI_Allgatherv(MPI_IN_PLACE, bnderef[Globals::my_rank], MPI_BYTE,
                                       lderef.data(), bnderef.data(), bddisp.data(),
                                       MPI_BYTE, MPI_COMM_WORLD));
  }
#endif

  // calculate the list of the newly derefined blocks
  int ctnd = 0;
  if (tnderef >= nleaf) {
    int lk = !mesh_size.symmetry(X3DIR);
    int lj = !mesh_size.symmetry(X2DIR);
    for (int n = 0; n < tnderef; n++) {
      if (lderef[n].IsLowerLeftCornerOfParent()) {
        int r = n, rr = 0;
        // offsets from "bottom left" corner of parent block
        for (std::int64_t ok = 0; ok <= lk; ok++) {
          for (std::int64_t oj = 0; oj <= lj; oj++) {
            for (std::int64_t oi = 0; oi <= 1; oi++) {
              if (r < tnderef) {
                if (lderef[n].GetSameLevelNeighbor(oi, oj, ok) == lderef[r]) {
                  rr++;
                }
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
    std::sort(&(clderef[0]), &(clderef[ctnd - 1]),
              [](const LogicalLocation &left, const LogicalLocation &right) {
                return left.level() > right.level();
              });
  }

  // Now the lists of the blocks to be refined and derefined are completed
  // Start tree manipulation
  // Step 1. perform refinement
  for (int n = 0; n < tnref; n++) {
    nnew += forest.Refine(lref[n]);
  }

  // Step 2. perform derefinement
  for (int n = 0; n < ctnd; n++) {
    ndel += forest.Derefine(clderef[n]);
  }
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
  PARTHENON_INSTRUMENT
  // kill all old MeshData
  mesh_data.clear();

  // compute nleaf= number of leaf MeshBlocks per refined block
  int nleaf = 2;
  if (!mesh_size.symmetry(X2DIR)) nleaf = 4;
  if (!mesh_size.symmetry(X3DIR)) nleaf = 8;

  // construct new lists
  std::vector<LogicalLocation> newloc(ntot);
  std::vector<int> newrank(ntot);
  std::vector<double> newcost(ntot);
  std::vector<int> newtoold(ntot);
  std::vector<int> oldtonew(nbtotal);
  int nbtold = nbtotal;
  std::unordered_set<LogicalLocation> newly_refined;
  // store old nbstart and nbend before load balancing.
  int onbs = nslist[Globals::my_rank];
  int onbe = onbs + nblist[Globals::my_rank] - 1;

  { // Construct new list region
    PARTHENON_INSTRUMENT
    newloc = forest.GetMeshBlockListAndResolveGids();
    nbtotal = newloc.size();
    for (int ib = 0; ib < nbtotal; ++ib)
      newtoold[ib] = forest.GetOldGid(newloc[ib]);

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

  // The blocks of the old and new meshes that overlap follow from the old (loclist) and
  // new (newloc) lists of logical locations
  std::unordered_map<LogicalLocation, int> old_gid_of, new_gid_of;
  old_gid_of.reserve(nbtold);
  new_gid_of.reserve(nbtotal);
  for (int on = 0; on < nbtold; ++on)
    old_gid_of[loclist[on]] = on;
  for (int nn = 0; nn < nbtotal; ++nn)
    new_gid_of[newloc[nn]] = nn;

  // Calculate new load balance
  CalculateLoadBalance(newcost, newrank, nslist, nblist);

  int nbs = nslist[Globals::my_rank];
  int nbe = nbs + nblist[Globals::my_rank] - 1;

  // The new MeshBlock list, reusing the blocks that stay in place on this rank
  BlockList_t new_block_list(nbe - nbs + 1);

  // Prolongation from the coarse buffers of blocks that were newly refined (c2f). The
  // cache is built once the remesh data has been received, since receiving allocates
  // sparse variables on the new blocks
  bool any_newly_refined = false;
  for (int nn = nbs; nn <= nbe; nn++)
    any_newly_refined = any_newly_refined || newly_refined.count(newloc[nn]);
  ProResCache_t prolongation_cache;
  int nprolong = 0;
  auto build_prolongation_cache = [&]() {
    for (int nn = nbs; nn <= nbe; nn++) {
      if (newly_refined.count(newloc[nn]))
        nprolong += new_block_list[nn - nbs]->vars_cc_.size();
    }
    prolongation_cache.Initialize(nprolong, resolved_packages.get());
    int iprolong = 0;
    for (int nn = nbs; nn <= nbe; nn++) {
      if (newly_refined.count(newloc[nn])) {
        auto &pmb = new_block_list[nn - nbs];
        for (auto &var : pmb->vars_cc_) {
          prolongation_cache.RegisterRegionHost(
              iprolong++,
              ProResInfo::GetInteriorProlongate(pmb.get(), NeighborBlock(), var),
              var.get(), resolved_packages.get());
        }
      }
    }
    prolongation_cache.CopyToDevice();
  };

  { // AMR communicate field data from the old blocks to the new blocks
    PARTHENON_INSTRUMENT
    ResetCommunicationInfrastructure(this);
    auto new_remesh_neighbors = SetRemeshNeighbors_(block_list, onbs, nbs, nbe, newloc,
                                                    newrank, old_gid_of, new_gid_of);

    // The channels (and so their tags) follow from the remesh neighbor lists, so the
    // send side can be completely set up before the new blocks exist. Coalesced comms
    // identify the sending MeshData by its partition, so the old and new block lists
    // need their own (leaf grid) partitions.
    auto md_old = std::make_shared<MeshData<Real>>("base", block_list, this);
    md_old->partition =
        std::make_shared<BlockListPartition>(0, GridIdentifier::leaf(), block_list, this);
    AddToTagMap<BoundaryType::remesh_send>(md_old);
    LockCommLaneNumbers(BoundaryType::remesh_recv);
    const int recv_lanes = GetNumberOfCommLanes(BoundaryType::remesh_recv);
    for (int nn = nbs; nn <= nbe; nn++)
      tag_map.AddBlockToMap(nn, new_remesh_neighbors[nn - nbs], recv_lanes);
    tag_map.ResolveMap();
    BuildRemeshSendBuffers(md_old);
    if (do_coalesced_comms) pcoalesced_comms->ResolveAndSendSendBuffers();

    std::shared_ptr<MeshData<Real>> md_new;
    TaskCollection tc;
    TaskRegion &region = tc.AddRegion(1);
    auto &tl = region[0];
    // Send from the old blocks, restricting first where the new block is coarser
    auto send = tl.AddTask(TaskID(), TF(SendBoundBufs<BoundaryType::remesh_send>), md_old);
    tl.AddTask(send, TF(RemeshSendsComplete), md_old);

    // While the sends are in flight, create the new blocks (without allocating their
    // data) and set up receiving into them
    auto create = tl.AddTask(TaskID(), "CreateNewBlocks", [&]() {
      RegionSize block_size = GetDefaultBlockSize();
      for (int n = nbs; n <= nbe; n++) {
        int on = newtoold[n];
        if ((ranklist[on] == Globals::my_rank) && (loclist[on] == newloc[n])) {
          // on the same MPI rank and same location -> just move it
          new_block_list[n - nbs] = FindMeshBlock(on);
        } else {
          // on a different refinement level or MPI rank - create a new block
          BoundaryFlag block_bcs[6];
          SetBlockSizeAndBoundaries(newloc[n], block_size, block_bcs);
          constexpr bool defer_allocation = true;
          auto pmb = MeshBlock::Make(n, n - nbs, newloc[n], block_size, block_bcs, this,
                                     pin, app_in, packages, resolved_packages, gflag, 1.0,
                                     defer_allocation);
          pmb->remesh_neighbors = std::move(new_remesh_neighbors[n - nbs]);
          new_block_list[n - nbs] = pmb;
        }
      }
      md_new = std::make_shared<MeshData<Real>>("base", new_block_list, this);
      md_new->partition = std::make_shared<BlockListPartition>(
          0, GridIdentifier::leaf(), new_block_list, this);
      return BuildRemeshReceiveBuffers(md_new);
    });
    auto info = create;
    if (do_coalesced_comms) {
      info = tl.AddTask(create, "ReceiveCoalescedBufferInfo", [&]() {
        return pcoalesced_comms->TryReceiveBufferInfo() ? TaskStatus::complete
                                                        : TaskStatus::incomplete;
      });
    }
    auto start = tl.AddTask(info, "StartReceive", [&]() {
      return StartReceiveBoundBufs<BoundaryType::remesh_recv>(md_new);
    });
    auto alloc = tl.AddTask(create, "AllocateNewBlocks", [&]() {
      for (auto &pmb : new_block_list)
        pmb->AllocateDeferred(pin);
      return TaskStatus::complete;
    });

    // Receive into the new blocks and prolongate onto the newly refined ones
    auto recv = tl.AddTask(start | alloc, "Receive", [&]() {
      return ReceiveBoundBufs<BoundaryType::remesh_recv>(md_new);
    });
    auto set = tl.AddTask(
        recv, "SetBounds", [&]() { return SetBounds<BoundaryType::remesh_recv>(md_new); });
    if (any_newly_refined) {
      tl.AddTask(set, "ProlongateShared", [&]() {
        build_prolongation_cache();
        refinement::ProlongateShared(resolved_packages.get(), prolongation_cache,
                                     new_block_list[0]->cellbounds,
                                     new_block_list[0]->c_cellbounds);
        return TaskStatus::complete;
      });
    }
    TaskListStatus status = tc.Execute(task_collection_timeout_in_seconds);
    PARTHENON_REQUIRE(status == TaskListStatus::complete,
                      "Remesh communication failed.");

    ExchangeRemeshCounters_(block_list, onbs, new_block_list, nbs, oldtonew, newtoold,
                            newloc, newrank);
    Kokkos::fence();
  }

  // Replace the MeshBlock list
  auto old_block_list = std::move(block_list);
  block_list = std::move(new_block_list);

  // Ensure local and global ids are correct
  for (int n = nbs; n <= nbe; n++) {
    block_list[n - nbs]->gid = n;
    block_list[n - nbs]->lid = n - nbs;
    block_list[n - nbs]->remesh_neighbors.clear();
  }
  BuildBlockPartitions(GridIdentifier::leaf());

  { // AMR finalize new mesh
    PARTHENON_INSTRUMENT
    // Field data is transferred by the remesh channels above because the field layout
    // is tied directly to block topology. Swarms are different: after remesh, ownership
    // is determined by which new leaf block contains each particle. Run that
    // ownership-based redistribution only after the new leaf mesh and field state are
    // already in place.
    const SwarmRemeshContext swarm_remesh_context(onbs, onbe, oldtonew, loclist, newloc,
                                                  ranklist, newrank);
    RemeshSwarms(resolved_packages, old_block_list, this, swarm_remesh_context);
    ClearSwarmCachesAfterRemesh(this, block_list);

    // update the lists
    loclist = std::move(newloc);
    ranklist = std::move(newrank);
    costlist = std::move(newcost);

    BuildGMGBlockLists(pin, app_in);

    // init meshblock data
    for (auto &pmb : block_list) {
      if (pmb->InitMeshBlockUserData != nullptr) {
        pmb->InitMeshBlockUserData(pmb.get(), pin);
      }
    }

    // Find the non-cell centered fields that are communicated
    Metadata::FlagCollection fc;
    fc.TakeUnion(Metadata::Face, Metadata::Edge, Metadata::Node);
    fc.TakeIntersection(Metadata::FillGhost);
    std::vector<std::string> noncc_names = GetVariableNames(fc);

    if (noncc_names.size() > 0) {
      // A block newly refined and prolongated may have neighbors which were
      // already refined to the new level.
      // If so, the prolongated versions of shared elements will not reflect
      // the true, finer versions present in the neighbor block.
      // We must create any new fine buffers and fill them from these neighbors
      // in order to maintain a consistent global state.
      // Thus we rebuild and synchronize the mesh now, but using a unique
      // neighbor precedence favoring the "old" fine blocks over "new" ones
      SetMeshBlockNeighbors(this, GridIdentifier::leaf(), block_list, ranklist,
                            newly_refined);
      SetGMGNeighbors();
      BuildTagMapAndBoundaryBuffers();
      std::string noncc = "mesh_internal_noncc";
      for (auto &partition : GetDefaultBlockPartitions()) {
        auto &md = mesh_data.Add("base", partition);
        auto &md_noncc = mesh_data.AddShallow(noncc, md, noncc_names);
      }

      CommunicateBoundaries(
          noncc, noncc_names); // Called to make sure shared values are correct,
                               // ghosts of non-cell centered vars may get some junk
      // Now there is the correct data for prolongating on un-shared topological elements
      // on the new fine blocks
      if (nprolong > 0) {
        refinement::ProlongateInternal(resolved_packages.get(), prolongation_cache,
                                       block_list[0]->cellbounds,
                                       block_list[0]->c_cellbounds);
      }
    }

    // Rebuild just the ownership model, this time weighting the "new" fine blocks just
    // like any other blocks at their level.
    SetMeshBlockNeighbors(this, GridIdentifier::leaf(), block_list, ranklist);
    SetGMGNeighbors();
    // Ownership does not impact anything about the buffers, so we don't need to
    // rebuild them if they were built above
    if (noncc_names.size() == 0) BuildTagMapAndBoundaryBuffers();

    // Call to fill ghosts with real data and fill derived quantities
    PreCommFillDerived();
    CommunicateBoundaries();
    FillDerived();

    // Initialize the "base" MeshData object
    mesh_data.Add("base", GetBasePartition());
  } // AMR finalize new mesh

  ResetLoadBalanceVariables();
}
} // namespace parthenon
