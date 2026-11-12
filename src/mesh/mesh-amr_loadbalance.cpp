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
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "parthenon_mpi.hpp"

#include "bvals/boundary_conditions.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "interface/update.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_neighbors.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/remesh/inventory.hpp"
#include "mesh/remesh/placement.hpp"
#include "mesh/remesh/topology.hpp"
#include "mesh/remesh/transfer.hpp"
#include "mesh/swarm_amr_remesh.hpp"
#include "parthenon_arrays.hpp"
#include "utils/error_checking.hpp"
#include "utils/peer_transport.hpp"

namespace parthenon {

namespace {

remesh::CostPolicy GetRemeshCostPolicy(ParameterInput *pin) {
  const auto policy =
      pin->GetOrAddString("parthenon/loadbalancing", "remesh_cost_policy", "inherit",
                          std::vector<std::string>{"inherit", "parthenon", "application"},
                          "cost model used to place blocks after remeshing");
  if (policy == "parthenon") return remesh::CostPolicy::parthenon;
  if (policy == "application") return remesh::CostPolicy::application;
  return remesh::CostPolicy::inherit;
}

std::optional<std::uint64_t> ReadMemoryBudget(ParameterInput *pin,
                                              const std::string &name) {
  if (!pin->DoesParameterExist("parthenon/loadbalancing", name)) return std::nullopt;
  const auto mib = pin->GetReal("parthenon/loadbalancing", name);
  PARTHENON_REQUIRE(mib > 0.0 && std::isfinite(mib),
                    "Remesh memory budgets must be finite and positive");
  constexpr double bytes_per_mib = 1024.0 * 1024.0;
  PARTHENON_REQUIRE(mib <=
                        static_cast<double>(std::numeric_limits<std::uint64_t>::max()) /
                            bytes_per_mib,
                    "Remesh memory budget exceeds uint64_t");
  return static_cast<std::uint64_t>(mib * bytes_per_mib);
}

remesh::MemoryBudget GetRemeshMemoryBudget(ParameterInput *pin) {
  remesh::MemoryBudget budget;
  budget.host_bytes = ReadMemoryBudget(pin, "remesh_host_memory_budget_mib");
  budget.device_bytes = ReadMemoryBudget(pin, "remesh_device_memory_budget_mib");
  budget.require_hard_guarantee = pin->GetOrAddBoolean(
      "parthenon/loadbalancing", "remesh_require_hard_memory_guarantee", false,
      "reject remesh unless all prospective storage has a conservative bound");
  return budget;
}

std::vector<std::uint64_t> GatherSourceMemory(const Mesh &mesh) {
  std::vector<std::uint64_t> source_bytes(Globals::nranks, 0);
  for (const auto &block : mesh.block_list) {
    PARTHENON_REQUIRE(source_bytes[Globals::my_rank] <=
                          std::numeric_limits<std::uint64_t>::max() -
                              block->ReportMemUsage(),
                      "Source mesh memory accounting exceeds uint64_t");
    source_bytes[Globals::my_rank] += block->ReportMemUsage();
  }
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, source_bytes.data(), Globals::nranks,
                                    MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD));
#endif
  return source_bytes;
}

void ValidateCostsCollectively(const std::vector<double> &costs) {
#ifdef MPI_PARALLEL
  auto minimum = costs;
  auto maximum = costs;
  PARTHENON_REQUIRE(costs.size() <= std::numeric_limits<int>::max(),
                    "Remesh cost count exceeds MPI limits");
  const auto count = static_cast<int>(costs.size());
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, minimum.data(), count, MPI_DOUBLE,
                                    MPI_MIN, MPI_COMM_WORLD));
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, maximum.data(), count, MPI_DOUBLE,
                                    MPI_MAX, MPI_COMM_WORLD));
  PARTHENON_REQUIRE(minimum == maximum,
                    "Remesh cost callback returned rank-dependent results");
#endif
}

} // namespace

#ifdef MPI_PARALLEL
//----------------------------------------------------------------------------------------
//! \fn int CreateAMRMPITag(int lid, int ox1, int ox2, int ox3)
//  \brief calculate an MPI tag for AMR block transfer
// tag = local id of destination (remaining bits) + ox1(1 bit) + ox2(1 bit) + ox3(1 bit)
//       + physics(5 bits)

int CreateAMRMPITag(int lid, std::optional<LogicalLocation> loc = {}) {
  PARTHENON_REQUIRE(lid >= 0 && lid <= (std::numeric_limits<int>::max() >> 8),
                    "AMR destination local ID exceeds tag encoding range");
  // the trailing zero is used as "id" to indicate an AMR related tag
  int tag = lid << 8;
  if (loc) {
    auto offsets = (*loc).GetLocationInParent();
    tag = tag | (offsets[0] << 7) | (offsets[1] << 6) | (offsets[2] << 5) | 0;
  }
  return tag;
}

MPI_Request SendCoarseToFine(int lid_recv, int dest_rank, const LogicalLocation &fine_loc,
                             Variable<Real> *var, Mesh *pmesh) {
  MPI_Request req;
  MPI_Comm comm = pmesh->GetMPIComm(var->label());
  int tag = CreateAMRMPITag(lid_recv, fine_loc);
  peer_transport::ValidateTag(comm, tag);
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
  int test = 1;
#ifdef MPI_PARALLEL
  MPI_Comm comm = pmesh->GetMPIComm(var->label());
  int tag = CreateAMRMPITag(lid_recv, fine_loc);
  peer_transport::ValidateTag(comm, tag);
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

      auto &cellbounds = var->IsSet(Metadata::Fine) ? pmb->f_cellbounds : pmb->cellbounds;
      auto &c_cellbounds =
          var->IsSet(Metadata::Fine) ? pmb->cellbounds : pmb->c_cellbounds;
      for (auto te : var->GetTopologicalElements()) {
        const auto te_entire = var->IsSet(Metadata::CellMemAligned) ? TE::CC : te;
        IndexRange ib = c_cellbounds.GetBoundsI(IndexDomain::entire, te_entire);
        IndexRange jb = c_cellbounds.GetBoundsJ(IndexDomain::entire, te_entire);
        IndexRange kb = c_cellbounds.GetBoundsK(IndexDomain::entire, te_entire);

        IndexRange ib_int = cellbounds.GetBoundsI(IndexDomain::interior, te);
        IndexRange jb_int = cellbounds.GetBoundsJ(IndexDomain::interior, te);
        IndexRange kb_int = cellbounds.GetBoundsK(IndexDomain::interior, te);

        const auto [ox1, ox2, ox3] = fine_loc.GetLocationInParent();
        const int ks = (ox3 == 0) ? 0 : (kb_int.e - kb_int.s + 1) / 2;
        const int js = (ox2 == 0) ? 0 : (jb_int.e - jb_int.s + 1) / 2;
        const int is = (ox1 == 0) ? 0 : (ib_int.e - ib_int.s + 1) / 2;
        const int idx_te = static_cast<int>(te) % 3;
        parthenon::par_for(
            PARTHENON_AUTO_LABEL, 0, nt, 0, nu, 0, nv, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
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
  int tag = CreateAMRMPITag(lid_recv, fine_loc);
  peer_transport::ValidateTag(comm, tag);
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
  const int ndim = pmb->pmy_mesh->ndim;
  int test = 1;
#ifdef MPI_PARALLEL
  MPI_Comm comm = pmesh->GetMPIComm(var->label());
  int tag = CreateAMRMPITag(lid_recv, fine_loc);
  peer_transport::ValidateTag(comm, tag);
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

      auto &c_cellbounds =
          var->IsSet(Metadata::Fine) ? pmb->cellbounds : pmb->c_cellbounds;
      for (auto te : var->GetTopologicalElements()) {
        IndexRange ib = c_cellbounds.GetBoundsI(IndexDomain::interior, te);
        IndexRange jb = c_cellbounds.GetBoundsJ(IndexDomain::interior, te);
        IndexRange kb = c_cellbounds.GetBoundsK(IndexDomain::interior, te);
        // Deal with ownership of shared elements by removing right side of index
        // space if fine block is on the left side of a direction. I think this
        // should work fine even if the ownership model is changed elsewhere, since
        // the fine blocks should be consistent in their shared elements at this point
        const auto [ox1, ox2, ox3] = fine_loc.GetLocationInParent();
        if (ox3 == 0 && ndim > 2) kb.e -= TopologicalOffsetK(te);
        if (ox2 == 0 && ndim > 1) jb.e -= TopologicalOffsetJ(te);
        if (ox1 == 0) ib.e -= TopologicalOffsetI(te);
        const int ks =
            (ox3 == 0 || ndim < 3) ? 0 : (kb.e - kb.s + 1 - TopologicalOffsetK(te));
        const int js =
            (ox2 == 0 || ndim < 2) ? 0 : (jb.e - jb.s + 1 - TopologicalOffsetJ(te));
        const int is = (ox1 == 0) ? 0 : (ib.e - ib.s + 1 - TopologicalOffsetI(te));
        const int idx_te = static_cast<int>(te) % 3;
        parthenon::par_for(
            PARTHENON_AUTO_LABEL, 0, nt, 0, nu, 0, nv, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
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
std::array<MPI_Request, 2> SendSameToSame(int lid_recv, int dest_rank,
                                          Variable<Real> *var, MeshBlock *pmb,
                                          Mesh *pmesh) {
  std::array<MPI_Request, 2> requests;
  MPI_Comm comm = pmesh->GetMPIComm(var->label());
  int tag = CreateAMRMPITag(lid_recv);
  peer_transport::ValidateTag(comm, tag);
  peer_transport::ValidateTag(comm, tag + 1);
  var->com_state[0] = var->IsAllocated();
  var->com_state[1] = var->GetAllocationStatus();
  var->com_state[2] = pmb->pmr->DerefinementCount();
  var->com_state[3] = var->dealloc_count;
  PARTHENON_MPI_CHECK(
      MPI_Isend(var->com_state, 4, MPI_INT, dest_rank, tag + 1, comm, &requests[0]));
  if (var->IsAllocated()) {
    PARTHENON_MPI_CHECK(MPI_Isend(var->data.data(), var->data.size(), MPI_PARTHENON_REAL,
                                  dest_rank, tag, comm, &requests[1]));
  } else {
    PARTHENON_MPI_CHECK(MPI_Isend(var->data.data(), 0, MPI_PARTHENON_REAL, dest_rank, tag,
                                  comm, &requests[1]));
  }
  return requests;
}

bool TryRecvSameToSame(int lid_recv, int send_rank, Variable<Real> *var, MeshBlock *pmb,
                       Mesh *pmesh) {
  MPI_Comm comm = pmesh->GetMPIComm(var->label());
  int tag = CreateAMRMPITag(lid_recv);
  peer_transport::ValidateTag(comm, tag);
  peer_transport::ValidateTag(comm, tag + 1);
  int test;
  PARTHENON_MPI_CHECK(MPI_Iprobe(send_rank, tag + 1, comm, &test, MPI_STATUS_IGNORE));
  if (test) {
    PARTHENON_MPI_CHECK(MPI_Recv(var->com_state, 4, MPI_INT, send_rank, tag + 1, comm,
                                 MPI_STATUS_IGNORE));
    if (var->com_state[0]) {
      if (!pmb->IsAllocated(var->label())) pmb->AllocateSparse(var->label());
      PARTHENON_MPI_CHECK(MPI_Recv(var->data.data(), var->data.size(), MPI_PARTHENON_REAL,
                                   send_rank, tag, comm, MPI_STATUS_IGNORE));
      var->RestoreAllocationStatusForRemesh(var->com_state[1]);
    } else {
      if (pmb->IsAllocated(var->label()) &&
          !var->metadata().IsSet(Metadata::ForceAllocOnNewBlocks))
        pmb->DeallocateSparse(var->label());
      PARTHENON_MPI_CHECK(MPI_Recv(var->data.data(), 0, MPI_PARTHENON_REAL, send_rank,
                                   tag, comm, MPI_STATUS_IGNORE));
    }
    pmb->pmr->DerefinementCount() = var->com_state[2];
    var->dealloc_count = var->com_state[3];
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
  remesh::TopologyPlan topology;
  remesh::InventoryPlan inventory;
  remesh::PlacementPlan placement;

  if (adaptive) {
    topology = PlanAdaptiveMeshTopology();
    if (topology.changed) {
      inventory = remesh::PlanInventory(*this, topology);
    }
  }

  lb_flag_ |= lb_automatic_;

  UpdateCostList();

  modified = false;
  if (topology.changed) {
    GatherCostListAndCheckBalance();
    placement = remesh::PlanRemeshPlacement(
        topology, inventory, costlist, GatherSourceMemory(*this), Globals::nranks,
        GetRemeshCostPolicy(pin), GetRemeshMemoryBudget(pin),
        app_in == nullptr ? nullptr : app_in->EstimateRemeshBlockCost);
    ValidateCostsCollectively(placement.costs);
    nbnew += topology.added_blocks;
    nbdel += topology.removed_blocks;
    forest = std::move(*topology.candidate);
    ExecuteLegacyRemeshPlan(pin, app_in,
                            nbtotal + topology.added_blocks - topology.removed_blocks,
                            &topology, &inventory, &placement);
    modified = true;
  } else if (lb_flag_ && step_since_lb >= lb_interval_) {
    if (!GatherCostListAndCheckBalance()) { // load imbalance detected
      ExecuteLegacyRemeshPlan(pin, app_in, nbtotal);
      modified = true;
    }
    lb_flag_ = false;
  }
}

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

  auto ownership = remesh::PlaceBlocksContiguously(costlist, Globals::nranks);
  ranklist = std::move(ownership.owner);
  nslist = std::move(ownership.rank_starts);
  nblist = std::move(ownership.rank_counts);

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
// \!fn remesh::TopologyPlan Mesh::PlanAdaptiveMeshTopology()
// \brief collect effective tags and build an isolated candidate topology

remesh::TopologyPlan Mesh::PlanAdaptiveMeshTopology() {
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
    if (pmb->pmr->GetRefinementTag() == AmrTag::refine) nref[Globals::my_rank]++;
    if (pmb->pmr->GetRefinementTag() == AmrTag::derefine) nderef[Globals::my_rank]++;
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
  if (tnref == 0 && tnderef < nleaf) return {};

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
  std::vector<LogicalLocation> lref, lderef;
  if (tnref > 0) lref.resize(tnref);
  if (tnderef >= nleaf) lderef.resize(tnderef);

  // collect the locations and costs
  int iref = rdisp[Globals::my_rank], ideref = ddisp[Globals::my_rank];
  for (auto const &pmb : block_list) {
    if (pmb->pmr->GetRefinementTag() == AmrTag::refine) lref[iref++] = pmb->loc;
    if (pmb->pmr->GetRefinementTag() == AmrTag::derefine && tnderef >= nleaf)
      lderef[ideref++] = pmb->loc;
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

  std::vector<remesh::TaggedLeaf> tags;
  tags.reserve(lref.size() + lderef.size());
  for (const auto &loc : lref)
    tags.push_back({{loc}, AmrTag::refine});
  for (const auto &loc : lderef)
    tags.push_back({{loc}, AmrTag::derefine});
  const int ndim = mesh_size.symmetry(X2DIR) ? 1 : mesh_size.symmetry(X3DIR) ? 2 : 3;
  return remesh::PlanTopology(forest, tags, ranklist, ndim);
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
// \!fn void Mesh::ExecuteLegacyRemeshPlan(ParameterInput *pin, int ntot)
// \brief execute an accepted topology with the compatibility transfer path

void Mesh::ExecuteLegacyRemeshPlan(ParameterInput *pin, ApplicationInput *app_in,
                                   int ntot, const remesh::TopologyPlan *topology,
                                   const remesh::InventoryPlan *inventory,
                                   const remesh::PlacementPlan *placement) {
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
  std::optional<remesh::FieldTransferPlan> field_transfer;
  int nbtold = nbtotal;
  std::unordered_set<LogicalLocation> newly_refined;
  // store old nbstart and nbend before load balancing.
  int onbs = nslist[Globals::my_rank];
  int onbe = onbs + nblist[Globals::my_rank] - 1;

  { // Construct new list region
    PARTHENON_INSTRUMENT
    if (topology != nullptr && topology->changed) {
      PARTHENON_REQUIRE(topology->leaves.size() == ntot,
                        "Accepted remesh topology has the wrong leaf count");
      for (const auto &leaf : topology->leaves) {
        const int new_gid = leaf.new_gid;
        newloc[new_gid] = leaf.key.loc;
        newtoold[new_gid] = leaf.sources.front().source.gid;
      }
      for (int old_gid = 0; old_gid < topology->successors.size(); ++old_gid) {
        PARTHENON_REQUIRE(!topology->successors[old_gid].empty(),
                          "Accepted remesh source has no successor");
        oldtonew[old_gid] = topology->successors[old_gid].front();
      }
      nbtotal = topology->leaves.size();
    } else {
      newloc = forest.GetMeshBlockListAndResolveGids();
      nbtotal = newloc.size();
      for (int ib = 0; ib < nbtotal; ++ib)
        newtoold[ib] = forest.GetOldGid(newloc[ib]);

      oldtonew[0] = 0;
      int mb_idx = 1;
      for (int n = 1; n < ntot; n++) {
        if (newtoold[n] == newtoold[n - 1] + 1) {
          oldtonew[mb_idx++] = n;
        } else if (newtoold[n] == newtoold[n - 1] + nleaf) {
          for (int j = 0; j < nleaf - 1; j++)
            oldtonew[mb_idx++] = n - 1;
          oldtonew[mb_idx++] = n;
        }
      }
      for (; mb_idx < nbtold; mb_idx++)
        oldtonew[mb_idx] = ntot - 1;
    }

    if (placement != nullptr) {
      PARTHENON_REQUIRE(placement->costs.size() == newcost.size(),
                        "Planned remesh costs have the wrong size");
      newcost = placement->costs;
    }
    current_level = 0;
    for (int n = 0; n < ntot; n++) {
      // "on" = "old n" = "old gid" = "old global MeshBlock ID"
      int on = newtoold[n];
      if (newloc[n].level() > current_level) // set the current max level
        current_level = newloc[n].level();
      if (newloc[n].level() >= loclist[on].level()) { // same or refined
        if (placement == nullptr) newcost[n] = costlist[on];
        // Keep a list of all blocks refined for below
        if (newloc[n].level() > loclist[on].level()) {
          newly_refined.insert(newloc[n]);
        }
      } else {
        if (placement == nullptr) {
          double acost = 0.0;
          for (int l = 0; l < nleaf; l++)
            acost += costlist[on + l];
          newcost[n] = acost / nleaf;
        }
      }
    }
  } // Construct new list region

  if (topology != nullptr && topology->changed) {
    PARTHENON_REQUIRE(inventory != nullptr && placement != nullptr,
                      "Accepted remesh topology requires inventory and placement");
    field_transfer = remesh::PlanFieldTransfers(*topology, *inventory, ranklist,
                                                placement->ownership, do_coalesced_comms);
  }

  // Calculate new load balance
  if (placement == nullptr) {
    CalculateLoadBalance(newcost, newrank, nslist, nblist);
  } else {
    newrank = placement->ownership.owner;
    nslist = placement->ownership.rank_starts;
    nblist = placement->ownership.rank_counts;
  }

  int nbs = nslist[Globals::my_rank];
  int nbe = nbs + nblist[Globals::my_rank] - 1;

  // Restrict fine to coarse buffers
  if (block_list.size() > 0) {
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
          restriction_cache.RegisterRegionHost(
              irestrict++,
              ProResInfo::GetInteriorRestrict(pmb.get(), NeighborBlock(), var), var.get(),
              resolved_packages.get());
        }
      }
    }
    restriction_cache.CopyToDevice();
    refinement::Restrict(resolved_packages.get(), restriction_cache,
                         block_list[0]->cellbounds, block_list[0]->c_cellbounds);
  }
  Kokkos::fence();

#ifdef MPI_PARALLEL
  // Send data from old to new blocks
  std::vector<MPI_Request> send_reqs;
  if (!field_transfer) { // AMR Send region
    PARTHENON_INSTRUMENT
    for (int n = onbs; n <= onbe; n++) {
      int nn = oldtonew[n];
      LogicalLocation &oloc = loclist[n];
      LogicalLocation &nloc = newloc[nn];
      auto pb = FindMeshBlock(n);
      if (nloc.level() == oloc.level() &&
          newrank[nn] != Globals::my_rank) { // same level, different rank
        for (auto &var : pb->vars_cc_) {
          const auto requests = SendSameToSame(nn - nslist[newrank[nn]], newrank[nn],
                                               var.get(), pb.get(), this);
          send_reqs.insert(send_reqs.end(), requests.begin(), requests.end());
        }
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
  } // AMR Send region
#endif // MPI_PARALLEL

  // Construct a new MeshBlock list (moving the data within the MPI rank)
  BlockList_t new_block_list(nbe - nbs + 1);
  { // AMR Construct new MeshBlockList region
    PARTHENON_INSTRUMENT
    RegionSize block_size = GetDefaultBlockSize();

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

  if (field_transfer) {
    remesh::ApplyPlannedFieldAllocations(*inventory, new_block_list, nbs);
  }

  // Replace the MeshBlock list
  auto old_block_list = std::move(block_list);
  block_list = std::move(new_block_list);

  // Ensure local and global ids are correct
  for (int n = nbs; n <= nbe; n++) {
    block_list[n - nbs]->gid = n;
    block_list[n - nbs]->lid = n - nbs;
  }
  BuildBlockPartitions(GridIdentifier::leaf());

  if (field_transfer) {
    remesh::ExecuteFieldTransfers(*field_transfer, *inventory, old_block_list, onbs,
                                  block_list, nbs, this);
  }

  // Receive the data and load into MeshBlocks
  { // AMR Recv and unpack data
    PARTHENON_INSTRUMENT
    bool all_received;
    int niter = 0;
    if (!field_transfer && block_list.size() > 0) {
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
    if (nprolong > 0) {
      prolongation_cache.Initialize(nprolong, resolved_packages.get());
      int iprolong = 0;
      for (int nn = nbs; nn <= nbe; nn++) {
        int on = newtoold[nn];
        if (newloc[nn].level() > loclist[on].level()) {
          auto pmb = FindMeshBlock(nn);
          for (auto &var : pmb->vars_cc_) {
            prolongation_cache.RegisterRegionHost(
                iprolong++,
                ProResInfo::GetInteriorProlongate(pmb.get(), NeighborBlock(), var),
                var.get(), resolved_packages.get());
          }
        }
        prolongation_cache.CopyToDevice();
      }
      refinement::ProlongateShared(resolved_packages.get(), prolongation_cache,
                                   block_list[0]->cellbounds,
                                   block_list[0]->c_cellbounds);
    }

    // Field data is transferred by the usual AMR same-level, restriction, and
    // prolongation paths above because the field layout is tied directly to block
    // topology. Swarms are different: after remesh, ownership is determined by which new
    // leaf block contains each particle. Run that ownership-based redistribution only
    // after the new leaf mesh and field state are already in place.
    const SwarmRemeshContext swarm_remesh_context(onbs, onbe, oldtonew, loclist, newloc,
                                                  ranklist, newrank);
    RemeshSwarms(resolved_packages, old_block_list, this, swarm_remesh_context,
                 inventory == nullptr ? nullptr : &inventory->local_swarm_routes);
    ClearSwarmCachesAfterRemesh(this, block_list);

    // update the lists
    loclist = std::move(newloc);
    ranklist = std::move(newrank);
    costlist = std::move(newcost);

    BuildGMGBlockLists(pin, app_in);

    // Make sure all old sends/receives are done before we reconfigure the mesh
#ifdef MPI_PARALLEL
    if (send_reqs.size() != 0)
      PARTHENON_MPI_CHECK(
          MPI_Waitall(send_reqs.size(), send_reqs.data(), MPI_STATUSES_IGNORE));
#endif
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
  } // AMR Recv and unpack data

  ResetLoadBalanceVariables();
}
} // namespace parthenon
