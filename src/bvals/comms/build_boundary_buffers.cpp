//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2022 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2022-2023. Triad National Security, LLC. All rights reserved.
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
#include <random>
#include <string>
#include <vector>

#include "bvals_in_one.hpp"
#include "bvals_utils.hpp"
#include "config.hpp"
#include "globals.hpp"
#include "interface/variable.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock.hpp"
#include "utils/error_checking.hpp"
#include "utils/loop_utils.hpp"

namespace parthenon {

using namespace loops;
using namespace loops::shorthands;

namespace {
template <BoundaryType BTYPE>
void BuildBoundaryBufferSubset(std::shared_ptr<MeshData<Real>> &md,
                               Mesh::comm_buf_map_t &buf_map) {
  Mesh *pmesh = md->GetMeshPointer();
  #ifdef USE_NEIGHBORHOOD_COLLECTIVES
  if(BTYPE == BoundaryType::nonlocal){
    ForEachBoundary<BTYPE>(md, [&](auto pmb, sp_mbd_t /*rc*/, nb_t &nb, const sp_cv_t v) {
      int receiver_rank = nb.snb.rank;
      pmesh->neigh_token.add_buff_info(nb.snb.rank,GetBufferSize(pmb, nb, v),pmesh->tag_map.GetTag(pmb, nb));
      int tagg = pmesh->tag_map.GetTag(pmb, nb);
      auto comm_label = v->label();
      mpi_comm_t comm = pmesh->GetMPIComm(comm_label);
    });
    pmesh->neigh_token.calculate_off_prefix_sum();
    pmesh->neigh_token.alloc_comm_buffers();
  }
  #endif //USE_NEIGHBORHOOD_COLLECTIVES

  ForEachBoundary<BTYPE>(md, [&](auto pmb, sp_mbd_t /*rc*/, nb_t &nb, const sp_cv_t v) {
    // Calculate the required size of the buffer for this boundary
    int buf_size = GetBufferSize(pmb, nb, v);

    // Add a buffer pool if one does not exist for this size
    #ifdef USE_NEIGHBORHOOD_COLLECTIVES
    if (BTYPE != BoundaryType::nonlocal && pmesh->pool_map.count(buf_size) == 0) {
      pmesh->pool_map.emplace(std::make_pair(
          buf_size, buf_pool_t<Real>([buf_size](buf_pool_t<Real> *pool) {
            using buf_t = buf_pool_t<Real>::base_t;
            // TODO(LFR): Make nbuf a user settable parameter
            const int nbuf = 200;
            buf_t chunk("pool buffer", buf_size * nbuf);
            for (int i = 1; i < nbuf; ++i) {
              pool->AddFreeObjectToPool(
                  buf_t(chunk, std::make_pair(i * buf_size, (i + 1) * buf_size)));
            }
            return buf_t(chunk, std::make_pair(0, buf_size));
          })));
    }
    #else
    if (pmesh->pool_map.count(buf_size) == 0) {
      pmesh->pool_map.emplace(std::make_pair(
          buf_size, buf_pool_t<Real>([buf_size](buf_pool_t<Real> *pool) {
            using buf_t = buf_pool_t<Real>::base_t;
            // TODO(LFR): Make nbuf a user settable parameter
            const int nbuf = 200;
            buf_t chunk("pool buffer", buf_size * nbuf);
            for (int i = 1; i < nbuf; ++i) {
              pool->AddFreeObjectToPool(
                  buf_t(chunk, std::make_pair(i * buf_size, (i + 1) * buf_size)));
            }
            return buf_t(chunk, std::make_pair(0, buf_size));
          })));
    }
    #endif

    const int receiver_rank = nb.snb.rank;
    const int sender_rank = Globals::my_rank;

    int tag = 0;
#ifdef MPI_PARALLEL
    // Get a bi-directional mpi tag for this pair of blocks
    tag = pmesh->tag_map.GetTag(pmb, nb);
    auto comm_label = v->label();
    if constexpr (BTYPE == BoundaryType::flxcor_send ||
                  BTYPE == BoundaryType::flxcor_recv)
      comm_label += "_flcor";
    mpi_comm_t comm = pmesh->GetMPIComm(comm_label);
#else
      // Setting to zero is fine here since this doesn't actually get used when everything
      // is on the same rank
      mpi_comm_t comm = 0;
#endif

    bool use_sparse_buffers = v->IsSet(Metadata::Sparse);
    auto get_resource_method = [pmesh, buf_size]() {
      return buf_pool_t<Real>::owner_t(pmesh->pool_map.at(buf_size).Get());
    };

    #ifdef USE_NEIGHBORHOOD_COLLECTIVES
    int neigh_offset = -1;
    int end_neigh_offset = -1;
    if(BTYPE == BoundaryType::nonlocal){
      auto offset_info = pmesh->neigh_token.per_tag_offsets[receiver_rank][tag];
      neigh_offset = offset_info.first;
      end_neigh_offset   = offset_info.second;
    }
    #endif

    #ifdef USE_NEIGHBORHOOD_COLLECTIVES
    if(BTYPE == BoundaryType::nonlocal){
      auto neigh_get_resource_method = [pmesh, neigh_offset, end_neigh_offset]() {
        auto send_subview = subview(pmesh->neigh_token.send_comm_buffer,std::pair<size_t, size_t>(neigh_offset,end_neigh_offset));
        return buf_pool_t<Real>::owner_t( buf_pool_t<Real>::weak_t(std::move(send_subview)));
      };

      if constexpr (IsSender(BTYPE)) {
      auto s_key = SendKey(pmb, nb, v);
      if (buf_map.count(s_key) == 0)
        buf_map[s_key] = CommBuffer<buf_pool_t<Real>::owner_t>(
            tag, sender_rank, receiver_rank, comm, neigh_get_resource_method,
            use_sparse_buffers);
      }
    }
    else{
      if constexpr (IsSender(BTYPE)) {
        auto s_key = SendKey(pmb, nb, v);
        if (buf_map.count(s_key) == 0)
          buf_map[s_key] = CommBuffer<buf_pool_t<Real>::owner_t>(
              tag, sender_rank, receiver_rank, comm, get_resource_method,
              use_sparse_buffers);
      }
    }

    #else
    // Build send buffer (unless this is a receiving flux boundary)
    if constexpr (IsSender(BTYPE)) {
      auto s_key = SendKey(pmb, nb, v);
      if (buf_map.count(s_key) == 0)
        buf_map[s_key] = CommBuffer<buf_pool_t<Real>::owner_t>(
            tag, sender_rank, receiver_rank, comm, get_resource_method,
            use_sparse_buffers);
    }
    #endif // USE_NEIGHBORHOOD_COLLECTIVES (sender)

    // Also build the non-local receive buffers here
    #ifdef USE_NEIGHBORHOOD_COLLECTIVES
    if(BTYPE == BoundaryType::nonlocal){
      auto neigh_recv_get_resource_method = [pmesh, neigh_offset, end_neigh_offset]() {
        auto recv_subview = subview(pmesh->neigh_token.recv_comm_buffer,std::pair<size_t, size_t>(neigh_offset,end_neigh_offset));
        return buf_pool_t<Real>::owner_t( buf_pool_t<Real>::weak_t(recv_subview));
      };

      if constexpr (IsReceiver(BTYPE)) {
        if (sender_rank != receiver_rank) {
          auto r_key = ReceiveKey(pmb, nb, v);
          if (buf_map.count(r_key) == 0)
            buf_map[r_key] = CommBuffer<buf_pool_t<Real>::owner_t>(
                tag, receiver_rank, sender_rank, comm, neigh_recv_get_resource_method,
                use_sparse_buffers);
        }
      }

    }
    else{
      if constexpr (IsReceiver(BTYPE)) {
        if (sender_rank != receiver_rank) {
          auto r_key = ReceiveKey(pmb, nb, v);
          if (buf_map.count(r_key) == 0)
            buf_map[r_key] = CommBuffer<buf_pool_t<Real>::owner_t>(
                tag, receiver_rank, sender_rank, comm, get_resource_method,
                use_sparse_buffers);
        }
      }
    }
    #else
    if constexpr (IsReceiver(BTYPE)) {
      if (sender_rank != receiver_rank) {
        auto r_key = ReceiveKey(pmb, nb, v);
        if (buf_map.count(r_key) == 0)
          buf_map[r_key] = CommBuffer<buf_pool_t<Real>::owner_t>(
              tag, receiver_rank, sender_rank, comm, get_resource_method,
              use_sparse_buffers);
      }
    }
    #endif // USE_NEIGHBORHOOD_COLLECTIVES (receiver)
  });
}
} // namespace

// pmesh->boundary_comm_map.clear() after every remesh
// in InitializeBlockTimeStepsAndBoundaries()
TaskStatus BuildBoundaryBuffers(std::shared_ptr<MeshData<Real>> &md) {
  PARTHENON_INSTRUMENT
  Mesh *pmesh = md->GetMeshPointer();
  auto &all_caches = md->GetBvarsCache();

  // Clear the fast access vectors for this block since they are no longer valid
  // after all MeshData call BuildBoundaryBuffers
  all_caches.clear();

  #ifdef USE_NEIGHBORHOOD_COLLECTIVES
    pmesh->neigh_token.start_building_buffers();
  #endif
   // Moraru : separate local and nonlocal
  BuildBoundaryBufferSubset<BoundaryType::nonlocal>(md, pmesh->boundary_comm_map);
  #ifdef USE_NEIGHBORHOOD_COLLECTIVES
    pmesh->neigh_token.end_building_buffers();
  #endif

  BuildBoundaryBufferSubset<BoundaryType::local>(md, pmesh->boundary_comm_map);
  
  BuildBoundaryBufferSubset<BoundaryType::flxcor_send>(md,
                                                       pmesh->boundary_comm_flxcor_map);
  BuildBoundaryBufferSubset<BoundaryType::flxcor_recv>(md,
                                                       pmesh->boundary_comm_flxcor_map);

  return TaskStatus::complete;
}

TaskStatus BuildGMGBoundaryBuffers(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::Profiling::pushRegion("Task_BuildSendBoundBufs");
  Mesh *pmesh = md->GetMeshPointer();
  auto &all_caches = md->GetBvarsCache();

  std::cout<<"-- CALL TO BuildGMGBoundaryBuffers"<<std::endl;
  // Clear the fast access vectors for this block since they are no longer valid
  // after all MeshData call BuildBoundaryBuffers
  all_caches.clear();
  BuildBoundaryBufferSubset<BoundaryType::gmg_same>(md, pmesh->boundary_comm_map);
  BuildBoundaryBufferSubset<BoundaryType::gmg_prolongate_send>(md,
                                                               pmesh->boundary_comm_map);
  BuildBoundaryBufferSubset<BoundaryType::gmg_prolongate_recv>(md,
                                                               pmesh->boundary_comm_map);
  BuildBoundaryBufferSubset<BoundaryType::gmg_restrict_send>(md,
                                                             pmesh->boundary_comm_map);
  BuildBoundaryBufferSubset<BoundaryType::gmg_restrict_recv>(md,
                                                             pmesh->boundary_comm_map);

  Kokkos::Profiling::popRegion(); // "Task_BuildSendBoundBufs"
  return TaskStatus::complete;
}

} // namespace parthenon
