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

TaskStatus BuildBoundaryBuffers(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::Profiling::pushRegion("Task_BuildSendBoundaryBuffers"); 
  Mesh* pmesh = md->GetMeshPointer(); 

  // Check if mesh topology changed 
  
  int isize, jsize, ksize; 
  {
    auto& rc = md->GetBlockData(0);
    auto pmb = rc->GetBlockPointer(); 
    auto& cb = pmb->cellbounds; 
    IndexDomain in = IndexDomain::interior;
     isize = cb.ie(in) - cb.is(in) + 1;
     jsize = cb.je(in) - cb.js(in) + 1; 
     ksize = cb.ke(in) - cb.ks(in) + 1;  
  } 

  // Build buffers 
  for (int block = 0; block < md->NumBlocks(); ++block) {
    auto &rc = md->GetBlockData(block);
    auto pmb = rc->GetBlockPointer();
    for (auto &v : rc->GetCellVariableVector()) {
      if (v->IsSet(Metadata::FillGhost)) {
        for (int n = 0; n < pmb->pbval->nneighbor; ++n) {
          auto& nb = pmb->pbval->neighbor[n];

          // Calculate the buffer size, this should be safe even for unsame levels 
          int buf_size = (nb.ni.ox1 == 0 ? isize : Globals::nghost) 
                       * (nb.ni.ox2 == 0 ? jsize : Globals::nghost)
                       * (nb.ni.ox3 == 0 ? ksize : Globals::nghost);

          // Add a buffer pool if one does not exist for this size  
          if (pmesh->pool_map.count(buf_size) == 0) {
            pmesh->pool_map.emplace(std::make_pair(buf_size, Mesh::pool_t([buf_size]() {
                                                     return Mesh::pool_t::base_t("pool buffer", buf_size);
                                                   })));
          } 
          
          const int sender_id = pmb->gid; 
          const int receiver_id = nb.snb.gid;
          // This tag is still pretty limiting, since 2^7 = 128 
          int tag = (nb.snb.lid << 7) | pmb->lid; 

          const int receiver_rank = nb.snb.rank;
          const int sender_rank = Globals::my_rank; 

          // TODO: Fix this to use Philipp's communicators and deal with mpi or no mpi
          const int comm = 0; 
          

          // Build sending buffers  
          pmesh->boundary_comm_map[{sender_id, receiver_id, v->label()}] = 
                                                 CommBuffer<Mesh::pool_t::weak_t>(tag, sender_rank, receiver_rank, comm,
                                                 [&pmesh, buf_size]()
                                                 { return pmesh->pool_map.at(buf_size).Get();});
          
          // Also build the non-local receive buffers here          
          if (sender_rank != receiver_rank) {
            int tag = (pmb->lid << 7) | nb.snb.lid; 
            pmesh->boundary_comm_map[{receiver_id, sender_id, v->label()}] = 
                                                 CommBuffer<Mesh::pool_t::weak_t>(tag, receiver_rank, sender_rank, comm,
                                                 [&pmesh, buf_size]()
                                                 { return pmesh->pool_map.at(buf_size).Get();});
          }
        }
      }
    }
  }

  Kokkos::Profiling::popRegion();
  return TaskStatus::complete;
}

TaskStatus LoadAndSendBoundaryBuffers(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::Profiling::pushRegion("Task_LoadAndSendBoundaryBuffers"); 
  // Allocate buffers as necessary 
  for (int block = 0; block < md->NumBlocks(); ++block) {
    auto &rc = md->GetBlockData(block);
    auto pmb = rc->GetBlockPointer();
    for (auto &v : rc->GetCellVariableVector()) {
      if (v->IsSet(Metadata::FillGhost)) {
        for (int n = 0; n < pmb->pbval->nneighbor; ++n) {
          auto& nb = pmb->pbval->neighbor[n];
        }
      }
    }
  }
   
  Kokkos::Profiling::popRegion();
  return TaskStatus::complete;
}

TaskStatus ReceiveBoundaryBuffers(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::Profiling::pushRegion("Task_ReceiveBoundaryBuffers"); 

  Kokkos::Profiling::popRegion();
  return TaskStatus::complete;
}

TaskStatus ActivateBasedOnBoundaries(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::Profiling::pushRegion("Task_ActivateBasedOnBoundaries"); 

  Kokkos::Profiling::popRegion();
  return TaskStatus::complete;
}

TaskStatus SetInternalBoundaries(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::Profiling::pushRegion("Task_SetInternalBoundaries");  

  Kokkos::Profiling::popRegion();
  return TaskStatus::complete;
}

} // namespace cell_centered_bvars
} // namespace parthenon
