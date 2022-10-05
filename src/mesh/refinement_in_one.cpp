//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2022 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001
// for Los Alamos National Laboratory (LANL), which is operated by Triad
// National Security, LLC for the U.S. Department of Energy/National Nuclear
// Security Administration. All rights in the program are reserved by Triad
// National Security, LLC, and the U.S. Department of Energy/National Nuclear
// Security Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
// in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do
// so.
//========================================================================================

#include <algorithm>
#include <utility>

#include "kokkos_abstraction.hpp"
#include "mesh/mesh_refinement_ops.hpp"
#include "mesh/refinement_in_one.hpp"

namespace parthenon {
namespace refinement {

std::vector<bool> ComputePhysicalRestrictBoundsAllocStatus(MeshData<Real> *md) {
  Kokkos::Profiling::pushRegion("ComputePhysicalRestrictBoundsAllocStatus_MeshData");
  std::vector<bool> alloc_status;
  for (int block = 0; block < md->NumBlocks(); ++block) {
    auto &rc = md->GetBlockData(block);
    auto pmb = rc->GetBlockPointer();
    int nrestrictions = pmb->pbval->NumRestrictions();
    for (auto &v : rc->GetCellVariableVector()) {
      if (v->IsSet(parthenon::Metadata::FillGhost)) {
        int num_bufs = nrestrictions * (v->GetDim(6)) * (v->GetDim(5));
        for (int i = 0; i < num_bufs; ++i) {
          alloc_status.push_back(v->IsAllocated());
        }
      }
    }
  }

  Kokkos::Profiling::popRegion(); // ComputePhysicalRestrictBoundsAllocStatus_MeshData
  return alloc_status;
}

void ComputePhysicalRestrictBounds(MeshData<Real> *md) {
  Kokkos::Profiling::pushRegion("ComputePhysicalRestrictBounds_MeshData");
  auto alloc_status = ComputePhysicalRestrictBoundsAllocStatus(md);

  cell_centered_bvars::BufferCache_t info("physical restriction bounds",
                                          alloc_status.size());
  auto info_h = Kokkos::create_mirror_view(info);
  int idx = 0;
  for (int block = 0; block < md->NumBlocks(); ++block) {
    auto &rc = md->GetBlockData(block);
    auto pmb = rc->GetBlockPointer();
    for (auto &v : rc->GetCellVariableVector()) {
      if (v->IsSet(parthenon::Metadata::FillGhost)) {
        pmb->pbval->FillRestrictionMetadata(info_h, idx, v);
      }
    }
  }
  PARTHENON_DEBUG_REQUIRE(idx == alloc_status.size(), "All buffers accounted for");
  Kokkos::deep_copy(info, info_h);

  md->SetRestrictBuffers(info, info_h, alloc_status);

  Kokkos::Profiling::popRegion(); // ComputePhysicalRestrictBoundso_MeshData
}

// TODO(JMM): add explicit instantiations of the default prolongation/restriction functions

} // namespace refinement
} // namespace parthenon
