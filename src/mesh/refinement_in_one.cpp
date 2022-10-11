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
#include <tuple> // std::tuple
#include <utility>

#include "interface/mesh_data.hpp"
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
      if (v->IsSet(Metadat::FillGhost)) {
        pmb->pbval->FillRestrictionMetadata(info_h, idx, v);
      }
    }
  }
  PARTHENON_DEBUG_REQUIRE(idx == alloc_status.size(), "All buffers accounted for");
  Kokkos::deep_copy(info, info_h);

  md->SetRestrictBuffers(info, info_h, alloc_status);

  Kokkos::Profiling::popRegion(); // ComputePhysicalRestrictBoundso_MeshData
}

// This needs to be here to avoid circular dependencies with MeshData.
namespace impl {
std::tuple<cell_centered_bvars::BufferCache_t, cell_centered_bvars::BufferCacheHost_t,
           IndexShape, IndexShape>
GetAndUpdateRestrictionBuffers(MeshData<Real> *md,
                               const std::vector<bool> &alloc_status) {
  auto info_pair = md->GetRestrictBuffers();
  auto info = std::get<0>(info_pair);
  auto info_h = std::get<1>(info_pair);
  if (!info.is_allocated() || (alloc_status != md->GetRestrictBufAllocStatus())) {
    ComputePhysicalRestrictBounds(md);
    info_pair = md->GetRestrictBuffers();
    info = std::get<0>(info_pair);
    info_h = std::get<1>(info_pair);
  }
  auto &rc = md->GetBlockData(0);
  auto pmb = rc->GetBlockPointer();
  IndexShape cellbounds = pmb->cellbounds;
  IndexShape c_cellbounds = pmb->c_cellbounds;
  return std::make_tuple(info, info_h, cellbounds, c_cellbounds);
}
} // namespace impl

// explicit instantiations of the default prolongation/restriction
// functions
template <>
void Restrict<refinement_ops::RestrictCellAverage>(
    const cell_centered_bvars::BufferCache_t &info,
    const cell_centered_bvars::BufferCacheHost_t &info_h, const IndexShape &cellbnds,
    const IndexShape &c_cellbnds);
template <>
void Restrict<refinement_ops::RestrictCellAverage>(
    const cell_centered_bvars::BufferCacheHost_t &info_h, const IndexShape &cellbnds,
    const IndexShape &c_cellbnds);
template <>
TaskStatus
RestrictPhysicalBounds<refinement_ops::RestrictCellAverage>(MeshData<Real> *md);
template <>
void Prolongate<refinement_ops::ProlongateCellMinMod>(
    const cell_centered_bvars::BufferCache_t &info,
    const cell_centered_bvars::BufferCacheHost_t &info_h, const IndexShape &cellbnds,
    const IndexShape &c_cellbnds);
template <>
void Prolongate<refinement_ops::ProlongateCellMinMod>(
    const cell_centered_bvars::BufferCacheHost_t &info_h, const IndexShape &cellbnds,
    const IndexShape &c_cellbnds);

} // namespace refinement
} // namespace parthenon
