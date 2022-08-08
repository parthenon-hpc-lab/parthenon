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
#include "mesh/refinement_cc_in_one.hpp"

namespace parthenon {
namespace cell_centered_refinement {

/*
 * TODO(JMM): At some point we will want to be able to register
 * alternative prolongation/restriction operators per variable. For
 * example, face-centered fields, for higher-order
 * prolongation/restriction.
 * In this case, the restriction stencil, e.g., RestrictCellAverage<int>
 * should become a functor, on which the restriction loop can be templated.
 * Then users can register a restriction loop specialized to a given functor.
 */

void Restrict(const cell_centered_bvars::BufferCache_t &info,
              const cell_centered_bvars::BufferCacheHost_t &info_h,
              const IndexShape &cellbnds, const IndexShape &c_cellbnds) {
  const auto op = RefinementOp_t::Restriction;
  impl::DoProlongationRestrictionOp<refinement_ops::RestrictCellAverage>(
      cellbnds, info, info_h, cellbnds, c_cellbnds, op);
}
void Restrict(const cell_centered_bvars::BufferCacheHost_t &info_h,
              const IndexShape &cellbnds, const IndexShape &c_cellbnds) {
  const auto op = RefinementOp_t::Restriction;
  impl::DoProlongationRestrictionOp<refinement_ops::RestrictCellAverage>(
      cellbnds, info_h, cellbnds, c_cellbnds, op);
}

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

TaskStatus RestrictPhysicalBounds(MeshData<Real> *md) {
  Kokkos::Profiling::pushRegion("Task_RestrictPhysicalBounds_MeshData");

  // get alloc status
  auto alloc_status = ComputePhysicalRestrictBoundsAllocStatus(md);

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

  Restrict(info, info_h, cellbounds, c_cellbounds);

  Kokkos::Profiling::popRegion(); // Task_RestrictPhysicalBounds_MeshData
  return TaskStatus::complete;
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

  Kokkos::Profiling::popRegion(); // ComputePhysicalRestrictBounds_MeshData
}

void Prolongate(const cell_centered_bvars::BufferCache_t &info,
                const cell_centered_bvars::BufferCacheHost_t &info_h,
                const IndexShape &cellbnds, const IndexShape &c_cellbnds) {
  const auto op = RefinementOp_t::Prolongation;
  impl::DoProlongationRestrictionOp<refinement_ops::ProlongateCellMinMod>(
      cellbnds, info, info_h, cellbnds, c_cellbnds, op);
}
void Prolongate(const cell_centered_bvars::BufferCacheHost_t &info_h,
                const IndexShape &cellbnds, const IndexShape &c_cellbnds) {
  const auto op = RefinementOp_t::Prolongation;
  impl::DoProlongationRestrictionOp<refinement_ops::ProlongateCellMinMod>(
      cellbnds, info_h, cellbnds, c_cellbnds, op);
}

} // namespace cell_centered_refinement
} // namespace parthenon
