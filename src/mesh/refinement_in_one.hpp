//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2022 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2022. Triad National Security, LLC. All rights reserved.
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

#ifndef MESH_REFINEMENT_CC_IN_ONE_HPP_
#define MESH_REFINEMENT_CC_IN_ONE_HPP_

#include <algorithm>
#include <functional> // std::function
#include <tuple>      // std::tuple
#include <utility>    // std::forward
#include <vector>

#include "bvals/cc/bvals_cc_in_one.hpp" // for buffercache_t
#include "coordinates/coordinates.hpp"  // for coordinates
#include "globals.hpp"                  // for Globals
#include "mesh/domain.hpp"              // for IndexShape
#include "mesh/mesh_refinement_loops.hpp"
#include "mesh/mesh_refinement_ops.hpp"

namespace parthenon {
template <typename T>
class MeshData; // forward declaration

namespace refinement {
std::vector<bool> ComputePhysicalRestrictBoundsAllocStatus(MeshData<Real> *md);
void ComputePhysicalRestrictBounds(MeshData<Real> *md);

// The existence of this overload allows us to avoid a deep-copy in
// the per-meshblock calls
// TODO(JMM): I don't love having two overloads here.  However when
// we shift entirely to in-one machinery, the info_h only overload
// will go away.
template <template <int> class Op = refinement_ops::RestrictCellAverage>
void Restrict(const cell_centered_bvars::BufferCache_t &info,
              const cell_centered_bvars::BufferCacheHost_t &info_h,
              const IndexShape &cellbnds, const IndexShape &c_cellbnds) {
  const auto op = RefinementOp_t::Restriction;
  loops::DoProlongationRestrictionOp<Op>(cellbnds, info, info_h, cellbnds, c_cellbnds,
                                         op);
}
// The existence of this overload allows us to avoid a deep-copy in
// the per-meshblock calls
template <template <int> class Op = refinement_ops::RestrictCellAverage>
void Restrict(const cell_centered_bvars::BufferCacheHost_t &info_h,
              const IndexShape &cellbnds, const IndexShape &c_cellbnds) {
  const auto op = RefinementOp_t::Restriction;
  loops::DoProlongationRestrictionOp<Op>(cellbnds, info_h, cellbnds, c_cellbnds, op);
}

// Prototype for function that exists only to avoid circular dependency
namespace impl {
std::tuple<cell_centered_bvars::BufferCache_t, cell_centered_bvars::BufferCacheHost_t,
           IndexShape, IndexShape>
GetAndUpdateRestrictionBuffers(MeshData<Real> *md, const std::vector<bool> &alloc_status);
} // namespace impl
template <template <int> class Op = refinement_ops::RestrictCellAverage>
TaskStatus RestrictPhysicalBounds(MeshData<Real> *md) {
  Kokkos::Profiling::pushRegion("Task_RestrictPhysicalBounds_MeshData");

  // get alloc status
  auto alloc_status = ComputePhysicalRestrictBoundsAllocStatus(md);

  auto info_tuple = impl::GetAndUpdateRestrictionBuffers(md, alloc_status);
  auto info = std::get<0>(info_tuple);
  auto info_h = std::get<1>(info_tuple);
  auto cellbounds = std::get<2>(info_tuple);
  auto c_cellbounds = std::get<3>(info_tuple);

  Restrict<Op>(info, info_h, cellbounds, c_cellbounds);

  Kokkos::Profiling::popRegion(); // Task_RestrictPhysicalBounds_MeshData
  return TaskStatus::complete;
}

// TODO(JMM): I don't love having two overloads here.  However when
// we shift entirely to in-one machinery, the info_h only overload
// will go away.
template <template <int> class Op = refinement_ops::ProlongateCellMinMod>
void Prolongate(const cell_centered_bvars::BufferCache_t &info,
                const cell_centered_bvars::BufferCacheHost_t &info_h,
                const IndexShape &cellbnds, const IndexShape &c_cellbnds) {
  const auto op = RefinementOp_t::Prolongation;
  loops::DoProlongationRestrictionOp<Op>(cellbnds, info, info_h, cellbnds, c_cellbnds,
                                         op);
}
// The existence of this overload allows us to avoid a deep-copy in
// the per-meshblock calls
template <template <int> class Op = refinement_ops::ProlongateCellMinMod>
void Prolongate(const cell_centered_bvars::BufferCacheHost_t &info_h,
                const IndexShape &cellbnds, const IndexShape &c_cellbnds) {
  const auto op = RefinementOp_t::Prolongation;
  loops::DoProlongationRestrictionOp<Op>(cellbnds, info_h, cellbnds, c_cellbnds, op);
}

// std::function closures for the top-level restriction functions
using Restrictor_t = std::function<void(const cell_centered_bvars::BufferCache_t &,
                                        const cell_centered_bvars::BufferCacheHost_t &,
                                        const IndexShape &, const IndexShape &)>;
using RestrictorHost_t =
    std::function<void(const cell_centered_bvars::BufferCacheHost_t &, const IndexShape &,
                       const IndexShape &)>;
using BoundaryRestrictor_t = std::function<TaskStatus(MeshData<Real> *)>;
using Prolongator_t = std::function<void(const cell_centered_bvars::BufferCache_t &,
                                         const cell_centered_bvars::BufferCacheHost_t &,
                                         const IndexShape &, const IndexShape &)>;
using ProlongatorHost_t =
    std::function<void(const cell_centered_bvars::BufferCacheHost_t &, const IndexShape &,
                       const IndexShape &)>;
struct RefinementFunctions_t {
  RefinementFunctions_t() = default;

  template <template <int> class ProlongationOp, template <int> class RestrictionOp>
  static RefinementFunctions_t RegisterOps() {
    RefinementFunctions_t funcs;
    funcs.restrictor = [](const cell_centered_bvars::BufferCache_t &info,
                          const cell_centered_bvars::BufferCacheHost_t &info_h,
                          const IndexShape &cellbnds, const IndexShape &c_cellbnds) {
      Restrict<RestrictionOp>(info, info_h, cellbnds, c_cellbnds);
    };
    funcs.restrictor_host = [](const cell_centered_bvars::BufferCacheHost_t &info_h,
                               const IndexShape &cellbnds, const IndexShape &c_cellbnds) {
      Restrict<RestrictionOp>(info_h, cellbnds, c_cellbnds);
    };
    funcs.boundary_restrictor = RestrictPhysicalBounds<RestrictionOp>;
    funcs.prolongator = [](const cell_centered_bvars::BufferCache_t &info,
                           const cell_centered_bvars::BufferCacheHost_t &info_h,
                           const IndexShape &cellbnds, const IndexShape &c_cellbnds) {
      Prolongate<ProlongationOp>(info, info_h, cellbnds, c_cellbnds);
    };
    funcs.prolongator_host = [](const cell_centered_bvars::BufferCacheHost_t &info_h,
                                const IndexShape &cellbnds,
                                const IndexShape &c_cellbnds) {
      Prolongate<ProlongationOp>(info_h, cellbnds, c_cellbnds);
    };
    return funcs;
  }

  Restrictor_t restrictor;
  RestrictorHost_t restrictor_host;
  BoundaryRestrictor_t boundary_restrictor;
  Prolongator_t prolongator;
  ProlongatorHost_t prolongator_host;
};

} // namespace refinement
} // namespace parthenon

#endif // MESH_REFINEMENT_CC_IN_ONE_HPP_
