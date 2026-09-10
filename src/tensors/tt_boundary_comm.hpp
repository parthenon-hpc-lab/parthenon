//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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

#ifndef TENSORS_TT_BOUNDARY_COMM_HPP
#define TENSORS_TT_BOUNDARY_COMM_HPP

#include <memory>

#include "tensors/tt_boundary_cache.hpp"
#include "tensors/tt_container.hpp"

namespace parthenon {

// Build (or rebuild) the boundary-index cache for a MeshTTData partition: walk every
// boundary, derive the sender's interior-send box and the receiver's exterior-recv box
// (via CalcIndices + the reverse-neighbor descriptor), store the two indexers plus the
// whole-block extents as a TTBndInfo, and ensure a TTCommChannel exists in
// mesh->tt_comm_map for each boundary (recorded by index in cache.channels). The result is
// a single flat device array for one batched Send launch. Identity-transform, same-level
// boundaries only for now (Step 6 scope); asserts otherwise. Rebuilds when the mesh
// channel-map epoch has advanced (i.e. after (re)mesh).
void BuildTTBoundaryCache(std::shared_ptr<MeshTTData> &md, TTBoundaryCache *cache);

} // namespace parthenon

#endif // TENSORS_TT_BOUNDARY_COMM_HPP
