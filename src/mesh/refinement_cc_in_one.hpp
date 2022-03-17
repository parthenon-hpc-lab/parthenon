//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
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

#include <vector>

#include "bvals/cc/bvals_cc_in_one.hpp" // for buffercache_t
#include "coordinates/coordinates.hpp"  // for coordinates
#include "interface/mesh_data.hpp"
#include "mesh/domain.hpp" // for IndexShape

namespace parthenon {
namespace cell_centered_refinement {
void Restrict(cell_centered_bvars::CommBufferCache_t &info, IndexShape &cellbounds,
              IndexShape &c_cellbounds);
void Restrict(cell_centered_bvars::RefineBufferCache_t &info, IndexShape &cellbounds,
              IndexShape &c_cellbounds);
TaskStatus RestrictPhysicalBounds(MeshData<Real> *md);

std::vector<bool> ComputePhysicalRestrictBoundsAllocStatus(MeshData<Real> *md);
void ComputePhysicalRestrictBounds(MeshData<Real> *md);

} // namespace cell_centered_refinement
} // namespace parthenon

#endif // MESH_REFINEMENT_CC_IN_ONE_HPP_
