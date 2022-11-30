//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
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

#ifndef BVALS_CC_BVALS_CC_IN_ONE_HPP_
#define BVALS_CC_BVALS_CC_IN_ONE_HPP_

#include <memory>
#include <string>
#include <vector>

#include "basic_types.hpp"
#include "bvals/bvals_interfaces.hpp"
#include "coordinates/coordinates.hpp"
#include "interface/variable_state.hpp"
#include "utils/communication_buffer.hpp"
#include "utils/object_pool.hpp"

namespace parthenon {

template <typename T>
class MeshData;
class IndexRange;
class NeighborBlock;
template <typename T>
class CellVariable;

namespace cell_centered_bvars {

template <BoundaryType bound_type>
TaskStatus SendBoundBufs(std::shared_ptr<MeshData<Real>> &md);
template <BoundaryType bound_type>
TaskStatus StartReceiveBoundBufs(std::shared_ptr<MeshData<Real>> &md);
template <BoundaryType bound_type>
TaskStatus ReceiveBoundBufs(std::shared_ptr<MeshData<Real>> &md);
template <BoundaryType bound_type>
TaskStatus SetBounds(std::shared_ptr<MeshData<Real>> &md);

inline TaskStatus SendBoundaryBuffers(std::shared_ptr<MeshData<Real>> &md) {
  return SendBoundBufs<BoundaryType::any>(md);
}
inline TaskStatus StartReceiveBoundaryBuffers(std::shared_ptr<MeshData<Real>> &md) {
  return StartReceiveBoundBufs<BoundaryType::any>(md);
}
inline TaskStatus ReceiveBoundaryBuffers(std::shared_ptr<MeshData<Real>> &md) {
  return ReceiveBoundBufs<BoundaryType::any>(md);
}
inline TaskStatus SetBoundaries(std::shared_ptr<MeshData<Real>> &md) {
  return SetBounds<BoundaryType::any>(md);
}

TaskStatus StartReceiveFluxCorrections(std::shared_ptr<MeshData<Real>> &md);
TaskStatus LoadAndSendFluxCorrections(std::shared_ptr<MeshData<Real>> &md);
TaskStatus ReceiveFluxCorrections(std::shared_ptr<MeshData<Real>> &md);
TaskStatus SetFluxCorrections(std::shared_ptr<MeshData<Real>> &md);

// This task should not be called in down stream code
TaskStatus BuildBoundaryBuffers(std::shared_ptr<MeshData<Real>> &md);

} // namespace cell_centered_bvars
} // namespace parthenon

#endif // BVALS_CC_BVALS_CC_IN_ONE_HPP_
