//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2024. Triad National Security, LLC. All rights reserved.
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

#ifndef BVALS_COMMS_BVALS_IN_ONE_HPP_
#define BVALS_COMMS_BVALS_IN_ONE_HPP_

#include <memory>
#include <string>
#include <vector>

#include "basic_types.hpp"
#include "bvals/neighbor_block.hpp"
#include "coordinates/coordinates.hpp"

#include "tasks/tasks.hpp"
#include "utils/object_pool.hpp"

namespace parthenon {

template <typename T>
class MeshData;
class IndexRange;
class NeighborBlock;
template <typename T>
class Variable;

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

template <BoundaryType bound_type>
TaskStatus ProlongateBounds(std::shared_ptr<MeshData<Real>> &md);
inline TaskStatus ProlongateBoundaries(std::shared_ptr<MeshData<Real>> &md) {
  return ProlongateBounds<BoundaryType::any>(md);
}

static TaskStatus StartReceiveFluxCorrections(std::shared_ptr<MeshData<Real>> &md) {
  return StartReceiveBoundBufs<BoundaryType::flxcor_recv>(md);
}
static TaskStatus LoadAndSendFluxCorrections(std::shared_ptr<MeshData<Real>> &md) {
  return SendBoundBufs<BoundaryType::flxcor_send>(md);
}
static TaskStatus ReceiveFluxCorrections(std::shared_ptr<MeshData<Real>> &md) {
  return ReceiveBoundBufs<BoundaryType::flxcor_recv>(md);
}
static TaskStatus SetFluxCorrections(std::shared_ptr<MeshData<Real>> &md) {
  return SetBounds<BoundaryType::flxcor_recv>(md);
}

TaskStatus SendSwarmBoundaryBuffers(std::shared_ptr<MeshData<Real>> &md);
TaskStatus SendSwarmBoundaryBuffers(std::shared_ptr<MeshData<Real>> &md,
                                    std::vector<std::string> swarm_names);
TaskStatus ReceiveSwarmBoundaryBuffers(std::shared_ptr<MeshData<Real>> &md);
TaskStatus ReceiveSwarmBoundaryBuffers(std::shared_ptr<MeshData<Real>> &md,
                                       std::vector<std::string> swarm_names);

// Adds all relevant boundary communication to a single task list
template <BoundaryType bounds = BoundaryType::any>
TaskID AddBoundaryExchangeTasks(TaskID dependency, TaskList &tl,
                                std::shared_ptr<MeshData<Real>> &md, bool multilevel);

// Adds all relevant flux correction tasks to a single task list
TaskID AddFluxCorrectionTasks(TaskID dependency, TaskList &tl,
                              std::shared_ptr<MeshData<Real>> &md, bool multilevel);

// Adds all relevant swarm boundary communication to a single task list
TaskID AddSwarmBoundaryExchangeTasks(TaskID dependency, TaskList &tl,
                                     std::shared_ptr<MashData<Real>> &md);
TaskID AddSwarmBoundaryExchangeTasks(TaskID dependency, TaskList &tl,
                                     std::shared_ptr<MashData<Real>> &md,
                                     std::vector<std::string> swarm_names);

// These tasks should not be called in down stream code
TaskStatus BuildBoundaryBuffers(std::shared_ptr<MeshData<Real>> &md);
TaskStatus BuildGMGBoundaryBuffers(std::shared_ptr<MeshData<Real>> &md);

} // namespace parthenon

#endif // BVALS_COMMS_BVALS_IN_ONE_HPP_
