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

#ifndef BVALS_BOUNDARY_CONDITIONS_HPP_
#define BVALS_BOUNDARY_CONDITIONS_HPP_

#include <functional>
#include <memory>
#include <string>

#include "basic_types.hpp"
#include "interface/meshblock_data.hpp"
#include "interface/swarm_boundaries.hpp"
#include "mesh/domain.hpp"

namespace parthenon {

// Physical boundary conditions

using BValFunc = std::function<void(std::shared_ptr<MeshBlockData<Real>> &, bool)>;
using SBValFunc = std::function<
    std::unique_ptr<ParticleBound, DeviceDeleter<parthenon::DevMemSpace>>()>;

TaskStatus ApplyBoundaryConditionsOnCoarseOrFine(std::shared_ptr<MeshBlockData<Real>> &rc,
                                                 bool coarse);

inline TaskStatus ApplyBoundaryConditions(std::shared_ptr<MeshBlockData<Real>> &rc) {
  return ApplyBoundaryConditionsOnCoarseOrFine(rc, false);
}

namespace BoundaryFunction {

void OutflowInnerX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse);
void OutflowOuterX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse);
void OutflowInnerX2(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse);
void OutflowOuterX2(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse);
void OutflowInnerX3(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse);
void OutflowOuterX3(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse);
void ReflectInnerX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse);
void ReflectOuterX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse);
void ReflectInnerX2(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse);
void ReflectOuterX2(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse);
void ReflectInnerX3(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse);
void ReflectOuterX3(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse);

} // namespace BoundaryFunction
} // namespace parthenon

#endif // BVALS_BOUNDARY_CONDITIONS_HPP_
