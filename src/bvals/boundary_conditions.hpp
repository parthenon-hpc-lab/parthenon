//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
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

#include <memory>

#include "basic_types.hpp"
#include "interface/meshblock_data.hpp"

namespace parthenon {

TaskStatus ApplyBoundaryConditions(std::shared_ptr<MeshBlockData<Real>> &rc,
                                   bool coarse = false);

namespace BoundaryFunction {

void OutflowInnerX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse = false);
void OutflowOuterX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse = false);
void OutflowInnerX2(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse = false);
void OutflowOuterX2(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse = false);
void OutflowInnerX3(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse = false);
void OutflowOuterX3(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse = false);
void ReflectInnerX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse = false);
void ReflectOuterX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse = false);
void ReflectInnerX2(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse = false);
void ReflectOuterX2(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse = false);
void ReflectInnerX3(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse = false);
void ReflectOuterX3(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse = false);

} // namespace BoundaryFunction
} // namespace parthenon

#endif // BVALS_BOUNDARY_CONDITIONS_HPP_
