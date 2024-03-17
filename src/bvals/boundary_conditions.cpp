//========================================================================================
// (C) (or copyright) 2020-2023. Triad National Security, LLC. All rights reserved.
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

#include <memory>
#include <vector>

#include "bvals/boundary_conditions.hpp"
#include "bvals/boundary_conditions_generic.hpp"
#include "bvals/neighbor_block.hpp"
#include "defs.hpp"
#include "interface/meshblock_data.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock.hpp"

namespace parthenon {

namespace boundary_cond_impl {
bool DoPhysicalBoundary_(const BoundaryFlag flag, const BoundaryFace face,
                         const int ndim);
} // namespace boundary_cond_impl

TaskStatus ApplyBoundaryConditionsOnCoarseOrFine(std::shared_ptr<MeshBlockData<Real>> &rc,
                                                 bool coarse) {
  PARTHENON_INSTRUMENT
  using namespace boundary_cond_impl;
  MeshBlock *pmb = rc->GetBlockPointer();
  Mesh *pmesh = pmb->pmy_mesh;
  const int ndim = pmesh->ndim;

  for (int i = 0; i < BOUNDARY_NFACES; i++) {
    if (DoPhysicalBoundary_(pmb->boundary_flag[i], static_cast<BoundaryFace>(i), ndim)) {
      PARTHENON_DEBUG_REQUIRE(pmesh->MeshBndryFnctn[i] != nullptr,
                              "boundary function must not be null");
      pmesh->MeshBndryFnctn[i](rc, coarse);
      for (auto &bnd_func : pmesh->UserBoundaryFunctions[i]) {
        bnd_func(rc, coarse);
      }
    }
  }

  return TaskStatus::complete;
}

TaskStatus ApplyBoundaryConditionsMD(std::shared_ptr<MeshData<Real>> &pmd) {
  for (int b = 0; b < pmd->NumBlocks(); ++b)
    ApplyBoundaryConditions(pmd->GetBlockData(b));
  return TaskStatus::complete;
}

TaskStatus ApplyBoundaryConditionsOnCoarseOrFineMD(std::shared_ptr<MeshData<Real>> &pmd,
                                                   bool coarse) {
  for (int b = 0; b < pmd->NumBlocks(); ++b)
    ApplyBoundaryConditionsOnCoarseOrFine(pmd->GetBlockData(b), coarse);
  return TaskStatus::complete;
}

namespace BoundaryFunction {

void OutflowInnerX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  GenericBC<X1DIR, BCSide::Inner, BCType::Outflow, variable_names::any>(rc, coarse);
}

void OutflowOuterX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  GenericBC<X1DIR, BCSide::Outer, BCType::Outflow, variable_names::any>(rc, coarse);
}

void OutflowInnerX2(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  GenericBC<X2DIR, BCSide::Inner, BCType::Outflow, variable_names::any>(rc, coarse);
}

void OutflowOuterX2(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  GenericBC<X2DIR, BCSide::Outer, BCType::Outflow, variable_names::any>(rc, coarse);
}

void OutflowInnerX3(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  GenericBC<X3DIR, BCSide::Inner, BCType::Outflow, variable_names::any>(rc, coarse);
}

void OutflowOuterX3(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  GenericBC<X3DIR, BCSide::Outer, BCType::Outflow, variable_names::any>(rc, coarse);
}

void ReflectInnerX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  GenericBC<X1DIR, BCSide::Inner, BCType::Reflect, variable_names::any>(rc, coarse);
}

void ReflectOuterX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  GenericBC<X1DIR, BCSide::Outer, BCType::Reflect, variable_names::any>(rc, coarse);
}

void ReflectInnerX2(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  GenericBC<X2DIR, BCSide::Inner, BCType::Reflect, variable_names::any>(rc, coarse);
}

void ReflectOuterX2(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  GenericBC<X2DIR, BCSide::Outer, BCType::Reflect, variable_names::any>(rc, coarse);
}

void ReflectInnerX3(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  GenericBC<X3DIR, BCSide::Inner, BCType::Reflect, variable_names::any>(rc, coarse);
}

void ReflectOuterX3(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  GenericBC<X3DIR, BCSide::Outer, BCType::Reflect, variable_names::any>(rc, coarse);
}

void SwarmOutflowInnerX1(std::shared_ptr<SwarmContainer> &sc) {
  GenericSwarmBC<X1DIR, BCSide::Inner, BCType::Outflow>(sc);
}

void SwarmOutflowOuterX1(std::shared_ptr<SwarmContainer> &sc) {
  GenericSwarmBC<X1DIR, BCSide::Outer, BCType::Outflow>(sc);
}

void SwarmOutflowInnerX2(std::shared_ptr<SwarmContainer> &sc) {
  GenericSwarmBC<X2DIR, BCSide::Inner, BCType::Outflow>(sc);
}

void SwarmOutflowOuterX2(std::shared_ptr<SwarmContainer> &sc) {
  GenericSwarmBC<X2DIR, BCSide::Outer, BCType::Outflow>(sc);
}

void SwarmOutflowInnerX3(std::shared_ptr<SwarmContainer> &sc) {
  GenericSwarmBC<X3DIR, BCSide::Inner, BCType::Outflow>(sc);
}

void SwarmOutflowOuterX3(std::shared_ptr<SwarmContainer> &sc) {
  GenericSwarmBC<X3DIR, BCSide::Outer, BCType::Outflow>(sc);
}

void SwarmPeriodicInnerX1(std::shared_ptr<SwarmContainer> &sc) {
  GenericSwarmBC<X1DIR, BCSide::Inner, BCType::Periodic>(sc);
}

void SwarmPeriodicOuterX1(std::shared_ptr<SwarmContainer> &sc) {
  GenericSwarmBC<X1DIR, BCSide::Outer, BCType::Periodic>(sc);
}

void SwarmPeriodicInnerX2(std::shared_ptr<SwarmContainer> &sc) {
  GenericSwarmBC<X2DIR, BCSide::Inner, BCType::Periodic>(sc);
}

void SwarmPeriodicOuterX2(std::shared_ptr<SwarmContainer> &sc) {
  GenericSwarmBC<X2DIR, BCSide::Outer, BCType::Periodic>(sc);
}

void SwarmPeriodicInnerX3(std::shared_ptr<SwarmContainer> &sc) {
  GenericSwarmBC<X3DIR, BCSide::Inner, BCType::Periodic>(sc);
}

void SwarmPeriodicOuterX3(std::shared_ptr<SwarmContainer> &sc) {
  GenericSwarmBC<X3DIR, BCSide::Outer, BCType::Outflow>(sc);
}

} // namespace BoundaryFunction

namespace boundary_cond_impl {
bool DoPhysicalBoundary_(const BoundaryFlag flag, const BoundaryFace face,
                         const int ndim) {
  if (flag == BoundaryFlag::block) return false;
  if (flag == BoundaryFlag::undef) return false;
  if (flag == BoundaryFlag::periodic) return false;

  if (ndim < 3 && (face == BoundaryFace::inner_x3 || face == BoundaryFace::outer_x3)) {
    return false;
  }
  if (ndim < 2 && (face == BoundaryFace::inner_x2 || face == BoundaryFace::outer_x2)) {
    return false;
  } // ndim always at least 1

  return true; // reflect, outflow, user, dims correct
}

} // namespace boundary_cond_impl

} // namespace parthenon
