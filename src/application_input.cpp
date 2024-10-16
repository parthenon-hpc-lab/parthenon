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

#include <string>

#include "application_input.hpp"
#include "basic_types.hpp"
#include "bvals/boundary_conditions.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {
ApplicationInput::ApplicationInput() {
  using namespace BoundaryFunction;
  const std::string OUTFLOW = "outflow";
  const std::string PERIODIC = "periodic";

  // Periodic handled separately for mesh
  RegisterBoundaryCondition(BoundaryFace::inner_x1, OUTFLOW, &OutflowInnerX1);
  RegisterBoundaryCondition(BoundaryFace::outer_x1, OUTFLOW, &OutflowOuterX1);
  RegisterBoundaryCondition(BoundaryFace::inner_x2, OUTFLOW, &OutflowInnerX2);
  RegisterBoundaryCondition(BoundaryFace::outer_x2, OUTFLOW, &OutflowOuterX2);
  RegisterBoundaryCondition(BoundaryFace::inner_x3, OUTFLOW, &OutflowInnerX3);
  RegisterBoundaryCondition(BoundaryFace::outer_x3, OUTFLOW, &OutflowOuterX3);

  // Periodic is explicit function for swarms
  RegisterSwarmBoundaryCondition(BoundaryFace::inner_x1, OUTFLOW, &SwarmOutflowInnerX1);
  RegisterSwarmBoundaryCondition(BoundaryFace::outer_x1, OUTFLOW, &SwarmOutflowOuterX1);
  RegisterSwarmBoundaryCondition(BoundaryFace::inner_x2, OUTFLOW, &SwarmOutflowInnerX2);
  RegisterSwarmBoundaryCondition(BoundaryFace::outer_x2, OUTFLOW, &SwarmOutflowOuterX2);
  RegisterSwarmBoundaryCondition(BoundaryFace::inner_x3, OUTFLOW, &SwarmOutflowInnerX3);
  RegisterSwarmBoundaryCondition(BoundaryFace::outer_x3, OUTFLOW, &SwarmOutflowOuterX3);
  RegisterSwarmBoundaryCondition(BoundaryFace::inner_x1, PERIODIC, &SwarmPeriodicInnerX1);
  RegisterSwarmBoundaryCondition(BoundaryFace::outer_x1, PERIODIC, &SwarmPeriodicOuterX1);
  RegisterSwarmBoundaryCondition(BoundaryFace::inner_x2, PERIODIC, &SwarmPeriodicInnerX2);
  RegisterSwarmBoundaryCondition(BoundaryFace::outer_x2, PERIODIC, &SwarmPeriodicOuterX2);
  RegisterSwarmBoundaryCondition(BoundaryFace::inner_x3, PERIODIC, &SwarmPeriodicInnerX3);
  RegisterSwarmBoundaryCondition(BoundaryFace::outer_x3, PERIODIC, &SwarmPeriodicOuterX3);
}

void ApplicationInput::RegisterDefaultReflectingBoundaryConditions() {
  using namespace BoundaryFunction;
  const std::string REFLECTING = "reflecting";
  RegisterBoundaryCondition(BoundaryFace::inner_x1, REFLECTING, &ReflectInnerX1);
  RegisterBoundaryCondition(BoundaryFace::outer_x1, REFLECTING, &ReflectOuterX1);
  RegisterBoundaryCondition(BoundaryFace::inner_x2, REFLECTING, &ReflectInnerX2);
  RegisterBoundaryCondition(BoundaryFace::outer_x2, REFLECTING, &ReflectOuterX2);
  RegisterBoundaryCondition(BoundaryFace::inner_x3, REFLECTING, &ReflectInnerX3);
  RegisterBoundaryCondition(BoundaryFace::outer_x3, REFLECTING, &ReflectOuterX3);
}
void ApplicationInput::RegisterBoundaryCondition(BoundaryFace face,
                                                 const std::string &name,
                                                 BValFunc condition) {
  if (boundary_conditions_[face].count(name) > 0) {
    PARTHENON_THROW("Boundary condition " + name + " at face " + std::to_string(face) +
                    "already registered.");
  }
  boundary_conditions_[face][name] = condition;
}
void ApplicationInput::RegisterSwarmBoundaryCondition(BoundaryFace face,
                                                      const std::string &name,
                                                      SBValFunc condition) {
  if (swarm_boundary_conditions_[face].count(name) > 0) {
    PARTHENON_THROW("Swarm boundary condition " + name + " at face " +
                    std::to_string(face) + "already registered.");
  }
  swarm_boundary_conditions_[face][name] = condition;
}

BValFunc ApplicationInput::GetBoundaryCondition(BoundaryFace face,
                                                const std::string &name) const {
  if (boundary_conditions_[face].count(name) == 0) {
    std::stringstream msg;
    msg << "Boundary condition " << name << " at face " << face << "not registered!\n"
        << "Available conditions for this face are:\n";
    for (const auto &[name, func] : boundary_conditions_[face]) {
      msg << name << "\n";
    }
    PARTHENON_THROW(msg);
  }
  return boundary_conditions_[face].at(name);
}
SBValFunc ApplicationInput::GetSwarmBoundaryCondition(BoundaryFace face,
                                                      const std::string &name) const {
  if (swarm_boundary_conditions_[face].count(name) == 0) {
    std::stringstream msg;
    msg << "Swarm boundary condition " << name << " at face " << face
        << "not registered!\n"
        << "Available conditions for this face are:\n";
    for (const auto &[name, func] : swarm_boundary_conditions_[face]) {
      msg << name << "\n";
    }
    PARTHENON_THROW(msg);
  }
  return swarm_boundary_conditions_[face].at(name);
}

} // namespace parthenon
