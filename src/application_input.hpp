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
#ifndef APPLICATION_INPUT_HPP_
#define APPLICATION_INPUT_HPP_

#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "bvals/boundary_conditions.hpp"
#include "defs.hpp"
#include "interface/state_descriptor.hpp"
#include "interface/swarm_boundaries.hpp"
#include "kokkos_abstraction.hpp"
#include "parameter_input.hpp"
#include "parthenon_arrays.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

class ApplicationInput {
 public:
  ApplicationInput() {
    RegisterBoundaryCondition(BoundaryFace::inner_x1, "outflow",
                              &BoundaryFunction::OutflowInnerX1);
    RegisterBoundaryCondition(BoundaryFace::outer_x1, "outflow",
                              &BoundaryFunction::OutflowOuterX1);
    RegisterBoundaryCondition(BoundaryFace::inner_x2, "outflow",
                              &BoundaryFunction::OutflowInnerX2);
    RegisterBoundaryCondition(BoundaryFace::outer_x2, "outflow",
                              &BoundaryFunction::OutflowOuterX2);
    RegisterBoundaryCondition(BoundaryFace::inner_x3, "outflow",
                              &BoundaryFunction::OutflowInnerX3);
    RegisterBoundaryCondition(BoundaryFace::outer_x3, "outflow",
                              &BoundaryFunction::OutflowOuterX3);
    RegisterBoundaryCondition(BoundaryFace::inner_x1, "reflecting",
                              &BoundaryFunction::OutflowInnerX1);
    RegisterBoundaryCondition(BoundaryFace::outer_x1, "reflecting",
                              &BoundaryFunction::ReflectOuterX1);
    RegisterBoundaryCondition(BoundaryFace::inner_x2, "reflecting",
                              &BoundaryFunction::ReflectInnerX2);
    RegisterBoundaryCondition(BoundaryFace::outer_x2, "reflecting",
                              &BoundaryFunction::ReflectOuterX2);
    RegisterBoundaryCondition(BoundaryFace::inner_x3, "reflecting",
                              &BoundaryFunction::ReflectInnerX3);
    RegisterBoundaryCondition(BoundaryFace::outer_x3, "reflecting",
                              &BoundaryFunction::ReflectOuterX3);

    // Currently outflow and periodic are the only particle BCs available
    RegisterSwarmBoundaryCondition(BoundaryFace::inner_x1, "outflow",
                                   &DeviceAllocate<ParticleBoundIX1Outflow>);
    RegisterSwarmBoundaryCondition(BoundaryFace::outer_x1, "outflow",
                                   &DeviceAllocate<ParticleBoundOX1Outflow>);
    RegisterSwarmBoundaryCondition(BoundaryFace::inner_x2, "outflow",
                                   &DeviceAllocate<ParticleBoundIX2Outflow>);
    RegisterSwarmBoundaryCondition(BoundaryFace::outer_x2, "outflow",
                                   &DeviceAllocate<ParticleBoundOX2Outflow>);
    RegisterSwarmBoundaryCondition(BoundaryFace::inner_x3, "outflow",
                                   &DeviceAllocate<ParticleBoundIX3Outflow>);
    RegisterSwarmBoundaryCondition(BoundaryFace::outer_x3, "outflow",
                                   &DeviceAllocate<ParticleBoundOX3Outflow>);
    RegisterSwarmBoundaryCondition(BoundaryFace::inner_x1, "periodic",
                                   &DeviceAllocate<ParticleBoundIX1Periodic>);
    RegisterSwarmBoundaryCondition(BoundaryFace::outer_x1, "periodic",
                                   &DeviceAllocate<ParticleBoundOX1Periodic>);
    RegisterSwarmBoundaryCondition(BoundaryFace::inner_x2, "periodic",
                                   &DeviceAllocate<ParticleBoundIX2Periodic>);
    RegisterSwarmBoundaryCondition(BoundaryFace::outer_x2, "periodic",
                                   &DeviceAllocate<ParticleBoundOX2Periodic>);
    RegisterSwarmBoundaryCondition(BoundaryFace::inner_x3, "periodic",
                                   &DeviceAllocate<ParticleBoundIX3Periodic>);
    RegisterSwarmBoundaryCondition(BoundaryFace::outer_x3, "periodic",
                                   &DeviceAllocate<ParticleBoundOX3Periodic>);
  }

  // ParthenonManager functions
  std::function<Packages_t(std::unique_ptr<ParameterInput> &)> ProcessPackages = nullptr;

  // Mesh functions
  std::function<void(Mesh *, ParameterInput *)> InitUserMeshData = nullptr;
  std::function<void(Mesh *, ParameterInput *, MeshData<Real> *)> MeshProblemGenerator =
      nullptr;

  std::function<void(Mesh *, ParameterInput *, SimTime &)> PreStepMeshUserWorkInLoop =
      nullptr;
  std::function<void(Mesh *, ParameterInput *, SimTime const &)>
      PostStepMeshUserWorkInLoop = nullptr;

  std::function<void(Mesh *, ParameterInput *, SimTime const &)>
      PreStepDiagnosticsInLoop = nullptr;
  std::function<void(Mesh *, ParameterInput *, SimTime const &)>
      PostStepDiagnosticsInLoop = nullptr;

  std::function<void(Mesh *, ParameterInput *, SimTime &)> UserWorkAfterLoop = nullptr;
  std::function<void(Mesh *, ParameterInput *, SimTime &)> UserWorkBeforeLoop = nullptr;

  // MeshBlock functions
  std::function<std::unique_ptr<MeshBlockApplicationData>(MeshBlock *, ParameterInput *)>
      InitApplicationMeshBlockData = nullptr;
  std::function<void(MeshBlock *, ParameterInput *)> InitMeshBlockUserData = nullptr;
  std::function<void(MeshBlock *, ParameterInput *)> ProblemGenerator = nullptr;
  std::function<void(MeshBlock *, ParameterInput *)> MeshBlockUserWorkBeforeOutput =
      nullptr;

  // Boundary conditions
  void RegisterBoundaryCondition(BoundaryFace face, const std::string &name,
                                 BValFunc condition) {
    if (boundary_conditions_[face].count(name) > 0) {

      PARTHENON_THROW("Boundary condition " + name + " at face " + std::to_string(face) +
                      "already registered.");
    }
    boundary_conditions_[face][name] = condition;
  }
  void RegisterSwarmBoundaryCondition(BoundaryFace face, const std::string &name,
                                      SBValFunc condition) {
    if (swarm_boundary_conditions_[face].count(name) > 0) {
      PARTHENON_THROW("Swarm boundary condition " + name + " at face " +
                      std::to_string(face) + "already registered.");
    }
    swarm_boundary_conditions_[face][name] = condition;
  }
  template <typename T>
  void RegisterSwarmBoundaryCondition(BoundaryFace face, const std::string &name) {
    RegisterSwarmBoundaryCondition(face, name, &DeviceAllocate<T>);
  }
  void RegisterBoundaryCondition(BoundaryFace face, BValFunc condition) {
    RegisterBoundaryCondition(face, "user", condition);
  }
  template <typename T>
  void RegisterSwarmBoundaryCondition(BoundaryFace face) {
    RegisterSwarmBoundaryCondition(face, "user", &DeviceAllocate<T>);
  }
  void RegisterSwarmBoundaryCondition(BoundaryFace face, SBValFunc condition) {
    RegisterSwarmBoundaryCondition(face, "user", condition);
  }
  // Getters
  BValFunc GetBoundaryCondition(BoundaryFace face, const std::string &name) const {
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
  SBValFunc GetSwarmBoundaryCondition(BoundaryFace face, const std::string &name) const {
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

 private:
  Dictionary<BValFunc> boundary_conditions_[BOUNDARY_NFACES];
  Dictionary<SBValFunc> swarm_boundary_conditions_[BOUNDARY_NFACES];
};

} // namespace parthenon
#endif // APPLICATION_INPUT_HPP_
