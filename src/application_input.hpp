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
#ifndef APPLICATION_INPUT_HPP_
#define APPLICATION_INPUT_HPP_

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "basic_types.hpp"
#include "bvals/boundary_conditions.hpp"
#include "defs.hpp"
#include "parthenon_arrays.hpp"

namespace parthenon {
class Mesh;
class MeshBlock;
class MeshBlockApplicationData;
class OutputParameters;
class Packages_t;
class ParameterInput;
template <typename T>
class MeshData;

class ApplicationInput {
 public:
  ApplicationInput();

  // ParthenonManager functions
  std::function<Packages_t(std::unique_ptr<ParameterInput> &)> ProcessPackages = nullptr;

  // Mesh functions
  std::function<void(Mesh *, ParameterInput *)> InitUserMeshData = nullptr;
  std::function<void(Mesh *, ParameterInput *, MeshData<Real> *)> MeshProblemGenerator =
      nullptr;
  std::function<void(Mesh *, ParameterInput *, MeshData<Real> *)> MeshPostInitialization =
      nullptr;

  std::function<void(Mesh *, ParameterInput *, SimTime &)> PreStepMeshUserWorkInLoop =
      nullptr;
  std::function<void(Mesh *, ParameterInput *, SimTime const &)>
      PostStepMeshUserWorkInLoop = nullptr;

  std::function<void(Mesh *, ParameterInput *, SimTime const &)>
      UserMeshWorkBeforeOutput = nullptr;
  std::function<void(Mesh *, ParameterInput *, SimTime const &, OutputParameters *)>
      UserWorkBeforeRestartOutput = nullptr;

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
  std::function<void(MeshBlock *, ParameterInput *)> PostInitialization = nullptr;
  std::function<void(MeshBlock *, ParameterInput *, const SimTime &)>
      MeshBlockUserWorkBeforeOutput = nullptr;

  // Physical boundaries
  void RegisterBoundaryCondition(BoundaryFace face, const std::string &name,
                                 BValFunc condition);
  void RegisterBoundaryCondition(BoundaryFace face, BValFunc condition) {
    RegisterBoundaryCondition(face, "user", condition);
  }
  void RegisterDefaultReflectingBoundaryConditions();

  void RegisterSwarmBoundaryCondition(BoundaryFace face, const std::string &name,
                                      SBValFunc condition);
  void RegisterSwarmBoundaryCondition(BoundaryFace face, SBValFunc condition) {
    RegisterSwarmBoundaryCondition(face, "user", condition);
  }

  // Getters
  BValFunc GetBoundaryCondition(BoundaryFace face, const std::string &name) const;
  SBValFunc GetSwarmBoundaryCondition(BoundaryFace face, const std::string &name) const;

 private:
  Dictionary<BValFunc> boundary_conditions_[BOUNDARY_NFACES];
  Dictionary<SBValFunc> swarm_boundary_conditions_[BOUNDARY_NFACES];
};

} // namespace parthenon

#endif // APPLICATION_INPUT_HPP_
