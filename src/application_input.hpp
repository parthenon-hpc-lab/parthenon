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

#include "bvals/boundary_conditions.hpp"
#include "defs.hpp"
#include "interface/state_descriptor.hpp"
#include "parameter_input.hpp"
#include "parthenon_arrays.hpp"

namespace parthenon {

struct ApplicationInput {
 public:
  // ParthenonManager functions
  std::function<Packages_t(std::unique_ptr<ParameterInput> &)> ProcessPackages = nullptr;

  // Mesh functions
  std::function<void(Mesh *, ParameterInput *)> InitUserMeshData = nullptr;
  std::function<void(Mesh *, ParameterInput *, MeshData<Real> *)> MeshProblemGenerator =
      nullptr;
  std::function<void(Mesh *, MeshData<Real> *)> MeshPostInitialization = nullptr;

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
  BValFunc boundary_conditions[BOUNDARY_NFACES] = {nullptr};
  SBValFunc swarm_boundary_conditions[BOUNDARY_NFACES] = {nullptr};

  // MeshBlock functions
  std::function<std::unique_ptr<MeshBlockApplicationData>(MeshBlock *, ParameterInput *)>
      InitApplicationMeshBlockData = nullptr;
  std::function<void(MeshBlock *, ParameterInput *)> InitMeshBlockUserData = nullptr;
  std::function<void(MeshBlock *, ParameterInput *)> ProblemGenerator = nullptr;
  std::function<void(MeshBlockData<Real> *)> PostInitialization = nullptr;
  std::function<void(MeshBlock *, ParameterInput *)> MeshBlockUserWorkBeforeOutput =
      nullptr;
};

} // namespace parthenon

#endif // APPLICATION_INPUT_HPP_
