//========================================================================================
// (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
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
#include "interface/properties_interface.hpp"
#include "interface/state_descriptor.hpp"
#include "parameter_input.hpp"
#include "parthenon_arrays.hpp"

namespace parthenon {

struct ApplicationInput {
 public:
  // ParthenonManager functions
  std::function<Properties_t(std::unique_ptr<ParameterInput> &)> ProcessProperties =
      nullptr;
  std::function<Packages_t(std::unique_ptr<ParameterInput> &)> ProcessPackages = nullptr;

  // Mesh functions
  std::function<void(ParameterInput *)> InitUserMeshData = nullptr;

  std::function<void(Mesh *, ParameterInput *, SimTime const &)>
      PreStepMeshUserWorkInLoop = nullptr;
  std::function<void(Mesh *, ParameterInput *, SimTime const &)>
      PostStepMeshUserWorkInLoop = nullptr;

  std::function<void(Mesh *, ParameterInput *, SimTime const &)>
      PreStepDiagnosticsInLoop = nullptr;
  std::function<void(Mesh *, ParameterInput *, SimTime const &)>
      PostStepDiagnosticsInLoop = nullptr;

  std::function<void(Mesh *, ParameterInput *, SimTime &)> UserWorkAfterLoop = nullptr;
  BValFunc boundary_conditions[BOUNDARY_NFACES] = {nullptr, nullptr, nullptr,
                                                   nullptr, nullptr, nullptr};

  // MeshBlock functions
  std::function<std::unique_ptr<MeshBlockApplicationData>(MeshBlock *, ParameterInput *)>
      InitApplicationMeshBlockData = nullptr;
  std::function<void(ParameterInput *)> InitUserMeshBlockData = nullptr;
  std::function<void(MeshBlock *, ParameterInput *)> ProblemGenerator = nullptr;
  std::function<void()> MeshBlockUserWorkInLoop = nullptr;
  std::function<void(ParameterInput *)> UserWorkBeforeOutput = nullptr;
};

} // namespace parthenon

#endif // APPLICATION_INPUT_HPP_
