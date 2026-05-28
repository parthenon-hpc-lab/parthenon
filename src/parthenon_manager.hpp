//========================================================================================
// (C) (or copyright) 2020-2026. Triad National Security, LLC. All rights reserved.
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

// This file was made in part with generative AI.

#ifndef PARTHENON_MANAGER_HPP_
#define PARTHENON_MANAGER_HPP_

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "application_input.hpp"
#include "argument_parser.hpp"
#include "basic_types.hpp"
#include "driver/driver.hpp"
#include "interface/state_descriptor.hpp"
#include "interface/swarm.hpp"
#include "mesh/domain.hpp"
#include "mesh/forest/forest_topology.hpp"
#include "mesh/mesh.hpp"
#include "outputs/restart.hpp"
#include "outputs/restart_hdf5.hpp"
#include "parameter_input.hpp"
#include "utils/error_checking.hpp"
#include "utils/utils.hpp"

namespace parthenon {

enum class ParthenonStatus { ok, complete, error };

class ParthenonManager {
 public:
  ParthenonManager() { app_input.reset(new ApplicationInput()); }
  ParthenonStatus ParthenonInitEnv(int argc, char *argv[]);
  void
  ParthenonInitPackagesAndMesh(std::optional<forest::ForestDefinition> forest_def = {});
  ParthenonStatus ParthenonFinalize();

  static Packages_t ProcessPackagesDefault(std::unique_ptr<ParameterInput> &pin);
  void RestartPackages(Mesh &rm, RestartReaderHDF5 &resfile);

  std::function<Packages_t(std::unique_ptr<ParameterInput> &)> ProcessPackages =
      ProcessPackagesDefault;

  // member data
  std::unique_ptr<ParameterInput> pinput;
  std::unique_ptr<Mesh> pmesh;
  std::unique_ptr<RestartReader> restartReader;
  std::unique_ptr<ApplicationInput> app_input;

 private:
  ArgParse arg;
  bool called_init_env_ = false;
  bool called_init_packages_and_mesh_ = false;
};

} // namespace parthenon

#endif // PARTHENON_MANAGER_HPP_
