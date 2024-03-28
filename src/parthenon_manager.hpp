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

#ifndef PARTHENON_MANAGER_HPP_
#define PARTHENON_MANAGER_HPP_

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
#include "mesh/mesh.hpp"
#include "outputs/restart.hpp"
#include "parameter_input.hpp"
#include "utils/utils.hpp"

namespace parthenon {

enum class ParthenonStatus { ok, complete, error };

class ParthenonManager {
 public:
  ParthenonManager() { app_input.reset(new ApplicationInput()); }
  ParthenonStatus ParthenonInitEnv(int argc, char *argv[]);
  void ParthenonInitPackagesAndMesh();
  ParthenonStatus ParthenonFinalize();

  bool IsRestart() { return (arg.restart_filename == nullptr ? false : true); }
  static Packages_t ProcessPackagesDefault(std::unique_ptr<ParameterInput> &pin);
  void RestartPackages(Mesh &rm, RestartReader &resfile);

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

  template <typename T>
  void ReadSwarmVars_(const SP_Swarm &pswarm, const BlockList_t &block_list,
                      const std::size_t count_on_rank, const std::size_t offset) {
    const std::string &swarmname = pswarm->label();
    std::vector<T> dataVec;
    for (const auto &var : pswarm->GetVariableVector<T>()) {
      const std::string &varname = var->label();
      std::cout << "SwarmVar: " << varname << std::endl;
      const auto &m = var->metadata();
      auto arrdims = m.GetArrayDims(pswarm->GetBlockPointer(), false);

      try {
        restartReader->ReadSwarmVar(swarmname, varname, count_on_rank, offset, m,
                                    dataVec);
      } catch (std::exception &ex) {
        std::cout << StringPrintf("[%d] WARNING: Failed to read Swarm %s Variable %s "
                                  "from restart file:\n%s",
                                  Globals::my_rank, swarmname.c_str(), varname.c_str(),
                                  ex.what())
                  << std::endl;
        continue;
      }

      // Only safe because swarm starts completely defragged.
      // Note ordering here: block is second-inner-most loop.
      // If hdf5 output format changes, this needs to change too.
      std::size_t ivec = 0;
      for (int n6 = 0; n6 < arrdims[5]; ++n6) {
        for (int n5 = 0; n5 < arrdims[4]; ++n5) {
          for (int n4 = 0; n4 < arrdims[3]; ++n4) {
            for (int n3 = 0; n3 < arrdims[2]; ++n3) {
              for (int n2 = 0; n2 < arrdims[1]; ++n2) {
                for (auto &pmb : block_list) {
                  // 1 deep copy per tensor component per swarmvar per
                  // block, unfortunately. But only at initialization.
                  auto swarm_container = pmb->meshblock_data.Get()->swarm_data.Get();
                  auto pswarm_blk = swarm_container->Get(swarmname);
                  auto v = Kokkos::subview(pswarm_blk->Get<T>(varname).data, n6, n5, n4,
                                           n3, n2, Kokkos::ALL());
                  auto v_h = Kokkos::create_mirror_view(v);
                  for (int n1 = 0; n1 < pswarm_blk->GetNumActive(); ++n1) {
                    v_h(n1) = dataVec[ivec++];
                  }
                  Kokkos::deep_copy(v, v_h);
                }
              }
            }
          }
        }
      }
    }
  }
};

} // namespace parthenon

#endif // PARTHENON_MANAGER_HPP_
