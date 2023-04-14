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

#ifndef PARTHENON_MANAGER_HPP_
#define PARTHENON_MANAGER_HPP_

#include <memory>

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

namespace parthenon {

enum class ParthenonStatus { ok, complete, error };

class ParthenonManager {
 public:
  ParthenonManager() { app_input.reset(new ApplicationInput()); }
  ParthenonStatus ParthenonInit(int argc, char *argv[]);
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

  template <typename T>
  void ReadSwarmVars_(const SP_Swarm &pswarm, const BlockList_t &block_list,
                      const std::size_t count_on_rank, const std::size_t offset) {
    const std::string &swarmname = pswarm->label();
    std::vector<T> dataVec;
    for (const auto &var : pswarm->GetVariableVector<T>()) {
      const std::string &varname = var->label();
      const auto &m = var->metadata();

      restartReader->ReadSwarmVar(swarmname, varname, count_on_rank, offset, m, dataVec);

      std::size_t vidx = 0;
      for (auto &pmb : block_list) {
	auto swarm_container = pmb->swarm_data.Get();
	auto pswarm_blk = swarm_container->Get(swarmname);
        auto &v = pswarm_blk->Get<T>(varname);
	auto v_h = v.data.GetHostMirror();
	for (int n6 = 0; n6 < v_h.GetDim(6); ++n6) {
	  for (int n5 = 0; n5 < v_h.GetDim(5); ++n5) {
	    for (int n4 = 0; n4 < v_h.GetDim(4); ++n4) {
	      for (int n3 = 0; n3 < v_h.GetDim(3); ++n3) {
		for (int n2 = 0; n2 < v_h.GetDim(2); ++n2) {
		  // only safe because swarm starts completely defragged
		  for (int n1 = 0; n1 < pswarm_blk->GetNumActive(); ++n1) {
		    v_h(n6, n5, n4, n3, n2, n1) = dataVec[vidx++];
		  }
		}
	      }
	    }
	  }
	}
	v.data.DeepCopy(v_h);
      }
    }
  }
};

} // namespace parthenon

#endif // PARTHENON_MANAGER_HPP_
