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

// Standard Includes
#include <fstream>
#include <memory>
#include <string>
#include <vector>

// Parthenon Includes
#include <parthenon/driver.hpp>

// Local Includes
#include "calculate_pi.hpp"
#include "pi_driver.hpp"

// Preludes
using namespace parthenon::driver::prelude;

using pi::PiDriver;

Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin);

int main(int argc, char *argv[]) {
  ParthenonManager pman;

  pman.app_input->ProcessPackages = ProcessPackages;

  auto manager_status = pman.ParthenonInit(argc, argv);
  if (manager_status == ParthenonStatus::complete) {
    pman.ParthenonFinalize();
    return 0;
  }
  if (manager_status == ParthenonStatus::error) {
    pman.ParthenonFinalize();
    return 1;
  }

  PiDriver driver(pman.pinput.get(), pman.app_input.get(), pman.pmesh.get());

  auto driver_status = driver.Execute();

  // call MPI_Finalize if necessary
  pman.ParthenonFinalize();

  return 0;
}

// can be used to set global properties that all meshblocks want to know about
// no need in this app so use the weak version that ships with parthenon
// Properties_t ParthenonManager::ProcessProperties(std::unique_ptr<ParameterInput>& pin)
// {
//  Properties_t props;
//  return props;
//}

Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;
  // only have one package for this app, but will typically have more things added to
  packages["calculate_pi"] = calculate_pi::Initialize(pin.get());
  return packages;
}

// this should set up initial conditions of independent variables on the block
// this app only has one variable of derived type, so nothing to do here.
// in this case, just use the weak version
// void MeshBlock::ProblemGenerator(ParameterInput *pin) {
//  // nothing to do here for this app
//}

// applications can register functions to fill shared derived quantities
// before and/or after all the package FillDerived call backs
// in this case, just use the weak version that sets these to nullptr
// void ParthenonManager::SetFillDerivedFunctions() {
//  FillDerivedVariables::SetFillDerivedFunctions(nullptr,nullptr);
//}

parthenon::DriverStatus PiDriver::Execute() {
  // this is where the main work is orchestrated
  // No evolution in this driver.  Just calculates something once.
  // For evolution, look at the EvolutionDriver
  PreExecute();

  pouts->MakeOutputs(pmesh, pinput);

  // The tasks compute pi and store it in the param "pi_val"
  ConstructAndExecuteTaskLists<>(this);

  // retrieve "pi_val" and post execute.
  auto &pi_val = pmesh->packages["calculate_pi"]->Param<Real>("pi_val");
  pmesh->mbcnt = pmesh->nbtotal; // this is how many blocks were processed
  PostExecute(pi_val);
  return DriverStatus::complete;
}

void PiDriver::PostExecute(Real pi_val) {
  if (my_rank == 0) {
    std::cout << std::endl
              << std::endl
              << "PI = " << pi_val << "    rel error = " << (pi_val - M_PI) / M_PI
              << std::endl
              << std::endl;

    std::fstream fs;
    fs.open("summary.txt", std::fstream::out);
    fs << "PI = " << pi_val << std::endl;
    fs << "rel error = " << (pi_val - M_PI) / M_PI << std::endl;
    fs.close();
  }
  Driver::PostExecute();
}

template <typename T>
TaskCollection PiDriver::MakeTasks(T &blocks) {
  using calculate_pi::AccumulateAreas;
  using calculate_pi::ComputeArea;
  TaskCollection tc;

  // 1 means pack is 1 meshblock, <1 means use entire mesh
  int pack_size = pinput->GetOrAddInteger("Pi", "pack_size", 1);
  if (pack_size < 1) pack_size = blocks.size();

  std::vector<BlockList_t> partitions;
  partition::ToSizeN(blocks, pack_size, partitions);
  ParArrayHost<Real> areas("areas", partitions.size());

  TaskRegion &async_region = tc.AddRegion(partitions.size());
  {
    // asynchronous region where area is computed per mesh pack
    for (int i = 0; i < partitions.size(); i++) {
      TaskID none(0);
      auto pack = PackVariablesOnMesh(partitions[i], "base",
                                      std::vector<std::string>{"in_or_out"});
      auto get_area = async_region[i].AddTask(ComputeArea, none, pack, areas, i);
    }
  }
  TaskRegion &sync_region = tc.AddRegion(1);
  {
    TaskID none(0);
    auto accumulate_areas =
        sync_region[0].AddTask(AccumulateAreas, none, areas, pmesh->packages);
  }

  return tc;
}
