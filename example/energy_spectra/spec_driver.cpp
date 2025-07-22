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

// Standard Includes
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// Parthenon Includes
#include <parthenon/driver.hpp>

// Local Includes
#include "calc_spec.hpp"
#include "spec_driver.hpp"

// Preludes
using namespace parthenon::driver::prelude;

using pi::PiDriver;

Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin);

int main(int argc, char *argv[]) {
  ParthenonManager pman;

  pman.app_input->ProcessPackages = ProcessPackages;

  // This is called on each mesh block whenever the mesh changes.
  pman.app_input->ProblemGenerator = calculate_pi::ProblemGenerator;

  auto manager_status = pman.ParthenonInitEnv(argc, argv);
  if (manager_status == ParthenonStatus::complete) {
    pman.ParthenonFinalize();
    return 0;
  }
  if (manager_status == ParthenonStatus::error) {
    pman.ParthenonFinalize();
    return 1;
  }

  // This needs to be scoped so that the driver object is destructed before Finalize
  pman.ParthenonInitPackagesAndMesh();
  {
    PiDriver driver(pman.pinput.get(), pman.app_input.get(), pman.pmesh.get());

    auto driver_status = driver.Execute();
  }
  // call MPI_Finalize if necessary
  pman.ParthenonFinalize();

  return 0;
}

Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;
  // only have one package for this app, but will typically have more things added to
  packages.Add(calculate_pi::Initialize(pin.get()));
  return packages;
}

// this should set up initial conditions of independent variables on the block
// this app only has one variable of derived type, so nothing to do here.
// in this case, just use the weak version
// void MeshBlock::ProblemGenerator(ParameterInput *pin) {
//  // nothing to do here for this app
//}

parthenon::DriverStatus PiDriver::Execute() {
  // this is where the main work is orchestrated
  // No evolution in this driver.  Just calculates something once.
  // For evolution, look at the EvolutionDriver
  PreExecute();

  // pouts->MakeOutputs(pmesh, pinput);

  // The tasks compute pi and store it in the param "pi_val"
  ConstructAndExecuteTaskLists<>(this);

  // retrieve "pi_val" and post execute.
  // auto &pi_val = pmesh->packages.Get("calculate_pi")->Param<Real>("pi_val");
  // pmesh->mbcnt = pmesh->nbtotal; // this is how many blocks were processed
  PiPostExecute(42);
  return DriverStatus::complete;
}

void PiDriver::PiPostExecute(Real pi_val) {
  if (my_rank == 0) {
    std::cout << "We're done here!\n";
  }
  Driver::PostExecute(DriverStatus::complete);
}

template <typename T>
TaskCollection PiDriver::MakeTaskCollection(T &blocks) {
  using calculate_pi::CalcSpec;
  TaskCollection tc;

  auto partitions = pmesh->GetDefaultBlockPartitions();
  const int num_partitions = partitions.size();
  ParArrayHost<Real> areas("areas", num_partitions);
  for (int k = 0; k < 3; k++) {
    TaskRegion &async_region = tc.AddRegion(num_partitions);
    {
      // asynchronous region where area is computed per partition
      for (int i = 0; i < num_partitions; i++) {
        TaskID none(0);
        auto &md = pmesh->mesh_data.Add("base", partitions[i]);
        auto get_area = async_region[i].AddTask(none, CalcSpec, md, areas, k);
      }
    }
  }

  return tc;
}
