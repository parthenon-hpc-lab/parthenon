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

  auto manager_status = pman.ParthenonInitEnv(argc, argv);
  if (manager_status == ParthenonStatus::complete) {
    pman.ParthenonFinalize();
    return 0;
  }
  if (manager_status == ParthenonStatus::error) {
    pman.ParthenonFinalize();
    return 1;
  }

  if (pman.pinput->GetString("calc_spec", "input_file_format") == "athenak_multifile") {
    pman.app_input->ProblemGenerator = calculate_pi::ProblemGenerator;
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
  if (my_rank == 0) {
    auto out_num = pinput->GetInteger("calc_spec", "output_number");
    out_stream.open("spec_" + std::to_string(out_num) + ".bp", adios2::fstream::out,
                    MPI_COMM_SELF);
  }

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
    out_stream.close();
    std::cout << "We're done here!\n";
  }
  Driver::PostExecute(DriverStatus::complete);
}

template <typename T>
TaskCollection PiDriver::MakeTaskCollection(T &blocks) {
  using calculate_pi::CalcSpec;
  TaskCollection tc;

  PARTHENON_REQUIRE_THROWS(pmesh->DefaultNumPartitions() == 1,
                           "Only pack_size=-1 currently supported for heffte.")
  auto partitions = pmesh->GetDefaultBlockPartitions();
  const auto num_partitions = partitions.size();
  auto &md = pmesh->mesh_data.Add("base", partitions[0]);
  TaskRegion &region = tc.AddRegion(num_partitions);

  TaskID none(0);
  auto task_calc_stats =
      region[0].AddTask(none, calculate_pi::CalcStats, md, &out_stream);

  auto task_calc_spec = none;
  for (int spec_type = 0; spec_type < 3; spec_type++) {
    task_calc_spec =
        region[0].AddTask(task_calc_spec, CalcSpec, md, spec_type, &out_stream);
  }
  return tc;
}
