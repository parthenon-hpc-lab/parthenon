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

// Parthenon Includes
#include <parthenon/package.hpp>

// Local Includes
#include "calculate_pi.hpp"
#include "pi_driver.hpp"

// Preludes
using namespace parthenon::package::prelude;

using parthenon::BlockTaskFunc;

using pi::PiDriver;

int main(int argc, char *argv[]) {
  ParthenonManager pman;

  auto manager_status = pman.ParthenonInit(argc, argv);
  if (manager_status == ParthenonStatus::complete) {
    pman.ParthenonFinalize();
    return 0;
  }
  if (manager_status == ParthenonStatus::error) {
    pman.ParthenonFinalize();
    return 1;
  }

  PiDriver driver(pman.pinput.get(), pman.pmesh.get());

  auto driver_status = driver.Execute();

  // call MPI_Finalize if necessary
  pman.ParthenonFinalize();

  return 0;
}

parthenon::DriverStatus PiDriver::Execute() {
  // this is where the main work is orchestrated
  // No evolution in this driver.  Just calculates something once.
  // For evolution, look at the EvolutionDriver
  PreExecute();

  pouts->MakeOutputs(pmesh, pinput);

  ConstructAndExecuteBlockTasks<>(this);

  // All the blocks are done, now do a global reduce and spit out the answer
  // first sum over blocks on this rank
  Real area = 0.0;
  for (auto &mb : pmesh->pblock) {
    Container<Real> &rc = mb.real_containers.Get();
    ParArrayND<Real> v = rc.Get("in_or_out").data;

    // extract area from device memory
    Real block_area;
    Kokkos::deep_copy(mb.exec_space, block_area, v.Get(0, 0, 0, 0, 0, 0));
    mb.exec_space.fence(); // as the deep copy may be async

    const auto &radius = mb.packages["calculate_pi"]->Param<Real>("radius");
    // area must be reduced by r^2 to get the block's contribution to PI
    block_area /= (radius * radius);

    area += block_area;
  }
#ifdef MPI_PARALLEL
  Real pi_val;
  MPI_Reduce(&area, &pi_val, 1, MPI_PARTHENON_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
#else
  Real pi_val = area;
#endif
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

parthenon::TaskList PiDriver::MakeTaskList(MeshBlock *pmb) {
  // make a task list for this mesh block
  using calculate_pi::ComputeArea;
  TaskList tl;

  // make some lambdas that over overkill here but clean things up for more realistic code
  auto AddBlockTask = [pmb, &tl](BlockTaskFunc func, TaskID dependencies) {
    return tl.AddTask<BlockTask>(func, dependencies, pmb);
  };

  TaskID none(0);
  auto get_area = AddBlockTask(ComputeArea, none);

  // could add more tasks like:
  // auto next_task = tl.AddTask(FuncPtr, get_area, pmb);
  // for a task that executes the function FuncPtr (with argument MeshBlock *pmb)
  // that depends on task get_area
  return tl;
}
