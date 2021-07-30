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

#include <memory>
#include <string>
#include <vector>

// Local Includes
#include "bvals/cc/bvals_cc_in_one.hpp"
#include "interface/metadata.hpp"
#include "interface/update.hpp"
#include "mesh/meshblock_pack.hpp"
#include "mesh/refinement_cc_in_one.hpp"
#include "parthenon/driver.hpp"
#include "poisson_driver.hpp"
#include "poisson_package.hpp"
#include "refinement/refinement.hpp"

using namespace parthenon::driver::prelude;

namespace poisson_example {

parthenon::DriverStatus PoissonDriver::Execute() {
  pouts->MakeOutputs(pmesh, pinput);
  ConstructAndExecuteTaskLists<>(this);
  pouts->MakeOutputs(pmesh, pinput);
  return DriverStatus::complete;
}

TaskCollection PoissonDriver::MakeTaskCollection(BlockList_t &blocks) {
  using namespace parthenon::Update;
  TaskCollection tc;
  TaskID none(0);

  for (int i = 0; i < blocks.size(); i++) {
    auto &pmb = blocks[i];
    auto &base = pmb->meshblock_data.Get();
    pmb->meshblock_data.Add("delta", base);
  }

  int max_iters = pmesh->packages.Get("poisson_package")->Param<int>("max_iterations");
  int check_interval =
      pmesh->packages.Get("poisson_package")->Param<int>("check_interval");
  bool fail_flag =
      pmesh->packages.Get("poisson_package")->Param<bool>("fail_without_convergence");
  bool warn_flag =
      pmesh->packages.Get("poisson_package")->Param<bool>("warn_without_convergence");

  const int num_partitions = pmesh->DefaultNumPartitions();
  TaskRegion &solver_region = tc.AddRegion(num_partitions);

  for (int i = 0; i < num_partitions; i++) {
    // make/get a mesh_data container for the state
    auto &md = pmesh->mesh_data.GetOrAdd("base", i);
    auto &mdelta = pmesh->mesh_data.GetOrAdd("delta", i);

    TaskList &tl = solver_region[i];

    auto &solver = tl.AddIteration("poisson solver");
    solver.SetMaxIterations(max_iters);
    solver.SetCheckInterval(check_interval);
    solver.SetFailWithMaxIterations(fail_flag);
    solver.SetWarnWithMaxIterations(warn_flag);
    auto start_recv = solver.AddTask(none, &MeshData<Real>::StartReceiving, md.get(),
                                     BoundaryCommSubset::all);

    auto update = solver.AddTask(none, poisson_package::UpdatePhi<MeshData<Real>>,
                                 md.get(), mdelta.get());

    auto send =
        solver.AddTask(update, parthenon::cell_centered_bvars::SendBoundaryBuffers, md);

    auto recv = solver.AddTask(
        start_recv, parthenon::cell_centered_bvars::ReceiveBoundaryBuffers, md);

    auto setb =
        solver.AddTask(recv | update, parthenon::cell_centered_bvars::SetBoundaries, md);

    auto clear = solver.AddTask(send | setb, &MeshData<Real>::ClearBoundary, md.get(),
                                BoundaryCommSubset::all);

    auto check = solver.SetCompletionTask(
        update | clear, poisson_package::CheckConvergence<MeshData<Real>>, md.get(),
        mdelta.get());
    solver_region.AddRegionalDependencies(0, i, check);
  }

  return tc;
}

} // namespace poisson_example
