//========================================================================================
// (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
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

  // setup some reductions
  // initialize to zero
  total_mass.val = 0.0;
  // we'll also demonstrate how to reduce a vector
  vec_reduce.val.resize(10);
  for (int i = 0; i < 10; i++)
    vec_reduce.val[i] = 0;

  for (int i = 0; i < num_partitions; i++) {
    // make/get a mesh_data container for the state
    auto &md = pmesh->mesh_data.GetOrAdd("base", i);
    auto &mdelta = pmesh->mesh_data.GetOrAdd("delta", i);

    TaskList &tl = solver_region[i];

    // pass a pointer to the variable being reduced into
    auto loc_red = tl.AddTask(none, poisson_package::SumMass<MeshData<Real>>, md.get(),
                              &total_mass.val);
    // make it a regional dependency so dependent tasks can't execute until all lists do
    // this
    solver_region.AddRegionalDependencies(0, i, loc_red);
    // start a non-blocking MPI_Iallreduce
    TaskID start_global_reduce =
        (i == 0 ? tl.AddTask(loc_red, &parthenon::AllReduce<Real>::StartReduce,
                             &total_mass, MPI_SUM)
                : none);
    // test the reduction until it completes
    TaskID finish_global_reduce = tl.AddTask(
        start_global_reduce, &parthenon::AllReduce<Real>::CheckReduce, &total_mass);
    solver_region.AddRegionalDependencies(1, i, finish_global_reduce);

    // notice how we must always pass a pointer to the reduction value
    // since tasks capture args by value, this would print zero if we just passed in
    // the val since the tasks that compute the value haven't actually executed yet
    auto report_mass = (i == 0 && parthenon::Globals::my_rank == 0
                            ? tl.AddTask(
                                  finish_global_reduce,
                                  [](Real *mass) {
                                    std::cout << "Total mass = " << *mass << std::endl;
                                    return TaskStatus::complete;
                                  },
                                  &total_mass.val)
                            : none);

    auto mat_elem =
        tl.AddTask(none, poisson_package::SetMatrixElements<MeshData<Real>>, md.get());

    auto &solver = tl.AddIteration("poisson solver");
    solver.SetMaxIterations(max_iters);
    solver.SetCheckInterval(check_interval);
    solver.SetFailWithMaxIterations(fail_flag);
    solver.SetWarnWithMaxIterations(warn_flag);
    auto start_recv = solver.AddTask(none, &MeshData<Real>::StartReceiving, md.get(),
                                     BoundaryCommSubset::all);

    auto update =
        solver.AddTask(mat_elem, poisson_package::UpdatePhi<MeshData<Real>>,
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
        clear, poisson_package::CheckConvergence<MeshData<Real>>, md.get(), mdelta.get());
    // mark task so that dependent tasks (below) won't execute
    // until all task lists have completed it
    solver_region.AddRegionalDependencies(2, i, check);

    auto print = none;
    if (i == 0) { // only print once
      print = tl.AddTask(check, poisson_package::PrintComplete);
    }

    // do a vector reduction, just for fun
    // first fill it in
    auto fill_vec = tl.AddTask(
        none,
        [](std::vector<int> *vec) {
          auto &v = *vec;
          for (int n = 0; n < v.size(); n++)
            v[n] += n;
          return TaskStatus::complete;
        },
        &vec_reduce.val);
    solver_region.AddRegionalDependencies(3, i, fill_vec);

    TaskID start_vec_reduce =
        (i == 0
             ? tl.AddTask(fill_vec, &parthenon::AllReduce<std::vector<int>>::StartReduce,
                          &vec_reduce, MPI_SUM)
             : none);
    // test the reduction until it completes
    TaskID finish_vec_reduce =
        tl.AddTask(start_vec_reduce, &parthenon::AllReduce<std::vector<int>>::CheckReduce,
                   &vec_reduce);
    solver_region.AddRegionalDependencies(1, i, finish_vec_reduce);

    auto report_vec =
        (i == 0 && parthenon::Globals::my_rank == 0
             ? tl.AddTask(
                   finish_vec_reduce,
                   [num_partitions](std::vector<int> *vec) {
                     auto &v = *vec;
                     std::cout << "Vec reduction: ";
                     for (int n = 0; n < v.size(); n++) {
                       std::cout << v[n] << " ";
                     }
                     std::cout << std::endl;
                     std::cout << "Should be:     ";
                     for (int n = 0; n < v.size(); n++) {
                       std::cout << n * num_partitions * parthenon::Globals::nranks
                                 << " ";
                     }
                     std::cout << std::endl;
                     return TaskStatus::complete;
                   },
                   &vec_reduce.val)
             : none);
  }

  return tc;
}

} // namespace poisson_example
