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

#include <algorithm>
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
  using namespace parthenon;
  using poisson_package::HostArray1D;
  TaskCollection tc;
  TaskID none(0);

  for (int i = 0; i < blocks.size(); i++) {
    auto &pmb = blocks[i];
    auto &base = pmb->meshblock_data.Get();
    pmb->meshblock_data.Add("delta", base);
  }

  auto pkg = pmesh->packages.Get("poisson_package");
  auto max_iters = pkg->Param<int>("max_iterations");
  auto check_interval = pkg->Param<int>("check_interval");
  auto fail_flag = pkg->Param<bool>("fail_without_convergence");
  auto warn_flag = pkg->Param<bool>("warn_without_convergence");

  const int num_partitions = pmesh->DefaultNumPartitions();
  TaskRegion &solver_region = tc.AddRegion(num_partitions);

  // setup some reductions
  // initialize to zero
  total_mass.val = 0.0;
  update_norm.val = 0.0;
  max_rank.val = 0;
  // we'll also demonstrate how to reduce a vector
  vec_reduce.val.resize(10);
  for (int i = 0; i < 10; i++)
    vec_reduce.val[i] = 0;
  // and a kokkos view just for fun
  HostArray1D view_val = pkg->Param<HostArray1D>("view_reduce");
  view_reduce.val = ParHostUnmanaged1D<int>(view_val.data(), 10);
  int reg_dep_id;
  for (int i = 0; i < num_partitions; i++) {
    reg_dep_id = 0;
    // make/get a mesh_data container for the state
    auto &md = pmesh->mesh_data.GetOrAdd("base", i);
    auto &mdelta = pmesh->mesh_data.GetOrAdd("delta", i);

    TaskList &tl = solver_region[i];

    //--- Demo a few reductions
    // pass a pointer to the variable being reduced into
    auto loc_red = tl.AddTask(none, poisson_package::SumMass<MeshData<Real>>, md.get(),
                              &total_mass.val);
    // make it a regional dependency so dependent tasks can't execute until all lists do
    // this
    solver_region.AddRegionalDependencies(reg_dep_id, i, loc_red);
    reg_dep_id++;

    auto rank_red = tl.AddTask(
        none,
        [](int *max_rank) {
          *max_rank = std::max(*max_rank, Globals::my_rank);
          return TaskStatus::complete;
        },
        &max_rank.val);
    solver_region.AddRegionalDependencies(reg_dep_id, i, rank_red);
    reg_dep_id++;

    // start a non-blocking MPI_Iallreduce
    auto start_global_reduce =
        (i == 0 ? tl.AddTask(loc_red, &AllReduce<Real>::StartReduce, &total_mass, MPI_SUM)
                : none);

    auto start_rank_reduce =
        (i == 0 ? tl.AddTask(rank_red, &Reduce<int>::StartReduce, &max_rank, 0, MPI_MAX)
                : none);

    // test the reduction until it completes
    auto finish_global_reduce =
        tl.AddTask(start_global_reduce, &AllReduce<Real>::CheckReduce, &total_mass);
    solver_region.AddRegionalDependencies(reg_dep_id, i, finish_global_reduce);
    reg_dep_id++;

    auto finish_rank_reduce =
        tl.AddTask(start_rank_reduce, &Reduce<int>::CheckReduce, &max_rank);
    solver_region.AddRegionalDependencies(reg_dep_id, i, finish_rank_reduce);
    reg_dep_id++;

    // notice how we must always pass a pointer to the reduction value
    // since tasks capture args by value, this would print zero if we just passed in
    // the val since the tasks that compute the value haven't actually executed yet
    auto report_mass = (i == 0 && Globals::my_rank == 0
                            ? tl.AddTask(
                                  finish_global_reduce,
                                  [](Real *mass) {
                                    std::cout << "Total mass = " << *mass << std::endl;
                                    return TaskStatus::complete;
                                  },
                                  &total_mass.val)
                            : none);
    auto report_rank = (i == 0 && Globals::my_rank == 0
                            ? tl.AddTask(
                                  finish_rank_reduce,
                                  [](int *max_rank) {
                                    std::cout << "Max rank = " << *max_rank << std::endl;
                                    return TaskStatus::complete;
                                  },
                                  &max_rank.val)
                            : none);

    //--- Begining of tasks related to solving the Poisson eq.
    auto mat_elem =
        tl.AddTask(none, poisson_package::SetMatrixElements<MeshData<Real>>, md.get());

    auto &solver = tl.AddIteration("poisson solver");
    solver.SetMaxIterations(max_iters);
    solver.SetCheckInterval(check_interval);
    solver.SetFailWithMaxIterations(fail_flag);
    solver.SetWarnWithMaxIterations(warn_flag);
    auto start_recv = solver.AddTask(none, &MeshData<Real>::StartReceiving, md.get(),
                                     BoundaryCommSubset::all);

    auto update = solver.AddTask(mat_elem, poisson_package::UpdatePhi<MeshData<Real>>,
                                 md.get(), mdelta.get());

    auto norm = solver.AddTask(update, poisson_package::SumDeltaPhi<MeshData<Real>>,
                               mdelta.get(), &update_norm.val);
    solver_region.AddRegionalDependencies(reg_dep_id, i, norm);
    reg_dep_id++;
    auto start_reduce_norm = (i == 0 ? solver.AddTask(norm, &AllReduce<Real>::StartReduce,
                                                      &update_norm, MPI_SUM)
                                     : none);
    auto finish_reduce_norm =
        solver.AddTask(start_reduce_norm, &AllReduce<Real>::CheckReduce, &update_norm);
    auto report_norm = (i == 0 ? solver.AddTask(
                                     finish_reduce_norm,
                                     [](Real *norm) {
                                       if (Globals::my_rank == 0) {
                                         std::cout << "Update norm = " << *norm
                                                   << std::endl;
                                       }
                                       *norm = 0.0;
                                       return TaskStatus::complete;
                                     },
                                     &update_norm.val)
                               : none);

    auto send = solver.AddTask(update, cell_centered_bvars::SendBoundaryBuffers, md);

    auto recv =
        solver.AddTask(start_recv, cell_centered_bvars::ReceiveBoundaryBuffers, md);

    auto setb = solver.AddTask(recv | update, cell_centered_bvars::SetBoundaries, md);

    auto clear = solver.AddTask(send | setb | report_norm, &MeshData<Real>::ClearBoundary,
                                md.get(), BoundaryCommSubset::all);

    auto check = solver.SetCompletionTask(
        clear, poisson_package::CheckConvergence<MeshData<Real>>, md.get(), mdelta.get());
    // mark task so that dependent tasks (below) won't execute
    // until all task lists have completed it
    solver_region.AddRegionalDependencies(reg_dep_id, i, check);
    reg_dep_id++;

    auto print = none;
    if (i == 0) { // only print once
      print = tl.AddTask(check, poisson_package::PrintComplete);
    }
    //--- End of tasks related to solving the Poisson eq

    // do a vector reduction (everything below here), just for fun
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
    solver_region.AddRegionalDependencies(reg_dep_id, i, fill_vec);
    reg_dep_id++;

    TaskID start_vec_reduce =
        (i == 0 ? tl.AddTask(fill_vec, &AllReduce<std::vector<int>>::StartReduce,
                             &vec_reduce, MPI_SUM)
                : none);
    // test the reduction until it completes
    TaskID finish_vec_reduce = tl.AddTask(
        start_vec_reduce, &AllReduce<std::vector<int>>::CheckReduce, &vec_reduce);
    solver_region.AddRegionalDependencies(reg_dep_id, i, finish_vec_reduce);
    reg_dep_id++;

    auto report_vec = (i == 0 && Globals::my_rank == 0
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
                                     std::cout << n * num_partitions * Globals::nranks
                                               << " ";
                                   }
                                   std::cout << std::endl;
                                   return TaskStatus::complete;
                                 },
                                 &vec_reduce.val)
                           : none);

    // And lets do a view reduce too just for fun
    // The views are filled in the package
    TaskID start_view_reduce =
        (i == 0 ? tl.AddTask(none, &AllReduce<ParHostUnmanaged1D<int>>::StartReduce,
                             &view_reduce, MPI_SUM)
                : none);
    // test the reduction until it completes
    TaskID finish_view_reduce =
        tl.AddTask(start_view_reduce, &AllReduce<ParHostUnmanaged1D<int>>::CheckReduce,
                   &view_reduce);
    solver_region.AddRegionalDependencies(reg_dep_id, i, finish_view_reduce);
    reg_dep_id++;

    auto report_view = (i == 0 && Globals::my_rank == 0
                            ? tl.AddTask(
                                  finish_view_reduce,
                                  [num_partitions](ParHostUnmanaged1D<int> *view) {
                                    auto &v = *view;
                                    std::cout << "View reduction: ";
                                    for (int n = 0; n < v.size(); n++) {
                                      std::cout << v(n) << " ";
                                    }
                                    std::cout << std::endl;
                                    std::cout << "Should be:     ";
                                    for (int n = 0; n < v.size(); n++) {
                                      std::cout << n * num_partitions * Globals::nranks
                                                << " ";
                                    }
                                    std::cout << std::endl;
                                    return TaskStatus::complete;
                                  },
                                  &view_reduce.val)
                            : none);
  }

  return tc;
}

} // namespace poisson_example
