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
#include "amr_criteria/refinement_package.hpp"
#include "bvals/comms/bvals_in_one.hpp"
#include "interface/metadata.hpp"
#include "interface/update.hpp"
#include "mesh/meshblock_pack.hpp"
#include "parthenon/driver.hpp"
#include "poisson_driver.hpp"
#include "poisson_package.hpp"
#include "prolong_restrict/prolong_restrict.hpp"

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
  AllReduce<HostArray1D<Real>> *pview_reduce =
      pkg->MutableParam<AllReduce<HostArray1D<Real>>>("view_reduce");
  int reg_dep_id;
  for (int i = 0; i < num_partitions; i++) {
    reg_dep_id = 0;
    // make/get a mesh_data container for the state
    auto &md = pmesh->mesh_data.GetOrAdd("base", i);
    auto &mdelta = pmesh->mesh_data.GetOrAdd("delta", i);

    TaskList &tl = solver_region[i];

    //--- Demo a few reductions
    // pass a pointer to the variable being reduced into
    auto loc_red =
        tl.AddTask(TaskQualifier::local_sync, none,
                   poisson_package::SumMass<MeshData<Real>>, md.get(), &total_mass.val);

    auto rank_red = tl.AddTask(
        TaskQualifier::local_sync, none,
        [](int *max_rank) {
          *max_rank = std::max(*max_rank, Globals::my_rank);
          return TaskStatus::complete;
        },
        &max_rank.val);

    // start a non-blocking MPI_Iallreduce
    auto start_global_reduce =
        tl.AddTask(TaskQualifier::once_per_region, loc_red, &AllReduce<Real>::StartReduce,
                   &total_mass, MPI_SUM);

    auto start_rank_reduce = tl.AddTask(TaskQualifier::once_per_region, rank_red,
                                        &Reduce<int>::StartReduce, &max_rank, 0, MPI_MAX);

    // test the reduction until it completes
    auto finish_global_reduce =
        tl.AddTask(TaskQualifier::local_sync | TaskQualifier::once_per_region,
                   start_global_reduce, &AllReduce<Real>::CheckReduce, &total_mass);

    auto finish_rank_reduce =
        tl.AddTask(TaskQualifier::local_sync | TaskQualifier::once_per_region,
                   start_rank_reduce, &Reduce<int>::CheckReduce, &max_rank);

    // notice how we must always pass a pointer to the reduction value
    // since tasks capture args by value, this would print zero if we just passed in
    // the val since the tasks that compute the value haven't actually executed yet
    auto report_mass = tl.AddTask(
        TaskQualifier::once_per_region, finish_global_reduce,
        [](Real *mass) {
          if (Globals::my_rank == 0) std::cout << "Total mass = " << *mass << std::endl;
          return TaskStatus::complete;
        },
        &total_mass.val);
    auto report_rank = tl.AddTask(
        TaskQualifier::once_per_region, finish_rank_reduce,
        [](int *max_rank) {
          if (Globals::my_rank == 0) std::cout << "Max rank = " << *max_rank << std::endl;
          return TaskStatus::complete;
        },
        &max_rank.val);

    //--- Begining of tasks related to solving the Poisson eq.
    auto mat_elem =
        tl.AddTask(none, poisson_package::SetMatrixElements<MeshData<Real>>, md.get());

    auto [solver, solver_id] = tl.AddSublist(mat_elem, {1, max_iters});

    auto start_recv = solver.AddTask(none, parthenon::StartReceiveBoundaryBuffers, md);

    auto update = solver.AddTask(none, poisson_package::UpdatePhi<MeshData<Real>>,
                                 md.get(), mdelta.get());

    auto norm = solver.AddTask(TaskQualifier::local_sync, update,
                               poisson_package::SumDeltaPhi<MeshData<Real>>, mdelta.get(),
                               &update_norm.val);
    auto start_reduce_norm =
        solver.AddTask(TaskQualifier::once_per_region, norm,
                       &AllReduce<Real>::StartReduce, &update_norm, MPI_SUM);
    auto finish_reduce_norm =
        solver.AddTask(TaskQualifier::once_per_region, start_reduce_norm,
                       &AllReduce<Real>::CheckReduce, &update_norm);
    auto report_norm = solver.AddTask(
        TaskQualifier::once_per_region, finish_reduce_norm,
        [](Real *norm) {
          if (Globals::my_rank == 0) {
            std::cout << "Update norm = " << *norm << std::endl;
          }
          *norm = 0.0;
          return TaskStatus::complete;
        },
        &update_norm.val);

    auto send = solver.AddTask(update, SendBoundaryBuffers, md);

    auto recv = solver.AddTask(start_recv, ReceiveBoundaryBuffers, md);

    auto setb = solver.AddTask(recv | update, SetBoundaries, md);

    auto check = solver.AddTask(
        TaskQualifier::completion | TaskQualifier::global_sync, send | setb | report_norm,
        poisson_package::CheckConvergence<MeshData<Real>>, md.get(), mdelta.get());

    auto print = tl.AddTask(TaskQualifier::once_per_region, solver_id,
                            poisson_package::PrintComplete);
    //--- End of tasks related to solving the Poisson eq

    // do a vector reduction (everything below here), just for fun
    // first fill it in
    auto fill_vec = tl.AddTask(
        TaskQualifier::local_sync, none,
        [](std::vector<int> *vec) {
          auto &v = *vec;
          for (int n = 0; n < v.size(); n++)
            v[n] += n;
          return TaskStatus::complete;
        },
        &vec_reduce.val);

    TaskID start_vec_reduce =
        tl.AddTask(TaskQualifier::once_per_region, fill_vec,
                   &AllReduce<std::vector<int>>::StartReduce, &vec_reduce, MPI_SUM);
    // test the reduction until it completes
    TaskID finish_vec_reduce = tl.AddTask(
        TaskQualifier::once_per_region | TaskQualifier::local_sync, start_vec_reduce,
        &AllReduce<std::vector<int>>::CheckReduce, &vec_reduce);

    auto report_vec = tl.AddTask(
        TaskQualifier::once_per_region, finish_vec_reduce,
        [num_partitions](std::vector<int> *vec) {
          if (Globals::my_rank == 0) {
            auto &v = *vec;
            std::cout << "Vec reduction: ";
            for (int n = 0; n < v.size(); n++) {
              std::cout << v[n] << " ";
            }
            std::cout << std::endl;
            std::cout << "Should be:     ";
            for (int n = 0; n < v.size(); n++) {
              std::cout << n * num_partitions * Globals::nranks << " ";
            }
            std::cout << std::endl;
          }
          return TaskStatus::complete;
        },
        &vec_reduce.val);

    // And lets do a view reduce too just for fun
    // The views are filled in the package
    TaskID start_view_reduce =
        tl.AddTask(TaskQualifier::once_per_region, none,
                   &AllReduce<HostArray1D<Real>>::StartReduce, pview_reduce, MPI_SUM);
    // test the reduction until it completes
    TaskID finish_view_reduce = tl.AddTask(
        TaskQualifier::once_per_region | TaskQualifier::local_sync, start_view_reduce,
        &AllReduce<HostArray1D<Real>>::CheckReduce, pview_reduce);

    auto report_view = tl.AddTask(
        TaskQualifier::once_per_region, finish_view_reduce,
        [num_partitions](HostArray1D<Real> *view) {
          if (Globals::my_rank == 0) {
            auto &v = *view;
            std::cout << "View reduction: ";
            for (int n = 0; n < v.size(); n++) {
              std::cout << v(n) << " ";
            }
            std::cout << std::endl;
            std::cout << "Should be:     ";
            for (int n = 0; n < v.size(); n++) {
              std::cout << n * num_partitions * Globals::nranks << " ";
            }
            std::cout << std::endl;
          }
          return TaskStatus::complete;
        },
        &(pview_reduce->val));
  }

  return tc;
}

} // namespace poisson_example
