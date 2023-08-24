//========================================================================================
// (C) (or copyright) 2023. Triad National Security, LLC. All rights reserved.
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

void PoissonDriver::AddRestrictionProlongationLevel(TaskRegion &region, int level,
                                                    int max_level) {
  using namespace parthenon;
  using namespace poisson_package;
  TaskID none(0);
  const int num_partitions = pmesh->DefaultNumPartitions();

  auto pkg = pmesh->packages.Get("poisson_package");
  auto jacobi_iterations = pkg->Param<int>("jacobi_iterations");
  auto damping = pkg->Param<Real>("jacobi_damping");

  for (int i = 0; i < num_partitions; ++i) {
    TaskList &tl = region[i + (max_level - level) * num_partitions];

    auto &md = pmesh->gmg_mesh_data[level].GetOrAdd(level, "base", i);

    // 0. Receive residual from coarser level if there is one
    auto set_from_finer = none;
    if (level < max_level) {
      // Fill fields with restricted values
      auto recv_from_finer =
          tl.AddTask(none, ReceiveBoundBufs<BoundaryType::gmg_restrict_recv>, md);
      set_from_finer =
          tl.AddTask(recv_from_finer, SetBounds<BoundaryType::gmg_restrict_recv>, md);
    }

    auto print_post_prolongate = none;
    if (level > 0) {
      std::string label = "Pre-restrict field on level " + std::to_string(level);
      auto pre_print = tl.AddTask(set_from_finer, PrintChosenValues<res_err>, md, label);
      // Restrict and send data to next coarser level
      auto communicate_to_coarse =
          tl.AddTask(pre_print, SendBoundBufs<BoundaryType::gmg_restrict_send>, md);

      // Try to receive data from next finer level and prolongate
      auto zero_res = tl.AddTask(communicate_to_coarse, SetToZero<res_err>, md);
      auto recv_from_coarser =
          tl.AddTask(communicate_to_coarse | zero_res,
                     ReceiveBoundBufs<BoundaryType::gmg_prolongate_recv>, md);
      auto set_from_coarser =
          tl.AddTask(recv_from_coarser, SetBounds<BoundaryType::gmg_prolongate_recv>, md);
      auto prolongate = tl.AddTask(
          set_from_coarser, ProlongateBounds<BoundaryType::gmg_prolongate_recv>, md);

      // Print out the post-prolongation solution
      std::string label2 = "Post-prolongate field on level " + std::to_string(level);
      print_post_prolongate =
          tl.AddTask(prolongate, PrintChosenValues<res_err>, md, label2);
    } else {
      std::string label2 = "Field on last level " + std::to_string(level);
      print_post_prolongate =
          tl.AddTask(set_from_finer, PrintChosenValues<res_err>, md, label2);
    }

    if (level < max_level) {
      // If we aren't the finest level, communicate boundaries and then send data to
      // next finer level
      auto same_gmg_level_boundary_comm =
          AddBoundaryExchangeTasks(print_post_prolongate, tl, md, true);
      tl.AddTask(same_gmg_level_boundary_comm,
                 SendBoundBufs<BoundaryType::gmg_prolongate_send>, md);
    }
  }
}

void PoissonDriver::AddMultiGridTasksLevel(TaskRegion &region, int level, int min_level, int max_level, bool final) {
  using namespace parthenon;
  using namespace poisson_package;
  TaskID none(0);
  const int num_partitions = pmesh->DefaultNumPartitions();

  auto pkg = pmesh->packages.Get("poisson_package");
  auto jacobi_iterations = pkg->Param<int>("jacobi_iterations");
  auto damping = pkg->Param<Real>("jacobi_damping");
  
  //if (level == 0) jacobi_iterations = 200; 

  for (int i = 0; i < num_partitions; ++i) {
    TaskList &tl = region[i + (max_level - level) * num_partitions];

    auto &md = pmesh->gmg_mesh_data[level].GetOrAdd(level, "base", i);

    // 0. Receive residual from coarser level if there is one
    auto set_from_finer = none;
    if (level < max_level) {
      // Fill fields with restricted values
      auto recv_from_finer =
          tl.AddTask(none, ReceiveBoundBufs<BoundaryType::gmg_restrict_recv>, md);
      set_from_finer =
          tl.AddTask(recv_from_finer, SetBounds<BoundaryType::gmg_restrict_recv>, md);
    }

    // 0.1 Build the matrix on this level
    auto build_matrix = tl.AddTask(none, BuildMatrix, md);

    // 1. Copy residual from dual purpose communication field to the rhs, copy actual RHS
    // for finest level
    auto copy_rhs = tl.AddTask(set_from_finer, CopyData<res_err, rhs>, md);

    // 2. Do pre-smooth and fill solution on this level
    auto zero_u = tl.AddTask(none, AddFieldsAndStore<u, u, u>, md, 0.0, 0.0);
    
    // 2. Do pre-smooth and fill solution on this level
    auto pre_previous_iter = copy_rhs | build_matrix | zero_u;
    for (int jacobi_iter = 0; jacobi_iter < jacobi_iterations / 2; ++jacobi_iter) {
      auto comm1 = AddBoundaryExchangeTasks(pre_previous_iter, tl, md, true);
      auto jacobi1 =
          tl.AddTask(comm1, JacobiIteration<u, temp>, md, 1.0 - damping, level);
      auto comm2 = AddBoundaryExchangeTasks(jacobi1, tl, md, true);
      pre_previous_iter =
          tl.AddTask(comm2, JacobiIteration<temp, u>, md, 1.0 - damping, level);
    }
    auto pre_smooth = pre_previous_iter;
    
    // If we are finer than the coarsest level:
    auto post_smooth = none;
    if (level > min_level) {
      // 3. Communicate same level boundaries so that u is up to date everywhere
      auto comm_u = AddBoundaryExchangeTasks(pre_smooth, tl, md, true);

      // 4. Caclulate residual and store in communication field
      auto residual = tl.AddTask(comm_u, CalculateResidual, md);

      // 5. Restrict communication field and send to next level
      auto communicate_to_coarse =
          tl.AddTask(residual, SendBoundBufs<BoundaryType::gmg_restrict_send>, md);

      // 6. Receive error field into communication field and prolongate
      auto recv_from_coarser = tl.AddTask(
          communicate_to_coarse, ReceiveBoundBufs<BoundaryType::gmg_prolongate_recv>, md);
      auto set_from_coarser =
          tl.AddTask(recv_from_coarser, SetBounds<BoundaryType::gmg_prolongate_recv>, md);
      auto zero_res = tl.AddTask(communicate_to_coarse, SetToZero<res_err>, md);
      auto prolongate =
          tl.AddTask(set_from_coarser | zero_res,
                     ProlongateBounds<BoundaryType::gmg_prolongate_recv>, md);

      // 7. Correct solution on this level with res_err field and store in
      //    communication field
      auto update_err =
          tl.AddTask(prolongate, AddFieldsAndStore<u, res_err, u>, md, 1.0, 1.0);
      auto comm_err = AddBoundaryExchangeTasks(update_err, tl, md, true);

      // 8. Post smooth using communication field and stored RHS
      auto previous_iter = comm_err;
      for (int jacobi_iter = 0; jacobi_iter < jacobi_iterations / 2; ++jacobi_iter) {
        auto comm1 = AddBoundaryExchangeTasks(previous_iter, tl, md, true);
        auto jacobi1 =
            tl.AddTask(comm1, JacobiIteration<u, temp>, md, 1.0 - damping, level);
        auto comm2 = AddBoundaryExchangeTasks(jacobi1, tl, md, true);
        previous_iter =
            tl.AddTask(comm2, JacobiIteration<temp, u>, md, 1.0 - damping, level);
      }
      auto copy_over = tl.AddTask(previous_iter, CopyData<u, res_err>, md);
      auto boundary = AddBoundaryExchangeTasks(copy_over, tl, md, true);
      post_smooth = boundary;
    } else {
      post_smooth = tl.AddTask(pre_smooth, CopyData<u, res_err>, md);
    }

    // 9. Send communication field to next finer level (should be error field for that
    // level)
    //    or update the solution on the finest level
    if (level < max_level) {
      tl.AddTask(post_smooth, SendBoundBufs<BoundaryType::gmg_prolongate_send>, md);
    } else {
      auto comm = AddBoundaryExchangeTasks(post_smooth, tl, md, true);
      auto update_solution =
          tl.AddTask(comm, AddFieldsAndStore<solution, res_err, solution>, md, 1.0, 1.0);
      auto set_u = tl.AddTask(update_solution, CopyData<solution, u>, md);
      auto set_rhs = tl.AddTask(update_solution, CopyData<rhs_base, rhs>, md);
      auto res = tl.AddTask(set_u | set_rhs, CalculateResidual, md);
      if (final) { 
        std::string label = "Solution, rhs, residual";
        auto printout = tl.AddTask(res, PrintChosenValues<u, rhs_base, res_err>, md, label);
      }
      auto res_comm = AddBoundaryExchangeTasks(res, tl, md, true);
    }
  }
}

TaskCollection PoissonDriver::MakeTaskCollection(BlockList_t &blocks) {
  using namespace parthenon;
  TaskCollection tc;
  TaskID none(0);

  auto pkg = pmesh->packages.Get("poisson_package");
  auto max_iterations = pkg->Param<int>("max_iterations");

  const int num_partitions = pmesh->DefaultNumPartitions();
  int min_level = 0;//pmesh->GetGMGMaxLevel();
  int max_level = pmesh->GetGMGMaxLevel();
  for (int ivcycle = 0; ivcycle < max_iterations; ++ivcycle) {
    TaskRegion &region = tc.AddRegion(num_partitions * (max_level + 1));
    for (int level = max_level; level >= min_level; --level) {
      AddMultiGridTasksLevel(region, level, min_level, max_level, ivcycle == max_iterations - 1);
      // AddRestrictionProlongationLevel(region, level, max_level);
    }
  }

  return tc;
}

} // namespace poisson_example
