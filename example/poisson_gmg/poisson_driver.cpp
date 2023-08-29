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

TaskID AddRMSResidualPrintout(int level, TaskList &tl, TaskID depends_on, bool multilevel,
                              std::shared_ptr<MeshData<Real>> &md) {
  using namespace parthenon;
  using namespace poisson_package;
  auto comm = AddBoundaryExchangeTasks(depends_on, tl, md, multilevel);
  auto copy_res = tl.AddTask(comm, CopyData<res_err, temp>, md);
  auto res = tl.AddTask(copy_res, CalculateResidual, md);
  std::string ident = "level " + std::to_string(level);
  auto printout = tl.AddTask(res, RMSResidual, md, ident);
  auto copy_back = tl.AddTask(printout, CopyData<temp, res_err>, md);
  return copy_back;
}

TaskID AddSRJIteration(TaskList &tl, TaskID depends_on, int stages, bool multilevel,
                       std::shared_ptr<MeshData<Real>> &md) {
  using namespace parthenon;
  using namespace poisson_package;
  int ndim = md->GetParentPointer()->ndim;

  // Damping factors from Yang & Mittal (2017)
  std::array<std::array<Real, 3>, 3> omega_M2{
      {{0.8723, 0.5395, 0.0000}, {1.3895, 0.5617, 0.0000}, {1.7319, 0.5695, 0.0000}}};
  std::array<std::array<Real, 3>, 3> omega_M3{
      {{0.9372, 0.6667, 0.5173}, {1.6653, 0.8000, 0.5264}, {2.2473, 0.8571, 0.5296}}};
  auto omega = omega_M2;
  if (stages == 3) omega = omega_M3;

  auto comm1 = AddBoundaryExchangeTasks(depends_on, tl, md, multilevel);
  auto jacobi1 = tl.AddTask(comm1, JacobiIteration<u, temp>, md, omega[ndim - 1][0]);
  auto comm2 = AddBoundaryExchangeTasks(jacobi1, tl, md, multilevel);
  auto jacobi2 = tl.AddTask(comm2, JacobiIteration<temp, u>, md, omega[ndim - 1][1]);
  if (stages < 3) return jacobi2;
  auto comm3 = AddBoundaryExchangeTasks(jacobi2, tl, md, multilevel);
  auto jacobi3 = tl.AddTask(comm3, JacobiIteration<u, temp>, md, omega[ndim - 1][2]);
  return tl.AddTask(jacobi3, CopyData<temp, u>, md);
}

TaskID AddGSIteration(TaskList &tl, TaskID depends_on, int iters, bool multilevel,
                      std::shared_ptr<MeshData<Real>> &md) {
  using namespace parthenon;
  using namespace poisson_package;
  auto previous_iter = depends_on;
  for (int i = 0; i < iters; ++i) {
    auto comm = AddBoundaryExchangeTasks(previous_iter, tl, md, multilevel);
    previous_iter = tl.AddTask(comm, JacobiIteration<u, u>, md, 1.0);
  }
  return previous_iter;
}

TaskID AddRBGSIteration(TaskList &tl, TaskID depends_on, int iters, bool multilevel,
                        std::shared_ptr<MeshData<Real>> &md) {
  using namespace parthenon;
  using namespace poisson_package;
  auto previous_iter = depends_on;
  for (int i = 0; i < iters; ++i) {
    auto comm1 = AddBoundaryExchangeTasks(previous_iter, tl, md, multilevel);
    auto update_even = tl.AddTask(comm1, RBGSIteration<u, temp>, md, false);
    auto comm2 = AddBoundaryExchangeTasks(update_even, tl, md, multilevel);
    auto update_odd = tl.AddTask(comm2, RBGSIteration<temp, temp>, md, true);
    previous_iter = tl.AddTask(update_odd, CopyData<temp, u>, md);
  }
  return previous_iter;
}

void PoissonDriver::AddMultiGridTasksLevel(TaskRegion &region, int level, int min_level,
                                           int max_level, bool final) {
  using namespace parthenon;
  using namespace poisson_package;
  TaskID none(0);
  const int num_partitions = pmesh->DefaultNumPartitions();

  auto pkg = pmesh->packages.Get("poisson_package");
  auto damping = pkg->Param<Real>("jacobi_damping");
  auto smoother = pkg->Param<std::string>("smoother");
  int pre_stages = pkg->Param<int>("pre_smooth_iterations");
  int post_stages = pkg->Param<int>("post_smooth_iterations");
  if (smoother == "SRJ2") {
    pre_stages = 2;
    post_stages = 2;
  } else if (smoother == "SRJ3") {
    pre_stages = 3;
    post_stages = 3;
  }

  int ndim = pmesh->ndim;
  bool multilevel = (level != min_level);
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
      auto zero_u = tl.AddTask(none, SetToZero<u>, md);
      // 1. Copy residual from dual purpose communication field to the rhs, should be
      // actual RHS for finest level
      auto copy_rhs = tl.AddTask(set_from_finer, CopyData<res_err, rhs>, md);
      set_from_finer = zero_u | copy_rhs;
    }

    // 0.1 Build the matrix on this level
    auto build_matrix = tl.AddTask(none, BuildMatrix, md);

    // 2. Do pre-smooth and fill solution on this level
    auto pre_smooth = set_from_finer | build_matrix;
    if (smoother == "GS") {
      pre_smooth =
          AddGSIteration(tl, set_from_finer | build_matrix, pre_stages, multilevel, md);
    } else if (smoother == "RBGS") {
      pre_smooth =
          AddRBGSIteration(tl, set_from_finer | build_matrix, pre_stages, multilevel, md);
    } else if (smoother == "SRJ2" || smoother == "SRJ3") {
      pre_smooth =
          AddSRJIteration(tl, set_from_finer | build_matrix, pre_stages, multilevel, md);
    }

    // If we are finer than the coarsest level:
    auto post_smooth = none;
    if (level > min_level) {
      // 3. Communicate same level boundaries so that u is up to date everywhere
      auto comm_u = AddBoundaryExchangeTasks(pre_smooth, tl, md, multilevel);

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
      auto prolongate = tl.AddTask(
          set_from_coarser, ProlongateBounds<BoundaryType::gmg_prolongate_recv>, md);

      // 7. Correct solution on this level with res_err field and store in
      //    communication field
      auto update_sol =
          tl.AddTask(prolongate, AddFieldsAndStore<u, res_err, u>, md, 1.0, 1.0);

      // 8. Post smooth using communication field and stored RHS
      if (smoother == "GS") {
        post_smooth = AddGSIteration(tl, update_sol, post_stages, multilevel, md);
      } else if (smoother == "RBGS") {
        post_smooth = AddRBGSIteration(tl, update_sol, post_stages, multilevel, md);
      } else if (smoother == "SRJ2" || smoother == "SRJ3") {
        post_smooth = AddSRJIteration(tl, update_sol, post_stages, multilevel, md);
      }
    } else {
      post_smooth = tl.AddTask(pre_smooth, CopyData<u, res_err>, md);
    }

    // 9. Send communication field to next finer level (should be error field for that
    // level)
    if (level < max_level) {
      auto copy_over = tl.AddTask(post_smooth, CopyData<u, res_err>, md);
      auto boundary = AddBoundaryExchangeTasks(copy_over, tl, md, multilevel);
      tl.AddTask(boundary, SendBoundBufs<BoundaryType::gmg_prolongate_send>, md);
    } else {
      auto res = AddRMSResidualPrintout(level, tl, post_smooth, multilevel, md);
    }
  }
}

TaskCollection PoissonDriver::MakeTaskCollection(BlockList_t &blocks) {
  using namespace parthenon;
  using namespace poisson_package;
  TaskCollection tc;
  TaskID none(0);

  auto pkg = pmesh->packages.Get("poisson_package");
  auto max_iterations = pkg->Param<int>("max_iterations");

  const int num_partitions = pmesh->DefaultNumPartitions();
  int min_level = 0; // pmesh->GetGMGMaxLevel();
  int max_level = pmesh->GetGMGMaxLevel();

  TaskRegion &region_init = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; ++i) {
    TaskList &tl = region_init[i];
    auto &md = pmesh->gmg_mesh_data[max_level].GetOrAdd(max_level, "base", i);
    auto zero_u = tl.AddTask(none, SetToZero<u>, md);
  }

  for (int ivcycle = 0; ivcycle < max_iterations; ++ivcycle) {
    TaskRegion &region = tc.AddRegion(num_partitions * (max_level + 1));
    for (int level = max_level; level >= min_level; --level) {
      AddMultiGridTasksLevel(region, level, min_level, max_level,
                             ivcycle == max_iterations - 1);
    }
  }

  return tc;
}

} // namespace poisson_example
