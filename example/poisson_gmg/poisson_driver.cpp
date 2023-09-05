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
                                                    int min_level, int max_level) {
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
    if (level > min_level) {
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
      std::string label = "Pre-prolongate field on level " + std::to_string(level);
      auto print =
          tl.AddTask(same_gmg_level_boundary_comm, PrintChosenValues<res_err>, md, label);
      tl.AddTask(print,
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

template<class in_t, class out_t> 
TaskID AddJacobiIteration(TaskList &tl, TaskID depends_on, bool multilevel, Real omega,
                          std::shared_ptr<MeshData<Real>> &md) {
  using namespace parthenon;
  using namespace poisson_package;
  TaskID none(0);

  auto comm = AddBoundaryExchangeTasks(depends_on, tl, md, multilevel);
  auto flux = tl.AddTask(comm, CalculateFluxes<in_t>, md);
  auto start_flxcor = tl.AddTask(flux, StartReceiveFluxCorrections, md);
  auto send_flxcor = tl.AddTask(flux, LoadAndSendFluxCorrections, md);
  auto recv_flxcor = tl.AddTask(send_flxcor, ReceiveFluxCorrections, md);
  auto set_flxcor = tl.AddTask(recv_flxcor, SetFluxCorrections, md);
  auto mat_mult = tl.AddTask(set_flxcor, FluxMultiplyMatrix<in_t, out_t>, md); 
  return tl.AddTask(mat_mult, FluxJacobi<out_t, in_t, out_t>, md, omega);
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
  
  auto jacobi1 = AddJacobiIteration<u, temp>(tl, depends_on, multilevel, omega[ndim - 1][0], md); 
  auto jacobi2 = AddJacobiIteration<temp, u>(tl, jacobi1, multilevel, omega[ndim - 1][1], md); 
  if (stages < 3) return jacobi2; 
  auto jacobi3 = AddJacobiIteration<u, temp>(tl, jacobi2, multilevel, omega[ndim - 1][2], md); 
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

TaskID AddResidualCalc(TaskList &tl, TaskID depends_on, std::shared_ptr<MeshData<Real>> &md) {
  using namespace parthenon;
  using namespace poisson_package;
  auto flux_res = tl.AddTask(depends_on, CalculateFluxes<u>, md);
  auto start_flxcor = tl.AddTask(flux_res, StartReceiveFluxCorrections, md);
  auto send_flxcor = tl.AddTask(flux_res, LoadAndSendFluxCorrections, md);
  auto recv_flxcor = tl.AddTask(send_flxcor, ReceiveFluxCorrections, md);
  auto set_flxcor = tl.AddTask(recv_flxcor, SetFluxCorrections, md); 
  auto Ax_res = tl.AddTask(set_flxcor, FluxMultiplyMatrix<u, temp>, md); 
  return tl.AddTask(Ax_res, AddFieldsAndStore<rhs, temp, res_err>, md, 1.0, -1.0);
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
      auto residual = AddResidualCalc(tl, comm_u, md);
      
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
      AddBoundaryExchangeTasks(post_smooth, tl, md, multilevel);
    }
  }
}

template <class a_t, class b_t> 
TaskID DotProduct(TaskID dependency_in, TaskRegion &region, TaskList &tl, int partition, int &reg_dep_id, AllReduce<Real> *adotb, std::shared_ptr<MeshData<Real>> &md) {
  using namespace parthenon;
  using namespace poisson_package;
  auto zero_adotb = (partition == 0 ? tl.AddTask(dependency_in, [](AllReduce<Real> *r){r->val = 0.0; return TaskStatus::complete;}, adotb) : dependency_in);
  region.AddRegionalDependencies(reg_dep_id, partition, zero_adotb);
  reg_dep_id++;
  auto get_adotb = tl.AddTask(zero_adotb, DotProductLocal<a_t, b_t>, md, &(adotb->val));
  region.AddRegionalDependencies(reg_dep_id, partition, get_adotb);
  reg_dep_id++;
  auto start_global_adotb = (partition == 0 ? tl.AddTask(get_adotb, &AllReduce<Real>::StartReduce, adotb, MPI_SUM) : get_adotb);
  return tl.AddTask(start_global_adotb, &AllReduce<Real>::CheckReduce, adotb);
}

TaskCollection PoissonDriver::MakeTaskCollection(BlockList_t &blocks) {
  return MakeTaskCollectionMG(blocks);
}

TaskCollection PoissonDriver::MakeTaskCollectionProRes(BlockList_t &blocks) {
  using namespace parthenon;
  using namespace poisson_package;
  TaskCollection tc;
  TaskID none(0);

  auto pkg = pmesh->packages.Get("poisson_package");
  auto max_iterations = pkg->Param<int>("max_iterations");

  const int num_partitions = pmesh->DefaultNumPartitions();
  int min_level = 0; // pmesh->GetGMGMaxLevel();
  int max_level = pmesh->GetGMGMaxLevel();

  TaskRegion &region = tc.AddRegion(num_partitions * (max_level + 1));
  for (int level = max_level; level >= min_level; --level) {
    AddRestrictionProlongationLevel(region, level, min_level, max_level);
  }

  return tc;
}

TaskCollection PoissonDriver::MakeTaskCollectionMG(BlockList_t &blocks) {
  using namespace parthenon;
  using namespace poisson_package;
  TaskCollection tc;
  TaskID none(0);

  auto pkg = pmesh->packages.Get("poisson_package");
  auto max_iterations = pkg->Param<int>("max_iterations");

  const int num_partitions = pmesh->DefaultNumPartitions();
  int min_level = 0; // pmesh->GetGMGMaxLevel();
  int max_level = pmesh->GetGMGMaxLevel();

  {
    TaskRegion &region = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; ++i) {
      TaskList &tl = region[i];
      auto &md = pmesh->gmg_mesh_data[max_level].GetOrAdd(max_level, "base", i);
      auto zero_u = tl.AddTask(none, SetToZero<u>, md);
    }
  }
  
  for (int ivcycle = 0; ivcycle < max_iterations; ++ivcycle) {
    TaskRegion &region = tc.AddRegion(num_partitions * (max_level + 1));
    for (int level = max_level; level >= min_level; --level) {
      AddMultiGridTasksLevel(region, level, min_level, max_level,
                               level == min_level);
    }

    int reg_dep_id = 0; 
    TaskRegion &region_res = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; ++i) {
      TaskList &tl = region_res[i];
      auto &md = pmesh->gmg_mesh_data[max_level].GetOrAdd(max_level, "base", i);
      auto calc_pointwise_res = AddResidualCalc(tl, none, md); 
      auto get_res = DotProduct<res_err, res_err>(calc_pointwise_res, region_res, tl, i, reg_dep_id, &residual, md);
      if (i == 0) {
        tl.AddTask(get_res, [&](PoissonDriver *driver){
          Real rms_err = std::sqrt(driver->residual.val / pmesh->GetTotalCells()); 
          printf("RMS residual: %e\n", rms_err); 
          return TaskStatus::complete;
        }, this);
      }
    }
  }

  return tc;
}

TaskCollection PoissonDriver::MakeTaskCollectionMGCG(BlockList_t &blocks) {
  using namespace parthenon;
  using namespace poisson_package;
  TaskCollection tc;
  TaskID none(0);

  auto pkg = pmesh->packages.Get("poisson_package");
  auto max_iterations = pkg->Param<int>("max_iterations");

  const int num_partitions = pmesh->DefaultNumPartitions();
  int min_level = 0; // pmesh->GetGMGMaxLevel();
  int max_level = pmesh->GetGMGMaxLevel();

  auto AddGMGRegion = [&](){
    TaskRegion &region = tc.AddRegion(num_partitions * (max_level + 1));
    for (int level = max_level; level >= min_level; --level) {
      AddMultiGridTasksLevel(region, level, min_level, max_level,
                             level == min_level);
    }
  };
  
  {
    TaskRegion &region = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; ++i) {
      TaskList &tl = region[i];
      auto &md = pmesh->gmg_mesh_data[max_level].GetOrAdd(max_level, "base", i);
      auto zero_x = tl.AddTask(none, SetToZero<x>, md);
      auto zero_u = tl.AddTask(none, SetToZero<u>, md);
      auto copy_r = tl.AddTask(none, CopyData<rhs, r>, md);
    }
  }

  AddGMGRegion(); 

  {
    int reg_dep_id = 0;
    TaskRegion &region = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; ++i) {
      TaskList &tl = region[i];
      auto &md = pmesh->gmg_mesh_data[max_level].GetOrAdd(max_level, "base", i);
      auto copy_u = tl.AddTask(none, CopyData<u, p>, md);
      auto get_rtr = DotProduct<u, r>(none, region, tl, i, reg_dep_id, &rtr, md);
      if (i == 0) { 
        tl.AddTask(get_rtr, [this](PoissonDriver *driver){
          driver->rtr_old = driver->rtr.val; 
          driver->rtr.val = 0.0;  
          driver->pAp.val = 0.0;  
          driver->residual.val = 0.0;
          return TaskStatus::complete;}, this);
      }  
    }
  }


  for (int ivcycle = 0; ivcycle < max_iterations; ++ivcycle) {
    {
      TaskRegion &region = tc.AddRegion(num_partitions); 
      int reg_dep_id = 0;
      for (int i = 0; i < num_partitions; ++i) {
        TaskList &tl = region[i];
        auto &md = pmesh->gmg_mesh_data[max_level].GetOrAdd(max_level, "base", i);
        auto get_Adotp = tl.AddTask(none, MultiplyMatrix<p, Adotp>, md);
        auto get_pap = DotProduct<p, Adotp>(get_Adotp, region, tl, i, reg_dep_id, &pAp, md);

        auto correct_x = tl.AddTask(get_pap, [](PoissonDriver *driver, std::shared_ptr<MeshData<Real>> &md){
          Real alpha = driver->rtr_old / driver->pAp.val; 
          return AddFieldsAndStore<x, p, x>(md, 1.0, alpha); 
        }, this, md); 

        auto correct_r = tl.AddTask(get_pap, [](PoissonDriver *driver, std::shared_ptr<MeshData<Real>> &md){
          Real alpha = driver->rtr_old / driver->pAp.val; 
          return AddFieldsAndStore<r, Adotp, r>(md, 1.0, -alpha); 
        }, this, md);

        auto get_res = DotProduct<r, r>(correct_r, region, tl, i, reg_dep_id, &residual, md);
        
        if (i == 0) {
          tl.AddTask(get_res, [&](PoissonDriver *driver){
            Real rms_err = std::sqrt(driver->residual.val / pmesh->GetTotalCells()); 
            printf("RMS residual: %e\n", rms_err); 
            return TaskStatus::complete;
          }, this);
        }
        tl.AddTask(correct_r, CopyData<r, rhs>, md); 
        tl.AddTask(none, SetToZero<u>, md);
      }
    }
    AddGMGRegion();
    {
      TaskRegion &region = tc.AddRegion(num_partitions); 
      int reg_dep_id = 0;
      for (int i = 0; i < num_partitions; ++i) {
        TaskList &tl = region[i];
        auto &md = pmesh->gmg_mesh_data[max_level].GetOrAdd(max_level, "base", i);
        auto get_rtr = DotProduct<u, r>(none, region, tl, i, reg_dep_id, &rtr, md);
        
        auto correct_p = tl.AddTask(get_rtr, [](PoissonDriver *driver, std::shared_ptr<MeshData<Real>> &md){
          Real beta = driver->rtr.val / driver->rtr_old; 
          return AddFieldsAndStore<u, p, p>(md, 1.0, beta); 
        }, this, md); 
        region.AddRegionalDependencies(reg_dep_id, i, correct_p);
        reg_dep_id++;
        if (i == 0) { 
          tl.AddTask(correct_p, [](PoissonDriver *driver){
            driver->rtr_old = driver->rtr.val; 
            driver->rtr.val = 0.0;  
            driver->pAp.val = 0.0;  
            driver->residual.val = 0.0;
            return TaskStatus::complete;}, this);
        } 
      }
    }
  }

  return tc;
}

} // namespace poisson_example
