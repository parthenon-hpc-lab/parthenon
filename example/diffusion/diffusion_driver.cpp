//========================================================================================
// (C) (or copyright) 2023-2025. Triad National Security, LLC. All rights reserved.
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
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

// Local Includes
#include "amr_criteria/refinement_package.hpp"
#include "basic_types.hpp"
#include "bvals/comms/bvals_in_one.hpp"
#include "diffusion_driver.hpp"
#include "diffusion_equation.hpp"
#include "diffusion_hypre.hpp"
#include "diffusion_package.hpp"
#include "interface/metadata.hpp"
#include "interface/update.hpp"
#include "mesh/meshblock_pack.hpp"
#include "parthenon/driver.hpp"
#include "prolong_restrict/prolong_restrict.hpp"
#include "solvers/bicgstab_solver.hpp"
#include "solvers/cg_solver.hpp"
#include "solvers/mg_solver.hpp"
#include "solvers/solver_utils.hpp"
#include "utils/error_checking.hpp"

using namespace parthenon::driver::prelude;

namespace diffusion_example {

TaskListStatus DiffusionDriver::Step() {
  TaskListStatus status;

  PARTHENON_REQUIRE(integrator.nstages == 1,
                    "Only first order time integration supported!");

  BlockList_t &blocks = pmesh->block_list;
  auto num_task_lists_executed_independently = blocks.size();
  status = MakeTaskCollection().Execute();
  return status;
} // Step

TaskCollection DiffusionDriver::MakeTaskCollection() {
  using namespace parthenon;
  using namespace diffusion_package;
  TaskCollection tc;
  TaskID none(0);

  auto pkg = pmesh->packages.Get("diffusion_package");
  const auto use_hypre = pkg->Param<bool>("use_hypre");
  if (use_hypre) {
    return MakeTaskCollectionHypre();
  } else {
    return MakeTaskCollectionNative();
  }
}

TaskCollection DiffusionDriver::MakeTaskCollectionNative() {
  using namespace parthenon;
  using namespace diffusion_package;
  TaskCollection tc;
  TaskID none(0);

  auto pkg = pmesh->packages.Get("diffusion_package");
  auto psolver =
      pkg->Param<std::shared_ptr<parthenon::solvers::SolverBase>>("solver_pointer");
  const auto alpha = pkg->Param<Real>("diagonal_alpha");
  const auto rel_res = pkg->Param<Real>("rel_res");
  auto peqs = pkg->Param<std::shared_ptr<diffusion_package::DiffusionEquation<u, D>>>(
      "diffusion_equation");

  auto partitions = pmesh->GetDefaultBlockPartitions();
  const int num_partitions = partitions.size();
  TaskRegion &region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; ++i) {
    TaskList &tl = region[i];
    auto &md = pmesh->mesh_data.Add("base", partitions[i]);

    // SetDiffusionCoefficient
    auto set_d = tl.AddTask(none, TF(SetDiffusionCoefficient), md, tm.dt);

    auto &md_deltau = psolver->AddSolutionMeshData(pmesh, md, /*shallow=*/false);
    auto &md_rhs = psolver->AddRHSMeshData(pmesh, md);

    // Set the rhs
    // We are solving for Δu using  (alpha - dt ∇ D ∇) Δu = (dt ∇ D ∇) u_old. The
    // diffusion_equation class defines the operator A = alpha - dt ∇ D ∇, so that
    // we have A Δu = rhs with rhs = (alpha - A) u_old.
    auto comm =
        AddBoundaryExchangeTasks<BoundaryType::any>(none, tl, md, pmesh->multilevel);
    auto Au = peqs->Ax(tl, comm | set_d, md, md, md_rhs);
    auto set_rhs =
        tl.AddTask(Au, solvers::utils::AddFieldsAndStore<parthenon::TypeList<u>>, md,
                   md_rhs, md_rhs, alpha, -1.0);

    // Get the RHS scale for correct comparison to Hypre solver
    set_rhs = solvers::utils::DotProduct<parthenon::TypeList<u>>(set_rhs, tl, &u2, md, md,
                                                                 true);

    set_rhs = tl.AddTask(
        set_rhs,
        [alpha](parthenon::AllReduce<Real> *u2,
                std::shared_ptr<parthenon::solvers::SolverBase> psolver, Real rel_res) {
          *(psolver->absolute_residual_tolerance) = alpha * rel_res * sqrt(u2->val);
          return parthenon::TaskStatus::complete;
        },
        &u2, psolver, rel_res);

    // Set initial solution guess to zero
    auto zero_u = tl.AddTask(set_rhs, TF(solvers::utils::SetToZero<u>), md_deltau);
    psolver->initial_guess_is_zero = false;
    auto setup = psolver->AddSetupTasks(tl, zero_u, i, pmesh);
    auto solve = psolver->AddTasks(tl, setup, i, pmesh);

    // Update to u = u_0 + Δu
    auto update_u =
        tl.AddTask(solve, solvers::utils::AddFieldsAndStore<parthenon::TypeList<u>>, md,
                   md_deltau, md, 1.0, 1.0);

    // Update the timestep
    tl.AddTask(update_u, parthenon::Update::EstimateTimestep<MeshData<Real>>, md.get());
  }
  return tc;
}
TaskCollection DiffusionDriver::MakeTaskCollectionHypre() {
  using namespace parthenon;
  using namespace diffusion_package;
  TaskCollection tc;
  TaskID none(0);

#ifdef DIFFUSION_WITH_HYPRE

  auto pkg = pmesh->packages.Get("diffusion_package");
  auto hypre_solver = pkg->Param<std::shared_ptr<HypreSolver>>("hypre_solver");

  TaskRegion &grid_region = tc.AddRegion(1);
  grid_region[0].AddTask(
      none,
      [](HypreSolver *solver, parthenon::Mesh *pmesh) {
        if (pmesh->modified || solver->needs_grid_setup || !solver->grid_is_setup) {
          solver->DestroyGrid();
          solver->SetupGrid(pmesh);
        }
        return TaskStatus::complete;
      },
      hypre_solver.get(), pmesh);

  auto partitions = pmesh->GetDefaultBlockPartitions();
  const int num_partitions = partitions.size();
  TaskRegion &region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; ++i) {
    TaskList &tl = region[i];
    auto &md = pmesh->mesh_data.Add("base", partitions[i]);

    auto start_fluxcor = tl.AddTask(none, parthenon::StartReceiveFluxCorrections, md);

    // SetDiffusionCoefficient
    auto set_d = tl.AddTask(none, TF(SetDiffusionCoefficientHypre), md, tm.dt);

    auto set_fluxcor = parthenon::AddFluxCorrectionTasks(set_d | start_fluxcor, tl, md,
                                                         pmesh->multilevel);
  }

  auto &blocks = pmesh->block_list;
  TaskRegion &build_matrix_region = tc.AddRegion(blocks.size());
  for (int i = 0; i < blocks.size(); i++) {
    auto &tl = build_matrix_region[i];
    auto &pmb = blocks[i];
    auto build_block = tl.AddTask(none, TF(HypreSolver::BuildMatrixVector),
                                  hypre_solver.get(), i, pmb.get(), integrator.dt);
    // probably have a task for setting RHS and initial guess
  }

  TaskRegion &solve_region = tc.AddRegion(1);
  auto solve = solve_region[0].AddTask(none, TF(HypreSolver::Solve), hypre_solver.get());

  TaskRegion &update_region = tc.AddRegion(blocks.size());
  for (int i = 0; i < blocks.size(); ++i) {
    TaskList &tl = update_region[i];
    auto &pmb = blocks[i];
    auto update_block = tl.AddTask(none, TF(HypreSolver::UpdateSolution),
                                   hypre_solver.get(), i, pmb.get());
  }

  TaskRegion &dt_region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; ++i) {
    TaskList &tl = dt_region[i];
    auto &md = pmesh->mesh_data.Add("base", partitions[i]);

    // Update the timestep
    tl.AddTask(none, parthenon::Update::EstimateTimestep<MeshData<Real>>, md.get());
  }

#endif // DIFFUSION_WITH_HYPRE
  return tc;
}

} // namespace diffusion_example
