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
#include "poisson_equation.hpp"
#include "poisson_package.hpp"
#include "prolong_restrict/prolong_restrict.hpp"
#include "solvers/bicgstab_solver.hpp"
#include "solvers/mg_solver.hpp"

using namespace parthenon::driver::prelude;

namespace poisson_example {

parthenon::DriverStatus PoissonDriver::Execute() {
  using namespace parthenon;
  using namespace poisson_package;
  
  pouts->MakeOutputs(pmesh, pinput);
  ConstructAndExecuteTaskLists<>(this);
  pouts->MakeOutputs(pmesh, pinput);
  
  // After running, retrieve the final residual for checking in tests
  auto pkg = pmesh->packages.Get("poisson_package");
  auto solver = pkg->Param<std::string>("solver");
  if (solver == "BiCGSTAB") {
    auto *bicgstab_solver =
      pkg->MutableParam<parthenon::solvers::BiCGSTABSolver<u, rhs, PoissonEquation>>(
          "MGBiCGSTABsolver");
    final_rms_residual = bicgstab_solver->GetFinalResidual();
  } else if (solver == "MG") {
    auto *mg_solver =
      pkg->MutableParam<parthenon::solvers::MGSolver<u, rhs, PoissonEquation>>(
          "MGsolver");
    final_rms_residual = mg_solver->GetFinalResidual();
  }

  return DriverStatus::complete;
}

TaskCollection PoissonDriver::MakeTaskCollection(BlockList_t &blocks) {
  using namespace parthenon;
  using namespace poisson_package;
  TaskCollection tc;
  TaskID none(0);

  auto pkg = pmesh->packages.Get("poisson_package");
  auto solver = pkg->Param<std::string>("solver");
  auto flux_correct = pkg->Param<bool>("flux_correct");
  auto use_exact_rhs = pkg->Param<bool>("use_exact_rhs");
  auto *mg_solver =
      pkg->MutableParam<parthenon::solvers::MGSolver<u, rhs, PoissonEquation>>(
          "MGsolver");
  auto *bicgstab_solver =
      pkg->MutableParam<parthenon::solvers::BiCGSTABSolver<u, rhs, PoissonEquation>>(
          "MGBiCGSTABsolver");

  const int num_partitions = pmesh->DefaultNumPartitions();
  TaskRegion &region = tc.AddRegion(num_partitions);
  int reg_dep_id = 0;
  for (int i = 0; i < num_partitions; ++i) {
    TaskList &tl = region[i];
    auto &itl = tl.AddIteration("Solver");
    auto &md = pmesh->mesh_data.GetOrAdd("base", i);

    // Possibly set rhs <- A.u_exact for a given u_exact so that the exact solution is 
    // known when we solve A.u = rhs
    auto get_rhs = none;
    if (use_exact_rhs) {
      auto copy_exact = tl.AddTask(none, solvers::utils::CopyData<exact, u>, md);
      auto comm = AddBoundaryExchangeTasks<BoundaryType::any>(copy_exact, tl, md, true);
      PoissonEquation eqs;
      eqs.do_flux_cor = flux_correct;
      auto get_rhs = eqs.Ax<u, rhs>(tl, comm, md);
    }

    // Set initial solution guess to zero
    auto zero_u = tl.AddTask(get_rhs, solvers::utils::SetToZero<u>, md);
    
    // Add actual tasks for solving our matrix equation, depending on the solver
    // type requested. Note that the added tasks are iterative.
    auto solve = zero_u;
    if (solver == "BiCGSTAB") {
      solve = bicgstab_solver->AddTasks(tl, itl, zero_u, i, pmesh, region, reg_dep_id);
    } else if (solver == "MG") {
      solve = mg_solver->AddTasks(tl, itl, zero_u, i, pmesh, region, reg_dep_id);
    } else {
      PARTHENON_FAIL("Unknown solver type.");
    }

    // If we are using a rhs to which we know the exact solution, compare our computed
    // solution to the exact solution
    if (use_exact_rhs) {
      auto diff = tl.AddTask(solve, solvers::utils::AddFieldsAndStore<exact, u, u>, md,
                             1.0, -1.0);
      auto get_err =
          solvers::utils::DotProduct<u, u>(diff, region, tl, i, reg_dep_id, &err, md);
      tl.AddTask(
          get_err,
          [](PoissonDriver *driver, int partition) {
            if (partition != 0) return TaskStatus::complete;
            driver->final_rms_error =
                std::sqrt(driver->err.val / driver->pmesh->GetTotalCells());
            printf("Final rms error: %e\n", driver->final_rms_error);
            return TaskStatus::complete;
          },
          this, i);
    }
  }

  return tc;
}

} // namespace poisson_example
