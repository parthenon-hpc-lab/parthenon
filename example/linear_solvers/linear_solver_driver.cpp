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
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

// Local Includes
#include <amr_criteria/refinement_package.hpp>
#include <bvals/comms/bvals_in_one.hpp>
#include <interface/metadata.hpp>
#include <interface/update.hpp>
#include <mesh/meshblock_pack.hpp>
#include <parthenon/driver.hpp>
#include <prolong_restrict/prolong_restrict.hpp>
#include <solvers/bicgstab_solver.hpp>
#include <solvers/cg_solver.hpp>
#include <solvers/mg_solver.hpp>
#include <solvers/solver_utils.hpp>

#include "helmholtz_equation.hpp"
#include "helmholtz_package.hpp"
#include "linear_solver_driver.hpp"
#include "poisson_nodal_equation.hpp"
#include "poisson_nodal_package.hpp"

using namespace parthenon::driver::prelude;

namespace linear_solver_example {

parthenon::DriverStatus LinearSolverDriver::Execute() {
  using namespace parthenon;
  using namespace poisson_nodal_package;

  pouts->MakeOutputs(pmesh, pinput);
  ConstructAndExecuteTaskLists<>(this);
  pouts->MakeOutputs(pmesh, pinput);

  // After running, retrieve the final residual for checking in tests
  auto pkg = pmesh->packages.Get("poisson_nodal_package");
  auto psolver =
      pkg->Param<std::shared_ptr<parthenon::solvers::SolverBase>>("solver_pointer");
  final_rms_residual = psolver->GetFinalResidual();

  return DriverStatus::complete;
}

TaskCollection LinearSolverDriver::MakeTaskCollection(BlockList_t &blocks) {
  using namespace parthenon;
  using namespace poisson_nodal_package;
  TaskCollection tc;
  TaskID none(0);

  auto pkg = pmesh->packages.Get("poisson_nodal_package");
  auto use_exact_rhs = pkg->Param<bool>("use_exact_rhs");
  auto psolver =
      pkg->Param<std::shared_ptr<parthenon::solvers::SolverBase>>("solver_pointer");

  auto partitions = pmesh->GetDefaultBlockPartitions();
  const int num_partitions = partitions.size();
  TaskRegion &region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; ++i) {
    TaskList &tl = region[i];
    auto &md = pmesh->mesh_data.Add("base", partitions[i]);
    auto &md_u = pmesh->mesh_data.Add("u", md, {u::name()});
    auto &md_rhs = pmesh->mesh_data.Add("rhs", md, {u::name()});
    auto &md_exact = pmesh->mesh_data.Add("exact", md, {u::name()});

    // set the rhs
    auto set_rhs = tl.AddTask(none, SetVector, pinput, false, md_rhs);

    // Possibly set rhs <- A.u_exact for a given u_exact so that the exact solution is
    // known when we solve A.u = rhs
    if (use_exact_rhs) {
      auto set_exact = tl.AddTask(set_rhs, SetVector, pinput, true, md_exact);
      auto comm =
          AddBoundaryExchangeTasks<BoundaryType::any>(set_exact, tl, md_exact, true);
      auto *eqs = pkg->MutableParam<poisson_nodal_package::PoissonEquation<u>>(
          "poisson_nodal_equation");
      set_rhs = eqs->Ax(tl, comm, md, md_exact, md_rhs);
    }

    // Set initial solution guess to zero
    auto zero_u = tl.AddTask(set_rhs, TF(solvers::utils::SetToZero<u>), md_u);
    auto setup = psolver->AddSetupTasks(tl, zero_u, i, pmesh);
    auto solve = psolver->AddTasks(tl, setup, i, pmesh);

    // If we are using a rhs to which we know the exact solution, compare our computed
    // solution to the exact solution
    if (use_exact_rhs) {
      auto diff = tl.AddTask(solve, solvers::utils::AddFieldsAndStore<TypeList<u>>,
                             md_exact, md_u, md_exact, 1.0, -1.0);
      auto get_err =
          solvers::utils::DotProduct<TypeList<u>>(diff, tl, &err, md_exact, md_exact);
      tl.AddTask(
          get_err,
          [](LinearSolverDriver *driver, int partition) {
            if (partition != 0) return TaskStatus::complete;
            driver->final_rms_error =
                std::sqrt(driver->err.val / driver->pmesh->GetTotalCells());
            if (Globals::my_rank == 0)
              printf("Final rms error: %e\n", driver->final_rms_error);
            return TaskStatus::complete;
          },
          this, i);
    }
  }
  return tc;
}

} // namespace linear_solver_example
