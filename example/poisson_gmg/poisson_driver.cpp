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
  pouts->MakeOutputs(pmesh, pinput);
  ConstructAndExecuteTaskLists<>(this);
  pouts->MakeOutputs(pmesh, pinput);
  return DriverStatus::complete;
}

TaskCollection PoissonDriver::MakeTaskCollection(BlockList_t &blocks) {
  using namespace parthenon;
  using namespace poisson_package;
  TaskCollection tc;
  TaskID none(0);

  auto pkg = pmesh->packages.Get("poisson_package");
  auto solver = pkg->Param<std::string>("solver");
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
    // auto copy_exact = tl.AddTask(none, CopyData<exact, u>, md);
    // auto comm = AddBoundaryExchangeTasks<BoundaryType::any>(copy_exact, tl, md, true);
    // auto get_rhs = Axpy<u, u, rhs>(tl, comm, md, 1.0, 0.0, false, false);
    auto zero_u = tl.AddTask(none, solvers::utils::SetToZero<u>, md);
    if (solver == "BiCGSTAB") {
      bicgstab_solver->AddTasks(tl, itl, zero_u, i, pmesh, region, reg_dep_id);
    } else if (solver == "MG") {
      mg_solver->AddTasks(itl, zero_u, i, pmesh, region, reg_dep_id);
    } else {
      PARTHENON_FAIL("Unknown solver type.");
    }
  }

  return tc;
}

} // namespace poisson_example
