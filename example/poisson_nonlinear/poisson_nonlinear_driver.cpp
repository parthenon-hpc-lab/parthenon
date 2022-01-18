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

#include <solvers/cg_solver.hpp>
#include <solvers/newton_krylov.hpp>

// Local Includes
#include "bvals/cc/bvals_cc_in_one.hpp"
#include "interface/metadata.hpp"
#include "interface/update.hpp"
#include "mesh/meshblock_pack.hpp"
#include "mesh/refinement_cc_in_one.hpp"
#include "parthenon/driver.hpp"
#include "poisson_nonlinear_driver.hpp"
#include "poisson_nonlinear_package.hpp"
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
  using namespace parthenon::solvers;
  TaskCollection tc;
  TaskID none(0);

  auto psn_pkg = pmesh->packages.Get("poisson_package");
  auto poisson_solver = psn_pkg->Param<
      std::shared_ptr<NewtonKrylov<CG_Solver<SparseMatrixAccessor>, MeshData<Real>>>>(
      "PoissonSolver");
  std::string solver_name;
  std::vector<std::string> solver_vec_names;
  solver_name = poisson_solver->label();
  solver_vec_names = poisson_solver->SolverState();

  for (int i = 0; i < blocks.size(); i++) {
    auto &pmb = blocks[i];
    auto &base = pmb->meshblock_data.Get();
    pmb->meshblock_data.Add(solver_name, base, solver_vec_names);
  }

  const int num_partitions = pmesh->DefaultNumPartitions();
  TaskRegion &solver_region = tc.AddRegion(num_partitions);

  for (int i = 0; i < num_partitions; i++) {
    TaskList &tl = solver_region[i];
    // make/get a mesh_data container for the state
    auto &base = pmesh->mesh_data.GetOrAdd("base", i);
    auto &md = pmesh->mesh_data.GetOrAdd(solver_name, i);

    // create task list for solver.
    auto poisson_complete =
        poisson_solver->createTaskList(none, i, solver_region, base, md);

    auto print = none;
    if (i == 0) { // only print once
      print = tl.AddTask(poisson_complete, poisson_package::PrintComplete);
    }
  }

  return tc;
}

} // namespace poisson_example
