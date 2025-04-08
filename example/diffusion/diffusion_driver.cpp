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
#include "bvals/comms/bvals_in_one.hpp"
#include "diffusion_driver.hpp"
#include "diffusion_equation.hpp"
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
  auto psolver =
      pkg->Param<std::shared_ptr<parthenon::solvers::SolverBase>>("solver_pointer");

  auto partitions = pmesh->GetDefaultBlockPartitions();
  const int num_partitions = partitions.size();
  TaskRegion &region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; ++i) {
    TaskList &tl = region[i];
    auto &md = pmesh->mesh_data.Add("base", partitions[i]);

    // SetDiffusionCoefficient
    auto set_d = tl.AddTask(none, TF(SetDiffusionCoefficient), md, tm.dt);

    auto &md_u = pmesh->mesh_data.Add("u", md, {u::name()});
    auto &md_rhs = pmesh->mesh_data.Add("rhs", md, {u::name()});

    // SetRHS
    auto set_rhs = tl.AddTask(set_d, (SetRHS), md, md_rhs);

    // Set initial solution guess to zero
    auto zero_u = tl.AddTask(set_rhs, TF(solvers::utils::SetToZero<u>), md_u);
    auto setup = psolver->AddSetupTasks(tl, zero_u, i, pmesh);
    auto solve = psolver->AddTasks(tl, setup, i, pmesh);
    auto copy_back =
        tl.AddTask(solve, TF(solvers::utils::CopyData<parthenon::TypeList<u>>), md_u, md);

    auto new_dt = tl.AddTask(
        copy_back, parthenon::Update::EstimateTimestep<MeshData<Real>>, md.get());
  }
  return tc;
}

} // namespace diffusion_example
