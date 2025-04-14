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

#include "helmholtz_package.hpp"
#include "linear_solver_driver.hpp"
#include "poisson_cell_package.hpp"
#include "poisson_nodal_package.hpp"

using namespace parthenon::driver::prelude;

namespace linear_solver_example {

parthenon::DriverStatus LinearSolverDriver::Execute() {
  using namespace parthenon;

  pouts->MakeOutputs(pmesh, pinput);
  ConstructAndExecuteTaskLists<>(this);
  pouts->MakeOutputs(pmesh, pinput);

  return DriverStatus::complete;
}

TaskCollection LinearSolverDriver::MakeTaskCollection(BlockList_t &blocks) {
  using namespace parthenon;
  TaskCollection tc;

  poisson_nodal_package::AddTaskRegion(tc, this);
  // poisson_cell_package::AddTaskRegion(tc, this);

  return tc;
}

} // namespace linear_solver_example
