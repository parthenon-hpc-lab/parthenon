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

void PoissonDriver::AddMultiGridTasks(TaskCollection &tc, int level, int max_level) {
  using namespace parthenon;
  TaskID none(0);
  const int num_partitions = pmesh->DefaultNumPartitions();

  printf("Building level %i restriction.\n", level);
  TaskRegion &pre_region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; ++i) {
    TaskList &tl = pre_region[i];
    auto &md = pmesh->gmg_mesh_data[level].GetOrAdd(level, "base", i);
    printf("level = %i meshdata size = %i\n", level, md->NumBlocks());
    auto set_from_finer = none;
    if (level < max_level) {
      // Fill fields with restricted values
      auto recv_from_finer =
          tl.AddTask(none, ReceiveBoundBufs<BoundaryType::gmg_restrict_recv>, md);
      set_from_finer =
          tl.AddTask(recv_from_finer, SetBounds<BoundaryType::gmg_restrict_recv>, md);
    }
    // 1. Copy residual from dual purpose communication field, copy actual RHS for finest
    // level
    // 2. Do pre-smooth and fill solution on this level
    // 3. Communicate same level boundaries
    // 4. Caclulate residual and store in communication field
    // 5. Restrict communication field and send to next level
    // 6. Receive error field into communication field and prolongate
    // 7. Correct solution on this level with communication field and store in
    // communication field
    // 8. Post smooth using communication field and stored RHS
    // 9. Send communication field to next finer level (should be error field for that
    // level)

    // Communicate boundaries
    auto communicate_bounds = AddBoundaryExchangeTasks(set_from_finer, tl, md, true);

    // Apply pre-smoother
    auto smooth = tl.AddTask(communicate_bounds, poisson_package::PrintValues, md);

    // Communicate boundaries
    auto communicate_bounds_2 = AddBoundaryExchangeTasks(smooth, tl, md, true);

    // Calculate residual (do this in ghosts as well)
    auto residual = communicate_bounds_2;

    // Restrict residual (and others) to next coarser grid
    if (level > 0) {
      auto communicate_to_coarse =
          tl.AddTask(residual, SendBoundBufs<BoundaryType::gmg_restrict_send>, md);
      tl.AddTask(communicate_to_coarse, poisson_package::SetToZero, md);
    }
  }

  // Call recursive multi grid
  if (level > 0) AddMultiGridTasks(tc, level - 1, max_level);

  printf("Building level %i prolongation.\n", level);
  TaskRegion &post_region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; ++i) {
    TaskList &tl = post_region[i];
    auto &md = pmesh->gmg_mesh_data[level].GetOrAdd(level, "base", i);
    auto smooth = none;
    if (level > 0) {
      // Fill fields with prolongated values
      auto recv_from_coarser =
          tl.AddTask(none, ReceiveBoundBufs<BoundaryType::gmg_prolongate_recv>, md);
      auto set_from_coarser =
          tl.AddTask(recv_from_coarser, SetBounds<BoundaryType::gmg_prolongate_recv>, md);
      auto prolongate = tl.AddTask(
          set_from_coarser, ProlongateBounds<BoundaryType::gmg_prolongate_recv>, md);

      // Apply error to residual
      auto apply_error = prolongate;

      // Communicate boundaries
      auto communicate_bounds = AddBoundaryExchangeTasks(apply_error, tl, md, true);

      // Apply post-smoother
      auto smooth = tl.AddTask(communicate_bounds, poisson_package::PrintValues, md);
    }
    // Send values to the next coarser grid to be prolongated
    if (level < max_level)
      tl.AddTask(smooth, SendBoundBufs<BoundaryType::gmg_prolongate_send>, md);
  }
}

TaskCollection PoissonDriver::MakeTaskCollection(BlockList_t &blocks) {
  using namespace parthenon;
  TaskCollection tc;
  TaskID none(0);

  int max_level = pmesh->GetGMGMaxLevel();
  AddMultiGridTasks(tc, max_level, max_level);

  return tc;
}

} // namespace poisson_example
