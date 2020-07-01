//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
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

#include <iomanip>
#include <iostream>
#include <limits>

#include "poisson_driver.hpp"

using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;

namespace poisson {
TaskList PoissonDriver::MakeTaskList(MeshBlock *pmb) {
  TaskList tl;
  TaskID none(0);

  auto &base = pmb->real_containers.Get();
  pmb->real_containers.Add("update", base);
  auto &update = pmb->real_containers.Get("update");

  auto start_recv = tl.AddTask(Container<Real>::StartReceivingTask, none, update);

  auto smooth = tl.AddTask(Smooth, none, base, update);

  // update ghost cells
  auto send = tl.AddTask(Container<Real>::SendBoundaryBuffersTask, smooth, update);
  auto recv = tl.AddTask(Container<Real>::ReceiveBoundaryBuffersTask, send, update);
  auto fill_from_bufs = tl.AddTask(Container<Real>::SetBoundariesTask, recv, update);
  auto clear_comm_flags =
    tl.AddTask(Container<Real>::ClearBoundaryTask, fill_from_bufs, update);

  auto prolongBound = tl.AddTask(
      [](MeshBlock *pmb) {
        pmb->pbval->ProlongateBoundaries(0.0, 0.0);
        return TaskStatus::complete;
      },
      fill_from_bufs, pmb);

  // set physical boundaries
  auto set_bc =
    tl.AddTask(parthenon::ApplyBoundaryConditions, prolongBound, update);

  // fill in derived fields
  auto fill_derived =
    tl.AddTask(parthenon::FillDerivedVariables::FillDerived, set_bc, update);

  // swap containers
  auto swap = tl.AddTask(
      [](MeshBlock *pmb) {
        pmb->real_containers.Swap("base", "update");
        return TaskStatus::complete;
      },
      fill_derived,pmb);

  // Update refinement
  if (pmesh->adaptive) {
    auto tag_regine = tl.AddTask(
        [](MeshBlock *pmb) {
          pmb->pmr->CheckRefinementCondition();
          return TaskStatus::complete;
        },
        swap, pmb);
  }

  return tl;
}

void PoissonDriver::OutputCycleDiagnostics() {
  const int precision = std::numeric_limits<Real>::max_digits10 - 1;
  const int ratio_precision = 3;
  if ((ncycle_out > 0) && (ncycle % ncycle_out == 0) &&
      (parthenon::Globals::my_rank == 0)) {
    std::cout << "cycle=" << ncycle << std::scientific << std::setprecision(precision)
              << " reisidual=" << residual;

    // insert more diagnostics here
    std::cout << std::endl;
  }
  return;
}

}; // namespace poisson
