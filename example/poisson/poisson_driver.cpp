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

  // weighted Jacobi is:
  // next = base + w*residual/D
  // with residual = K*potential - div(grad(field)) and D = diagonal component
  auto &base = pmb->real_containers.Get(); // last iteration
  // intermediate container to hold flux divergences
  pmb->real_containers.Add("div", base);
  auto &div = pmb->real_containers.Get("div");
  pmb->real_containers.Add("update", base); // next iteration
  auto &update = pmb->real_containers.Get("update");

  auto start_recv = tl.AddTask(&Container<Real>::StartReceiving, update.get(), none,
                               BoundaryCommSubset::all);

  // calculate flux psi = grad(phi)
  auto calc_flux = tl.AddTask(poisson::CalculateFluxes, none, base);

  // flux correction
  auto send_flux =
      tl.AddTask(&Container<Real>::SendFluxCorrection, base.get(), calc_flux);
  auto recv_flux =
      tl.AddTask(&Container<Real>::ReceiveFluxCorrection, base.get(), calc_flux);

  // flux divergence
  auto flux_div = tl.AddTask(parthenon::Update::FluxDivergence, recv_flux, base, div);

  // compute residual and D^{-1}
  auto compute_residual =
    tl.AddTask(ComputeResidualAndDiagonal, flux_div, div, update);

  // u^{n+1}_{i,j,k} = u^{n}_{i,j,k} + w*((1/D^n_{i,j,k}) * residual^{n}_{i,j,k})
  // depends only on state from previous step
  auto smooth = tl.AddTask(Smooth, compute_residual, base, update);

  // update ghost cells
  auto send = tl.AddTask(&Container<Real>::SendBoundaryBuffers, update.get(), smooth);
  auto recv = tl.AddTask(&Container<Real>::ReceiveBoundaryBuffers, update.get(), send);
  auto fill_from_bufs = tl.AddTask(&Container<Real>::SetBoundaries, update.get(), recv);
  auto clear_comm_flags = tl.AddTask(&Container<Real>::ClearBoundary, update.get(),
                                     fill_from_bufs, BoundaryCommSubset::all);

  // prolongation and restriction
  auto prolongBound = tl.AddTask(
      [](MeshBlock *pmb) {
        pmb->pbval->ProlongateBoundaries(0.0, 0.0);
        return TaskStatus::complete;
      },
      fill_from_bufs, pmb);

  // set physical boundaries
  auto set_bc = tl.AddTask(parthenon::ApplyBoundaryConditions, prolongBound, update);

  // fill in derived fields
  auto fill_derived = tl.AddTask(parthenon::FillDerivedVariables::FillDerived,
                                 set_bc, update);

  // swap containers
  auto swap = tl.AddTask(
      [](MeshBlock *pmb) {
        pmb->real_containers.Swap("base", "update");
        return TaskStatus::complete;
      },
      fill_derived, pmb);

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
