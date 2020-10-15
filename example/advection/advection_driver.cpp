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

#include <memory>
#include <string>
#include <vector>

// Local Includes
#include "advection_driver.hpp"
#include "advection_package.hpp"

using namespace parthenon::driver::prelude;

namespace advection_example {

// *************************************************//
// define the application driver. in this case,    *//
// that mostly means defining the MakeTaskList     *//
// function.                                       *//
// *************************************************//
AdvectionDriver::AdvectionDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm)
    : MultiStageBlockTaskDriver(pin, app_in, pm) {
  // fail if these are not specified in the input file
  pin->CheckRequired("parthenon/mesh", "ix1_bc");
  pin->CheckRequired("parthenon/mesh", "ox1_bc");
  pin->CheckRequired("parthenon/mesh", "ix2_bc");
  pin->CheckRequired("parthenon/mesh", "ox2_bc");

  // warn if these fields aren't specified in the input file
  pin->CheckDesired("parthenon/mesh", "refinement");
  pin->CheckDesired("parthenon/mesh", "numlevel");
  pin->CheckDesired("Advection", "cfl");
  pin->CheckDesired("Advection", "vx");
  pin->CheckDesired("Advection", "refine_tol");
  pin->CheckDesired("Advection", "derefine_tol");
}
// first some helper tasks
TaskStatus UpdateContainer(MeshBlock *pmb, int stage,
                           std::vector<std::string> &stage_name, Integrator *integrator) {
  // const Real beta = stage_wghts[stage-1].beta;
  const Real beta = integrator->beta[stage - 1];
  const Real dt = integrator->dt;
  auto &base = pmb->real_containers.Get();
  auto &cin = pmb->real_containers.Get(stage_name[stage - 1]);
  auto &cout = pmb->real_containers.Get(stage_name[stage]);
  auto &dudt = pmb->real_containers.Get("dUdt");
  parthenon::Update::AverageContainers(cin, base, beta);
  parthenon::Update::UpdateContainer(cin, dudt, beta * dt, cout);
  return TaskStatus::complete;
}

// See the advection.hpp declaration for a description of how this function gets called.
TaskList AdvectionDriver::MakeTaskList(MeshBlock *pmb, int stage) {
  TaskList tl;

  TaskID none(0);
  // first make other useful containers
  if (stage == 1) {
    auto &base = pmb->real_containers.Get();
    pmb->real_containers.Add("dUdt", base);
    for (int i = 1; i < integrator->nstages; i++)
      pmb->real_containers.Add(stage_name[i], base);
  }

  // pull out the container we'll use to get fluxes and/or compute RHSs
  auto &sc0 = pmb->real_containers.Get(stage_name[stage - 1]);
  // pull out a container we'll use to store dU/dt.
  // This is just -flux_divergence in this example
  auto &dudt = pmb->real_containers.Get("dUdt");
  // pull out the container that will hold the updated state
  // effectively, sc1 = sc0 + dudt*dt
  auto &sc1 = pmb->real_containers.Get(stage_name[stage]);

  auto start_recv = tl.AddTask(none, &Container<Real>::StartReceiving, sc1.get(),
                               BoundaryCommSubset::all);

  auto advect_flux = tl.AddTask(none, advection_package::CalculateFluxes, sc0);

  auto send_flux =
      tl.AddTask(advect_flux, &Container<Real>::SendFluxCorrection, sc0.get());
  auto recv_flux =
      tl.AddTask(advect_flux, &Container<Real>::ReceiveFluxCorrection, sc0.get());

  // compute the divergence of fluxes of conserved variables
  auto flux_div = tl.AddTask(recv_flux, parthenon::Update::FluxDivergence, sc0, dudt);

  // apply du/dt to all independent fields in the container
  auto update_container =
      tl.AddTask(flux_div, UpdateContainer, pmb, stage, stage_name, integrator);

  // update ghost cells
  auto send =
      tl.AddTask(update_container, &Container<Real>::SendBoundaryBuffers, sc1.get());
  auto recv = tl.AddTask(send, &Container<Real>::ReceiveBoundaryBuffers, sc1.get());
  auto fill_from_bufs = tl.AddTask(recv, &Container<Real>::SetBoundaries, sc1.get());
  auto clear_comm_flags = tl.AddTask(fill_from_bufs, &Container<Real>::ClearBoundary,
                                     sc1.get(), BoundaryCommSubset::all);

  auto prolong_bound = tl.AddTask(
      fill_from_bufs,
      [](MeshBlock *pmb) {
        pmb->pbval->ProlongateBoundaries(0.0, 0.0);
        return TaskStatus::complete;
      },
      pmb);

  // set physical boundaries
  auto set_bc = tl.AddTask(prolong_bound, parthenon::ApplyBoundaryConditions, sc1);

  // fill in derived fields
  auto fill_derived =
      tl.AddTask(set_bc, parthenon::FillDerivedVariables::FillDerived, sc1);

  // estimate next time step
  if (stage == integrator->nstages) {
    auto new_dt = tl.AddTask(
        fill_derived,
        [](std::shared_ptr<Container<Real>> &rc) {
          auto pmb = rc->GetBlockPointer();
          pmb->SetBlockTimestep(parthenon::Update::EstimateTimestep(rc));
          return TaskStatus::complete;
        },
        sc1);

    // Update refinement
    if (pmesh->adaptive) {
      auto tag_refine = tl.AddTask(
          fill_derived,
          [](MeshBlock *pmb) {
            pmb->pmr->CheckRefinementCondition();
            return TaskStatus::complete;
          },
          pmb);
    }
  }
  return tl;
}

} // namespace advection_example
