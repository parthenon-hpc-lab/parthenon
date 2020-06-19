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
AdvectionDriver::AdvectionDriver(ParameterInput *pin, Mesh *pm)
    : MultiStageBlockTaskDriver(pin, pm) {
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
  Container<Real> &base = pmb->real_containers.Get();
  Container<Real> &cin = pmb->real_containers.Get(stage_name[stage - 1]);
  Container<Real> &cout = pmb->real_containers.Get(stage_name[stage]);
  Container<Real> &dudt = pmb->real_containers.Get("dUdt");
  parthenon::Update::AverageContainers(cin, base, beta);
  parthenon::Update::UpdateContainer(cin, dudt, beta * dt, cout);
  return TaskStatus::complete;
}

// See the advection.hpp declaration for a description of how this function gets called.
TaskList AdvectionDriver::MakeTaskList(MeshBlock *pmb, int stage) {
  TaskList tl;
  // we're going to populate our list with multiple kinds of tasks
  // these lambdas just clean up the interface to adding tasks of the relevant kinds
  auto AddMyTask = [&tl, pmb, stage, this](BlockStageNamesIntegratorTaskFunc func,
                                           TaskID dep) {
    return tl.AddTask<BlockStageNamesIntegratorTask>(func, dep, pmb, stage, stage_name,
                                                     integrator);
  };
  auto AddContainerTask = [&tl](ContainerTaskFunc func, TaskID dep, Container<Real> &rc) {
    return tl.AddTask<ContainerTask>(func, dep, rc);
  };
  auto AddTwoContainerTask = [&tl](TwoContainerTaskFunc f, TaskID dep,
                                   Container<Real> &rc1, Container<Real> &rc2) {
    return tl.AddTask<TwoContainerTask>(f, dep, rc1, rc2);
  };

  TaskID none(0);
  // first make other useful containers
  if (stage == 1) {
    Container<Real> &base = pmb->real_containers.Get();
    pmb->real_containers.Add("dUdt", base);
    for (int i = 1; i < integrator->nstages; i++)
      pmb->real_containers.Add(stage_name[i], base);
  }

  // pull out the container we'll use to get fluxes and/or compute RHSs
  Container<Real> &sc0 = pmb->real_containers.Get(stage_name[stage - 1]);
  // pull out a container we'll use to store dU/dt.
  // This is just -flux_divergence in this example
  Container<Real> &dudt = pmb->real_containers.Get("dUdt");
  // pull out the container that will hold the updated state
  // effectively, sc1 = sc0 + dudt*dt
  Container<Real> &sc1 = pmb->real_containers.Get(stage_name[stage]);

  auto start_recv = AddContainerTask(Container<Real>::StartReceivingTask, none, sc1);

  auto advect_flux = AddContainerTask(advection_package::CalculateFluxes, none, sc0);

  auto send_flux =
      AddContainerTask(Container<Real>::SendFluxCorrectionTask, advect_flux, sc0);
  auto recv_flux =
      AddContainerTask(Container<Real>::ReceiveFluxCorrectionTask, advect_flux, sc0);

  // compute the divergence of fluxes of conserved variables
  auto flux_div =
      AddTwoContainerTask(parthenon::Update::FluxDivergence, recv_flux, sc0, dudt);

  // apply du/dt to all independent fields in the container
  auto update_container = AddMyTask(UpdateContainer, flux_div);

  // update ghost cells
  auto send =
      AddContainerTask(Container<Real>::SendBoundaryBuffersTask, update_container, sc1);
  auto recv = AddContainerTask(Container<Real>::ReceiveBoundaryBuffersTask, send, sc1);
  auto fill_from_bufs = AddContainerTask(Container<Real>::SetBoundariesTask, recv, sc1);
  auto clear_comm_flags =
      AddContainerTask(Container<Real>::ClearBoundaryTask, fill_from_bufs, sc1);

  auto prolongBound = tl.AddTask<BlockTask>(
      [](MeshBlock *pmb) {
        pmb->pbval->ProlongateBoundaries(0.0, 0.0);
        return TaskStatus::complete;
      },
      fill_from_bufs, pmb);

  // set physical boundaries
  auto set_bc = AddContainerTask(parthenon::ApplyBoundaryConditions, prolongBound, sc1);

  // fill in derived fields
  auto fill_derived =
      AddContainerTask(parthenon::FillDerivedVariables::FillDerived, set_bc, sc1);

  // estimate next time step
  if (stage == integrator->nstages) {
    auto new_dt = AddContainerTask(
        [](Container<Real> &rc) {
          MeshBlock *pmb = rc.pmy_block;
          pmb->SetBlockTimestep(parthenon::Update::EstimateTimestep(rc));
          return TaskStatus::complete;
        },
        fill_derived, sc1);

    // Update refinement
    if (pmesh->adaptive) {
      auto tag_refine = tl.AddTask<BlockTask>(
          [](MeshBlock *pmb) {
            pmb->pmr->CheckRefinementCondition();
            return TaskStatus::complete;
          },
          fill_derived, pmb);
    }
  }
  return tl;
}

} // namespace advection_example
