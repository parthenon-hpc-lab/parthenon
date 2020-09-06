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
#include "parthenon/driver.hpp"

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
TaskStatus UpdateContainer(std::vector<MeshBlock *> &blocks, const int stage,
                           std::vector<std::string> stage_name, Integrator *integrator) {
  // const Real beta = stage_wghts[stage-1].beta;
  const Real beta = integrator->beta[stage - 1];
  const Real dt = integrator->dt;
  parthenon::Update::AverageContainers(blocks, stage_name[stage - 1], "base", beta);
  parthenon::Update::UpdateContainer(blocks, stage_name[stage - 1], "dUdt", beta * dt,
                                     stage_name[stage]);
  return TaskStatus::complete;
}

// See the advection.hpp declaration for a description of how this function gets called.
TaskCollection AdvectionDriver::MakeTaskCollection(std::vector<MeshBlock *> &blocks,
                                                   const int stage) {
  TaskCollection tc;

  TaskID none(0);

  // Number of task lists that can be executed indepenently and thus *may*
  // be executed in parallel and asynchronous.
  // Being extra verbose here in this example to highlight that this is not
  // required to be 1 or blocks.size() but could also only apply to a subset of blocks.
  auto num_task_lists_executed_independently = blocks.size();
  TaskRegion &async_region1 = tc.AddRegion(num_task_lists_executed_independently);

  for (int i = 0; i < blocks.size(); i++) {
    auto *pmb = blocks[i];
    auto &tl = async_region1[i];
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

    auto start_recv = tl.AddTask(&Container<Real>::StartReceiving, sc1.get(), none,
                                 BoundaryCommSubset::all);

    auto advect_flux = tl.AddTask(advection_package::CalculateFluxes, none, sc0);

    auto send_flux =
        tl.AddTask(&Container<Real>::SendFluxCorrection, sc0.get(), advect_flux);
    auto recv_flux =
        tl.AddTask(&Container<Real>::ReceiveFluxCorrection, sc0.get(), advect_flux);
  }

  // note that task within this region that contains only a single task list
  // could still be executed in parallel
  TaskRegion &single_tasklist_region = tc.AddRegion(1);
  {
    auto &tl = single_tasklist_region[0];
    // compute the divergence of fluxes of conserved variables
    auto flux_div = tl.AddTask(parthenon::Update::FluxDivergenceMesh, none, blocks,
                               stage_name[stage - 1], "dUdt");
    // apply du/dt to all independent fields in the container
    auto update_container =
        tl.AddTask(UpdateContainer, flux_div, blocks, stage, stage_name, integrator);
  }
  TaskRegion &async_region2 = tc.AddRegion(num_task_lists_executed_independently);

  for (int i = 0; i < blocks.size(); i++) {
    auto *pmb = blocks[i];
    auto &tl = async_region2[i];
    auto &sc1 = pmb->real_containers.Get(stage_name[stage]);

    // update ghost cells
    auto send = tl.AddTask(&Container<Real>::SendBoundaryBuffers, sc1.get(), none);
    auto recv = tl.AddTask(&Container<Real>::ReceiveBoundaryBuffers, sc1.get(), send);
    auto fill_from_bufs = tl.AddTask(&Container<Real>::SetBoundaries, sc1.get(), recv);
    auto clear_comm_flags = tl.AddTask(&Container<Real>::ClearBoundary, sc1.get(),
                                       fill_from_bufs, BoundaryCommSubset::all);

    auto prolongBound = tl.AddTask(
        [](MeshBlock *pmb) {
          pmb->pbval->ProlongateBoundaries(0.0, 0.0);
          return TaskStatus::complete;
        },
        fill_from_bufs, pmb);

    // set physical boundaries
    auto set_bc = tl.AddTask(parthenon::ApplyBoundaryConditions, prolongBound, sc1);

    // fill in derived fields
    auto fill_derived =
        tl.AddTask(parthenon::FillDerivedVariables::FillDerived, set_bc, sc1);

    // estimate next time step
    if (stage == integrator->nstages) {
      auto new_dt = tl.AddTask(
          [](std::shared_ptr<Container<Real>> &rc) {
            MeshBlock *pmb = rc->pmy_block;
            pmb->SetBlockTimestep(parthenon::Update::EstimateTimestep(rc));
            return TaskStatus::complete;
          },
          fill_derived, sc1);

      // Update refinement
      if (pmesh->adaptive) {
        auto tag_refine = tl.AddTask(
            [](MeshBlock *pmb) {
              pmb->pmr->CheckRefinementCondition();
              return TaskStatus::complete;
            },
            fill_derived, pmb);
      }
    }
  }
  return tc;
}

} // namespace advection_example
