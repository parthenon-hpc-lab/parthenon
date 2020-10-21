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
#include "bvals/cc/bvals_cc_in_one.hpp"
#include "interface/metadata.hpp"
#include "mesh/meshblock_pack.hpp"
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
TaskStatus UpdateContainer(const int stage, Integrator *integrator,
                           std::shared_ptr<parthenon::MeshData<Real>> &in,
                           std::shared_ptr<parthenon::MeshData<Real>> &base,
                           std::shared_ptr<parthenon::MeshData<Real>> &dudt,
                           std::shared_ptr<parthenon::MeshData<Real>> &out) {
  const Real beta = integrator->beta[stage - 1];
  const Real dt = integrator->dt;
  parthenon::Update::AverageContainers(in, base, beta);
  parthenon::Update::UpdateContainer(in, dudt, beta * dt, out);
  return TaskStatus::complete;
}

// See the advection.hpp declaration for a description of how this function gets called.
TaskCollection AdvectionDriver::MakeTaskCollection(BlockList_t &blocks, const int stage) {
  TaskCollection tc;

  TaskID none(0);

  // Number of task lists that can be executed indepenently and thus *may*
  // be executed in parallel and asynchronous.
  // Being extra verbose here in this example to highlight that this is not
  // required to be 1 or blocks.size() but could also only apply to a subset of blocks.
  auto num_task_lists_executed_independently = blocks.size();
  TaskRegion &async_region1 = tc.AddRegion(num_task_lists_executed_independently);

  for (int i = 0; i < blocks.size(); i++) {
    auto &pmb = blocks[i];
    auto &tl = async_region1[i];
    // first make other useful containers
    if (stage == 1) {
      auto &base = pmb->meshblock_data.Get();
      pmb->meshblock_data.Add("dUdt", base);
      for (int i = 1; i < integrator->nstages; i++)
        pmb->meshblock_data.Add(stage_name[i], base);
    }

    // pull out the container we'll use to get fluxes and/or compute RHSs
    auto &sc0 = pmb->meshblock_data.Get(stage_name[stage - 1]);
    // pull out a container we'll use to store dU/dt.
    // This is just -flux_divergence in this example
    auto &dudt = pmb->meshblock_data.Get("dUdt");
    // pull out the container that will hold the updated state
    // effectively, sc1 = sc0 + dudt*dt
    auto &sc1 = pmb->meshblock_data.Get(stage_name[stage]);

    auto start_recv = tl.AddTask(none, &MeshBlockData<Real>::StartReceiving, sc1.get(),
                                 BoundaryCommSubset::all);

    auto advect_flux = tl.AddTask(none, advection_package::CalculateFluxes, sc0);

    auto send_flux =
        tl.AddTask(advect_flux, &MeshBlockData<Real>::SendFluxCorrection, sc0.get());
    auto recv_flux =
        tl.AddTask(advect_flux, &MeshBlockData<Real>::ReceiveFluxCorrection, sc0.get());
  }

  const int pack_size = pmesh->DefaultPackSize();
  auto partitions = partition::ToSizeN(pmesh->block_list, pack_size);
  if (stage == 1) {
    auto setup_mesh_data = [=](const std::string &name, BlockList_t partition,
                              const std::string &mb_name) {
      auto md = pmesh->mesh_data.Add(name);
      md->Set(partition, mb_name);
    };
    for (int i = 0; i < partitions.size(); i++) {
      setup_mesh_data("base" + std::to_string(i), partitions[i], "base");
      setup_mesh_data("dUdt" + std::to_string(i), partitions[i], "dUdt");
      for (int j = 1; j < integrator->nstages; j++) {
        setup_mesh_data(stage_name[j] + std::to_string(i), partitions[i], stage_name[j]);
      }
    }
  }

  const auto use_pack_in_one =
      blocks[0]->packages["advection_package"]->Param<bool>("use_pack_in_one");

  // note that task within this region that contains one tasklist per pack
  // could still be executed in parallel
  TaskRegion &single_tasklist_per_pack_region = tc.AddRegion(1);
  for (int i = 0; i < partitions.size(); i++) {
    auto &tl = single_tasklist_per_pack_region[i];
    auto &mbase = pmesh->mesh_data.Get("base" + std::to_string(i));
    auto &mc0 = pmesh->mesh_data.Get(stage_name[stage - 1] + std::to_string(i));
    auto &mc1 = pmesh->mesh_data.Get(stage_name[stage] + std::to_string(i));
    auto &mdudt = pmesh->mesh_data.Get("dUdt" + std::to_string(i));

    // compute the divergence of fluxes of conserved variables
    auto flux_div = tl.AddTask(none, parthenon::Update::FluxDivergenceMesh, mc0, mdudt);

    // apply du/dt to all independent fields in the container
    auto update_container =
        tl.AddTask(flux_div, UpdateContainer, stage, integrator, mc0, mbase, mdudt, mc1);

    if (use_pack_in_one) {
      // update ghost cells
      auto send = tl.AddTask(update_container,
                             parthenon::cell_centered_bvars::SendBoundaryBuffers, mc1);

      auto recv =
          tl.AddTask(send, parthenon::cell_centered_bvars::ReceiveBoundaryBuffers, mc1);
      auto fill_from_bufs =
          tl.AddTask(recv, parthenon::cell_centered_bvars::SetBoundaries, mc1);
    }
  }
  TaskRegion &async_region2 = tc.AddRegion(num_task_lists_executed_independently);

  for (int i = 0; i < blocks.size(); i++) {
    auto &pmb = blocks[i];
    auto &tl = async_region2[i];
    auto &sc1 = pmb->meshblock_data.Get(stage_name[stage]);

    auto prev_task = none;
    if (!use_pack_in_one) {
      // update ghost cells
      auto send = tl.AddTask(none, &MeshBlockData<Real>::SendBoundaryBuffers, sc1.get());
      auto recv =
          tl.AddTask(send, &MeshBlockData<Real>::ReceiveBoundaryBuffers, sc1.get());
      auto fill_from_bufs =
          tl.AddTask(recv, &MeshBlockData<Real>::SetBoundaries, sc1.get());
      prev_task = fill_from_bufs;
    }

    auto clear_comm_flags = tl.AddTask(prev_task, &MeshBlockData<Real>::ClearBoundary,
                                       sc1.get(), BoundaryCommSubset::all);

    auto prolongBound = tl.AddTask(
        prev_task,
        [](std::shared_ptr<MeshBlock> pmb) {
          pmb->pbval->ProlongateBoundaries(0.0, 0.0);
          return TaskStatus::complete;
        },
        pmb);

    // set physical boundaries
    auto set_bc = tl.AddTask(prolongBound, parthenon::ApplyBoundaryConditions, sc1);

    // fill in derived fields
    auto fill_derived =
        tl.AddTask(set_bc, parthenon::FillDerivedVariables::FillDerived, sc1);

    // estimate next time step
    if (stage == integrator->nstages) {
      auto new_dt = tl.AddTask(
          fill_derived,
          [](std::shared_ptr<MeshBlockData<Real>> &rc) {
            auto pmb = rc->GetBlockPointer();
            pmb->SetBlockTimestep(parthenon::Update::EstimateTimestep(rc));
            return TaskStatus::complete;
          },
          sc1);

      // Update refinement
      if (pmesh->adaptive) {
        auto tag_refine = tl.AddTask(
            fill_derived,
            [](std::shared_ptr<MeshBlock> pmb) {
              pmb->pmr->CheckRefinementCondition();
              return TaskStatus::complete;
            },
            pmb);
      }
    }
  }
  return tc;
}

} // namespace advection_example
