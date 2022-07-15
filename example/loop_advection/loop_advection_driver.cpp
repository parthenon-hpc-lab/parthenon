//========================================================================================
// (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
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
#include "loop_advection.hpp"

//#include "bvals/cc/bvals_cc_in_one.hpp"
//#include "interface/metadata.hpp"
//#include "interface/update.hpp"
//#include "mesh/meshblock_pack.hpp"
//#include "mesh/refinement_cc_in_one.hpp"
//#include "parthenon/driver.hpp"
//#include "refinement/refinement.hpp"

using namespace parthenon::driver::prelude;

namespace loop_advection_example {

LoopAdvectionDriver::LoopAdvectionDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm)
    : MultiStageDriver(pin, app_in, pm) {
}

TaskCollection LoopAdvectionDriver::MakeTaskCollection(BlockList_t &blocks, const int stage) {
  using namespace parthenon::Update;
  TaskCollection tc;
  TaskID none(0);

  const Real beta = integrator->beta[stage - 1];
  const Real dt = integrator->dt;
  const auto &stage_name = integrator->stage_name;

  // Number of task lists that can be executed independently and thus *may*
  // be executed in parallel and asynchronous.
  // Being extra verbose here in this example to highlight that this is not
  // required to be 1 or blocks.size() but could also only apply to a subset of blocks.
  auto num_task_lists_executed_independently = blocks.size();
  TaskRegion &async_region1 = tc.AddRegion(num_task_lists_executed_independently);

  assert(blocks.size() == async_region1.size());
  for (int i = 0; i < blocks.size(); i++) {
    auto &pmb = blocks[i];
    auto &tl = async_region1[i];
    // first make other useful containers
    if (stage == 1) {
      auto &base = pmb->meshblock_data.Get();
      pmb->meshblock_data.Add("dBdt", base);
      for (int i = 1; i < integrator->nstages; i++)
        pmb->meshblock_data.Add(stage_name[i], base);
    }

    // pull out the container we'll use to compute RHSs
    auto &sc0 = pmb->meshblock_data.Get(stage_name[stage - 1]);
    // pull out a container we'll use to store dU/dt.
    // This will contain the face centered dBdt
    auto &dBdt = pmb->meshblock_data.Get("dBdt");
    // pull out the container that will hold the updated state
    // effectively, sc1 = sc0 + dBdt*dt
    auto &sc1 = pmb->meshblock_data.Get(stage_name[stage]);

    auto start_recv = tl.AddTask(none, &MeshBlockData<Real>::StartReceiving, sc1.get(),
                                 BoundaryCommSubset::all);

    //Calculate dBdt from the magnetic field already in the registers in dBdt
    auto calc_dBdt = tl.AddTask(none, LoopAdvection::calc_dBdt, dBdt.get());

    //FIXME(forrestglines): Update to "EMF" Corrections on edges - into which container should this go?
    auto send_edge_correction =
        tl.AddTask(calc_dBdt, &MeshBlockData<Real>::SendEdgeCorrection, dBdt.get());
    auto recv_edge_correction =
        tl.AddTask(calc_dBdt, &MeshBlockData<Real>::ReceiveEdgeCorrection, dBdt.get());

  }

  
  //FIXME(forrestglines) What is this region doing? Is it needed?
  const int num_partitions = pmesh->DefaultNumPartitions();
  // note that task within this region that contains one tasklist per pack
  // could still be executed in parallel
  TaskRegion &single_tasklist_per_pack_region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = single_tasklist_per_pack_region[i];
    auto &mbase = pmesh->mesh_data.GetOrAdd("base", i);
    auto &mc0 = pmesh->mesh_data.GetOrAdd(stage_name[stage - 1], i);
    auto &mc1 = pmesh->mesh_data.GetOrAdd(stage_name[stage], i);
    auto &mdudt = pmesh->mesh_data.GetOrAdd("dBdt", i);

    //auto send_flx = tl.AddTask(
    //    none, parthenon::cell_centered_bvars::LoadAndSendSparseFluxCorrectionBuffers,
    //    mc0);
    //auto recv_flx = tl.AddTask(
    //    none, parthenon::cell_centered_bvars::ReceiveSparseFluxCorrectionBuffers, mc0);
    //auto set_flx =
    //    tl.AddTask(recv_flx, parthenon::cell_centered_bvars::SetFluxCorrections, mc0);

    // compute the divergence of fluxes of conserved variables
    //auto flux_div =
    //    tl.AddTask(set_flx, FluxDivergence<MeshData<Real>>, mc0.get(), mdudt.get());
    auto update = tl.AddTask(none, calc_dBdt, mdudt.get());

    //auto avg_data = tl.AddTask(flux_div, AverageIndependentData<MeshData<Real>>,
    //                           mc0.get(), mbase.get(), beta);
    // apply du/dt to all independent fields in the container -- including to faces?
    auto update = tl.AddTask(avg_data, UpdateIndependentData<MeshData<Real>>, mc0.get(),
                             mdudt.get(), beta * dt, mc1.get());

    // do boundary exchange
    //auto send_cell =
    //    tl.AddTask(update, parthenon::cell_centered_bvars::SendBoundaryBuffers, mc1);
    //auto recv_cell =
    //    tl.AddTask(update, parthenon::cell_centered_bvars::ReceiveBoundaryBuffers, mc1);
    //auto set_cell = tl.AddTask(recv_cell, parthenon::cell_centered_bvars::SetBoundaries, mc1);

    //FIXME(forrestglines) How many of these are necessary additions?
    auto send_face =
        tl.AddTask(update, parthenon::face_centered_bvars::SendBoundaryBuffers, mc1);
    auto recv_face =
        tl.AddTask(update, parthenon::face_centered_bvars::ReceiveBoundaryBuffers, mc1);
    auto set_face = tl.AddTask(recv_face, parthenon::face_centered_bvars::SetBoundaries, mc1);

    //auto send_edge =
    //    tl.AddTask(update, parthenon::edge_centered_bvars::SendBoundaryBuffers, mc1);
    //auto recv_edge =
    //    tl.AddTask(update, parthenon::edge_centered_bvars::ReceiveBoundaryBuffers, mc1);
    //auto set_edge = tl.AddTask(recv_edge, parthenon::edge_centered_bvars::SetBoundaries, mc1);

    if (pmesh->multilevel) {
      tl.AddTask(set_cell, parthenon::cell_centered_refinement::RestrictPhysicalBounds,
                 mc1.get());
      tl.AddTask(set_face, parthenon::face_centered_refinement::RestrictPhysicalBounds,
                 mc1.get());
      //tl.AddTask(set_edge, parthenon::edge_centered_refinement::RestrictPhysicalBounds,
      //           mc1.get());
    }
  }

  TaskRegion &async_region2 = tc.AddRegion(num_task_lists_executed_independently);
  assert(blocks.size() == async_region2.size());
  for (int i = 0; i < blocks.size(); i++) {
    auto &pmb = blocks[i];
    auto &tl = async_region2[i];
    auto &sc1 = pmb->meshblock_data.Get(stage_name[stage]);

    auto clear_comms_flags = tl.AddTask(none, &MeshBlockData<Real>::ClearBoundary,
                                        sc1.get(), BoundaryCommSubset::all);

    auto prolongBound = none;
    if (pmesh->multilevel) {
      prolongBound = tl.AddTask(none, parthenon::ProlongateBoundaries, sc1);
    }

    // set physical boundaries
    auto set_bc = tl.AddTask(prolongBound, parthenon::ApplyBoundaryConditions, sc1);

    // fill in derived fields
    auto fill_derived = tl.AddTask(
        set_bc, parthenon::Update::FillDerived<MeshBlockData<Real>>, sc1.get());

    // estimate next time step
    if (stage == integrator->nstages) {
      auto new_dt =
          tl.AddTask(fill_derived, EstimateTimestep<MeshBlockData<Real>>, sc1.get());

      // Update refinement
      if (pmesh->adaptive) {
        auto tag_refine = tl.AddTask(
            fill_derived, parthenon::Refinement::Tag<MeshBlockData<Real>>, sc1.get());
      }
    }
  }
  return tc;
}

} // namespace loop_advection
