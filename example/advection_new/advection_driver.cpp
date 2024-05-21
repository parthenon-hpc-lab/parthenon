//========================================================================================
// (C) (or copyright) 2020-2024. Triad National Security, LLC. All rights reserved.
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
#include "amr_criteria/refinement_package.hpp"
#include "bvals/comms/bvals_in_one.hpp"
#include "interface/metadata.hpp"
#include "interface/update.hpp"
#include "mesh/meshblock_pack.hpp"
#include "parthenon/driver.hpp"
#include "prolong_restrict/prolong_restrict.hpp"

using namespace parthenon::driver::prelude;

namespace advection_example {

// *************************************************//
// define the application driver. in this case,    *//
// that mostly means defining the MakeTaskList     *//
// function.                                       *//
// *************************************************//
AdvectionDriver::AdvectionDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm)
    : MultiStageDriver(pin, app_in, pm) {
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

// See the advection.hpp declaration for a description of how this function gets called.
TaskCollection AdvectionDriver::MakeTaskCollection(BlockList_t &blocks, const int stage) {
  using namespace parthenon::Update;
  TaskCollection tc;
  TaskID none(0);

  // Build MeshBlockData containers that will be included in MeshData containers. It is
  // gross that this has to be done by hand.
  const auto &stage_name = integrator->stage_name;
  if (stage == 1) {
    for (int i = 0; i < blocks.size(); i++) {
      auto &pmb = blocks[i];
      // first make other useful containers
      auto &base = pmb->meshblock_data.Get();
      pmb->meshblock_data.Add("dUdt", base);
      for (int s = 1; s < integrator->nstages; s++)
        pmb->meshblock_data.Add(stage_name[s], base);
    }
  }

  const Real beta = integrator->beta[stage - 1];
  const Real dt = integrator->dt;
  
  const int num_partitions = pmesh->DefaultNumPartitions();
  TaskRegion &single_tasklist_per_pack_region = tc.AddRegion(num_partitions);
  
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = single_tasklist_per_pack_region[i];
    auto &mbase = pmesh->mesh_data.GetOrAdd("base", i);
    auto &mc0 = pmesh->mesh_data.GetOrAdd(stage_name[stage - 1], i);
    auto &mc1 = pmesh->mesh_data.GetOrAdd(stage_name[stage], i);
    auto &mdudt = pmesh->mesh_data.GetOrAdd("dUdt", i);
    
    auto start_send = tl.AddTask(none, parthenon::StartReceiveBoundaryBuffers, mc1);
    auto start_flxcor = tl.AddTask(none, parthenon::StartReceiveFluxCorrections, mc0);

    using TE = parthenon::TopologicalElement; 
    auto flx1 = tl.AddTask(none, advection_package::CalculateFluxes<TE::F1>, mc0.get());
    auto flx2 = none; 
    if (pmesh->ndim > 1) 
      flx2 = tl.AddTask(none, advection_package::CalculateFluxes<TE::F2>, mc0.get());
    auto flx3 = none; 
    if (pmesh->ndim > 2) 
      flx3 = tl.AddTask(none, advection_package::CalculateFluxes<TE::F3>, mc0.get());

    auto set_flx = parthenon::AddFluxCorrectionTasks(start_flxcor | flx1 | flx2 | flx3, tl, mc0, pmesh->multilevel);

    static auto desc = parthenon::MakePackDescriptor<advection_package::Conserved::scalar>(pmesh->resolved_packages.get(), 
                                                                                           {parthenon::Metadata::WithFluxes},
                                                                                           {parthenon::PDOpt::WithFluxes});
    using pack_desc_t = decltype(desc); 

    auto flux_div = tl.AddTask(set_flx, advection_package::Stokes<pack_desc_t>, 
                               parthenon::CellLevel::same,
                               parthenon::TopologicalType::Cell,
                               desc, pmesh->ndim, 
                               mc0.get(), mdudt.get());

    auto avg_data = tl.AddTask(flux_div, advection_package::WeightedSumData<pack_desc_t>, parthenon::CellLevel::same, 
        parthenon::TopologicalElement::CC, desc, mc0.get(), mbase.get(), beta, 1.0 - beta, mc0.get());
    auto update = tl.AddTask(avg_data, advection_package::WeightedSumData<pack_desc_t>, parthenon::CellLevel::same, 
        parthenon::TopologicalElement::CC, desc, mc0.get(), mdudt.get(), 1.0, beta * dt, mc1.get());

    auto boundaries = parthenon::AddBoundaryExchangeTasks(update | start_send, tl, mc1, pmesh->multilevel);

    auto fill_derived = tl.AddTask(
        boundaries, parthenon::Update::FillDerived<MeshData<Real>>, mc1.get());

    if (stage == integrator->nstages) {
      auto new_dt = tl.AddTask(fill_derived, EstimateTimestep<MeshData<Real>>, mc1.get());
      if (pmesh->adaptive) {
        auto tag_refine = tl.AddTask(
            new_dt, parthenon::Refinement::Tag<MeshData<Real>>, mc1.get());
      }
    }
  }

  return tc;
}

} // namespace advection_example
