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
#include "stokes.hpp"

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

    static auto desc =
        parthenon::MakePackDescriptor<advection_package::Conserved::scalar>(
            pmesh->resolved_packages.get(), {parthenon::Metadata::WithFluxes},
            {parthenon::PDOpt::WithFluxes});
    using pack_desc_t = decltype(desc);

    static auto desc_fine =
        parthenon::MakePackDescriptor<advection_package::Conserved::scalar_fine>(
            pmesh->resolved_packages.get(), {parthenon::Metadata::WithFluxes},
            {parthenon::PDOpt::WithFluxes});
    using pack_desc_fine_t = decltype(desc_fine);

    using TE = parthenon::TopologicalElement;
    std::vector<TE> faces{TE::F1};
    if (pmesh->ndim > 1) faces.push_back(TE::F2);
    if (pmesh->ndim > 2) faces.push_back(TE::F3);
    auto flx = none;
    auto flx_fine = none;
    for (auto face : faces) {
      flx = flx | tl.AddTask(none, advection_package::CalculateFluxes<pack_desc_t>, desc,
                             face, parthenon::CellLevel::same, mc0.get());
      flx_fine = flx_fine |
                 tl.AddTask(none, advection_package::CalculateFluxes<pack_desc_fine_t>,
                            desc_fine, face, parthenon::CellLevel::fine, mc0.get());
    }

    auto set_flx = parthenon::AddFluxCorrectionTasks(start_flxcor | flx | flx_fine, tl,
                                                     mc0, pmesh->multilevel);

    auto flux_div = tl.AddTask(set_flx, Stokes<pack_desc_t>, parthenon::CellLevel::same,
                               parthenon::TopologicalType::Cell, desc, pmesh->ndim,
                               mc0.get(), mdudt.get());

    auto flux_div_fine = tl.AddTask(
        set_flx, Stokes<pack_desc_fine_t>, parthenon::CellLevel::fine,
        parthenon::TopologicalType::Cell, desc_fine, pmesh->ndim, mc0.get(), mdudt.get());

    auto avg_data =
        tl.AddTask(flux_div, WeightedSumData<pack_desc_t>, parthenon::CellLevel::same,
                   parthenon::TopologicalElement::CC, desc, mc0.get(), mbase.get(), beta,
                   1.0 - beta, mc0.get());
    auto avg_data_fine =
        tl.AddTask(flux_div_fine, WeightedSumData<pack_desc_fine_t>,
                   parthenon::CellLevel::fine, parthenon::TopologicalElement::CC,
                   desc_fine, mc0.get(), mbase.get(), beta, 1.0 - beta, mc0.get());

    auto update =
        tl.AddTask(avg_data, WeightedSumData<pack_desc_t>, parthenon::CellLevel::same,
                   parthenon::TopologicalElement::CC, desc, mc0.get(), mdudt.get(), 1.0,
                   beta * dt, mc1.get());
    auto update_fine =
        tl.AddTask(avg_data_fine, WeightedSumData<pack_desc_fine_t>,
                   parthenon::CellLevel::fine, parthenon::TopologicalElement::CC,
                   desc_fine, mc0.get(), mdudt.get(), 1.0, beta * dt, mc1.get());

    auto boundaries = parthenon::AddBoundaryExchangeTasks(
        update | update_fine | start_send, tl, mc1, pmesh->multilevel);

    auto fill_derived =
        tl.AddTask(boundaries, parthenon::Update::FillDerived<MeshData<Real>>, mc1.get());

    if (stage == integrator->nstages) {
      auto new_dt = tl.AddTask(fill_derived, EstimateTimestep<MeshData<Real>>, mc1.get());
      if (pmesh->adaptive) {
        auto tag_refine =
            tl.AddTask(new_dt, parthenon::Refinement::Tag<MeshData<Real>>, mc1.get());
      }
    }
  }

  return tc;
}

} // namespace advection_example
