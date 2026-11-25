//========================================================================================
// (C) (or copyright) 2020-2024. Triad National Security, LLC. All rights
// reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001
// for Los Alamos National Laboratory (LANL), which is operated by Triad
// National Security, LLC for the U.S. Department of Energy/National Nuclear
// Security Administration. All rights in the program are reserved by Triad
// National Security, LLC, and the U.S. Department of Energy/National Nuclear
// Security Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
// in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do
// so.
//========================================================================================

#include <memory>
#include <string>
#include <vector>

// Local Includes
#include "advection/advection_package.hpp"
#include "amr_criteria/refinement_package.hpp"
#include "bvals/comms/bvals_in_one.hpp"
#include "driver.hpp"
#include "interface/metadata.hpp"
#include "interface/state_descriptor.hpp"
#include "interface/update.hpp"
#include "mesh/meshblock_pack.hpp"
#include "parthenon/driver.hpp"
#include "prolong_restrict/prolong_restrict.hpp"
#include "utilities/stokes.hpp"

using namespace parthenon::driver::prelude;

namespace scalar_imex {

// Load the required sub-packages
Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;

  if (pin->GetOrAddBoolean("scalar_imex", "advection", true))
    packages.Add(advection_package::Initialize(pin.get()));

  auto app = std::make_shared<StateDescriptor>("scalar_imex_app");
  packages.Add(app);

  return packages;
}

// *************************************************//
// define the application driver. in this case,    *//
// that mostly means defining the MakeTaskList     *//
// function.                                       *//
// *************************************************//
ScalarIMEXDriver::ScalarIMEXDriver(ParameterInput *pin, ApplicationInput *app_in,
                                   Mesh *pm)
    : parthenon::MultiStageDriverGeneric<IMEXRKIntegrator>(pin, app_in, pm) {

  // fail if these are not specified in the input file
  pin->CheckRequired("parthenon/mesh", "ix1_bc");
  pin->CheckRequired("parthenon/mesh", "ox1_bc");
  pin->CheckRequired("parthenon/mesh", "ix2_bc");
  pin->CheckRequired("parthenon/mesh", "ox2_bc");

  // warn if these fields aren't specified in the input file
  pin->CheckDesired("parthenon/mesh", "refinement");
  pin->CheckDesired("parthenon/mesh", "numlevel");

  // Determine which packages to include in driver, allowing for packages to be loaded but
  // not run in the driver
  do_advection = pin->GetOrAddBoolean("scalar_imex", "do_advection",
                                      pin->GetBoolean("scalar_imex", "advection"));
}

// See the advection.hpp declaration for a description of how this function gets
// called.
TaskCollection ScalarIMEXDriver::MakeTaskCollection(BlockList_t &, const int stage) {
  using namespace parthenon::Update;
  TaskCollection tc;
  TaskID none(0);

  const Real dt = tm.dt;
  const int ndim = pmesh->ndim;

  auto partitions = pmesh->GetDefaultBlockPartitions();
  TaskRegion &single_tasklist_per_pack_region = tc.AddRegion(partitions.size());

  // Integration steps:
  // for i = 1, ..., nstages
  //  1. if i == 1, copy data from base register to first stage register
  //  2. Do *local* implicit update from current state of register using U^(i) =
  //  U^(i)* + dt * \tilde a_{ii} R(U^(i))
  //  3. Communicate and fill derived on U^(i)
  //  4. Calculate partial_x F(U^(i)) and R(U^(i))
  //  5. for j = i + 1, ..., nstages
  //     a. if i == 1, add to base register and store in j register, otherwise
  //     add and store in j register b. Add -dt * \tilde a_{ji} * partial_x
  //     F(U^(i)) + dt * a_{ji} * R(U^(i))
  //  6. Update U^base += -dt * \tilde b_{i} * partial_x F(U^(i)) + dt * b_{i} *
  //  R(U^(i))
  //  7. if i == nstages, communicate and fill derived on base register

  using namespace advection_package::Conserved;
  static auto desc_phi = parthenon::MakePackDescriptor<phi>(
      pmesh->resolved_packages.get(),
      {parthenon::Metadata::WithFluxes, parthenon::Metadata::Cell},
      {parthenon::PDOpt::WithFluxes});
  using pack_desc_phi_t = decltype(desc_phi);

  for (int i = 0; i < partitions.size(); i++) {
    auto &tl = single_tasklist_per_pack_region[i];
    auto &mbase = pmesh->mesh_data.Add("base", partitions[i]);
    auto &mc0 = pmesh->mesh_data.Add(integrator->GetStageName(stage), mbase);
    using namespace advection_package::Conserved;

    auto set_stage0 = none;
    if (stage == 1) {
      // Copy base data into this stage, need to be careful that it is a full
      // copy
      set_stage0 = tl.AddTask(none, WeightedSumDataAll, mbase.get(), mbase.get(), 1.0,
                              0.0, mc0.get());
    }

    // Do implicit update here to get conserved variables for the current stage
    auto implicit_update = set_stage0;
    const auto stage_dt = integrator->a(stage, stage) * dt;
    if (do_advection)
      implicit_update = tl.AddTask(set_stage0, advection_package::ImplicitSourceUpdate,
                                   stage_dt, mc0.get(), mc0.get());

    // Update state of current stage here, as this is the first place it is
    // finalized
    auto boundaries_stage =
        parthenon::AddBoundaryExchangeTasks(implicit_update, tl, mc0, pmesh->multilevel);

    auto fill_derived_stage = tl.AddTask(
        boundaries_stage, parthenon::Update::FillDerived<MeshData<Real>>, mc0.get());

    // Calculate fluxes for the current stage
    using TT = parthenon::TopologicalType;
    using TE = parthenon::TopologicalElement;
    std::vector<TE> faces{TE::F1};
    if (pmesh->ndim > 1) faces.push_back(TE::F2);
    if (pmesh->ndim > 2) faces.push_back(TE::F3);
    auto flx = none;
    for (auto face : faces) {
      if (do_advection)
        flx = flx | tl.AddTask(fill_derived_stage,
                               advection_package::CalculateFluxes<pack_desc_phi_t>,
                               desc_phi, face, parthenon::CellLevel::same, mc0.get());
    }

    auto set_flux = parthenon::AddFluxCorrectionTasks(flx, tl, mc0, pmesh->multilevel);

    auto &mdudt_F = pmesh->mesh_data.Add("dUdtF", mbase);
    auto &mdudt_R = pmesh->mesh_data.Add("dUdtR", mbase);
    auto flux_div = tl.AddTask(set_flux, StokesAll, mc0.get(), mdudt_F.get());

    // Calculate source terms for the current stage
    auto source = set_flux;
    if (do_advection)
      source = source |
               tl.AddTask(set_flux, advection_package::Source, mc0.get(), mdudt_R.get());

    // Add the contribution from this stage to the updates for subsequent stages
    auto update_stages = flux_div;
    for (int stage2 = stage + 1; stage2 <= integrator->nstages; ++stage2) {
      auto label2 = integrator->GetStageName(stage2);
      auto &md_in = stage == 1 ? pmesh->mesh_data.Add("base", partitions[i])
                               : pmesh->mesh_data.Add(label2, mbase);
      auto &md_out = pmesh->mesh_data.Add(label2, mbase);
      auto add_flux_div =
          tl.AddTask(source, WeightedSumDataAll, md_in.get(), mdudt_F.get(), 1.0,
                     integrator->at(stage2, stage) * dt, md_out.get());
      auto add_source =
          tl.AddTask(add_flux_div, WeightedSumDataAll, md_out.get(), mdudt_R.get(), 1.0,
                     integrator->a(stage2, stage) * dt, md_out.get());
      update_stages = update_stages | add_source;
    }

    // Add this contribution to the base stage
    auto add_flux_div = tl.AddTask(source, WeightedSumDataAll, mbase.get(), mdudt_F.get(),
                                   1.0, integrator->bt(stage) * dt, mbase.get());
    auto add_source =
        tl.AddTask(add_flux_div, WeightedSumDataAll, mbase.get(), mdudt_R.get(), 1.0,
                   integrator->b(stage) * dt, mbase.get());

    // Perform the last stage cleanup tasks
    if (stage == integrator->nstages) {
      auto boundaries =
          parthenon::AddBoundaryExchangeTasks(add_source, tl, mbase, pmesh->multilevel);

      auto fill_derived = tl.AddTask(
          boundaries, parthenon::Update::FillDerived<MeshData<Real>>, mbase.get());

      auto dealloc = tl.AddTask(fill_derived, SparseDealloc, mbase.get());
      auto new_dt = tl.AddTask(dealloc, EstimateTimestep<MeshData<Real>>, mbase.get());
      if (pmesh->adaptive) {
        auto tag_refine =
            tl.AddTask(new_dt, parthenon::Refinement::Tag<MeshData<Real>>, mbase.get());
      }
    }
  }

  return tc;
}

} // namespace scalar_imex
