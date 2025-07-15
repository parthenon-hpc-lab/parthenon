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
#include "interface/state_descriptor.hpp"
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

template <class pack_desc_t>
TaskID AddUpdateTasks(TaskID dep, TaskList &tl, parthenon::CellLevel cl,
                      parthenon::TopologicalType tt, Real beta, Real dt,
                      pack_desc_t &desc, MeshData<Real> *mbase, MeshData<Real> *mc0,
                      MeshData<Real> *mdudt, MeshData<Real> *mc1) {
  const int ndim = mc0->GetParentPointer()->ndim;
  auto flux_div = tl.AddTask(dep, Stokes<pack_desc_t>, cl, tt, desc, ndim, mc0, mdudt);
  auto avg_data = tl.AddTask(flux_div, WeightedSumData<pack_desc_t>, cl, tt, desc, mc0,
                             mbase, beta, 1.0 - beta, mc0);
  return tl.AddTask(avg_data, WeightedSumData<pack_desc_t>, cl, tt, desc, mc0, mdudt, 1.0,
                    beta * dt, mc1);
}

// See the advection.hpp declaration for a description of how this function gets called.
TaskCollection AdvectionDriver::MakeTaskCollection(BlockList_t &blocks, const int stage) {
  using namespace parthenon::Update;
  TaskCollection tc;
  TaskID none(0);

  std::shared_ptr<parthenon::StateDescriptor> pkg =
      pmesh->packages.Get("advection_package");
  const auto do_regular_advection = pkg->Param<bool>("do_regular_advection");
  const auto do_fine_advection = pkg->Param<bool>("do_fine_advection");
  const auto do_CT_advection = pkg->Param<bool>("do_CT_advection");

  // Build MeshBlockData containers that will be included in MeshData containers. It is
  // gross that this has to be done by hand.
  const auto &stage_name = integrator->stage_name;
  const Real beta = integrator->beta[stage - 1];
  const Real dt = integrator->dt;

  auto partitions = pmesh->GetDefaultBlockPartitions();
  TaskRegion &single_tasklist_per_pack_region = tc.AddRegion(partitions.size());

  for (int i = 0; i < partitions.size(); i++) {
    auto &tl = single_tasklist_per_pack_region[i];
    auto &mbase = pmesh->mesh_data.Add("base", partitions[i]);
    auto &mc0 = pmesh->mesh_data.Add(stage_name[stage - 1], mbase);
    auto &mc1 = pmesh->mesh_data.Add(stage_name[stage], mbase);
    auto &mdudt = pmesh->mesh_data.Add("dUdt", mbase);

    auto start_send = tl.AddTask(none, parthenon::StartReceiveBoundaryBuffers, mc1);
    auto start_flxcor = tl.AddTask(none, parthenon::StartReceiveFluxCorrections, mc0);

    // Make a sparse variable pack descriptors that can be used to build packs
    // including some subset of the fields in this example. This will be passed
    // to the Stokes update routines, so that they can internally create variable
    // packs that operate on only the desired set of variables.
    using namespace advection_package::Conserved;
    auto desc = parthenon::MakePackDescriptor<phi>(
        pmesh, {parthenon::Metadata::WithFluxes}, {parthenon::PDOpt::WithFluxes});
    using pack_desc_t = decltype(desc);

    auto desc_fine = parthenon::MakePackDescriptor<phi_fine>(
        pmesh, {parthenon::Metadata::WithFluxes}, {parthenon::PDOpt::WithFluxes});
    using pack_desc_fine_t = decltype(desc_fine);

    auto desc_vec = parthenon::MakePackDescriptor<C, D>(
        pmesh, {parthenon::Metadata::WithFluxes}, {parthenon::PDOpt::WithFluxes});

    using TT = parthenon::TopologicalType;
    using TE = parthenon::TopologicalElement;
    std::vector<TE> faces{TE::F1};
    if (pmesh->ndim > 1) faces.push_back(TE::F2);
    if (pmesh->ndim > 2) faces.push_back(TE::F3);
    auto flx = none;
    auto flx_fine = none;
    for (auto face : faces) {
      if (do_regular_advection) {
        flx = flx | tl.AddTask(none, advection_package::CalculateFluxes<pack_desc_t>,
                               desc, face, parthenon::CellLevel::same, mc0.get());
      }
      if (do_fine_advection) {
        flx_fine = flx_fine |
                   tl.AddTask(none, advection_package::CalculateFluxes<pack_desc_fine_t>,
                              desc_fine, face, parthenon::CellLevel::fine, mc0.get());
      }
    }

    auto vf_dep = none;
    if (do_CT_advection) {
      for (auto edge : std::vector<TE>{TE::E1, TE::E2, TE::E3}) {
        vf_dep = tl.AddTask(vf_dep, advection_package::CalculateVectorFluxes<C, D>, edge,
                            parthenon::CellLevel::same, 1.0, mc0.get());
        vf_dep = tl.AddTask(vf_dep, advection_package::CalculateVectorFluxes<D, C>, edge,
                            parthenon::CellLevel::same, -1.0, mc0.get());
      }
    }

    auto set_flx = parthenon::AddFluxCorrectionTasks(
        start_flxcor | flx | flx_fine | vf_dep, tl, mc0, pmesh->multilevel);

    auto update = set_flx;
    if (do_regular_advection) {
      update = AddUpdateTasks(set_flx, tl, parthenon::CellLevel::same, TT::Cell, beta, dt,
                              desc, mbase.get(), mc0.get(), mdudt.get(), mc1.get());
    }

    auto update_fine = set_flx;
    if (do_fine_advection) {
      update_fine =
          AddUpdateTasks(set_flx, tl, parthenon::CellLevel::fine, TT::Cell, beta, dt,
                         desc_fine, mbase.get(), mc0.get(), mdudt.get(), mc1.get());
    }

    auto update_vec = set_flx;
    if (do_CT_advection) {
      update_vec =
          AddUpdateTasks(set_flx, tl, parthenon::CellLevel::same, TT::Face, beta, dt,
                         desc_vec, mbase.get(), mc0.get(), mdudt.get(), mc1.get());
    }

    auto boundaries = parthenon::AddBoundaryExchangeTasks(
        update | update_vec | update_fine | start_send, tl, mc1, pmesh->multilevel);

    auto fill_derived =
        tl.AddTask(boundaries, parthenon::Update::FillDerived<MeshData<Real>>, mc1.get());

    if (stage == integrator->nstages) {
      auto dealloc = tl.AddTask(fill_derived, SparseDealloc, mc1.get());
      auto new_dt = tl.AddTask(dealloc, EstimateTimestep<MeshData<Real>>, mc1.get());
      if (pmesh->adaptive) {
        auto tag_refine =
            tl.AddTask(new_dt, parthenon::Refinement::Tag<MeshData<Real>>, mc1.get());
      }
    }
  }

  return tc;
}

} // namespace advection_example
