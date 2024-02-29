//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2024 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2024. Triad National Security, LLC. All rights reserved.
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
//! \file openpmd.cpp
//  \brief Output for OpenPMD https://www.openpmd.org/ (supporting various backends)

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/types.h>
#include <vector>

// Parthenon headers
#include "coordinates/coordinates.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "interface/variable_state.hpp"
#include "mesh/mesh.hpp"
#include "openPMD/Datatype.hpp"
#include "openPMD/IO/Access.hpp"
#include "openPMD/Mesh.hpp"
#include "openPMD/Series.hpp"
#include "openPMD/backend/MeshRecordComponent.hpp"
#include "outputs/output_utils.hpp"
#include "outputs/outputs.hpp"
#include "parthenon_array_generic.hpp"
#include "utils/error_checking.hpp"

// OpenPMD headers
#ifdef PARTHENON_ENABLE_OPENPMD
#include <openPMD/openPMD.hpp>
#endif // ifdef PARTHENON_ENABLE_OPENPMD

namespace parthenon {

using namespace OutputUtils;

//----------------------------------------------------------------------------------------
//! \fn void OpenPMDOutput:::WriteOutputFile(Mesh *pm)
//  \brief  Expose mesh and all Cell variables for processing with Ascent
void OpenPMDOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin, SimTime *tm,
                                    const SignalHandler::OutputSignal signal) {
#ifndef PARTHENON_ENABLE_OPENPMD
  if (Globals::my_rank == 0) {
    PARTHENON_WARN("OpenPMD output requested by input file, but OpenPMD support not "
                   "compiled in. Skipping this output type.");
  }
#else
  using openPMD::Access;
  using openPMD::Series;

  // TODO(pgrete) .h5 for hd5 and .bp for ADIOS2 or .json for JSON
  // TODO(pgrete) check if CREATE is the correct pattern (for not overwriting the series
  // but an interation) This just describes the pattern of the filename. The correct file
  // will be accessed through the iteration idx below. The file suffix maps to the chosen
  // backend.
  Series series = Series("adios_test_%05T.bp", Access::CREATE);

  // TODO(pgrete) How to handle downstream info, e.g.,  on how/what defines a vector?
  // TODO(pgrete) Should we update for restart or only set this once? Or make it per
  // iteration?
  // ... = pin->GetString(output_params.block_name, "actions_file");
  series.setAuthor("My Name <mail@addre.es");
  series.setComment("Hello world!");
  series.setMachine("bla");
  series.setSoftware("Parthenon + Downstream info");
  series.setDate("2024-02-29");

  // TODO(pgrete) Units?

  // TODO(pgrete) We probably want this for params!
  series.setAttribute("bla", true);

  // In line with existing outputs, we write one file per iteration/snapshot
  series.setIterationEncoding(openPMD::IterationEncoding::fileBased);

  // open iteration (corresponding to a timestep in OpenPMD naming)
  // TODO(pgrete) fix iteration name <-> file naming
  auto it = series.iterations[output_params.file_number];
  it.open(); // explicit open() is important when run in parallel

  it.setTime(tm->time);
  it.setDt(tm->dt);

  // TODO(pgrete) check var name standard compatiblity
  // e.g., description: names of records and their components are only allowed to contain
  // the characters a-Z, the numbers 0-9 and the underscore _

  const int num_blocks_local = static_cast<int>(pm->block_list.size());
  // TODO(pgrete) adjust for single prec output
  // openPMD::Datatype dtype = openPMD::determineDatatype<Real>();
  // using OutT = typename std::conditional<WRITE_SINGLE_PRECISION, float, Real>::type;
  // Need to create the buffer outside so that it's persistent until the data is flushed.
  // Dynamically allocating the inner vector should not be a performance bottleneck given
  // that the hdf5 backend also uses an explicit per block and per variable
  // GetHostMirrorAndCopy, but still this could be optimized.
  std::vector<std::vector<Real>> buffer_list;

  for (size_t b_idx = 0; b_idx < num_blocks_local; ++b_idx) {
    const auto &pmb = pm->block_list[b_idx];
    // create a unique id for this MeshBlock
    const std::string &meshblock_name = "block_" + std::to_string(pmb->gid);
    auto block_record = it.meshes[meshblock_name];

    // TODO(pgrete) check if we should update the logic (e.g., defining axes) for 1D and
    // 2D outputs

    auto &bounds = pmb->cellbounds;
    auto ib = bounds.GetBoundsI(IndexDomain::interior);
    auto jb = bounds.GetBoundsJ(IndexDomain::interior);
    auto kb = bounds.GetBoundsK(IndexDomain::interior);
    uint64_t ni = ib.e - ib.s + 1;
    uint64_t nj = jb.e - jb.s + 1;
    uint64_t nk = kb.e - kb.s + 1;

    uint64_t ncells = ni * nj * nk;
    auto &coords = pmb->coords;
    Real dx1 = coords.CellWidth<X1DIR>(ib.s, jb.s, kb.s);
    Real dx2 = coords.CellWidth<X2DIR>(ib.s, jb.s, kb.s);
    Real dx3 = coords.CellWidth<X3DIR>(ib.s, jb.s, kb.s);

    // These attributes are shared across all components.
    // In general, we should check if there's a performance bottleneck with writing so
    // many meshes rather than writing one mesh per level and dump all data there (which
    // relies on support for sparse storage by the backend.).

    // TODO(pgrete) check if this should be tied to the MemoryLayout
    block_record.setDataOrder(openPMD::Mesh::DataOrder::C);
    block_record.setGridSpacing(std::vector<Real>{dx3, dx2, dx1});
    block_record.setAxisLabels({"z", "y", "x"});
    std::array<Real, 3> corner = coords.GetXmin();
    block_record.setGridGlobalOffset({corner[2], corner[1], corner[0]});
    PARTHENON_REQUIRE_THROWS(typeid(Coordinates_t) == typeid(UniformCartesian),
                             "Ascent currently only supports Cartesian coordinates.");
    block_record.setGeometry(openPMD::Mesh::Geometry::cartesian);

    // create a field for each component of each variable pack
    auto &mbd = pmb->meshblock_data.Get();

    for (const auto &var : mbd->GetVariableVector()) {
      // ensure that only cell vars are added (for now) as the topology below is only
      // valid for cell centered vars
      if (!var->IsSet(Metadata::Cell)) {
        continue;
      }
      const auto var_info = VarInfo(var);

      for (int icomp = 0; icomp < var_info.num_components; ++icomp) {
        // Pick a subview of the active cells of this component
        auto const data = Kokkos::subview(
            var->data, 0, 0, icomp, std::make_pair(kb.s, kb.e + 1),
            std::make_pair(jb.s, jb.e + 1), std::make_pair(ib.s, ib.e + 1));

        // Map a view onto a host allocation (so that we can call deep_copy)
        auto component_buffer = buffer_list.emplace_back(ncells);
        Kokkos::View<Real ***, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            component_buffer_view(component_buffer.data(), nk, nj, ni);
        Kokkos::deep_copy(component_buffer_view, data);

        const std::string varname = var_info.component_labels.at(icomp);
        auto component_record = block_record[varname];
        // TODO(pgrete) needs to be updated for face and edges etc
        component_record.setPosition(std::vector<Real>{0.5, 0.5, 0.5});
        auto const dataset =
            openPMD::Dataset(openPMD::determineDatatype<Real>(), {nk, nj, ni});
        component_record.resetDataset(dataset);

        component_record.storeChunkRaw(component_buffer.data(), {0, 0, 0}, {nk, nj, ni});
      }
    }
  }
  // The iteration can be closed in order to help free up resources.
  // The iteration's content will be flushed automatically.
  // An iteration once closed cannot (yet) be reopened.
  it.close();
  // No need to close series as it's done in the desctructor of the object when it runs
  // out of scope.
#endif // ifndef PARTHENON_ENABLE_OPENPMD

  // advance output parameters
  output_params.file_number++;
  output_params.next_time += output_params.dt;
  pin->SetInteger(output_params.block_name, "file_number", output_params.file_number);
  pin->SetReal(output_params.block_name, "next_time", output_params.next_time);
}

} // namespace parthenon
