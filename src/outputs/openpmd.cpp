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

#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

// Parthenon headers
#include "coordinates/coordinates.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "interface/variable_state.hpp"
#include "mesh/mesh.hpp"
#include "openPMD/IO/Access.hpp"
#include "openPMD/Series.hpp"
#include "outputs/output_utils.hpp"
#include "outputs/outputs.hpp"
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
  Series series = Series("test_%05T.h5", Access::CREATE);

  // TODO(pgrete) How to handle downstream info, e.g.,  on how/what defines a vector?
  // TODO(pgrete) Should we update for restart or only set this once? Or make it per
  // iteration?
  // ... = pin->GetString(output_params.block_name, "actions_file");
  series.setAuthor("My Name <mail@addre.es");
  series.setComment("Hello world!");
  series.setMachine("bla");
  series.setSoftware("Parthenon + Downstream info");
  series.setDate("today");

  // TODO(pgrete) Units?

  // TODO(pgrete) We probably want this for params!
  series.setAttribute("bla", true);

  // open iteration (corresponding to a timestep in OpenPMD naming)
  // TODO(pgrete) fix iteration name <-> file naming
  auto it = series.iterations[output_params.file_number];
  it.open(); // explicit open() is important when run in parallel

  it.setTime(tm->time);
  it.setDt(tm->dt);
  for (auto &pmb : pm->block_list) {
    // create a unique id for this MeshBlock
    const std::string &meshblock_name = "domain_" + std::to_string(pmb->gid);
    auto mesh = it.meshes[meshblock_name];

    // add basic state info
    mesh["state/domain_id"] = pmb->gid;
    mesh["state/cycle"] = tm->ncycle;
    mesh["state/time"] = tm->time;

    auto &bounds = pmb->cellbounds;
    auto ib = bounds.GetBoundsI(IndexDomain::entire);
    auto jb = bounds.GetBoundsJ(IndexDomain::entire);
    auto kb = bounds.GetBoundsK(IndexDomain::entire);
    auto ni = ib.e - ib.s + 1;
    auto nj = jb.e - jb.s + 1;
    auto nk = kb.e - kb.s + 1;
    uint64_t ncells = ni * nj * nk;

    auto ib_int = bounds.GetBoundsI(IndexDomain::interior);
    auto jb_int = bounds.GetBoundsJ(IndexDomain::interior);
    auto kb_int = bounds.GetBoundsK(IndexDomain::interior);

    auto &coords = pmb->coords;
    Real dx1 = coords.CellWidth<X1DIR>(ib.s, jb.s, kb.s);
    Real dx2 = coords.CellWidth<X2DIR>(ib.s, jb.s, kb.s);
    Real dx3 = coords.CellWidth<X3DIR>(ib.s, jb.s, kb.s);
    std::array<Real, 3> corner = coords.GetXmin();

    // create the coordinate set
    mesh["coordsets/coords/type"] = "uniform";
    PARTHENON_REQUIRE_THROWS(typeid(Coordinates_t) == typeid(UniformCartesian),
                             "Ascent currently only supports Cartesian coordinates.");

    mesh["coordsets/coords/dims/i"] = ni + 1;
    mesh["coordsets/coords/dims/j"] = nj + 1;
    if (nk > 1) {
      mesh["coordsets/coords/dims/k"] = nk + 1;
    }

    // add origin and spacing to the coordset (optional)
    mesh["coordsets/coords/origin/x"] = corner[0];
    mesh["coordsets/coords/origin/y"] = corner[1];
    if (nk > 1) {
      mesh["coordsets/coords/origin/z"] = corner[2];
    }

    mesh["coordsets/coords/spacing/dx"] = dx1;
    mesh["coordsets/coords/spacing/dy"] = dx2;
    if (nk > 1) {
      mesh["coordsets/coords/spacing/dz"] = dx3;
    }

    // create a field for each component of each variable pack
    auto &mbd = pmb->meshblock_data.Get();

    for (const auto &var : mbd->GetVariableVector()) {
      // ensure that only cell vars are added (for now) as the topology above is only
      // valid for cell centered vars
      if (!var->IsSet(Metadata::Cell)) {
        continue;
      }
      const auto var_info = VarInfo(var);

      for (int icomp = 0; icomp < var_info.num_components; ++icomp) {
        auto const data = Kokkos::subview(var->data, 0, 0, icomp, Kokkos::ALL(),
                                          Kokkos::ALL(), Kokkos::ALL());
        const std::string varname = var_info.component_labels.at(icomp);
        mesh["fields/" + varname + "/association"] = "element";
        mesh["fields/" + varname + "/topology"] = "topo";
        mesh["fields/" + varname + "/values"].set_external(data.data(), ncells);
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
