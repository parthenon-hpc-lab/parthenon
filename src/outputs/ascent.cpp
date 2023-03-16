//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2023 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2023. Triad National Security, LLC. All rights reserved.
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
//! \file ascent.cpp
//  \brief Ascent situ visualization and analysis interop

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
#include "mesh/mesh.hpp"
#include "outputs/outputs.hpp"
#include "utils/error_checking.hpp"

// Ascent headers
#ifdef PARTHENON_ENABLE_ASCENT
#include "ascent.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_relay_io.hpp"
#include "conduit_relay_io_blueprint.hpp"
#endif // ifdef PARTHENON_ENABLE_ASCENT

namespace parthenon {

//----------------------------------------------------------------------------------------
//! \fn void AscentOutput:::WriteOutputFile(Mesh *pm)
//  \brief  Expose mesh and all Cell variables for processing with Ascent
void AscentOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin, SimTime *tm,
                                   const SignalHandler::OutputSignal signal) {
#ifndef PARTHENON_ENABLE_ASCENT
  if (Globals::my_rank == 0) {
    PARTHENON_WARN("Ascent output requested by input file, but Ascent support not "
                   "compiled in. Skipping this output type.");
  }
#else
  // reference:
  // https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh.html#complete-uniform-example

  using conduit::Node;

  // Ascent needs the MPI communicator we are using
  ascent::Ascent ascent;
  Node ascent_opts;
  ascent_opts["mpi_comm"] = MPI_Comm_c2f(MPI_COMM_WORLD);
  ascent_opts["actions_file"] = pin->GetString(output_params.block_name, "actions_file");
  // Only publish fields that are used within actions to reduce memory footprint.
  // A user might need to override this, e.g., in a runtime ascent_options.yaml, if
  // the required fields cannot be resolved by Ascent.
  // See https://ascent.readthedocs.io/en/latest/AscentAPI.html#field-filtering
  // TODO(some in mid 2023) Reenable this as this currently only works in develop of
  // Ascent and not in published release (expected in 0.9.1), see
  // https://github.com/Alpine-DAV/ascent/pull/1109
  // ascent_opts["field_filtering"] = "true";
  ascent.open(ascent_opts);

  // create root node for the whole mesh
  Node root;

  for (auto &pmb : pm->block_list) {
    // create a unique id for this MeshBlock
    const std::string &meshblock_name = "domain_" + std::to_string(pmb->gid);
    Node &mesh = root[meshblock_name];

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

    // add the topology
    mesh["topologies/topo/type"] = "uniform";
    mesh["topologies/topo/coordset"] = "coords";

    // indicate ghost zones with ascent_ghosts set to 1
    Node &n_field = mesh["fields/ascent_ghosts"];
    n_field["association"] = "element";
    n_field["topology"] = "topo";

    // allocate ghost mask if not already done
    if (ghost_mask.data() == nullptr) {
      ghost_mask = ParArray1D<std::int32_t>("Ascent ghost mask", ncells);

      const int njni = nj * ni;
      pmb->par_for(
          "Set ascent ghost mask", 0, ncells, KOKKOS_LAMBDA(const int &idx) {
            const int k = idx / (njni);
            const int j = (idx - k * njni) / ni;
            const int i = idx - k * njni - j * nj;

            if ((i < ib_int.s) || (ib_int.e < i) || (j < jb_int.s) || (jb_int.e < j) ||
                ((nk > 1) && ((k < kb_int.s) || (kb_int.e < k)))) {
              ghost_mask(idx) = 1;
            } else {
              ghost_mask(idx) = 0;
            }
          });
    }
    // Set ghost mask
    n_field["values"].set_external(ghost_mask.data(), ncells);

    // create a field for each component of each variable pack
    auto &mbd = pmb->meshblock_data.Get();

    for (const auto &var : mbd->GetCellVariableVector()) {
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

  // make sure we conform:
  Node verify_info;
  if (!conduit::blueprint::mesh::verify(root, verify_info)) {
    if (parthenon::Globals::my_rank == 0) {
      PARTHENON_WARN("Ascent output: blueprint::mesh::verify failed!");
    }
    verify_info.print();
  }
  ascent.publish(root);

  // Create dummy action as we need to "execute" to override the actions defined in the
  // yaml file.
  Node actions;
  // execute the actions
  ascent.execute(actions);

  // close ascent
  ascent.close();
#endif // ifndef PARTHENON_ENABLE_ASCENT

  // advance output parameters
  output_params.file_number++;
  output_params.next_time += output_params.dt;
  pin->SetInteger(output_params.block_name, "file_number", output_params.file_number);
  pin->SetReal(output_params.block_name, "next_time", output_params.next_time);
}

} // namespace parthenon
