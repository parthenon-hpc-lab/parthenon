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

#include <atomic>
#include <cstddef>
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
#include "Kokkos_Core_fwd.hpp"
#include "basic_types.hpp"
#include "coordinates/coordinates.hpp"
#include "defs.hpp"
#include "driver/driver.hpp"
#include "globals.hpp"
#include "interface/variable_state.hpp"
#include "mesh/mesh.hpp"
#include "openPMD/Dataset.hpp"
#include "openPMD/Datatype.hpp"
#include "openPMD/IO/Access.hpp"
#include "openPMD/Mesh.hpp"
#include "openPMD/Series.hpp"
#include "openPMD/backend/MeshRecordComponent.hpp"
#include "outputs/output_utils.hpp"
#include "outputs/outputs.hpp"
#include "parthenon_array_generic.hpp"
#include "utils/error_checking.hpp"
#include "utils/instrument.hpp"

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
  Series series = Series("opmd.%05T.bp", Access::CREATE);

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

  auto const &first_block = *(pm->block_list.front());

  // TODO(?) in principle, we could abstract this to a more general WriteAttributes place
  // and reuse for hdf5 and OpenPMD output with corresponing calls
  // -------------------------------------------------------------------------------- //
  //   WRITING ATTRIBUTES                                                             //
  // -------------------------------------------------------------------------------- //

  // Note, that profiling is likely skewed as data is actually written to disk/flushed
  // only later.
  Kokkos::Profiling::pushRegion("write Attributes");
  // First the ones required by the OpenPMD standard
  if (tm != nullptr) {
    it.setTime(tm->time);
    it.setDt(tm->dt);
    it.setAttribute("NCycle", tm->ncycle);
  } else {
    it.setTime(-1.0);
    it.setDt(-1.0);
  }
  { // TESTING REMOVE
    const auto view_d = Kokkos::View<Real **, Kokkos::DefaultExecutionSpace>("blub", 5,3);
    // Map a view onto a host allocation (so that we can call deep_copy)
    auto host_vec = std::vector<Real>(view_d.size());
    Kokkos::View<Real **, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        view_h(host_vec.data(), view_d.extent_int(0), view_d.extent_int(1));
    Kokkos::deep_copy(view_h, view_d);
    it.setAttribute("blub", host_vec);
  }
  // Then our own
  {
    PARTHENON_INSTRUMENT_REGION("write input");
    // write input key-value pairs
    std::ostringstream oss;
    pin->ParameterDump(oss);
    it.setAttribute("InputFile", oss.str());
  }

  {
    it.setAttribute("WallTime", Driver::elapsed_main());
    it.setAttribute("NumDims", pm->ndim);
    it.setAttribute("NumMeshBlocks", pm->nbtotal);
    it.setAttribute("MaxLevel", pm->GetCurrentLevel() - pm->GetRootLevel());
    // write whether we include ghost cells or not
    it.setAttribute("IncludesGhost", output_params.include_ghost_zones ? 1 : 0);
    // write number of ghost cells in simulation
    it.setAttribute("NGhost", Globals::nghost);
    it.setAttribute("Coordinates", std::string(first_block.coords.Name()).c_str());

    // restart info, write always
    it.setAttribute("NBNew", pm->nbnew);
    it.setAttribute("NBDel", pm->nbdel);
    it.setAttribute("RootLevel", pm->GetRootLevel());
    it.setAttribute("Refine", pm->adaptive ? 1 : 0);
    it.setAttribute("Multilevel", pm->multilevel ? 1 : 0);

    it.setAttribute("BlocksPerPE", pm->GetNbList());

    // Mesh block size
    const auto base_block_size = pm->GetBlockSize();
    it.setAttribute("MeshBlockSize",
                    std::vector<int>{base_block_size.nx(X1DIR), base_block_size.nx(X2DIR),
                                     base_block_size.nx(X3DIR)});

    // RootGridDomain - float[9] array with xyz mins, maxs, rats (dx(i)/dx(i-1))
    it.setAttribute(
        "RootGridDomain",
        std::vector<Real>{pm->mesh_size.xmin(X1DIR), pm->mesh_size.xmax(X1DIR),
                          pm->mesh_size.xrat(X1DIR), pm->mesh_size.xmin(X2DIR),
                          pm->mesh_size.xmax(X2DIR), pm->mesh_size.xrat(X2DIR),
                          pm->mesh_size.xmin(X3DIR), pm->mesh_size.xmax(X3DIR),
                          pm->mesh_size.xrat(X3DIR)});

    // Root grid size (number of cells at root level)
    it.setAttribute("RootGridSize",
                    std::vector<int>{pm->mesh_size.nx(X1DIR), pm->mesh_size.nx(X2DIR),
                                     pm->mesh_size.nx(X3DIR)});

    // Boundary conditions
    std::vector<std::string> boundary_condition_str(BOUNDARY_NFACES);
    for (size_t i = 0; i < boundary_condition_str.size(); i++) {
      boundary_condition_str[i] = GetBoundaryString(pm->mesh_bcs[i]);
    }

    it.setAttribute("BoundaryConditions", boundary_condition_str);
    Kokkos::Profiling::popRegion(); // write Info
  }                                 // Info section

  Kokkos::Profiling::popRegion(); // write Attributes

  // TODO(pgrete) check var name standard compatiblity
  // e.g., description: names of records and their components are only allowed to contain
  // the characters a-Z, the numbers 0-9 and the underscore _

  const int num_blocks_local = static_cast<int>(pm->block_list.size());

  // -------------------------------------------------------------------------------- //
  //   WRITING VARIABLES DATA                                                         //
  // -------------------------------------------------------------------------------- //
  Kokkos::Profiling::pushRegion("write all variable data");

  // get list of all vars, just use the first block since the list is the same for all
  // blocks
  // TODO(pgrete) add restart_ var to output
  auto all_vars_info = GetAllVarsInfo(
      GetVarsToWrite(pm->block_list.front(), true, output_params.variables));

  // We're currently writing (flushing) one var at a time. This saves host memory but
  // results more smaller write. Might be updated in the future.
  // Allocate space for largest size variable
  int var_size_max = 0;
  for (auto &vinfo : all_vars_info) {
    const auto var_size = vinfo.Size();
    var_size_max = std::max(var_size_max, var_size);
  }

  // TODO(pgrete) adjust for single prec output
  // openPMD::Datatype dtype = openPMD::determineDatatype<Real>();
  using OutT =
      Real; // typename std::conditional<WRITE_SINGLE_PRECISION, float, Real>::type;
  std::vector<OutT> tmp_data(var_size_max * num_blocks_local);

  // TODO(pgrete) This needs to be in the loop for non-cell-centered vars
  auto &bounds = pm->block_list.front()->cellbounds;
  auto ib = bounds.GetBoundsI(IndexDomain::interior);
  auto jb = bounds.GetBoundsJ(IndexDomain::interior);
  auto kb = bounds.GetBoundsK(IndexDomain::interior);
  // for each variable we write
  for (auto &vinfo : all_vars_info) {
    PARTHENON_INSTRUMENT_REGION("Write variable loop")

    // Reset host write bufer. Not really necessary, but doesn't hurt.
    memset(tmp_data.data(), 0, tmp_data.size() * sizeof(OutT));
    uint64_t tmp_offset = 0;

    const bool is_scalar = vinfo.nx4 == 1 && vinfo.nx5 == 1 && vinfo.nx6 == 1;
    if (vinfo.is_vector) {
      // sanity check
      PARTHENON_REQUIRE_THROWS(
          vinfo.nx4 == pm->ndim && vinfo.nx5 == 1 && vinfo.nx6 == 1,
          "A 'standard' vector is expected to only have components matching the "
          "dimensionality of the simulation.")
    }

    // TODO(pgrete) need to make sure that var names are allowed within standard
    const std::string var_name = vinfo.label;
    for (auto &pmb : pm->block_list) {
      // TODO(pgrete) check if we should skip the suffix for level 0
      const auto level = pmb->loc.level() - pm->GetRootLevel();
      const std::string &mesh_record_name = var_name + "_lvl" + std::to_string(level);

      // Create the mesh_record for this variable at the given level (if it doesn't exist
      // yet)
      if (!it.meshes.contains(mesh_record_name)) {
        auto mesh_record = it.meshes[mesh_record_name];

        // These following attributes are shared across all components of the record.

        PARTHENON_REQUIRE_THROWS(
            typeid(Coordinates_t) == typeid(UniformCartesian),
            "OpenPMD in Parthenon currently only supports Cartesian coordinates.");
        mesh_record.setGeometry(openPMD::Mesh::Geometry::cartesian);
        auto &coords = pmb->coords;
        // For uniform Cartesian, all dxN are const across the block so we just pick the
        // first index.
        Real dx1 = coords.CellWidth<X1DIR>(0, 0, 0);
        Real dx2 = coords.CellWidth<X2DIR>(0, 0, 0);
        Real dx3 = coords.CellWidth<X3DIR>(0, 0, 0);

        // TODO(pgrete) check if this should be tied to the MemoryLayout
        mesh_record.setDataOrder(openPMD::Mesh::DataOrder::C);
        mesh_record.setGridSpacing(std::vector<Real>{dx3, dx2, dx1});
        mesh_record.setAxisLabels({"z", "y", "x"});
        mesh_record.setGridGlobalOffset({
            pm->mesh_size.xmin(X3DIR),
            pm->mesh_size.xmin(X2DIR),
            pm->mesh_size.xmin(X1DIR),
        });

        // TODO(pgrete) need unitDimension and timeOffset for this record?
      }

      auto mesh_record = it.meshes[mesh_record_name];

      // Now that the mesh record exists, actually write the data
      auto out_var = pmb->meshblock_data.Get()->GetVarPtr(var_name);
      PARTHENON_REQUIRE_THROWS(out_var->metadata().Where() ==
                                   MetadataFlag(Metadata::Cell),
                               "Currently only cell centered vars are supported.");

      if (out_var->IsAllocated()) {
        // TODO(pgrete) check if we can work with a direct copy from a subview to not
        // duplicate the memory footprint here
#if 0        
        // Pick a subview of the active cells of this component
        auto const data = Kokkos::subview(
            var->data, 0, 0, icomp, std::make_pair(kb.s, kb.e + 1),
            std::make_pair(jb.s, jb.e + 1), std::make_pair(ib.s, ib.e + 1));

        // Map a view onto a host allocation (so that we can call deep_copy)
        auto component_buffer = buffer_list.emplace_back(ncells);
        Kokkos::View<Real ***, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            component_buffer_view(component_buffer.data(), nk, nj, ni);
        Kokkos::deep_copy(component_buffer_view, data);
#endif
        auto out_var_h = out_var->data.GetHostMirrorAndCopy();
        int idx_component = 0;
        const auto &Nt = out_var->GetDim(6);
        const auto &Nu = out_var->GetDim(5);
        const auto &Nv = out_var->GetDim(4);
        // loop over all components
        for (int t = 0; t < Nt; ++t) {
          for (int u = 0; u < Nu; ++u) {
            for (int v = 0; v < Nv; ++v) {
              // Get the correct record
              std::string comp_name;
              if (is_scalar) {
                comp_name = openPMD::MeshRecordComponent::SCALAR;
              } else if (vinfo.is_vector) {
                if (v == 0) {
                  comp_name = "x";
                } else if (v == 1) {
                  comp_name = "y";
                } else if (v == 2) {
                  comp_name = "z";
                } else {
                  PARTHENON_THROW("Expected v index doesn't match vector expectation.");
                }
              } else {
                comp_name = vinfo.component_labels[idx_component];
              }
              auto mesh_comp = mesh_record[comp_name];

              // TODO(pgrete) needs to be updated for face and edges etc
              mesh_comp.setPosition(std::vector<Real>{0.5, 0.5, 0.5});
              // TODO(pgrete) needs to be updated for face and edges etc
              // Also this feels wrong for deep hierachies...
              auto effective_nx = static_cast<std::uint64_t>(std::pow(2, level));
              openPMD::Extent global_extent = {
                  static_cast<std::uint64_t>(pm->mesh_size.nx(X3DIR)) * effective_nx,
                  static_cast<std::uint64_t>(pm->mesh_size.nx(X2DIR)) * effective_nx,
                  static_cast<std::uint64_t>(pm->mesh_size.nx(X1DIR)) * effective_nx,
              };
              auto const dataset =
                  openPMD::Dataset(openPMD::determineDatatype<OutT>(), global_extent);
              mesh_comp.resetDataset(dataset);

              const auto comp_offset = tmp_offset;
              for (int k = kb.s; k <= kb.e; ++k) {
                for (int j = jb.s; j <= jb.e; ++j) {
                  for (int i = ib.s; i <= ib.e; ++i) {
                    tmp_data[tmp_offset] = static_cast<OutT>(out_var_h(t, u, v, k, j, i));
                    tmp_offset++;
                  }
                }
              }
              openPMD::Offset chunk_offset = {
                  pmb->loc.lx3() * static_cast<uint64_t>(pmb->block_size.nx(X3DIR)),
                  pmb->loc.lx2() * static_cast<uint64_t>(pmb->block_size.nx(X2DIR)),
                  pmb->loc.lx1() * static_cast<uint64_t>(pmb->block_size.nx(X1DIR))};
              openPMD::Extent chunk_extent = {
                  static_cast<uint64_t>(pmb->block_size.nx(X3DIR)),
                  static_cast<uint64_t>(pmb->block_size.nx(X2DIR)),
                  static_cast<uint64_t>(pmb->block_size.nx(X1DIR))};
              mesh_comp.storeChunkRaw(&tmp_data[comp_offset], chunk_offset, chunk_extent);
              idx_component += 1;
            }
          }
        } // loop over components
      }   // out_var->IsAllocated()
    }     // loop over blocks
    it.seriesFlush();
  }                               // loop over vars
  Kokkos::Profiling::popRegion(); // write all variable data

  // The iteration can be closed in order to help free up resources.
  // The iteration's content will be flushed automatically.
  // An iteration once closed cannot (yet) be reopened.
  it.close();
  series.close();
#endif // ifndef PARTHENON_ENABLE_OPENPMD

  // advance output parameters
  output_params.file_number++;
  output_params.next_time += output_params.dt;
  pin->SetInteger(output_params.block_name, "file_number", output_params.file_number);
  pin->SetReal(output_params.block_name, "next_time", output_params.next_time);
}

} // namespace parthenon
