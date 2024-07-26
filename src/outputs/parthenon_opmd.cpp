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
//! \file parthenon_openpmd.cpp
//  \brief Output for OpenPMD https://www.openpmd.org/ (supporting various backends)

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/types.h>
#include <tuple>
#include <vector>

// Parthenon headers
#include "Kokkos_Core_fwd.hpp"
#include "basic_types.hpp"
#include "coordinates/coordinates.hpp"
#include "defs.hpp"
#include "driver/driver.hpp"
#include "globals.hpp"
#include "interface/state_descriptor.hpp"
#include "interface/swarm_default_names.hpp"
#include "interface/variable_state.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "openPMD/Dataset.hpp"
#include "openPMD/Datatype.hpp"
#include "openPMD/IO/Access.hpp"
#include "openPMD/Iteration.hpp"
#include "openPMD/Mesh.hpp"
#include "openPMD/ParticleSpecies.hpp"
#include "openPMD/Series.hpp"
#include "openPMD/backend/MeshRecordComponent.hpp"
#include "outputs/output_utils.hpp"
#include "outputs/outputs.hpp"
#include "outputs/parthenon_hdf5.hpp" // needd for VALId_VEC_TYPES -> move
#include "parthenon_array_generic.hpp"
#include "utils/error_checking.hpp"
#include "utils/instrument.hpp"

// OpenPMD headers
#ifdef PARTHENON_ENABLE_OPENPMD
#include <openPMD/openPMD.hpp>
#endif // ifdef PARTHENON_ENABLE_OPENPMD

namespace parthenon {

using namespace OutputUtils;

template <typename T>
void WriteAllParamsOfType(std::shared_ptr<StateDescriptor> pkg, openPMD::Iteration *it) {
  const std::string prefix = "Params/" + pkg->label() + "/";
  const auto &params = pkg->AllParams();
  for (const auto &key : params.GetKeys()) {
    const auto type = params.GetType(key);
    if (type == std::type_index(typeid(T))) {
      // auto typed_ptr = dynamic_cast<Params::object_t<T> *>((p.second).get());
      it->setAttribute(prefix + key, params.Get<T>(key));
    }
  }
}

template <typename... Ts>
void WriteAllParamsOfMultipleTypes(std::shared_ptr<StateDescriptor> pkg,
                                   openPMD::Iteration *it) {
  ([&] { WriteAllParamsOfType<Ts>(pkg, it); }(), ...);
}

template <typename T>
void WriteAllParams(std::shared_ptr<StateDescriptor> pkg, openPMD::Iteration *it) {
  WriteAllParamsOfMultipleTypes<T, std::vector<T>>(pkg, it);
  // TODO(pgrete) check why this doens't work, i.e., which type is causing problems
  // WriteAllParamsOfMultipleTypes<PARTHENON_ATTR_VALID_VEC_TYPES(T)>(pkg, it);
}

namespace OpenPMDUtils {

template <typename T>
void WriteSwarmVar(const SwarmInfo &swinfo, openPMD::ParticleSpecies swm,
                   openPMD::Iteration it) {
  auto &vars_of_type_T = std::get<SwarmInfo::MapToVarVec<T>>(swinfo.vars);
  for (const auto &[vname, swmvarvec] : vars_of_type_T) {
    const auto &vinfo = swinfo.var_info.at(vname);
    auto host_data = swinfo.FillHostBuffer<T>(vname, swmvarvec);

    auto const dataset = openPMD::Dataset(openPMD::determineDatatype(host_data.data()),
                                          {swinfo.global_count});
    // TODO(pgrete) ask OpenPMD group if this is the right approach of if our non-scalar
    // partices should be a multi-D `dataset` if is scalar
    if (vinfo.tensor_rank == 0) {
      // special sauce to align "positions" with standard
      std::string particle_record;
      std::string particle_record_component;
      if (vname == swarm_position::x::name()) {
        particle_record = "position";
        particle_record_component = "x";
      } else if (vname == swarm_position::y::name()) {
        particle_record = "position";
        particle_record_component = "y";
      } else if (vname == swarm_position::z::name()) {
        particle_record = "position";
        particle_record_component = "z";
      } else {
        particle_record = vname;
        particle_record_component = openPMD::MeshRecordComponent::SCALAR;
      }
      openPMD::RecordComponent rc = swm[particle_record][particle_record_component];
      rc.resetDataset(dataset);
      // only write if there's sth to write (otherwise the host_data nullptr is caught)
      if (swinfo.count_on_rank != 0) {
        rc.storeChunk(host_data, {swinfo.global_offset}, {host_data.size()});
      }

      // if positional, add offsets
      if (particle_record_component != openPMD::MeshRecordComponent::SCALAR) {
        auto rc_offset = swm["positionOffset"][particle_record_component];
        rc_offset.resetDataset(dataset);
        rc_offset.makeConstant(0.0);
      }

      // else flatten components
    } else {
      for (auto n = 0; n < vinfo.nvar; n++) {
        openPMD::RecordComponent rc = swm[vname][std::to_string(n)];
        rc.resetDataset(dataset);
        // only write if there's sth to write (otherwise the host_data nullptr is caught)
        if (swinfo.count_on_rank != 0) {
          rc.storeChunkRaw(&host_data[n * swinfo.count_on_rank], {swinfo.global_offset},
                           {swinfo.count_on_rank});
        }
      }
    }
    // Flush because the host buffer is temporary
    it.seriesFlush();
  }
}
std::tuple<std::string, std::string> GetMeshRecordAndComponentNames(const VarInfo &vinfo,
                                                                    const int comp_idx,
                                                                    const int level) {
  std::string comp_name;
  if (vinfo.is_vector) {
    if (comp_idx == 0) {
      comp_name = "x";
    } else if (comp_idx == 1) {
      comp_name = "y";
    } else if (comp_idx == 2) {
      comp_name = "z";
    } else {
      PARTHENON_THROW("Expected component index doesn't match vector expectation.");
    }
    //  Current unclear how to properly handle other vectors and tensors, so everything
    //  that not's a proper vector is a a scalar for now.
  } else {
    comp_name = openPMD::MeshRecordComponent::SCALAR;
  }
  // TODO(pgrete) need to make sure that var names are allowed within standard
  const std::string &mesh_record_name = vinfo.label + "_" +
                                        vinfo.component_labels[comp_idx] + "_lvl" +
                                        std::to_string(level);
  // return std::make_tuple(mesh_record_name, comp_name);
  return {mesh_record_name, comp_name};
}

std::tuple<openPMD::Offset, openPMD::Extent>
GetChunkOffsetAndExtent(Mesh *pm, std::shared_ptr<MeshBlock> pmb) {
  openPMD::Offset chunk_offset;
  openPMD::Extent chunk_extent;
  const auto loc = pm->Forest().GetLegacyTreeLocation(pmb->loc);
  if (pm->ndim == 3) {
    chunk_offset = {loc.lx3() * static_cast<uint64_t>(pmb->block_size.nx(X3DIR)),
                    loc.lx2() * static_cast<uint64_t>(pmb->block_size.nx(X2DIR)),
                    loc.lx1() * static_cast<uint64_t>(pmb->block_size.nx(X1DIR))};
    chunk_extent = {static_cast<uint64_t>(pmb->block_size.nx(X3DIR)),
                    static_cast<uint64_t>(pmb->block_size.nx(X2DIR)),
                    static_cast<uint64_t>(pmb->block_size.nx(X1DIR))};
  } else if (pm->ndim == 2) {
    chunk_offset = {loc.lx2() * static_cast<uint64_t>(pmb->block_size.nx(X2DIR)),
                    loc.lx1() * static_cast<uint64_t>(pmb->block_size.nx(X1DIR))};
    chunk_extent = {static_cast<uint64_t>(pmb->block_size.nx(X2DIR)),
                    static_cast<uint64_t>(pmb->block_size.nx(X1DIR))};
  } else {
    PARTHENON_THROW("1D output for openpmd not yet supported.");
  }
  return {chunk_offset, chunk_extent};
}
} // namespace OpenPMDUtils

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
  // TODO(pgrete) add final and now logic
  // Prepending @ indicates that the config is a file to be read and parsed.
  std::string backend_config =
      backend_config_ == "default" ? "{}" : "@" + backend_config_;

  Series series =
      Series(output_params.file_basename + "." + output_params.file_id + ".%05T.bp",
             Access::CREATE, MPI_COMM_WORLD, backend_config);

  // TODO(pgrete) How to handle downstream info, e.g.,  on how/what defines a vector?
  // TODO(pgrete) Should we update for restart or only set this once? Or make it per
  // iteration?
  // ... = pin->GetString(output_params.block_name, "actions_file");
  series.setAuthor("My Name <mail@addre.es");
  series.setComment("Hello world!");
  series.setMachine("bla");
  series.setSoftware("Parthenon + Downstream info");
  series.setDate("2024-02-29 17:48:42 +0100");

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
  { // FIXME move this to dump params

    // TODO(pgrete) Make this test piece work on devices
    if constexpr (false) {
      PARTHENON_INSTRUMENT_REGION("Dump Params");
      const auto view_d =
          Kokkos::View<Real **, Kokkos::DefaultExecutionSpace>("blub", 5, 3);
      // Map a view onto a host allocation (so that we can call deep_copy)
      auto host_vec = std::vector<Real>(view_d.size());
      Kokkos::View<Real **, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
          view_h(host_vec.data(), view_d.extent_int(0), view_d.extent_int(1));
      Kokkos::deep_copy(view_h, view_d);
      it.setAttribute("blub", host_vec);
    }

    for (const auto &[key, pkg] : pm->packages.AllPackages()) {
      // WriteAllParams<bool>(pkg, &it); // check why this (vector of bool) doesn't work
      WriteAllParams<int32_t>(pkg, &it);
      WriteAllParams<int64_t>(pkg, &it);
      WriteAllParams<uint32_t>(pkg, &it);
      WriteAllParams<uint64_t>(pkg, &it);
      WriteAllParams<float>(pkg, &it);
      WriteAllParams<double>(pkg, &it);
      WriteAllParams<std::string>(pkg, &it);
      WriteAllParamsOfType<bool>(pkg, &it);
      // WriteAllParamsOfType<std::vector<bool>>(pkg, &it);
    }
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
    // It's not clear we need all these attributes, but they mirror what's done in the
    // hdf5 output.
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
    it.setAttribute("RootLevel", pm->GetLegacyTreeRootLevel());
    it.setAttribute("Refine", pm->adaptive ? 1 : 0);
    it.setAttribute("Multilevel", pm->multilevel ? 1 : 0);

    it.setAttribute("BlocksPerPE", pm->GetNbList());

    // Mesh block size
    const auto base_block_size = pm->GetDefaultBlockSize();
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
  } // Info section

  Kokkos::Profiling::popRegion(); // write Attributes

  // Write block metadata
  {
    // Manually gather all block data first as it allows to use the (simpler)
    // Attribute interface rather than writing a distributed dataset -- especially as all
    // data is being read on restart by every rank anyway.
    std::vector<int64_t> loc_local = OutputUtils::ComputeLocs(pm);
    auto loc_global = FlattendedLocalToGlobal<int64_t>(pm, loc_local);
    it.setAttribute("loc.lx123", loc_global);

    std::vector<int> id_local = OutputUtils::ComputeIDsAndFlags(pm);
    auto id_global = FlattendedLocalToGlobal<int>(pm, id_local);
    it.setAttribute("loc.level-gid-lid-cnghost-gflag", id_global);

    // derefinement count
    std::vector<int> derefcnt_local = OutputUtils::ComputeDerefinementCount(pm);
    auto derefcnt_global = FlattendedLocalToGlobal<int>(pm, derefcnt_local);
    it.setAttribute("derefinement_count", derefcnt_global);
  }

  // TODO(pgrete) check var name standard compatiblity
  // e.g., description: names of records and their components are only allowed to contain
  // the characters a-Z, the numbers 0-9 and the underscore _

  const int num_blocks_local = static_cast<int>(pm->block_list.size());

  // -------------------------------------------------------------------------------- //
  //   WRITING VARIABLES DATA                                                         //
  // -------------------------------------------------------------------------------- //
  Kokkos::Profiling::pushRegion("write all variable data");

  auto &bounds = pm->block_list.front()->cellbounds;
  // get list of all vars, just use the first block since the list is the same for all
  // blocks
  // TODO(pgrete) add restart_ var to output
  // TODO(pgrete) check if this needs to be updated/unifed with get_var logic in hdf5
  auto all_vars_info = GetAllVarsInfo(
      GetVarsToWrite(pm->block_list.front(), true, output_params.variables), bounds);

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
  auto ib = bounds.GetBoundsI(IndexDomain::interior);
  auto jb = bounds.GetBoundsJ(IndexDomain::interior);
  auto kb = bounds.GetBoundsK(IndexDomain::interior);
  // for each variable we write
  for (auto &vinfo : all_vars_info) {
    PARTHENON_INSTRUMENT_REGION("Write variable loop")

    // Reset host write bufer. Not really necessary, but doesn't hurt.
    memset(tmp_data.data(), 0, tmp_data.size() * sizeof(OutT));
    uint64_t tmp_offset = 0;

    if (vinfo.is_vector) {
      // sanity check
      PARTHENON_REQUIRE_THROWS(
          vinfo.GetDim(4) == pm->ndim && vinfo.GetDim(5) == 1 && vinfo.GetDim(6) == 1,
          "A 'standard' vector is expected to only have components matching the "
          "dimensionality of the simulation.")
    }

    for (auto &pmb : pm->block_list) {
      // TODO(pgrete) check if we should skip the suffix for level 0
      const auto level = pmb->loc.level() - pm->GetRootLevel();

      for (int comp_idx = 0; comp_idx < vinfo.component_labels.size(); comp_idx++) {
        const auto [record_name, comp_name] =
            OpenPMDUtils::GetMeshRecordAndComponentNames(vinfo, comp_idx, level);

        // Create the mesh_record for this variable at the given level (if it doesn't
        // exist yet)
        if (!it.meshes.contains(record_name)) {
          auto mesh_record = it.meshes[record_name];

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

          auto mesh_comp = mesh_record[comp_name];
          // TODO(pgrete) needs to be updated for face and edges etc
          // Also this feels wrong for deep hierachies...
          auto effective_nx = static_cast<std::uint64_t>(std::pow(2, level));
          openPMD::Extent global_extent;
          if (pm->ndim == 3) {
            mesh_record.setGridSpacing(std::vector<Real>{dx3, dx2, dx1});
            mesh_record.setAxisLabels({"z", "y", "x"});
            mesh_record.setGridGlobalOffset({
                pm->mesh_size.xmin(X3DIR),
                pm->mesh_size.xmin(X2DIR),
                pm->mesh_size.xmin(X1DIR),
            });
            // TODO(pgrete) needs to be updated for face and edges etc
            mesh_comp.setPosition(std::vector<Real>{0.5, 0.5, 0.5});
            global_extent = {
                static_cast<std::uint64_t>(pm->mesh_size.nx(X3DIR)) * effective_nx,
                static_cast<std::uint64_t>(pm->mesh_size.nx(X2DIR)) * effective_nx,
                static_cast<std::uint64_t>(pm->mesh_size.nx(X1DIR)) * effective_nx,
            };
          } else if (pm->ndim == 2) {
            mesh_record.setGridSpacing(std::vector<Real>{dx2, dx1});
            mesh_record.setAxisLabels({"y", "x"});
            mesh_record.setGridGlobalOffset({
                pm->mesh_size.xmin(X2DIR),
                pm->mesh_size.xmin(X1DIR),
            });

            // TODO(pgrete) needs to be updated for face and edges etc
            mesh_comp.setPosition(std::vector<Real>{0.5, 0.5});
            global_extent = {
                static_cast<std::uint64_t>(pm->mesh_size.nx(X2DIR)) * effective_nx,
                static_cast<std::uint64_t>(pm->mesh_size.nx(X1DIR)) * effective_nx,
            };

          } else {
            PARTHENON_THROW("1D output for openpmd not yet supported.");
          }
          // Handling this here to now re-reset dataset later when iterating through the
          // blocks
          auto const dataset =
              openPMD::Dataset(openPMD::determineDatatype<OutT>(), global_extent);
          // TODO(pgrete) check whether this should/need to be a collective so that the
          // mesh generation should be done across all ranks prior to writing data, rather
          // than in-situ for the local blocks only
          mesh_comp.resetDataset(dataset);

          // TODO(pgrete) need unitDimension and timeOffset for this record?
        }
      }

      // Now that the mesh record exists, actually write the data
      auto out_var = pmb->meshblock_data.Get()->GetVarPtr(vinfo.label);
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
        int comp_idx = 0;
        const auto &Nt = out_var->GetDim(6);
        const auto &Nu = out_var->GetDim(5);
        const auto &Nv = out_var->GetDim(4);
        // loop over all components
        for (int t = 0; t < Nt; ++t) {
          for (int u = 0; u < Nu; ++u) {
            for (int v = 0; v < Nv; ++v) {
              const auto [record_name, comp_name] =
                  OpenPMDUtils::GetMeshRecordAndComponentNames(vinfo, comp_idx, level);
              auto mesh_comp = it.meshes[record_name][comp_name];

              const auto comp_offset = tmp_offset;
              for (int k = kb.s; k <= kb.e; ++k) {
                for (int j = jb.s; j <= jb.e; ++j) {
                  for (int i = ib.s; i <= ib.e; ++i) {
                    tmp_data[tmp_offset] = static_cast<OutT>(out_var_h(t, u, v, k, j, i));
                    tmp_offset++;
                  }
                }
              }
              const auto [chunk_offset, chunk_extent] =
                  OpenPMDUtils::GetChunkOffsetAndExtent(pm, pmb);
              mesh_comp.storeChunkRaw(&tmp_data[comp_offset], chunk_offset, chunk_extent);
              comp_idx += 1;
            }
          }
        } // loop over components
      }   // out_var->IsAllocated()
    }     // loop over blocks
    it.seriesFlush();
  }                               // loop over vars
  Kokkos::Profiling::popRegion(); // write all variable data

  // -------------------------------------------------------------------------------- //
  //   WRITING PARTICLE DATA                                                          //
  // -------------------------------------------------------------------------------- //

  Kokkos::Profiling::pushRegion("write particle data");
  // TODO(pgrete) as above, first wrt differentiating between restart_ (last arg)
  AllSwarmInfo all_swarm_info(pm->block_list, output_params.swarms, true);
  for (auto &[swname, swinfo] : all_swarm_info.all_info) {
    openPMD::ParticleSpecies swm = it.particles[swname];
    // These indicate particles/meshblock and location in global index
    // space where each meshblock starts
    auto counts_global = FlattendedLocalToGlobal<std::size_t>(pm, swinfo.counts);
    swm.setAttribute("counts", counts_global);
    auto offsets_global = FlattendedLocalToGlobal<std::size_t>(pm, swinfo.offsets);
    swm.setAttribute("offsets", offsets_global);

    if (swinfo.global_count == 0) {
      continue;
    }

    OpenPMDUtils::WriteSwarmVar<int>(swinfo, swm, it);
    OpenPMDUtils::WriteSwarmVar<Real>(swinfo, swm, it);

    // From the HDF5 output:
    // If swarm does not contain an "id" object, generate a sequential
    // one for vis.
    // BUT PG: this may break things in unpredicable ways
    // I'm in favor of enforcing a global id somehow. We shold discuss.
    PARTHENON_REQUIRE_THROWS(swinfo.var_info.count("id") != 0,
                             "Particles should always carry a unique, persistent id!");
  }
  Kokkos::Profiling::popRegion(); // write particle data

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
