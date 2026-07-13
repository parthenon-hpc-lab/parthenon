//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2024-2026 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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
// This file was made in part with generative AI.

// This file was made in part with generative AI.

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

// OpenPMD headers
#include <openPMD/openPMD.hpp>

// Parthenon headers
#include "basic_types.hpp"
#include "coordinates/coordinates.hpp"
#include "defs.hpp"
#include "driver/driver.hpp"
#include FS_HEADER
#include "globals.hpp"
#include "interface/state_descriptor.hpp"
#include "interface/variable_state.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "outputs/output_attr.hpp"
#include "outputs/output_parameters.hpp"
#include "outputs/output_utils.hpp"
#include "outputs/outputs.hpp"
#include "outputs/parthenon_opmd.hpp"
#include "pack/default_names.hpp"
#include "parthenon_array_generic.hpp"
#include "provenance.hpp"
#include "utils/error_checking.hpp"
#include "utils/instrument.hpp"

namespace fs = FS_NAMESPACE;

namespace parthenon {

using namespace OutputUtils;

namespace OpenPMDUtils {

void CheckValidName(const std::string &name) {
  // also including \v as special char used for scalar records
  auto is_alnum_underscore = [](char c) {
    return (isalnum(c) || (c == '_') || c == '\v');
  };
  PARTHENON_REQUIRE_THROWS(
      find_if_not(name.begin(), name.end(), is_alnum_underscore) == name.end(),
      "Generated OpenPMD mesh or particle record'" + name +
          "' is not standard compliant. Please contact Parthenon developers for a fix.");
}

template <typename T>
  requires(KokkosView<T>)
auto GetFlatHostVecFromView(T view) {
  // Take a view and return a vector containing rank and dims and a flattened (1D)
  // std::vector that can then easily be passed to OpenPMD.
  using base_t = T::non_const_value_type;
  using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
  const std::size_t n = view.size();
  std::vector<base_t> host_vec(n);
  Kokkos::View<base_t *, HostMemSpace, Unmanaged> view_h(host_vec.data(), n);
  Kokkos::View<base_t *, Unmanaged> dev_flat(view.data(), n);
  Kokkos::deep_copy(view_h, dev_flat);

  // cpplint demands compile constants be all caps
  constexpr auto RANK = static_cast<size_t>(T::rank);
  std::vector<size_t> rank_and_dims(RANK + 1);
  rank_and_dims[0] = RANK;
  for (size_t d = 0; d < RANK; ++d) {
    rank_and_dims[1 + d] = view.extent_int(d);
  }
  return std::make_tuple(rank_and_dims, host_vec);
}

template <typename T>
void WriteAllParamsOfType(const Params &params, const std::string &prefix,
                          openPMD::Iteration *it) {
  for (const auto &key : params.GetKeys()) {
    const auto type = params.GetType(key);
    if (type == std::type_index(typeid(T))) {
      auto full_path = prefix + delim + key;
      // The '/' is kind of a reserved character in the OpenPMD standard, which results
      // in attribute keys with said character not being exposed.
      // Thus we replace it.
      std::replace(full_path.begin(), full_path.end(), '/', delim[0]);

      if constexpr (::KokkosView<T>) {
        const auto &view = params.Get<T>(key);
        auto [rank_and_dims, host_vec] = GetFlatHostVecFromView(view);
        it->setAttribute(full_path + ".rankdims", rank_and_dims);
        it->setAttribute(full_path, host_vec);
      } else if constexpr (is_specialization_of<T, ParArrayGeneric>::value) {
        const auto &view = params.Get<T>(key).KokkosView();
        auto [rank_and_dims, host_vec] = GetFlatHostVecFromView(view);
        it->setAttribute(full_path + ".rankdims", rank_and_dims);
        it->setAttribute(full_path, host_vec);
      } else {
        it->setAttribute(full_path, params.Get<T>(key));
      }
    }
  }
}

template <typename... Ts>
void WriteAllParamsOfMultipleTypes(const Params &params, const std::string &prefix,
                                   openPMD::Iteration *it) {
  ([&] { WriteAllParamsOfType<Ts>(params, prefix, it); }(), ...);
}

template <typename T>
void WriteAllParams(const Params &params, const std::string &prefix,
                    openPMD::Iteration *it) {
  WriteAllParamsOfMultipleTypes<PARTHENON_ATTR_VALID_VEC_TYPES(T)>(params, prefix, it);
}

void WriteAllParams(const Params &params, const std::string &pkg_name,
                    openPMD::Iteration *it) {
  using OpenPMDUtils::delim;
  const std::string prefix = "Params" + delim + pkg_name;
  // check why this (vector of bool) doesn't work
  // WriteAllParams<bool>(params, prefix, it);
  WriteAllParamsOfType<bool>(params, prefix, it);
  WriteAllParams<int32_t>(params, prefix, it);
  WriteAllParams<int64_t>(params, prefix, it);
  WriteAllParams<uint32_t>(params, prefix, it);
  WriteAllParams<uint64_t>(params, prefix, it);
  WriteAllParams<float>(params, prefix, it);
  WriteAllParams<double>(params, prefix, it);

  // strings (not supported in Kokkos Views)
  WriteAllParamsOfType<std::string>(params, prefix, it);
  WriteAllParamsOfType<std::vector<std::string>>(params, prefix, it);
}

template <typename T>
void WriteSwarmVar(const SwarmInfo &swinfo, openPMD::ParticleSpecies swm,
                   openPMD::Iteration it) {
  auto &vars_of_type_T = std::get<SwarmInfo::MapToVarVec<T>>(swinfo.vars);
  for (const auto &[vname, swmvarvec] : vars_of_type_T) {
    const auto &vinfo = swinfo.var_info.at(vname);
    auto host_data = swinfo.FillHostBuffer<T>(vname, swmvarvec);

    auto const dataset = openPMD::Dataset(openPMD::determineDatatype(host_data.data()),
                                          {swinfo.global_count});
    // TODO(pgrete) ask OpenPMD group if this is the right approach (flatten vector and
    // tensors with flattened indices as string component names) or if our non-scalar
    // particle variables should be a multi-D `dataset` (if possible)
    for (auto n = 0; n < vinfo.nvar; n++) {
      auto [particle_record, particle_record_component] =
          OpenPMDUtils::GetParticleRecordAndComponentNames(vname, vinfo.tensor_rank, n);

      openPMD::RecordComponent rc = swm[particle_record][particle_record_component];
      rc.resetDataset(dataset);
      // only write if there's sth to write (otherwise the host_data nullptr is caught)
      if (swinfo.count_on_rank != 0) {
        rc.storeChunkRaw(&host_data[n * swinfo.count_on_rank], {swinfo.global_offset},
                         {swinfo.count_on_rank});
      }

      // if positional, add offsets
      if (particle_record == "position") {
        auto rc_offset = swm["positionOffset"][particle_record_component];
        rc_offset.resetDataset(dataset);
        rc_offset.makeConstant(0.0);
      }
    }
    // Flush because the host buffer is temporary
    it.seriesFlush();
  }
}

std::tuple<std::string, std::string>
GetParticleRecordAndComponentNames(const std::string &vname, const int rank,
                                   const int flat_comp_idx) {
  std::string particle_record;
  std::string particle_record_component;

  // Map swarm positions to OpenPMD standard "position" record with x1/x2/x3 components
  if (vname == swarm_position::x1::name()) {
    particle_record = "position";
    particle_record_component = "x1";
  } else if (vname == swarm_position::x2::name()) {
    particle_record = "position";
    particle_record_component = "x2";
  } else if (vname == swarm_position::x3::name()) {
    particle_record = "position";
    particle_record_component = "x3";
  } else if (vname == swarm_position::id::name()) {
    particle_record = "id";
    particle_record_component = openPMD::MeshRecordComponent::SCALAR;
    // Backwards compatibility: support old position names (swarm.x/y/z -> position/x,y,z)
  } else if (vname == "swarm.x") {
    particle_record = "position";
    particle_record_component = "x";
  } else if (vname == "swarm.y") {
    particle_record = "position";
    particle_record_component = "y";
  } else if (vname == "swarm.z") {
    particle_record = "position";
    particle_record_component = "z";
  } else {
    particle_record = vname;
    particle_record_component =
        rank == 0 ? openPMD::MeshRecordComponent::SCALAR : std::to_string(flat_comp_idx);
  }
  CheckValidName(particle_record);
  CheckValidName(particle_record_component);
  return {particle_record, particle_record_component};
}

std::tuple<std::string, std::string>
GetMeshRecordAndComponentNames(const VarInfo &vinfo, const TopologicalElement te,
                               const int comp_idx, const int level,
                               const int format_version) {
  // Default for cell centered fields is an empty string
  // to maintain backwards compatiblity with first iteration of
  // OpenPMD outputs.
  std::string te_str = "";
  if (te == TopologicalElement::F1) {
    te_str = "F1_";
  } else if (te == TopologicalElement::F2) {
    te_str = "F2_";
  } else if (te == TopologicalElement::F3) {
    te_str = "F3_";
  } else if (te == TopologicalElement::E1) {
    te_str = "E1_";
  } else if (te == TopologicalElement::E2) {
    te_str = "E2_";
  } else if (te == TopologicalElement::E3) {
    te_str = "E3_";
  } else if (te == TopologicalElement::NN) {
    te_str = "NN_";
  } else {
    PARTHENON_REQUIRE_THROWS(te == TopologicalElement::CC,
                             "Outputs for this type of TE not implemented.")
  }

  std::string mesh_record_name;
  std::string comp_name;

  if (format_version >= 2) {
    // Standard-compliant: one shared record per variable+te+level, components
    // distinguished only by sub-group name.
    mesh_record_name = vinfo.GetBaseName() + "_" + te_str + "lvl" + std::to_string(level);

    if (vinfo.is_vector && vinfo.num_components == 3) {
      if (comp_idx == 0) {
        comp_name = "x";
      } else if (comp_idx == 1) {
        comp_name = "y";
      } else if (comp_idx == 2) {
        comp_name = "z";
      } else {
        PARTHENON_THROW("Expected component index doesn't match vector expectation.");
      }
    } else if (vinfo.num_components == 1) {
      comp_name = openPMD::MeshRecordComponent::SCALAR;
    } else {
      // Multi-component non-vector (tensor or arbitrary shape): extract the
      // user-provided suffix from the full component label.
      // component_labels are built as GetBaseName() + "_" + suffix in VarInfo.
      const auto &full = vinfo.component_labels[comp_idx];
      const auto prefix_len = vinfo.GetBaseName().length() + 1; // skip "<base_name>_"
      comp_name = full.substr(prefix_len);
    }
  } else {
    // Legacy format (version 1): each component gets its own record with the
    // component label embedded in the record name.
    mesh_record_name = vinfo.label + "_" + te_str + vinfo.component_labels[comp_idx] +
                       "_lvl" + std::to_string(level);

    if (vinfo.is_vector && vinfo.num_components == 3) {
      if (comp_idx == 0) {
        comp_name = "x";
      } else if (comp_idx == 1) {
        comp_name = "y";
      } else if (comp_idx == 2) {
        comp_name = "z";
      } else {
        PARTHENON_THROW("Expected component index doesn't match vector expectation.");
      }
    } else {
      comp_name = openPMD::MeshRecordComponent::SCALAR;
    }
  }

  CheckValidName(mesh_record_name);
  CheckValidName(comp_name);
  return {mesh_record_name, comp_name};
}

std::tuple<openPMD::Offset, openPMD::Extent>
GetChunkOffsetAndExtent(Mesh *pm, std::shared_ptr<MeshBlock> pmb,
                        const TopologicalElement te, const int coarsening_factor,
                        const DumpOutputMode mode) {
  openPMD::Offset chunk_offset;
  openPMD::Extent chunk_extent;
  const auto loc = pm->Forest().GetLegacyTreeLocation(pmb->loc);
  uint64_t nx1_eff = pmb->block_size.nx(X1DIR) / coarsening_factor;
  uint64_t nx2_eff = pmb->block_size.nx(X2DIR) / coarsening_factor;
  uint64_t nx3_eff = pmb->block_size.nx(X3DIR) / coarsening_factor;
  if (pm->ndim == 3) {
    chunk_offset = {loc.lx3() * nx3_eff, loc.lx2() * nx2_eff, loc.lx1() * nx1_eff};
    chunk_extent = {nx3_eff + TopologicalOffsetK(te), nx2_eff + TopologicalOffsetJ(te),
                    nx1_eff + TopologicalOffsetI(te)};
  } else if (pm->ndim == 2) {
    chunk_offset = {loc.lx2() * nx2_eff, loc.lx1() * nx1_eff};
    chunk_extent = {static_cast<uint64_t>(nx2_eff + TopologicalOffsetJ(te)),
                    static_cast<uint64_t>(nx1_eff + TopologicalOffsetI(te))};
  } else {
    PARTHENON_THROW("1D output for openpmd not yet supported.");
  }
  int remove_comp = -1;
  if (mode == DumpOutputMode::X1Slice) {
    remove_comp = 2;
  } else if (mode == DumpOutputMode::X2Slice) {
    remove_comp = 1;
  } else if (mode == DumpOutputMode::X3Slice) {
    remove_comp = 0;
  }
  if (remove_comp >= 0) {
    chunk_extent.erase(chunk_extent.begin() + remove_comp);
    chunk_offset.erase(chunk_offset.begin() + remove_comp);
  }
  return {chunk_offset, chunk_extent};
}
} // namespace OpenPMDUtils

//----------------------------------------------------------------------------------------
//! \fn void OpenPMDOutput:::WriteOutputFile(Mesh *pm)
//  \brief  Write output in OpenPMD format
void OpenPMDOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin, SimTime *tm,
                                    const SignalHandler::OutputSignal signal) {
  if (output_params.single_precision_output) {
    this->template WriteOutputFileImpl<true>(pm, pin, tm, signal);
  } else {
    this->template WriteOutputFileImpl<false>(pm, pin, tm, signal);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void OpenPMDOutput:::WriteOutputFile(Mesh *pm)
//  \brief  Write output in OpenPMD format
template <bool WRITE_SINGLE_PRECISION>
void OpenPMDOutput::WriteOutputFileImpl(Mesh *pm, ParameterInput *pin, SimTime *tm,
                                        const SignalHandler::OutputSignal signal) {
  if constexpr (WRITE_SINGLE_PRECISION) {
    Kokkos::Profiling::pushRegion("OPMD::WriteOutputFileSinglePrec");
  } else {
    Kokkos::Profiling::pushRegion("OPMD::WriteOutputFileRealPrec");
  }
  // Check that the parameter input is safe to write (i.e., consistent across ranks)
  OutputUtils::CheckParameterInputConsistent(pin);

  using openPMD::Access;
  using openPMD::Series;

  // TODO(pgrete) check if CREATE is the correct pattern (for not overwriting the series
  // but an interation) This just describes the pattern of the filename. The correct file
  // will be accessed through the iteration idx below. The file suffix maps to the chosen
  // backend.
  // Prepending @ indicates that the config is a file to be read and parsed.
  std::string backend_config =
      backend_config_ == "default" ? "{}" : "@" + backend_config_;

  auto filename = output_params.file_basename + "." + output_params.file_id;

  // Write meta file (to be used by ParaView and Visit to recognize time series)
  if (Globals::my_rank == 0) {
    const auto meta_filename = filename + ".pmd";
    if (!fs::is_regular_file(meta_filename)) {
      std::ofstream outfile(meta_filename);
      outfile << filename << ".%05T.bp\n";
      outfile.close();
    }
  }
  if (signal == SignalHandler::OutputSignal::now) {
    filename.append(".now");
  } else if (signal == SignalHandler::OutputSignal::final &&
             output_params.file_label_final) {
    filename.append(".final");
  }
  filename.append(".%05T");

  filename.append(".bp");
  Series series = Series(filename, Access::CREATE,
#ifdef MPI_PARALLEL
                         MPI_COMM_WORLD,
#endif
                         backend_config);
  // TODO(pgrete) How to handle downstream info, e.g.,  on how/what defines a vector?
  // TODO(pgrete) Should we update for restart or only set this once? Or make it per
  // iteration?

  // Set default info when present
  if (pin->DoesParameterExist("parthenon/job", "author")) {
    const auto author = pin->Get<std::string>("parthenon/job", "author");
    series.setAuthor(author);
  }
  if (pin->DoesParameterExist("parthenon/job", "comment")) {
    const auto comment = pin->Get<std::string>("parthenon/job", "comment");
    series.setComment(comment);
  }
  if (pin->DoesParameterExist("parthenon/job", "machine")) {
    const auto machine = pin->Get<std::string>("parthenon/job", "machine");
    series.setMachine(machine);
  }
  series.setSoftware("Parthenon + X");
  const auto now = std::chrono::system_clock::now();
  series.setDate(std::format("{:%F %T}", now));

  // TODO(someone) Handle units

  // In line with existing outputs, we write one file per iteration/snapshot
  series.setIterationEncoding(openPMD::IterationEncoding::fileBased);

  // open iteration (corresponding to a timestep in OpenPMD naming)
  auto it = series.iterations[output_params.file_number];
  it.open(); // explicit open() is important when run in parallel

  if (signal == SignalHandler::OutputSignal::none) {
    // After file has been opened with the current number, already advance output
    // parameters so that for restarts the file is not immediatly overwritten again.
    // Only applies to default time-based data dumps, so that writing "now" and "final"
    // outputs does not change the desired output numbering.
    UpdateNextOutput_(pm, tm);
  }

  auto const &first_block = *(pm->block_list.front());

  // TODO(?) in principle, we could abstract this to a more general WriteAttributes place
  // and reuse for hdf5 and OpenPMD output with corresponing calls
  // -------------------------------------------------------------------------------- //
  //   WRITING ATTRIBUTES                                                             //
  // -------------------------------------------------------------------------------- //

  // Note, that profiling is likely skewed as data is actually written to disk/flushed
  // only later.
  Kokkos::Profiling::pushRegion("write Attributes");
  it.setAttribute("OutputFormatVersion", format_version_);

  // First the ones required by the OpenPMD standard
  if (tm != nullptr) {
    it.setTime(tm->time);
    it.setDt(tm->dt);
    it.setAttribute("NCycle", tm->ncycle);
  } else {
    it.setTime(-1.0);
    it.setDt(-1.0);
  }

  using enum DumpOutputMode;
  const auto is_slice = output_params.mode == X1Slice || output_params.mode == X2Slice ||
                        output_params.mode == X3Slice;
  auto slice_loc = std::numeric_limits<Real>::signaling_NaN();
  if (is_slice) {
    PARTHENON_REQUIRE_THROWS(pm->ndim == 3, "Slices are only implemented in 3D");
    slice_loc = pin->GetReal(output_params.block_name, "slice_loc");
  }

  auto in_output = [&](const Coordinates_t &coords, const int k, const int j, const int i,
                       const int width) {
    if (!is_slice) return true;

    if (output_params.mode == X1Slice) {
      return slice_loc >= coords.Xf<X1DIR>(k, j, i) &&
             slice_loc < coords.Xf<X1DIR>(k, j, i + width);
    } else if (output_params.mode == X2Slice) {
      return slice_loc >= coords.Xf<X2DIR>(k, j, i) &&
             slice_loc < coords.Xf<X2DIR>(k, j + width, i);
    } else if (output_params.mode == X3Slice) {
      return slice_loc >= coords.Xf<X3DIR>(k, j, i) &&
             slice_loc < coords.Xf<X3DIR>(k + width, j, i);
    }
    PARTHENON_FAIL("Unclear how I got here.");
  };

  if (!is_slice) {
    PARTHENON_INSTRUMENT_REGION("Dump Params");

    for (const auto &[pkg_name, pkg] : pm->packages.AllPackages()) {
      const auto &params = pkg->AllParams();
      OpenPMDUtils::WriteAllParams(params, pkg_name, &it);
    }
  }
  // Then our own
  if (!is_slice) {
    PARTHENON_INSTRUMENT_REGION("write input");
    // write input key-value pairs
    std::ostringstream oss;
    pin->ParameterDump(oss);
    it.setAttribute("InputFile", oss.str());
  }

  if (!is_slice) {
    // It's not clear we need all these attributes, but they mirror what's done in the
    // hdf5 output.

    // Writing build and provenance information
    it.setAttribute("ParthenonGitHash", provenance::PARTHENON_GIT_HASH);
    it.setAttribute("ParthenonGitBranch", provenance::PARTHENON_GIT_BRANCH);
    it.setAttribute("ParthenonCompiler", provenance::PARTHENON_COMPILER);
    it.setAttribute("ParthenonBuildTimestamp", provenance::PARTHENON_BUILD_TIMESTAMP);
    it.setAttribute("ParthenonBuildArch", provenance::PARTHENON_ARCH);
    it.setAttribute("ParthenonBuildOptLevel", provenance::PARTHENON_OPTIMIZATION);

    // Pull out Kokkos config which can contain GPU information
    std::ostringstream kokkos_config;
    Kokkos::print_configuration(kokkos_config);
    it.setAttribute("KokkosConfig", kokkos_config.str());

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
    it.setAttribute("CoarseningFactor", coarsening_factor_);

    // Mesh block size
    // TODO(pgrete) Check if we potentially can modify this to restart from coarse outs
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
    auto arr_to_vec = [](const auto &arr) {
      std::vector<std::string> vec(BOUNDARY_NFACES);
      for (int i = 0; i < BOUNDARY_NFACES; i++) {
        vec[i] = arr.at(i);
      }
      return vec;
    };
    it.setAttribute("BoundaryConditions", arr_to_vec(pm->mesh_bc_names));
    it.setAttribute("SwarmBoundaryConditions", arr_to_vec(pm->mesh_swarm_bc_names));
  } // Info section

  Kokkos::Profiling::popRegion(); // write Attributes

  // Write block metadata
  if (!is_slice) {
    // Manually gather all block data first as it allows to use the (simpler)
    // Attribute interface rather than writing a distributed dataset -- especially as all
    // data is being read on restart by every rank anyway.
    std::vector<int64_t> loc_local = OutputUtils::ComputeLocs(pm);
    auto loc_global = FlattenedLocalToGlobal<int64_t>(pm, loc_local);
    it.setAttribute("loc.lx123", loc_global);

    std::vector<int> id_local = OutputUtils::ComputeIDsAndFlags(pm);
    auto id_global = FlattenedLocalToGlobal<int>(pm, id_local);
    it.setAttribute("loc.level-gid-lid-cnghost-gflag", id_global);

    // derefinement count
    std::vector<int> derefcnt_local = OutputUtils::ComputeDerefinementCount(pm);
    auto derefcnt_global = FlattenedLocalToGlobal<int>(pm, derefcnt_local);
    it.setAttribute("derefinement_count", derefcnt_global);
  }

  const int num_blocks_local = static_cast<int>(pm->block_list.size());

  // -------------------------------------------------------------------------------- //
  //   WRITING VARIABLES DATA                                                         //
  // -------------------------------------------------------------------------------- //
  Kokkos::Profiling::pushRegion("write all variable data");

  const auto &bounds = pm->block_list.front()->cellbounds;
  const auto &f_bounds = pm->block_list.front()->f_cellbounds;

  // All blocks have the same list of variable metadata that exist in the entire
  // simulation, but not all variables may be allocated on all blocks

  auto get_vars = [=, this](const std::shared_ptr<MeshBlock> pmb) {
    const auto &data = pmb->meshblock_data.Get("base");
    const VariableVector<Real> &var_vec = data->GetVariableVector();
    VariableVector<Real> coords_vars =
        GetAnyVariables(var_vec, {parthenon::Metadata::CoordinatesVec});
    PARTHENON_DEBUG_REQUIRE(
        coords_vars.size() == 0,
        "Writing/handling explicit coordinate is currently not handled in OpenPMD "
        "output. Please get in touch on GitHub if there's a use case.");
    VariableVector<Real> fine_vars =
        GetAnyVariables(var_vec, {parthenon::Metadata::Fine});
    PARTHENON_DEBUG_REQUIRE(
        fine_vars.size() == 0,
        "Writing/handling explicit Fine fields is currently not handled in OpenPMD "
        "output. Please get in touch on GitHub if there's a use case.");

    VariableVector<Real> out;
    // Dump required vars for restarts or use those vars as default if none are given
    // (e.g, for slices or data dumps)
    if (output_params.mode == Restart || output_params.variables.empty()) {
      // get all vars with flag Independent OR restart
      out = GetAnyVariables(
          var_vec, {parthenon::Metadata::Independent, parthenon::Metadata::Restart});
    }

    // Always add any (additional) variables specified manually
    auto extra_vars = GetAnyVariables(var_vec, output_params.variables);
    for (auto &pextra_var : extra_vars) {
      if (std::none_of(out.begin(), out.end(), [&](const auto &pout_var) {
            return pextra_var->label() == pout_var->label();
          })) {
        out.push_back(pextra_var);
      }
    }

    return out;
  };

  // get list of all vars, just use first block as the list is the same for all blocks
  auto all_vars_info =
      VarInfo::GetAll(get_vars(pm->block_list.front()), bounds, f_bounds);

  // Mirroring the SparseInfo handling in HDF5 here.
  // Could probably made easier by just sequentially filling vectors, but better be safe
  // than sorry.
  //
  // We need to add information about the sparse variables to the output file, namely:
  // 1) Which variables are sparse
  // 2) Is a sparse id of a particular sparse variable allocated on a given block
  //
  // This information is stored in the dataset called "SparseInfo". The data set
  // contains an attribute "SparseFields" that is a vector of strings with the names
  // of the sparse fields (field name with sparse id, i.e. "bar_28", "bar_7", foo_1",
  // "foo_145"). The field names are in alphabetical order, which is the same order
  // they show up in all_unique_vars (because it's a sorted set).
  //
  // The dataset SparseInfo itself is a 2D array of bools. The first index is the
  // global block index and the second index is the sparse field (same order as the
  // SparseFields attribute). SparseInfo[b][v] is true if the sparse field with index
  // v is allocated on the block with index b, otherwise the value is false.
  // If the logic here is ever updated, ensure to update the HDF5 logic, too.
  std::vector<std::string> sparse_names;
  std::unordered_map<std::string, size_t> sparse_field_idx;
  for (auto &vinfo : all_vars_info) {
    if (vinfo.is_sparse) {
      sparse_field_idx.insert({vinfo.label, sparse_names.size()});
      sparse_names.push_back(vinfo.label);
    }
  }
  auto num_sparse = sparse_names.size();
  // Note, we're using int8_t here to circument the global reduction of a bool vector,
  // which would require much more boilerplate.
  std::vector<int8_t> sparse_allocated(num_blocks_local * num_sparse);
  std::vector<int> sparse_dealloc_count(num_blocks_local * num_sparse);

  // We're currently writing (flushing) one var at a time. This saves host memory but
  // results more smaller write. Might be updated in the future.
  // Allocate space for largest size variable
  // Could in principle be reduced for coarsended outputs, but lets better be safe than
  // sorry given the edge cases with non cell centered vars.
  std::size_t var_size_max = 0;
  for (auto &vinfo : all_vars_info) {
    const auto var_size = vinfo.Size();
    var_size_max = std::max(var_size_max, var_size);
  }

  using OutT = typename std::conditional<WRITE_SINGLE_PRECISION, float, Real>::type;
  std::vector<OutT> tmp_data(var_size_max * static_cast<std::size_t>(num_blocks_local));

  // Pre-pass (sparse only): determine globally which (sparse var, level) combinations
  // have allocated data so all ranks create or skip mesh records uniformly.
  // Dense variables are always allocated on every block — no reduction needed.
  // Skipped entirely when sparse is disabled at compile time.
  const int num_levels = pm->GetCurrentLevel() - pm->GetRootLevel() + 1;
  // Flat layout: sparse_level_has_data[s_idx * num_levels + level] = 1 if any block
  // on any rank has sparse_names[s_idx] allocated at that level.
  std::vector<int8_t> sparse_level_has_data;
#ifndef PARTHENON_DISABLE_SPARSE
  if (num_sparse > 0) {
    sparse_level_has_data.assign(num_sparse * num_levels, 0);
    for (const auto &[label, s_idx] : sparse_field_idx) {
      for (const auto &pmb : pm->block_list) {
        const int level = pmb->loc.level() - pm->GetRootLevel();
        if (pmb->meshblock_data.Get()->GetVarPtr(label)->IsAllocated())
          sparse_level_has_data[s_idx * num_levels + level] = 1;
      }
    }
#ifdef MPI_PARALLEL
    PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, sparse_level_has_data.data(),
                                      static_cast<int>(sparse_level_has_data.size()),
                                      MPI_INT8_T, MPI_MAX, MPI_COMM_WORLD));
#endif
  }
#endif

  // for each variable we write
  for (auto &vinfo : all_vars_info) {
    PARTHENON_INSTRUMENT_REGION("Write variable loop")

    // Reset host write bufer. Not really necessary, but doesn't hurt.
    memset(tmp_data.data(), 0, tmp_data.size() * sizeof(OutT));
    uint64_t tmp_offset = 0;

    if (vinfo.is_vector) {
      // sanity check
      PARTHENON_REQUIRE_THROWS(
          vinfo.GetDim(5) == 1 && vinfo.GetDim(6) == 1,
          "A 'standard' vector is expected to not have higher dimensional indices.")
    }

    // for each local mesh block
    for (size_t b_idx = 0; b_idx < num_blocks_local; ++b_idx) {
      const auto &pmb = pm->block_list[b_idx];
      auto pmb_ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
      auto pmb_jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
      auto pmb_kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
      const int pmb_width = output_params.mode == X1Slice   ? pmb_ib.e - pmb_ib.s + 1
                            : output_params.mode == X2Slice ? pmb_jb.e - pmb_jb.s + 1
                                                            : pmb_kb.e - pmb_kb.s + 1;
      if (!in_output(pmb->coords, pmb_kb.s, pmb_jb.s, pmb_ib.s, pmb_width)) {
        continue;
      }

      // TODO(pgrete) check if we should skip the suffix for level 0
      const auto level = pmb->loc.level() - pm->GetRootLevel();

      auto out_var = pmb->meshblock_data.Get()->GetVarPtr(vinfo.label);

      for (const auto &te : vinfo.topological_elements) {
        for (int comp_idx = 0; comp_idx < vinfo.component_labels.size(); comp_idx++) {
          const auto [record_name, comp_name] =
              OpenPMDUtils::GetMeshRecordAndComponentNames(vinfo, te, comp_idx, level,
                                                           format_version_);

          // Skip creating a new mesh record when no rank has data for this
          // (sparse var, level) combination. Dense vars are always allocated so
          // they never hit this guard. The global flag (set by the MPI_Allreduce
          // pre-pass above) ensures all ranks make the same decision for a level.
#ifndef PARTHENON_DISABLE_SPARSE
          if (vinfo.is_sparse &&
              !sparse_level_has_data[sparse_field_idx.at(vinfo.label) * num_levels +
                                     level] &&
              !it.meshes.contains(record_name))
            continue;
#endif

          const bool new_record = !it.meshes.contains(record_name);
          auto mesh_record = it.meshes[record_name];

          // Set record-level attributes once (shared across all components)
          if (new_record) {
            PARTHENON_REQUIRE_THROWS(
                typeid(Coordinates_t) == typeid(UniformCartesian),
                "OpenPMD in Parthenon currently only supports Cartesian coordinates.");
            mesh_record.setGeometry(openPMD::Mesh::Geometry::cartesian);
            auto &coords = pmb->coords;
            // For Cartesian geometry, all dxN are const across the block so we just pick
            // the first index.
            Real dx1 = coords.CellWidth<X1DIR>(0, 0, 0) * coarsening_factor_;
            Real dx2 = coords.CellWidth<X2DIR>(0, 0, 0) * coarsening_factor_;
            Real dx3 = coords.CellWidth<X3DIR>(0, 0, 0) * coarsening_factor_;

            // TODO(pgrete) check if this should be tied to the MemoryLayout
            mesh_record.setDataOrder(openPMD::Mesh::DataOrder::C);

            if (pm->ndim == 3) {
              auto grid_spacing = std::vector<Real>{dx3, dx2, dx1};
              auto axis_labels = std::vector<std::string>{"z", "y", "x"};
              auto global_offset = std::vector<Real>{
                  pm->mesh_size.xmin(X3DIR),
                  pm->mesh_size.xmin(X2DIR),
                  pm->mesh_size.xmin(X1DIR),
              };
              int remove_comp = -1;
              if (output_params.mode == X1Slice) {
                remove_comp = 2;
              } else if (output_params.mode == X2Slice) {
                remove_comp = 1;
              } else if (output_params.mode == X3Slice) {
                remove_comp = 0;
              }
              if (remove_comp >= 0) {
                grid_spacing.erase(grid_spacing.begin() + remove_comp);
                axis_labels.erase(axis_labels.begin() + remove_comp);
                global_offset.erase(global_offset.begin() + remove_comp);
              }
              mesh_record.setGridSpacing(grid_spacing);
              mesh_record.setAxisLabels(axis_labels);
              mesh_record.setGridGlobalOffset(global_offset);
            } else if (pm->ndim == 2) {
              mesh_record.setGridSpacing(std::vector<Real>{dx2, dx1});
              mesh_record.setAxisLabels({"y", "x"});
              mesh_record.setGridGlobalOffset({
                  pm->mesh_size.xmin(X2DIR),
                  pm->mesh_size.xmin(X1DIR),
              });
            } else {
              PARTHENON_THROW("1D output for openpmd not yet supported.");
            }
            // TODO(pgrete) need unitDimension and timeOffset for this record?
          }

          // Per-component setup: position and dataset (each component needs its own)
          const bool new_comp = !mesh_record.contains(comp_name);
          if (new_comp) {
            auto mesh_comp = mesh_record[comp_name];
            auto effective_nx = static_cast<std::uint64_t>(std::pow(2, level));
            openPMD::Extent global_extent;
            if (pm->ndim == 3) {
              auto position = std::vector<Real>{0.5 - 0.5 * TopologicalOffsetK(te),
                                                0.5 - 0.5 * TopologicalOffsetJ(te),
                                                0.5 - 0.5 * TopologicalOffsetI(te)};
              global_extent = {
                  static_cast<std::uint64_t>(pm->mesh_size.nx(X3DIR) /
                                             coarsening_factor_) *
                          effective_nx +
                      TopologicalOffsetK(te),
                  static_cast<std::uint64_t>(pm->mesh_size.nx(X2DIR) /
                                             coarsening_factor_) *
                          effective_nx +
                      TopologicalOffsetJ(te),
                  static_cast<std::uint64_t>(pm->mesh_size.nx(X1DIR) /
                                             coarsening_factor_) *
                          effective_nx +
                      TopologicalOffsetI(te),
              };
              int remove_comp = -1;
              if (output_params.mode == X1Slice) {
                remove_comp = 2;
              } else if (output_params.mode == X2Slice) {
                remove_comp = 1;
              } else if (output_params.mode == X3Slice) {
                remove_comp = 0;
              }
              if (remove_comp >= 0) {
                position.erase(position.begin() + remove_comp);
                global_extent.erase(global_extent.begin() + remove_comp);
              }
              mesh_comp.setPosition(position);
            } else if (pm->ndim == 2) {
              mesh_comp.setPosition(
                  std::vector<Real>{0.5 - 0.5 * TopologicalOffsetJ(te),
                                    0.5 - 0.5 * TopologicalOffsetI(te)});
              global_extent = {
                  static_cast<std::uint64_t>(pm->mesh_size.nx(X2DIR) /
                                             coarsening_factor_) *
                          effective_nx +
                      TopologicalOffsetJ(te),
                  static_cast<std::uint64_t>(pm->mesh_size.nx(X1DIR) /
                                             coarsening_factor_) *
                          effective_nx +
                      TopologicalOffsetI(te),
              };
            } else {
              PARTHENON_THROW("1D output for openpmd not yet supported.");
            }
            // Handling this here to now re-reset dataset later when iterating through the
            // blocks
            auto const dataset =
                openPMD::Dataset(openPMD::determineDatatype<OutT>(), global_extent);
            // TODO(pgrete) check whether this should/need to be a collective so that the
            // mesh generation should be done across all ranks prior to writing data,
            // rather than in-situ for the local blocks only
            mesh_comp.resetDataset(dataset);
          }
        }
      }

      // Now that the mesh record exists, actually write the data
      if (out_var->IsAllocated()) {
        auto &coords = pmb->coords;
        auto out_var_h = out_var->data.GetHostMirrorAndCopy();
        for (const auto &te : vinfo.topological_elements) {
          auto ib = bounds.GetBoundsI(IndexDomain::interior, te);
          auto jb = bounds.GetBoundsJ(IndexDomain::interior, te);
          auto kb = bounds.GetBoundsK(IndexDomain::interior, te);
          int comp_idx = 0;
          const auto &Nt = out_var->GetDim(6);
          const auto &Nu = out_var->GetDim(5);
          const auto &Nv = out_var->GetDim(4);
          // loop over all components
          for (int t = 0; t < Nt; ++t) {
            for (int u = 0; u < Nu; ++u) {
              for (int v = 0; v < Nv; ++v) {
                const auto [record_name, comp_name] =
                    OpenPMDUtils::GetMeshRecordAndComponentNames(vinfo, te, comp_idx,
                                                                 level, format_version_);
                auto mesh_comp = it.meshes[record_name][comp_name];

                const auto comp_offset = tmp_offset;
                for (int k = kb.s; k <= kb.e; ++k) {
                  for (int j = jb.s; j <= jb.e; ++j) {
                    for (int i = ib.s; i <= ib.e; ++i) {
                      if (((i - ib.s) % coarsening_factor_ != 0) ||
                          ((j - jb.s) % coarsening_factor_ != 0) ||
                          ((k - kb.s) % coarsening_factor_ != 0) ||
                          !in_output(coords, k, j, i, coarsening_factor_)) {
                        continue;
                      }

                      tmp_data[tmp_offset] = static_cast<OutT>(
                          out_var_h(static_cast<int>(te) % 3, t, u, v, k, j, i));

                      tmp_offset++;
                    }
                  }
                }
                // if no data was being selected
                if (comp_offset == tmp_offset) {
                  continue;
                }
                const auto [chunk_offset, chunk_extent] =
                    OpenPMDUtils::GetChunkOffsetAndExtent(pm, pmb, te, coarsening_factor_,
                                                          output_params.mode);

                mesh_comp.storeChunkRaw(&tmp_data[comp_offset], chunk_offset,
                                        chunk_extent);
                comp_idx += 1;
              }
            }
          } // loop over components
        } // loop over topological elements
      } // out_var->IsAllocated()
      if (vinfo.is_sparse) {
        auto sparse_idx = sparse_field_idx.at(vinfo.label);
        sparse_allocated.at(b_idx * num_sparse + sparse_idx) =
            static_cast<int8_t>(out_var->IsAllocated());
        sparse_dealloc_count.at(b_idx * num_sparse + sparse_idx) = out_var->dealloc_count;
      }
    } // loop over blocks
    it.seriesFlush();
  } // loop over vars
  Kokkos::Profiling::popRegion(); // write all variable data

  // -------------------------------------------------------------------------------- //
  //   WRITING Sparse metadata                                                        //
  // -------------------------------------------------------------------------------- //
  if (!is_slice && num_sparse > 0) {
    auto sparse_allocated_global = FlattenedLocalToGlobal<int8_t>(pm, sparse_allocated);
    it.setAttribute("SparseInfo", sparse_allocated_global);
    it.setAttribute("SparseFields", sparse_names);
    auto sparse_dealloc_count_global =
        FlattenedLocalToGlobal<int>(pm, sparse_dealloc_count);
    it.setAttribute("SparseDeallocCount", sparse_dealloc_count_global);
  }

  // -------------------------------------------------------------------------------- //
  //   WRITING PARTICLE DATA                                                          //
  // -------------------------------------------------------------------------------- //
  if (!is_slice) {
    Kokkos::Profiling::pushRegion("write particle data");
    std::map<std::string, SwarmInfo> swarm_infos;

    // Dump required vars for restarts or use those vars as default if none are given
    if (output_params.mode == Restart || output_params.swarms.empty()) {
      AllSwarmInfo all_swarm_info(pm->block_list, output_params.swarms,
                                  DumpOutputMode::Restart);
      std::copy_if(
          std::make_move_iterator(all_swarm_info.all_info.begin()),
          std::make_move_iterator(all_swarm_info.all_info.end()),
          std::inserter(swarm_infos, swarm_infos.end()),
          [&swarm_infos](auto const &kv) { return swarm_infos.count(kv.first) == 0; });
    }

    // Always add any (additional) variables specified manually
    {
      AllSwarmInfo all_swarm_info(pm->block_list, output_params.swarms,
                                  DumpOutputMode::Data);
      std::copy_if(
          std::make_move_iterator(all_swarm_info.all_info.begin()),
          std::make_move_iterator(all_swarm_info.all_info.end()),
          std::inserter(swarm_infos, swarm_infos.end()),
          [&swarm_infos](auto const &kv) { return swarm_infos.count(kv.first) == 0; });
    }

    for (auto &[swname, swinfo] : swarm_infos) {
      openPMD::ParticleSpecies swm = it.particles[swname];
      // These indicate particles/meshblock and location in global index
      // space where each meshblock starts
      auto counts_global = FlattenedLocalToGlobal<std::size_t>(pm, swinfo.counts);
      swm.setAttribute("counts", counts_global);
      auto offsets_global = FlattenedLocalToGlobal<std::size_t>(pm, swinfo.offsets);
      swm.setAttribute("offsets", offsets_global);

      if (swinfo.global_count == 0) {
        continue;
      }

      OpenPMDUtils::WriteSwarmVar<int>(swinfo, swm, it);
      OpenPMDUtils::WriteSwarmVar<uint64_t>(swinfo, swm, it);
      OpenPMDUtils::WriteSwarmVar<Real>(swinfo, swm, it);

      // From the HDF5 output:
      // If swarm does not contain an "id" object, generate a sequential
      // one for vis.
      // BUT PG: this may break things in unpredicable ways
      // I'm in favor of enforcing a global id somehow. We shold discuss.
      PARTHENON_REQUIRE_THROWS(swinfo.var_info.count(swarm_position::id::name()) != 0 ||
                                   swinfo.var_info.count("id") != 0,
                               "Particles should always carry a unique, persistent id!");
    }
    Kokkos::Profiling::popRegion(); // write particle data
  }
  // The iteration can be closed in order to help free up resources.
  // The iteration's content will be flushed automatically.
  // An iteration once closed cannot (yet) be reopened.
  it.close();
  series.close();
  Kokkos::Profiling::popRegion(); // WriteOutputFile???Prec
}
// explicit template instantiation
template void
OpenPMDOutput::WriteOutputFileImpl<true>(Mesh *pm, ParameterInput *pin, SimTime *tm,
                                         const SignalHandler::OutputSignal signal);
template void
OpenPMDOutput::WriteOutputFileImpl<false>(Mesh *pm, ParameterInput *pin, SimTime *tm,
                                          const SignalHandler::OutputSignal signal);

} // namespace parthenon
