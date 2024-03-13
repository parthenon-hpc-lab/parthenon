//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2023 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
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

// options for building
#include "config.hpp"
#include "globals.hpp"
#include "utils/error_checking.hpp"

// Only proceed if HDF5 output enabled
#ifdef ENABLE_HDF5

#include <algorithm>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <tuple>
#include <type_traits>
#include <unordered_map>

#include "driver/driver.hpp"
#include "interface/metadata.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "outputs/output_utils.hpp"
#include "outputs/outputs.hpp"
#include "outputs/parthenon_hdf5.hpp"
#include "outputs/parthenon_xdmf.hpp"
#include "utils/string_utils.hpp"

namespace parthenon {

void PHDF5Output::WriteOutputFile(Mesh *pm, ParameterInput *pin, SimTime *tm,
                                  const SignalHandler::OutputSignal signal) {
  using namespace HDF5;
  if (output_params.single_precision_output) {
    this->template WriteOutputFileImpl<true>(pm, pin, tm, signal);
  } else {
    this->template WriteOutputFileImpl<false>(pm, pin, tm, signal);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void PHDF5Output:::WriteOutputFileImpl(Mesh *pm, ParameterInput *pin, bool flag)
//  \brief Cycles over all MeshBlocks and writes OutputData in the Parthenon HDF5 format,
//         one file per output using parallel IO.
template <bool WRITE_SINGLE_PRECISION>
void PHDF5Output::WriteOutputFileImpl(Mesh *pm, ParameterInput *pin, SimTime *tm,
                                      const SignalHandler::OutputSignal signal) {
  using namespace HDF5;
  using namespace OutputUtils;

  if constexpr (WRITE_SINGLE_PRECISION) {
    Kokkos::Profiling::pushRegion("PHDF5::WriteOutputFileSinglePrec");
  } else {
    Kokkos::Profiling::pushRegion("PHDF5::WriteOutputFileRealPrec");
  }

  // writes all graphics variables to hdf file
  // HDF5 structures
  // Also writes companion xdmf file

  const int max_blocks_global = pm->nbtotal;
  const int num_blocks_local = static_cast<int>(pm->block_list.size());

  const IndexDomain theDomain =
      (output_params.include_ghost_zones ? IndexDomain::entire : IndexDomain::interior);

  auto const &first_block = *(pm->block_list.front());

  const IndexRange out_ib = first_block.cellbounds.GetBoundsI(theDomain);
  const IndexRange out_jb = first_block.cellbounds.GetBoundsJ(theDomain);
  const IndexRange out_kb = first_block.cellbounds.GetBoundsK(theDomain);

  auto const nx1 = out_ib.e - out_ib.s + 1;
  auto const nx2 = out_jb.e - out_jb.s + 1;
  auto const nx3 = out_kb.e - out_kb.s + 1;

  const int rootLevel = pm->GetRootLevel();
  const int max_level = pm->GetCurrentLevel() - rootLevel;
  const auto &nblist = pm->GetNbList();

  // open HDF5 file
  // Define output filename
  auto filename = GenerateFilename_(pin, tm, signal);

  // set file access property list
  H5P const acc_file = H5P::FromHIDCheck(HDF5::GenerateFileAccessProps());

  // now create the file
  H5F file;
  try {
    file = H5F::FromHIDCheck(
        H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, acc_file));
  } catch (std::exception &ex) {
    std::stringstream err;
    err << "### ERROR: Failed to create HDF5 output file '" << filename
        << "' with the following error:" << std::endl
        << ex.what() << std::endl;
    PARTHENON_THROW(err)
  }

  // -------------------------------------------------------------------------------- //
  //   WRITING ATTRIBUTES                                                             //
  // -------------------------------------------------------------------------------- //
  Kokkos::Profiling::pushRegion("write Attributes");
  {
    Kokkos::Profiling::pushRegion("write input");
    // write input key-value pairs
    std::ostringstream oss;
    pin->ParameterDump(oss);

    // Mesh information
    const H5G input_group = MakeGroup(file, "/Input");

    HDF5WriteAttribute("File", oss.str().c_str(), input_group);
    Kokkos::Profiling::popRegion(); // write input
  }                                 // Input section

  // we'll need this again at the end
  const H5G info_group = MakeGroup(file, "/Info");
  {
    Kokkos::Profiling::pushRegion("write Info");
    HDF5WriteAttribute("OutputFormatVersion", OUTPUT_VERSION_FORMAT, info_group);

    if (tm != nullptr) {
      HDF5WriteAttribute("NCycle", tm->ncycle, info_group);
      HDF5WriteAttribute("Time", tm->time, info_group);
      HDF5WriteAttribute("dt", tm->dt, info_group);
    }

    HDF5WriteAttribute("WallTime", Driver::elapsed_main(), info_group);
    HDF5WriteAttribute("NumDims", pm->ndim, info_group);
    HDF5WriteAttribute("NumMeshBlocks", pm->nbtotal, info_group);
    HDF5WriteAttribute("MaxLevel", max_level, info_group);
    // write whether we include ghost cells or not
    HDF5WriteAttribute("IncludesGhost", output_params.include_ghost_zones ? 1 : 0,
                       info_group);
    // write number of ghost cells in simulation
    HDF5WriteAttribute("NGhost", Globals::nghost, info_group);
    HDF5WriteAttribute("Coordinates", std::string(first_block.coords.Name()).c_str(),
                       info_group);

    // restart info, write always
    HDF5WriteAttribute("NBNew", pm->nbnew, info_group);
    HDF5WriteAttribute("NBDel", pm->nbdel, info_group);
    HDF5WriteAttribute("RootLevel", rootLevel, info_group);
    HDF5WriteAttribute("Refine", pm->adaptive ? 1 : 0, info_group);
    HDF5WriteAttribute("Multilevel", pm->multilevel ? 1 : 0, info_group);

    HDF5WriteAttribute("BlocksPerPE", nblist, info_group);

    // Mesh block size
    HDF5WriteAttribute("MeshBlockSize", std::vector<int>{nx1, nx2, nx3}, info_group);

    // RootGridDomain - float[9] array with xyz mins, maxs, rats (dx(i)/dx(i-1))
    HDF5WriteAttribute(
        "RootGridDomain",
        std::vector<Real>{pm->mesh_size.xmin(X1DIR), pm->mesh_size.xmax(X1DIR),
                          pm->mesh_size.xrat(X1DIR), pm->mesh_size.xmin(X2DIR),
                          pm->mesh_size.xmax(X2DIR), pm->mesh_size.xrat(X2DIR),
                          pm->mesh_size.xmin(X3DIR), pm->mesh_size.xmax(X3DIR),
                          pm->mesh_size.xrat(X3DIR)},
        info_group);

    // Root grid size (number of cells at root level)
    HDF5WriteAttribute("RootGridSize",
                       std::vector<int>{pm->mesh_size.nx(X1DIR), pm->mesh_size.nx(X2DIR),
                                        pm->mesh_size.nx(X3DIR)},
                       info_group);

    // Boundary conditions
    std::vector<std::string> boundary_condition_str(BOUNDARY_NFACES);
    for (size_t i = 0; i < boundary_condition_str.size(); i++) {
      boundary_condition_str[i] = GetBoundaryString(pm->mesh_bcs[i]);
    }

    HDF5WriteAttribute("BoundaryConditions", boundary_condition_str, info_group);
    Kokkos::Profiling::popRegion(); // write Info
  }                                 // Info section

  // write Params
  {
    Kokkos::Profiling::pushRegion("behold: write Params");
    const H5G params_group = MakeGroup(file, "/Params");

    for (const auto &package : pm->packages.AllPackages()) {
      const auto state = package.second;
      // Write all params that can be written as HDF5 attributes
      state->AllParams().WriteAllToHDF5(state->label(), params_group);
    }
    Kokkos::Profiling::popRegion(); // behold: write Params
  }                                 // Params section
  Kokkos::Profiling::popRegion();   // write Attributes

  // -------------------------------------------------------------------------------- //
  //   WRITING MESHBLOCK METADATA                                                     //
  // -------------------------------------------------------------------------------- //

  // set local offset, always the same for all data sets
  hsize_t my_offset = 0;
  for (int i = 0; i < Globals::my_rank; i++) {
    my_offset += nblist[i];
  }

  H5P const pl_xfer = H5P::FromHIDCheck(H5Pcreate(H5P_DATASET_XFER));
  H5P const pl_dcreate = H5P::FromHIDCheck(H5Pcreate(H5P_DATASET_CREATE));

  // Never write fill values to the dataset
  PARTHENON_HDF5_CHECK(H5Pset_fill_time(pl_dcreate, H5D_FILL_TIME_NEVER));

#ifdef MPI_PARALLEL
  PARTHENON_HDF5_CHECK(H5Pset_dxpl_mpio(pl_xfer, H5FD_MPIO_COLLECTIVE));
#endif

  WriteBlocksMetadata_(pm, file, pl_xfer, my_offset, max_blocks_global);
  WriteCoordinates_(pm, theDomain, file, pl_xfer, my_offset, max_blocks_global);
  WriteLevelsAndLocs_(pm, file, pl_xfer, my_offset, max_blocks_global);

  // -------------------------------------------------------------------------------- //
  //   WRITING VARIABLES DATA                                                         //
  // -------------------------------------------------------------------------------- //
  Kokkos::Profiling::pushRegion("write all variable data");

  // All blocks have the same list of variable metadata that exist in the entire
  // simulation, but not all variables may be allocated on all blocks

  auto get_vars = [=](const std::shared_ptr<MeshBlock> pmb) {
    auto &var_vec = pmb->meshblock_data.Get()->GetVariableVector();
    if (restart_) {
      // get all vars with flag Independent OR restart
      return GetAnyVariables(
          var_vec, {parthenon::Metadata::Independent, parthenon::Metadata::Restart});
    } else {
      return GetAnyVariables(var_vec, output_params.variables);
    }
  };

  // get list of all vars, just use the first block since the list is the same for all
  // blocks
  std::vector<VarInfo> all_vars_info;
  const auto vars = get_vars(pm->block_list.front());
  for (auto &v : vars) {
    all_vars_info.emplace_back(v);
  }

  // sort alphabetically
  std::sort(all_vars_info.begin(), all_vars_info.end(),
            [](const VarInfo &a, const VarInfo &b) { return a.label < b.label; });

  // We need to add information about the sparse variables to the HDF5 file, namely:
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
  // v is allocated on the block with index b, otherwise the value is false

  std::vector<std::string> sparse_names;
  std::unordered_map<std::string, size_t> sparse_field_idx;
  for (auto &vinfo : all_vars_info) {
    if (vinfo.is_sparse) {
      sparse_field_idx.insert({vinfo.label, sparse_names.size()});
      sparse_names.push_back(vinfo.label);
    }
  }

  hsize_t num_sparse = sparse_names.size();
  // can't use std::vector here because std::vector<hbool_t> is the same as
  // std::vector<bool> and it doesn't have .data() member
  std::unique_ptr<hbool_t[]> sparse_allocated(new hbool_t[num_blocks_local * num_sparse]);

  // allocate space for largest size variable
  int varSize_max = 0;
  for (auto &vinfo : all_vars_info) {
    const int varSize = vinfo.Size();
    varSize_max = std::max(varSize_max, varSize);
  }

  using OutT = typename std::conditional<WRITE_SINGLE_PRECISION, float, Real>::type;
  std::vector<OutT> tmpData(varSize_max * num_blocks_local);

  // for each variable we write
  for (auto &vinfo : all_vars_info) {
    Kokkos::Profiling::pushRegion("write variable loop");
    // not really necessary, but doesn't hurt
    memset(tmpData.data(), 0, tmpData.size() * sizeof(OutT));

    const std::string var_name = vinfo.label;

    hsize_t local_offset[H5_NDIM];
    std::fill(local_offset + 1, local_offset + H5_NDIM, 0);
    local_offset[0] = my_offset;

    hsize_t local_count[H5_NDIM];
    local_count[0] = static_cast<hsize_t>(num_blocks_local);

    hsize_t global_count[H5_NDIM];
    global_count[0] = static_cast<hsize_t>(max_blocks_global);

    auto alldims = vinfo.GetShape<hsize_t>();

    int ndim = -1;
#ifndef PARTHENON_DISABLE_HDF5_COMPRESSION
    // we need chunks to enable compression
    std::array<hsize_t, H5_NDIM> chunk_size;
    std::fill(chunk_size.begin(), chunk_size.end(), 1);
#endif
    if (vinfo.where == MetadataFlag(Metadata::Cell)) {
      ndim = 3 + vinfo.tensor_rank + 1;
      for (int i = 0; i < vinfo.tensor_rank; i++) {
        local_count[1 + i] = global_count[1 + i] = alldims[alldims.size() - 3 - vinfo.tensor_rank + i];
      }
      local_count[vinfo.tensor_rank + 1] = global_count[vinfo.tensor_rank + 1] = nx3;
      local_count[vinfo.tensor_rank + 2] = global_count[vinfo.tensor_rank + 2] = nx2;
      local_count[vinfo.tensor_rank + 3] = global_count[vinfo.tensor_rank + 3] = nx1;

#ifndef PARTHENON_DISABLE_HDF5_COMPRESSION
      if (output_params.hdf5_compression_level > 0) {
        for (int i = ndim - 3; i < ndim; i++) {
          chunk_size[i] = local_count[i];
        }
      }
#endif
    } else if (vinfo.where == MetadataFlag(Metadata::None)) {
      ndim = vinfo.tensor_rank + 1;
      for (int i = 0; i < vinfo.tensor_rank; i++) {
        local_count[1 + i] = global_count[1 + i] = alldims[alldims.size() - vinfo.tensor_rank + i];
      }

#ifndef PARTHENON_DISABLE_HDF5_COMPRESSION
      if (output_params.hdf5_compression_level > 0) {
        int nchunk_indices = std::min<int>(vinfo.tensor_rank, 3);
        for (int i = ndim - nchunk_indices; i < ndim; i++) {
          chunk_size[i] = alldims[6 - nchunk_indices + i];
        }
      }
#endif
    } else {
      PARTHENON_THROW("Only Cell and None locations supported!");
    }

#ifndef PARTHENON_DISABLE_HDF5_COMPRESSION
    PARTHENON_HDF5_CHECK(H5Pset_chunk(pl_dcreate, ndim, chunk_size.data()));
    // Do not run the pipeline if compression is soft disabled.
    // By default data would still be passed, which may result in slower output.
    if (output_params.hdf5_compression_level > 0) {
      PARTHENON_HDF5_CHECK(
          H5Pset_deflate(pl_dcreate, std::min(9, output_params.hdf5_compression_level)));
    }
#endif

    // load up data
    hsize_t index = 0;

    Kokkos::Profiling::pushRegion("fill host output buffer");
    // for each local mesh block
    for (size_t b_idx = 0; b_idx < num_blocks_local; ++b_idx) {
      const auto &pmb = pm->block_list[b_idx];
      bool is_allocated = false;

      // for each variable that this local meshblock actually has
      const auto vars = get_vars(pmb);
      for (auto &v : vars) {
        // For reference, if we update the logic here, there's also
        // a similar block in parthenon_manager.cpp
        if (v->IsAllocated() && (var_name == v->label())) {
          auto v_h = v->data.GetHostMirrorAndCopy();
          OutputUtils::PackOrUnpackVar(
              pmb.get(), v.get(), output_params.include_ghost_zones, index, tmpData,
              [&](auto index, int t, int u, int v, int k, int j, int i) {
                tmpData[index] = static_cast<OutT>(v_h(t, u, v, k, j, i));
              });

          is_allocated = true;
          break;
        }
      }

      if (vinfo.is_sparse) {
        size_t sparse_idx = sparse_field_idx.at(vinfo.label);
        sparse_allocated[b_idx * num_sparse + sparse_idx] = is_allocated;
      }

      if (!is_allocated) {
        if (vinfo.is_sparse) {
          hsize_t varSize{};
          if (vinfo.where == MetadataFlag(Metadata::Cell)) {
            varSize = vinfo.TensorSize() * (out_kb.e - out_kb.s + 1) *
                      (out_jb.e - out_jb.s + 1) * (out_ib.e - out_ib.s + 1);
          } else {
            varSize = vinfo.Size();
          }
          auto fill_val =
              output_params.sparse_seed_nans ? std::numeric_limits<OutT>::quiet_NaN() : 0;
          std::fill(tmpData.data() + index, tmpData.data() + index + varSize, fill_val);
          index += varSize;
        } else {
          std::stringstream msg;
          msg << "### ERROR: Unable to find dense variable " << var_name << std::endl;
          PARTHENON_FAIL(msg);
        }
      }
    }
    Kokkos::Profiling::popRegion(); // fill host output buffer

    Kokkos::Profiling::pushRegion("write variable data");
    // write data to file
    HDF5WriteND(file, var_name, tmpData.data(), ndim, &local_offset[0], &local_count[0],
                &global_count[0], pl_xfer, pl_dcreate);
    Kokkos::Profiling::popRegion(); // write variable data
    Kokkos::Profiling::popRegion(); // write variable loop
  }
  Kokkos::Profiling::popRegion(); // write all variable data

  // names of variables
  std::vector<std::string> var_names;
  var_names.reserve(all_vars_info.size());

  // number of components within each dataset
  std::vector<size_t> num_components;
  num_components.reserve(all_vars_info.size());

  // names of components within each dataset
  std::vector<std::string> component_names;
  component_names.reserve(all_vars_info.size()); // may be larger

  for (const auto &vi : all_vars_info) {
    var_names.push_back(vi.label);

    const auto &component_labels = vi.component_labels;
    PARTHENON_REQUIRE_THROWS(component_labels.size() > 0, "Got 0 component labels");

    num_components.push_back(component_labels.size());
    for (const auto &label : component_labels) {
      component_names.push_back(label);
    }
  }

  HDF5WriteAttribute("NumComponents", num_components, info_group);
  HDF5WriteAttribute("ComponentNames", component_names, info_group);
  HDF5WriteAttribute("OutputDatasetNames", var_names, info_group);

  // write SparseInfo and SparseFields (we can't write a zero-size dataset, so only write
  // this if we have sparse fields)
  if (num_sparse > 0) {
    WriteSparseInfo_(pm, sparse_allocated.get(), sparse_names, num_sparse, file, pl_xfer,
                     my_offset, max_blocks_global);
  } // SparseInfo and SparseFields sections

  // -------------------------------------------------------------------------------- //
  //   WRITING PARTICLE DATA                                                          //
  // -------------------------------------------------------------------------------- //

  Kokkos::Profiling::pushRegion("write particle data");
  AllSwarmInfo swarm_info(pm->block_list, output_params.swarms, restart_);
  for (auto &[swname, swinfo] : swarm_info.all_info) {
    const H5G g_swm = MakeGroup(file, swname);
    // offsets/counts are NOT the same here vs the grid data
    hsize_t local_offset[6] = {static_cast<hsize_t>(my_offset), 0, 0, 0, 0, 0};
    hsize_t local_count[6] = {static_cast<hsize_t>(num_blocks_local), 0, 0, 0, 0, 0};
    hsize_t global_count[6] = {static_cast<hsize_t>(max_blocks_global), 0, 0, 0, 0, 0};
    // These indicate particles/meshblock and location in global index
    // space where each meshblock starts
    HDF5Write1D(g_swm, "counts", swinfo.counts.data(), local_offset, local_count,
                global_count, pl_xfer);
    HDF5Write1D(g_swm, "offsets", swinfo.offsets.data(), local_offset, local_count,
                global_count, pl_xfer);

    const H5G g_var = MakeGroup(g_swm, "SwarmVars");
    if (swinfo.global_count == 0) {
      continue;
    }
    auto SetCounts = [&](const SwarmInfo &swinfo, const SwarmVarInfo &vinfo) {
      const int rank = vinfo.tensor_rank;
      for (int i = 0; i < 6; ++i) {
        local_offset[i] = 0; // reset everything
        local_count[i] = 0;
        global_count[i] = 0;
      }
      for (int i = 0; i < rank; ++i) {
        local_count[i] = global_count[i] = vinfo.GetN(rank + 1 - i);
      }
      local_offset[rank] = swinfo.global_offset;
      local_count[rank] = swinfo.count_on_rank;
      global_count[rank] = swinfo.global_count;
    };
    auto &int_vars = std::get<SwarmInfo::MapToVarVec<int>>(swinfo.vars);
    for (auto &[vname, swmvarvec] : int_vars) {
      const auto &vinfo = swinfo.var_info.at(vname);
      auto host_data = swinfo.FillHostBuffer(vname, swmvarvec);
      SetCounts(swinfo, vinfo);
      HDF5WriteND(g_var, vname, host_data.data(), vinfo.tensor_rank + 1, local_offset,
                  local_count, global_count, pl_xfer, H5P_DEFAULT);
    }
    auto &rvars = std::get<SwarmInfo::MapToVarVec<Real>>(swinfo.vars);
    for (auto &[vname, swmvarvec] : rvars) {
      const auto &vinfo = swinfo.var_info.at(vname);
      auto host_data = swinfo.FillHostBuffer(vname, swmvarvec);
      SetCounts(swinfo, vinfo);
      HDF5WriteND(g_var, vname, host_data.data(), vinfo.tensor_rank + 1, local_offset,
                  local_count, global_count, pl_xfer, H5P_DEFAULT);
    }
    // If swarm does not contain an "id" object, generate a sequential
    // one for vis.
    if (swinfo.var_info.count("id") == 0) {
      std::vector<int> ids(swinfo.global_count);
      std::iota(std::begin(ids), std::end(ids), swinfo.global_offset);
      local_offset[0] = swinfo.global_offset;
      local_count[0] = swinfo.count_on_rank;
      global_count[0] = swinfo.global_count;
      HDF5Write1D(g_var, "id", ids.data(), local_offset, local_count, global_count,
                  pl_xfer);
    }
  }
  Kokkos::Profiling::popRegion(); // write particle data

  if (output_params.write_xdmf) {
    Kokkos::Profiling::pushRegion("genXDMF");
    // generate XDMF companion file
    XDMF::genXDMF(filename, pm, tm, nx1, nx2, nx3, all_vars_info, swarm_info);
    Kokkos::Profiling::popRegion(); // genXDMF
  }

  Kokkos::Profiling::popRegion(); // WriteOutputFile???Prec
}
// explicit template instantiation
template void PHDF5Output::WriteOutputFileImpl<false>(Mesh *, ParameterInput *, SimTime *,
                                                      SignalHandler::OutputSignal);
template void PHDF5Output::WriteOutputFileImpl<true>(Mesh *, ParameterInput *, SimTime *,
                                                     SignalHandler::OutputSignal);

std::string PHDF5Output::GenerateFilename_(ParameterInput *pin, SimTime *tm,
                                           const SignalHandler::OutputSignal signal) {
  using namespace HDF5;

  auto filename = std::string(output_params.file_basename);
  filename.append(".");
  filename.append(output_params.file_id);
  filename.append(".");
  if (signal == SignalHandler::OutputSignal::now) {
    filename.append("now");
  } else if (signal == SignalHandler::OutputSignal::final &&
             output_params.file_label_final) {
    filename.append("final");
    // default time based data dump
  } else {
    std::stringstream file_number;
    file_number << std::setw(output_params.file_number_width) << std::setfill('0')
                << output_params.file_number;
    filename.append(file_number.str());
  }
  filename.append(restart_ ? ".rhdf" : ".phdf");

  if (signal == SignalHandler::OutputSignal::none) {
    // After file has been opened with the current number, already advance output
    // parameters so that for restarts the file is not immediatly overwritten again.
    // Only applies to default time-based data dumps, so that writing "now" and "final"
    // outputs does not change the desired output numbering.
    output_params.file_number++;
    output_params.next_time += output_params.dt;
    pin->SetInteger(output_params.block_name, "file_number", output_params.file_number);
    pin->SetReal(output_params.block_name, "next_time", output_params.next_time);
  }
  return filename;
}

void PHDF5Output::WriteBlocksMetadata_(Mesh *pm, hid_t file, const HDF5::H5P &pl,
                                       hsize_t offset, hsize_t max_blocks_global) const {
  using namespace HDF5;
  Kokkos::Profiling::pushRegion("I/O HDF5: write block metadata");
  const H5G gBlocks = MakeGroup(file, "/Blocks");
  const hsize_t num_blocks_local = pm->block_list.size();
  const hsize_t ndim = pm->ndim;
  const hsize_t loc_offset[2] = {offset, 0};

  // write Xmin[ndim] for blocks
  {
    // JMM: These arrays chould be shared, but I think this is clearer
    // as to what's going on.
    hsize_t loc_cnt[2] = {num_blocks_local, ndim};
    hsize_t glob_cnt[2] = {max_blocks_global, ndim};

    std::vector<Real> tmpData = OutputUtils::ComputeXminBlocks(pm);
    HDF5Write2D(gBlocks, "xmin", tmpData.data(), &loc_offset[0], &loc_cnt[0],
                &glob_cnt[0], pl);
  }

  {
    // LOC.lx1,2,3
    hsize_t loc_cnt[2] = {num_blocks_local, 3};
    hsize_t glob_cnt[2] = {max_blocks_global, 3};
    std::vector<int64_t> tmpLoc = OutputUtils::ComputeLocs(pm);
    HDF5Write2D(gBlocks, "loc.lx123", tmpLoc.data(), &loc_offset[0], &loc_cnt[0],
                &glob_cnt[0], pl);
  }

  {
    // (LOC.)level, GID, LID, cnghost, gflag
    hsize_t loc_cnt[2] = {num_blocks_local, 5};
    hsize_t glob_cnt[2] = {max_blocks_global, 5};
    std::vector<int> tmpID = OutputUtils::ComputeIDsAndFlags(pm);
    HDF5Write2D(gBlocks, "loc.level-gid-lid-cnghost-gflag", tmpID.data(), &loc_offset[0],
                &loc_cnt[0], &glob_cnt[0], pl);
  }
  Kokkos::Profiling::popRegion(); // write block metadata
}

void PHDF5Output::WriteCoordinates_(Mesh *pm, const IndexDomain &domain, hid_t file,
                                    const HDF5::H5P &pl, hsize_t offset,
                                    hsize_t max_blocks_global) const {
  using namespace HDF5;
  Kokkos::Profiling::pushRegion("write mesh coords");
  const IndexShape &shape = pm->GetLeafBlockCellBounds();
  const IndexRange ib = shape.GetBoundsI(domain);
  const IndexRange jb = shape.GetBoundsJ(domain);
  const IndexRange kb = shape.GetBoundsK(domain);

  const hsize_t num_blocks_local = pm->block_list.size();
  const hsize_t loc_offset[2] = {offset, 0};
  hsize_t loc_cnt[2] = {num_blocks_local, 1};
  hsize_t glob_cnt[2] = {max_blocks_global, 1};

  for (const bool face : {true, false}) {
    const H5G gLocations = MakeGroup(file, face ? "/Locations" : "/VolumeLocations");

    std::vector<Real> loc_x, loc_y, loc_z;
    OutputUtils::ComputeCoords(pm, face, ib, jb, kb, loc_x, loc_y, loc_z);

    loc_cnt[1] = glob_cnt[1] = (ib.e - ib.s + 1) + face;
    HDF5Write2D(gLocations, "x", loc_x.data(), &loc_offset[0], &loc_cnt[0], &glob_cnt[0],
                pl);

    loc_cnt[1] = glob_cnt[1] = (jb.e - jb.s + 1) + face;
    HDF5Write2D(gLocations, "y", loc_y.data(), &loc_offset[0], &loc_cnt[0], &glob_cnt[0],
                pl);

    loc_cnt[1] = glob_cnt[1] = (kb.e - kb.s + 1) + face;
    HDF5Write2D(gLocations, "z", loc_z.data(), &loc_offset[0], &loc_cnt[0], &glob_cnt[0],
                pl);
  }
  Kokkos::Profiling::popRegion(); // write mesh coords
}

void PHDF5Output::WriteLevelsAndLocs_(Mesh *pm, hid_t file, const HDF5::H5P &pl,
                                      hsize_t offset, hsize_t max_blocks_global) const {
  using namespace HDF5;
  Kokkos::Profiling::pushRegion("write levels and locations");
  auto [levels, logicalLocations] = pm->GetLevelsAndLogicalLocationsFlat();

  // Only write levels on rank 0 since it has data for all ranks
  const hsize_t num_blocks_local = pm->block_list.size();
  const hsize_t loc_offset[2] = {offset, 0};
  const hsize_t loc_cnt[2] = {(Globals::my_rank == 0) ? max_blocks_global : 0, 3};
  const hsize_t glob_cnt[2] = {max_blocks_global, 3};

  HDF5Write1D(file, "Levels", levels.data(), &loc_offset[0], &loc_cnt[0], &glob_cnt[0],
              pl);
  HDF5Write2D(file, "LogicalLocations", logicalLocations.data(), &loc_offset[0],
              &loc_cnt[0], &glob_cnt[0], pl);

  Kokkos::Profiling::popRegion(); // write levels and locations
}

void PHDF5Output::WriteSparseInfo_(Mesh *pm, hbool_t *sparse_allocated,
                                   const std::vector<std::string> &sparse_names,
                                   hsize_t num_sparse, hid_t file, const HDF5::H5P &pl,
                                   size_t offset, hsize_t max_blocks_global) const {
  using namespace HDF5;
  Kokkos::Profiling::pushRegion("write sparse info");

  const hsize_t num_blocks_local = pm->block_list.size();
  const hsize_t loc_offset[2] = {offset, 0};
  const hsize_t loc_cnt[2] = {num_blocks_local, num_sparse};
  const hsize_t glob_cnt[2] = {max_blocks_global, num_sparse};

  HDF5Write2D(file, "SparseInfo", sparse_allocated, &loc_offset[0], &loc_cnt[0],
              &glob_cnt[0], pl);

  // write names of sparse fields as attribute, first convert to vector of const char*
  std::vector<const char *> names(num_sparse);
  for (size_t i = 0; i < num_sparse; ++i)
    names[i] = sparse_names[i].c_str();

  const H5D dset = H5D::FromHIDCheck(H5Dopen2(file, "SparseInfo", H5P_DEFAULT));
  HDF5WriteAttribute("SparseFields", names, dset);
  Kokkos::Profiling::popRegion(); // write sparse info
}

// Utility functions implemented
namespace HDF5 {
hid_t GenerateFileAccessProps() {
#ifdef MPI_PARALLEL
  /* set the file access template for parallel IO access */
  hid_t acc_file = H5Pcreate(H5P_FILE_ACCESS);

  /* ---------------------------------------------------------------------
     platform dependent code goes here -- the access template must be
     tuned for a particular filesystem blocksize.  some of these
     numbers are guesses / experiments, others come from the file system
     documentation.

     ---------------------------------------------------------------------- */

  // use collective metadata optimizations
#if H5_VERSION_GE(1, 10, 0)
  PARTHENON_HDF5_CHECK(H5Pset_coll_metadata_write(acc_file, true));
  PARTHENON_HDF5_CHECK(H5Pset_all_coll_metadata_ops(acc_file, true));
#endif

  bool exists, exists2;

  // Set the HDF5 format versions used when creating objects
  // Note, introducing API calls that create objects or features that are
  // only available to versions of the library greater than 1.8.x release will fail.
  // For that case, the highest version value will need to be increased.
  H5Pset_libver_bounds(acc_file, H5F_LIBVER_V18, H5F_LIBVER_V18);

  // Sets the maximum size of the data sieve buffer, in bytes.
  // The sieve_buf_size should be equal to a multiple of the disk block size
  // Default: Disabled
  size_t sieve_buf_size = Env::get<size_t>("H5_sieve_buf_size", 256 * KiB, exists);
  if (exists) {
    PARTHENON_HDF5_CHECK(H5Pset_sieve_buf_size(acc_file, sieve_buf_size));
  }

  // Sets the minimum metadata block size, in bytes.
  // Default: Disabled
  hsize_t meta_block_size = Env::get<hsize_t>("H5_meta_block_size", 8 * MiB, exists);
  if (exists) {
    PARTHENON_HDF5_CHECK(H5Pset_meta_block_size(acc_file, meta_block_size));
  }

  // Sets alignment properties of a file access property list.
  // Choose an alignment which is a multiple of the disk block size.
  // Default: Disabled
  hsize_t threshold; // Threshold value. Setting to 0 forces everything to be aligned.
  hsize_t alignment; // Alignment value.

  threshold = Env::get<hsize_t>("H5_alignment_threshold", 0, exists);
  alignment = Env::get<hsize_t>("H5_alignment_alignment", 8 * MiB, exists2);
  if (exists || exists2) {
    PARTHENON_HDF5_CHECK(H5Pset_alignment(acc_file, threshold, alignment));
  }

  // Defer metadata flush
  // Default: Disabled
  bool defer_metadata_flush = Env::get<bool>("H5_defer_metadata_flush", false, exists);
  if (defer_metadata_flush) {
    H5AC_cache_config_t cache_config;
    cache_config.version = H5AC__CURR_CACHE_CONFIG_VERSION;
    PARTHENON_HDF5_CHECK(H5Pget_mdc_config(acc_file, &cache_config));
    cache_config.set_initial_size = 1;
    cache_config.initial_size = 16 * MiB;
    cache_config.evictions_enabled = 0;
    cache_config.incr_mode = H5C_incr__off;
    cache_config.flash_incr_mode = H5C_flash_incr__off;
    cache_config.decr_mode = H5C_decr__off;
    PARTHENON_HDF5_CHECK(H5Pset_mdc_config(acc_file, &cache_config));
  }

  /* create an MPI_INFO object -- on some platforms it is useful to
     pass some information onto the underlying MPI_File_open call */
  MPI_Info FILE_INFO_TEMPLATE;
  PARTHENON_MPI_CHECK(MPI_Info_create(&FILE_INFO_TEMPLATE));

  // Free MPI_Info on error on return or throw
  struct MPI_InfoDeleter {
    MPI_Info info;
    ~MPI_InfoDeleter() { MPI_Info_free(&info); }
  } delete_info{FILE_INFO_TEMPLATE};

  // Hint specifies the manner in which the file will be accessed until the file is closed
  const auto access_style =
      Env::get<std::string>("MPI_access_style", "write_once", exists);
  PARTHENON_MPI_CHECK(
      MPI_Info_set(FILE_INFO_TEMPLATE, "access_style", access_style.c_str()));

  // Specifies whether the application may benefit from collective buffering
  // Default :: collective_buffering is disabled
  bool collective_buffering = Env::get<bool>("MPI_collective_buffering", false, exists);
  if (exists) {
    PARTHENON_MPI_CHECK(MPI_Info_set(FILE_INFO_TEMPLATE, "collective_buffering", "true"));
    // Specifies the block size to be used for collective buffering file acces
    const auto cb_block_size =
        Env::get<std::string>("MPI_cb_block_size", "1048576", exists);
    PARTHENON_MPI_CHECK(
        MPI_Info_set(FILE_INFO_TEMPLATE, "cb_block_size", cb_block_size.c_str()));
    // Specifies the total buffer space that can be used for collective buffering on each
    // target node, usually a multiple of cb_block_size
    const auto cb_buffer_size =
        Env::get<std::string>("MPI_cb_buffer_size", "4194304", exists);
    PARTHENON_MPI_CHECK(
        MPI_Info_set(FILE_INFO_TEMPLATE, "cb_buffer_size", cb_buffer_size.c_str()));
  }

  /* tell the HDF5 library that we want to use MPI-IO to do the writing */
  PARTHENON_HDF5_CHECK(H5Pset_fapl_mpio(acc_file, MPI_COMM_WORLD, FILE_INFO_TEMPLATE));
#else
  hid_t acc_file = H5P_DEFAULT;
#endif // ifdef MPI_PARALLEL
  return acc_file;
}
} // namespace HDF5
} // namespace parthenon

#endif // ifdef ENABLE_HDF5
