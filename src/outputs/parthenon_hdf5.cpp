//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2023 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2023. Triad National Security, LLC. All rights reserved.
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
  {
    // write input key-value pairs
    std::ostringstream oss;
    pin->ParameterDump(oss);

    // Mesh information
    const H5G input_group = MakeGroup(file, "/Input");

    HDF5WriteAttribute("File", oss.str().c_str(), input_group);
  } // Input section

  // we'll need this again at the end
  const H5G info_group = MakeGroup(file, "/Info");
  {
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
        std::vector<Real>{pm->mesh_size.x1min, pm->mesh_size.x1max, pm->mesh_size.x1rat,
                          pm->mesh_size.x2min, pm->mesh_size.x2max, pm->mesh_size.x2rat,
                          pm->mesh_size.x3min, pm->mesh_size.x3max, pm->mesh_size.x3rat},
        info_group);

    // Root grid size (number of cells at root level)
    HDF5WriteAttribute(
        "RootGridSize",
        std::vector<int>{pm->mesh_size.nx1, pm->mesh_size.nx2, pm->mesh_size.nx3},
        info_group);

    // Boundary conditions
    std::vector<std::string> boundary_condition_str(BOUNDARY_NFACES);
    for (size_t i = 0; i < boundary_condition_str.size(); i++) {
      boundary_condition_str[i] = GetBoundaryString(pm->mesh_bcs[i]);
    }

    HDF5WriteAttribute("BoundaryConditions", boundary_condition_str, info_group);
  } // Info section

  // write Params
  {
    const H5G params_group = MakeGroup(file, "/Params");

    for (const auto &package : pm->packages.AllPackages()) {
      const auto state = package.second;
      // Write all params that can be written as HDF5 attributes
      state->AllParams().WriteAllToHDF5(state->label(), params_group);
    }
  } // Params section

  // -------------------------------------------------------------------------------- //
  //   WRITING MESHBLOCK METADATA                                                     //
  // -------------------------------------------------------------------------------- //

  // set local offset, always the same for all data sets
  hsize_t my_offset = 0;
  for (int i = 0; i < Globals::my_rank; i++) {
    my_offset += nblist[i];
  }

  const std::array<hsize_t, H5_NDIM> local_offset({my_offset, 0, 0, 0, 0, 0, 0});

  // these can vary by data set, except index 0 is always the same
  std::array<hsize_t, H5_NDIM> local_count(
      {static_cast<hsize_t>(num_blocks_local), 1, 1, 1, 1, 1, 1});
  std::array<hsize_t, H5_NDIM> global_count(
      {static_cast<hsize_t>(max_blocks_global), 1, 1, 1, 1, 1, 1});

  // for convenience
  const hsize_t *const p_loc_offset = local_offset.data();
  const hsize_t *const p_loc_cnt = local_count.data();
  const hsize_t *const p_glob_cnt = global_count.data();

  H5P const pl_xfer = H5P::FromHIDCheck(H5Pcreate(H5P_DATASET_XFER));
  H5P const pl_dcreate = H5P::FromHIDCheck(H5Pcreate(H5P_DATASET_CREATE));

  // Never write fill values to the dataset
  PARTHENON_HDF5_CHECK(H5Pset_fill_time(pl_dcreate, H5D_FILL_TIME_NEVER));

#ifdef MPI_PARALLEL
  PARTHENON_HDF5_CHECK(H5Pset_dxpl_mpio(pl_xfer, H5FD_MPIO_COLLECTIVE));
#endif

  // write Blocks metadata
  {
    const H5G gBlocks = MakeGroup(file, "/Blocks");

    // write Xmin[ndim] for blocks
    {
      std::vector<Real> tmpData(num_blocks_local * 3);
      ComputeXminBlocks_(pm, tmpData);
      local_count[1] = global_count[1] = pm->ndim;
      HDF5Write2D(gBlocks, "xmin", tmpData.data(), p_loc_offset, p_loc_cnt, p_glob_cnt,
                  pl_xfer);
    }

    // write Block ID
    {
      // LOC.lx1,2,3
      hsize_t n = 3;
      std::vector<int64_t> tmpLoc(num_blocks_local * n);
      local_count[1] = global_count[1] = n;
      ComputeLocs_(pm, tmpLoc);
      HDF5Write2D(gBlocks, "loc.lx123", tmpLoc.data(), p_loc_offset, p_loc_cnt,
                  p_glob_cnt, pl_xfer);

      // (LOC.)level, GID, LID, cnghost, gflag
      n = 5; // this is NOT H5_NDIM
      std::vector<int> tmpID(num_blocks_local * n);
      local_count[1] = global_count[1] = n;
      ComputeIDsAndFlags_(pm, tmpID);
      HDF5Write2D(gBlocks, "loc.level-gid-lid-cnghost-gflag", tmpID.data(), p_loc_offset,
                  p_loc_cnt, p_glob_cnt, pl_xfer);
    }
  } // Block section

  // Write mesh coordinates to file
  for (const bool face : {true, false}) {
    const H5G gLocations = MakeGroup(file, face ? "/Locations" : "/VolumeLocations");

    // write X coordinates
    std::vector<Real> loc_x((nx1 + face) * num_blocks_local);
    std::vector<Real> loc_y((nx2 + face) * num_blocks_local);
    std::vector<Real> loc_z((nx3 + face) * num_blocks_local);

    ComputeCoords_(pm, face, out_ib, out_jb, out_kb, loc_x, loc_y, loc_z);

    local_count[1] = global_count[1] = nx1 + face;
    HDF5Write2D(gLocations, "x", loc_x.data(), p_loc_offset, p_loc_cnt, p_glob_cnt,
                pl_xfer);

    local_count[1] = global_count[1] = nx2 + face;
    HDF5Write2D(gLocations, "y", loc_y.data(), p_loc_offset, p_loc_cnt, p_glob_cnt,
                pl_xfer);

    local_count[1] = global_count[1] = nx3 + face;
    HDF5Write2D(gLocations, "z", loc_z.data(), p_loc_offset, p_loc_cnt, p_glob_cnt,
                pl_xfer);
  }

  // Write Levels and Logical Locations with the level for each Meshblock loclist contains
  // levels and logical locations for all meshblocks on all ranks
  {
    const auto &loclist = pm->GetLocList();

    std::vector<std::int64_t> levels;
    levels.reserve(pm->nbtotal);

    std::vector<std::int64_t> logicalLocations;
    logicalLocations.reserve(pm->nbtotal * 3);

    for (const auto &loc : loclist) {
      levels.push_back(loc.level - pm->GetRootLevel());
      logicalLocations.push_back(loc.lx1);
      logicalLocations.push_back(loc.lx2);
      logicalLocations.push_back(loc.lx3);
    }

    // Only write levels on rank 0 since it has data for all ranks
    local_count[0] = (Globals::my_rank == 0) ? pm->nbtotal : 0;
    HDF5WriteND(file, "Levels", levels.data(), 1, local_offset.data(), local_count.data(),
                global_count.data(), pl_xfer, H5P_DEFAULT);

    local_count[1] = global_count[1] = 3;
    HDF5Write2D(file, "LogicalLocations", logicalLocations.data(), local_offset.data(),
                local_count.data(), global_count.data(), pl_xfer);

    // reset for collective output
    local_count[0] = num_blocks_local;
  }

  // -------------------------------------------------------------------------------- //
  //   WRITING VARIABLES DATA                                                         //
  // -------------------------------------------------------------------------------- //

  // All blocks have the same list of variable metadata that exist in the entire
  // simulation, but not all variables may be allocated on all blocks

  auto get_vars = [=](const std::shared_ptr<MeshBlock> pmb) {
    auto &var_vec = pmb->meshblock_data.Get()->GetCellVariableVector();
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
    const int varSize =
        vinfo.nx6 * vinfo.nx5 * vinfo.nx4 * vinfo.nx3 * vinfo.nx2 * vinfo.nx1;
    varSize_max = std::max(varSize_max, varSize);
  }

  using OutT = typename std::conditional<WRITE_SINGLE_PRECISION, float, Real>::type;
  std::vector<OutT> tmpData(varSize_max * num_blocks_local);

  // create persistent spaces
  local_count[0] = num_blocks_local;
  global_count[0] = max_blocks_global;
  local_count[4] = global_count[4] = nx3;
  local_count[5] = global_count[5] = nx2;
  local_count[6] = global_count[6] = nx1;

  // for each variable we write
  for (auto &vinfo : all_vars_info) {
    // not really necessary, but doesn't hurt
    memset(tmpData.data(), 0, tmpData.size() * sizeof(OutT));

    const std::string var_name = vinfo.label;
    const hsize_t nx6 = vinfo.nx6;
    const hsize_t nx5 = vinfo.nx5;
    const hsize_t nx4 = vinfo.nx4;

    local_count[1] = global_count[1] = nx6;
    local_count[2] = global_count[2] = nx5;
    local_count[3] = global_count[3] = nx4;

    std::vector<hsize_t> alldims({nx6, nx5, nx4, static_cast<hsize_t>(vinfo.nx3),
                                  static_cast<hsize_t>(vinfo.nx2),
                                  static_cast<hsize_t>(vinfo.nx1)});

    int ndim = -1;
#ifndef PARTHENON_DISABLE_HDF5_COMPRESSION
    // we need chunks to enable compression
    std::array<hsize_t, H5_NDIM> chunk_size({1, 1, 1, 1, 1, 1, 1});
#endif
    if (vinfo.where == MetadataFlag(Metadata::Cell)) {
      ndim = 3 + vinfo.tensor_rank + 1;
      for (int i = 0; i < vinfo.tensor_rank; i++) {
        local_count[1 + i] = global_count[1 + i] = alldims[3 - vinfo.tensor_rank + i];
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
        local_count[1 + i] = global_count[1 + i] = alldims[6 - vinfo.tensor_rank + i];
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
    PARTHENON_HDF5_CHECK(
        H5Pset_deflate(pl_dcreate, std::min(9, output_params.hdf5_compression_level)));
#endif

    // load up data
    hsize_t index = 0;

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
          for (int t = 0; t < nx6; ++t) {
            for (int u = 0; u < nx5; ++u) {
              for (int v = 0; v < nx4; ++v) {
                if (vinfo.where == MetadataFlag(Metadata::Cell)) {
                  for (int k = out_kb.s; k <= out_kb.e; ++k) {
                    for (int j = out_jb.s; j <= out_jb.e; ++j) {
                      for (int i = out_ib.s; i <= out_ib.e; ++i) {
                        tmpData[index++] = static_cast<OutT>(v_h(t, u, v, k, j, i));
                      }
                    }
                  }
                } else {
                  for (int k = 0; k < vinfo.nx3; ++k) {
                    for (int j = 0; j < vinfo.nx2; ++j) {
                      for (int i = 0; i < vinfo.nx1; ++i) {
                        tmpData[index++] = static_cast<OutT>(v_h(t, u, v, k, j, i));
                      }
                    }
                  }
                }
              }
            }
          }

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
            varSize = vinfo.nx6 * vinfo.nx5 * vinfo.nx4 * (out_kb.e - out_kb.s + 1) *
                      (out_jb.e - out_jb.s + 1) * (out_ib.e - out_ib.s + 1);
          } else {
            varSize =
                vinfo.nx6 * vinfo.nx5 * vinfo.nx4 * vinfo.nx3 * vinfo.nx2 * vinfo.nx1;
          }
          memset(tmpData.data() + index, 0, varSize * sizeof(OutT));
          index += varSize;
        } else {
          std::stringstream msg;
          msg << "### ERROR: Unable to find dense variable " << var_name << std::endl;
          PARTHENON_FAIL(msg);
        }
      }
    }

    // write data to file
    HDF5WriteND(file, var_name, tmpData.data(), ndim, p_loc_offset, p_loc_cnt, p_glob_cnt,
                pl_xfer, pl_dcreate);
  }

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
    local_count[1] = global_count[1] = num_sparse;

    HDF5Write2D(file, "SparseInfo", sparse_allocated.get(), p_loc_offset, p_loc_cnt,
                p_glob_cnt, pl_xfer);

    // write names of sparse fields as attribute, first convert to vector of const char*
    std::vector<const char *> names(num_sparse);
    for (size_t i = 0; i < num_sparse; ++i)
      names[i] = sparse_names[i].c_str();

    const H5D dset = H5D::FromHIDCheck(H5Dopen2(file, "SparseInfo", H5P_DEFAULT));
    HDF5WriteAttribute("SparseFields", names, dset);
  } // SparseInfo and SparseFields sections

  // -------------------------------------------------------------------------------- //
  //   WRITING PARTICLE DATA                                                          //
  // -------------------------------------------------------------------------------- //

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

  // generate XDMF companion file
  XDMF::genXDMF(filename, pm, tm, nx1, nx2, nx3, all_vars_info, swarm_info);
}

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
// TODO(JMM): Should this live in the base class or output_utils?
void PHDF5Output::ComputeXminBlocks_(Mesh *pm, std::vector<Real> &data) {
  int i = 0;
  for (auto &pmb : pm->block_list) {
    auto xmin = pmb->coords.GetXmin();
    data[i++] = xmin[0];
    if (pm->ndim > 1) {
      data[i++] = xmin[1];
    }
    if (pm->ndim > 2) {
      data[i++] = xmin[2];
    }
  }
}
// TODO(JMM): Should this live in the base class or output_utils?
void PHDF5Output::ComputeLocs_(Mesh *pm, std::vector<int64_t> &locs) {
  int i = 0;
  for (auto &pmb : pm->block_list) {
    locs[i++] = pmb->loc.lx1;
    locs[i++] = pmb->loc.lx2;
    locs[i++] = pmb->loc.lx3;
  }
}
// TODO(JMM): Should this live in the base class or output_utils?
void PHDF5Output::ComputeIDsAndFlags_(Mesh *pm, std::vector<int> &data) {
  int i = 0;
  for (auto &pmb : pm->block_list) {
    data[i++] = pmb->loc.level;
    data[i++] = pmb->gid;
    data[i++] = pmb->lid;
    data[i++] = pmb->cnghost;
    data[i++] = pmb->gflag;
  }
}
// TODO(JMM): Should this live in the base class or output_utils?
void PHDF5Output::ComputeCoords_(Mesh *pm, bool face, const IndexRange &ib,
                                 const IndexRange &jb, const IndexRange &kb,
                                 std::vector<Real> &x, std::vector<Real> &y,
                                 std::vector<Real> &z) {
  std::size_t idx_x = 0, idx_y = 0, idx_z = 0;

  // note relies on casting of bool to int
  for (auto &pmb : pm->block_list) {
    for (int i = ib.s; i <= ib.e + face; ++i) {
      x[idx_x++] = face ? pmb->coords.Xf<1>(i) : pmb->coords.Xc<1>(i);
    }
    for (int j = jb.s; j <= jb.e + face; ++j) {
      y[idx_y++] = face ? pmb->coords.Xf<2>(j) : pmb->coords.Xc<2>(j);
    }
    for (int k = kb.s; k <= kb.e + face; ++k) {
      z[idx_z++] = face ? pmb->coords.Xf<3>(k) : pmb->coords.Xc<3>(k);
    }
  }
}

// explicit template instantiation
template void PHDF5Output::WriteOutputFileImpl<false>(Mesh *, ParameterInput *, SimTime *,
                                                      SignalHandler::OutputSignal);
template void PHDF5Output::WriteOutputFileImpl<true>(Mesh *, ParameterInput *, SimTime *,
                                                     SignalHandler::OutputSignal);

// Utility functions implemented
namespace HDF5 {
// template specializations for std::string
template <>
void HDF5WriteAttribute(const std::string &name, const std::vector<std::string> &values,
                        hid_t location) {
  std::vector<const char *> char_ptrs(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    char_ptrs[i] = values[i].c_str();
  }
  HDF5WriteAttribute(name, char_ptrs, location);
}

template <>
std::vector<std::string> HDF5ReadAttributeVec(hid_t location, const std::string &name) {
  // get strings as char pointers, HDF5 will allocate the memory and we need to free it
  auto char_ptrs = HDF5ReadAttributeVec<char *>(location, name);

  // make strings out of char pointers, which copies the memory and then free the memeory
  std::vector<std::string> res(char_ptrs.size());
  for (size_t i = 0; i < res.size(); ++i) {
    res[i] = std::string(char_ptrs[i]);
    free(char_ptrs[i]);
  }

  return res;
}

// template specialization for bool
template <>
void HDF5WriteAttribute(const std::string &name, const std::vector<bool> &values,
                        hid_t location) {
  // can't use std::vector here because std::vector<bool>  doesn't have .data() member
  std::unique_ptr<hbool_t[]> data(new hbool_t[values.size()]);
  for (size_t i = 0; i < values.size(); ++i) {
    data[i] = values[i];
  }
  HDF5WriteAttribute(name, values.size(), data.get(), location);
}

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
