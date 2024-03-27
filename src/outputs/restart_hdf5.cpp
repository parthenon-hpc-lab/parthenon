//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2024 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
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
//! \file restart.cpp
//  \brief writes restart files

#include <memory>
#include <numeric>
#include <string>
#include <utility>

#include "basic_types.hpp"
#include "globals.hpp"
#include "interface/params.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "outputs/outputs.hpp"
#ifdef ENABLE_HDF5
#include "outputs/parthenon_hdf5.hpp"
#endif
#include "outputs/restart.hpp"
#include "outputs/restart_hdf5.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

//----------------------------------------------------------------------------------------
//! \fn void RestartReader::RestartReader(const std::string filename)
//  \brief Opens the restart file and stores appropriate file handle in fh_
RestartReaderHDF5::RestartReaderHDF5(const char *filename) : filename_(filename) {
#ifndef ENABLE_HDF5
  std::stringstream msg;
  msg << "### FATAL ERROR in Restart (Reader) constructor" << std::endl
      << "Executable not configured for HDF5 outputs, but HDF5 file format "
      << "is required for restarts" << std::endl;
  PARTHENON_FAIL(msg);
#else  // HDF5 enabled
  // Open the HDF file in read only mode
  fh_ = H5F::FromHIDCheck(H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT));
  params_group_ = H5G::FromHIDCheck(H5Oopen(fh_, "Params", H5P_DEFAULT));

  hasGhost = GetAttr<int>("Info", "IncludesGhost");
#endif // ENABLE_HDF5
}

int RestartReaderHDF5::GetOutputFormatVersion() const {
#ifndef ENABLE_HDF5
  PARTHENON_FAIL("Restart functionality is not available because HDF5 is disabled");
#else  // HDF5 enabled
  const H5O obj = H5O::FromHIDCheck(H5Oopen(fh_, "Info", H5P_DEFAULT));
  auto status = PARTHENON_HDF5_CHECK(H5Aexists(obj, "OutputFormatVersion"));
  // file contains version info
  if (status > 0) {
    return GetAttr<int>("Info", "OutputFormatVersion");
  } else {
    return -1;
  }
#endif // ENABLE_HDF5
}

RestartReaderHDF5::SparseInfo RestartReaderHDF5::GetSparseInfo() const {
#ifndef ENABLE_HDF5
  PARTHENON_FAIL("Restart functionality is not available because HDF5 is disabled");
#else  // HDF5 enabled
  SparseInfo info;

  // check if SparseInfo exists, if not, return the default-constructed SparseInfo
  // instance
  auto status = PARTHENON_HDF5_CHECK(H5Lexists(fh_, "SparseInfo", H5P_DEFAULT));
  if (status > 0) {
    // SparseInfo exists, read its contents
    auto hdl = OpenDataset<bool>("SparseInfo");
    PARTHENON_REQUIRE_THROWS(hdl.rank == 2, "SparseInfo expected to have rank 2");

    info.labels = HDF5ReadAttributeVec<std::string>(hdl.dataset, "SparseFields");
    info.num_sparse = static_cast<int>(info.labels.size());
    PARTHENON_REQUIRE_THROWS(info.num_sparse == static_cast<int>(hdl.dims[1]),
                             "Mismatch in number of sparse fields");

    // Note: We cannot use ReadData, because std::vector<bool> doesn't have a data()
    // member
    info.allocated.reset(new hbool_t[hdl.count]);
    info.num_blocks = static_cast<int>(hdl.dims[0]);

    const H5S memspace =
        H5S::FromHIDCheck(H5Screate_simple(hdl.rank, hdl.dims.data(), NULL));

    // Read data from file
    PARTHENON_HDF5_CHECK(H5Dread(hdl.dataset, hdl.type, memspace, hdl.dataspace,
                                 H5P_DEFAULT, static_cast<void *>(info.allocated.get())));
  }

  return info;
#endif // ENABLE_HDF5
}

RestartReaderHDF5::MeshInfo RestartReaderHDF5::GetMeshInfo() const {
  RestartReaderHDF5::MeshInfo mesh_info;
  mesh_info.nbnew = GetAttr<int>("Info", "NBNew");
  mesh_info.nbdel = GetAttr<int>("Info", "NBDel");
  mesh_info.nbtotal = GetAttr<int>("Info", "NumMeshBlocks");
  mesh_info.root_level = GetAttr<int>("Info", "RootLevel");

  mesh_info.bound_cond = GetAttrVec<std::string>("Info", "BoundaryConditions");

  mesh_info.block_size = GetAttrVec<int>("Info", "MeshBlockSize");
  mesh_info.includes_ghost = GetAttr<int>("Info", "IncludesGhost");
  mesh_info.n_ghost = GetAttr<int>("Info", "NGhost");

  mesh_info.grid_dim = GetAttrVec<Real>("Info", "RootGridDomain");

  mesh_info.lx123 = ReadDataset<int64_t>("/Blocks/loc.lx123");
  mesh_info.level_gid_lid_cnghost_gflag =
      ReadDataset<int>("/Blocks/loc.level-gid-lid-cnghost-gflag");

  return mesh_info;
}

SimTime RestartReaderHDF5::GetTimeInfo() const {
  SimTime time_info{};

  time_info.time = GetAttr<Real>("Info", "Time");
  time_info.dt = GetAttr<Real>("Info", "dt");
  time_info.ncycle = GetAttr<int>("Info", "NCycle");

  return time_info;
}
// Gets the counts and offsets for MPI ranks for the meshblocks set
// by the indexrange. Returns the total count on this rank.
std::size_t RestartReaderHDF5::GetSwarmCounts(const std::string &swarm,
                                              const IndexRange &range,
                                              std::vector<std::size_t> &counts,
                                              std::vector<std::size_t> &offsets) {
#ifndef ENABLE_HDF5
  PARTHENON_FAIL("Restart functionality is not available because HDF5 is disabled");
  return 0;
#else  // HDF5 enabled
  // datasets
  auto counts_dset = OpenDataset<std::size_t>(swarm + "/counts");
  auto offsets_dset = OpenDataset<std::size_t>(swarm + "/offsets");
  // hyperslab
  hsize_t h5slab_offset[1] = {static_cast<hsize_t>(range.s)};
  hsize_t h5slab_count[1] = {static_cast<hsize_t>(range.e - range.s + 1)};
  PARTHENON_HDF5_CHECK(H5Sselect_hyperslab(counts_dset.dataspace, H5S_SELECT_SET,
                                           h5slab_offset, NULL, h5slab_count, NULL));
  PARTHENON_HDF5_CHECK(H5Sselect_hyperslab(offsets_dset.dataspace, H5S_SELECT_SET,
                                           h5slab_offset, NULL, h5slab_count, NULL));
  const H5S memspace = H5S::FromHIDCheck(H5Screate_simple(1, h5slab_count, NULL));

  // Read data
  counts.resize(range.e - range.s + 1);
  offsets.resize(range.e - range.s + 1);
  PARTHENON_HDF5_CHECK(H5Dread(counts_dset.dataset, counts_dset.type, memspace,
                               counts_dset.dataspace, H5P_DEFAULT, counts.data()));
  PARTHENON_HDF5_CHECK(H5Dread(offsets_dset.dataset, offsets_dset.type, memspace,
                               offsets_dset.dataspace, H5P_DEFAULT, offsets.data()));

  // Compute total count rank
  std::size_t total_count_on_rank = std::accumulate(counts.begin(), counts.end(), 0);
  return total_count_on_rank;
#endif // ENABLE_HDF5
}

void RestartReaderHDF5::ReadParams(const std::string &name, Params &p) {
#ifdef ENABLE_HDF5
  p.ReadFromRestart(name, params_group_);
#endif // ENABLE_HDF5
}
void RestartReaderHDF5::ReadBlocks(const std::string &name, IndexRange range,
                                   std::vector<Real> &dataVec,
                                   const std::vector<size_t> &bsize,
                                   int file_output_format_version, MetadataFlag where,
                                   const std::vector<int> &shape) const {
#ifndef ENABLE_HDF5
  PARTHENON_FAIL("Restart functionality is not available because HDF5 is disabled");
#else  // HDF5 enabled
  auto hdl = OpenDataset<Real>(name);

  constexpr int CHUNK_MAX_DIM = 7;

  /** Select hyperslab in dataset **/
  hsize_t offset[CHUNK_MAX_DIM] = {static_cast<hsize_t>(range.s), 0, 0, 0, 0, 0, 0};
  hsize_t count[CHUNK_MAX_DIM];
  int total_dim = 0;
  if (file_output_format_version == -1) {
    size_t vlen = 1;
    for (int i = 0; i < shape.size(); i++) {
      vlen *= shape[i];
    }
    count[0] = static_cast<hsize_t>(range.e - range.s + 1);
    count[1] = bsize[2];
    count[2] = bsize[1];
    count[3] = bsize[0];
    count[4] = vlen;
    total_dim = 5;
  } else if (file_output_format_version == 2) {
    PARTHENON_REQUIRE(shape.size() <= 1,
                      "Higher than vector datatypes are unstable in output versions < 3");
    size_t vlen = 1;
    for (int i = 0; i < shape.size(); i++) {
      vlen *= shape[i];
    }
    count[0] = static_cast<hsize_t>(range.e - range.s + 1);
    count[1] = vlen;
    count[2] = bsize[2];
    count[3] = bsize[1];
    count[4] = bsize[0];
    total_dim = 5;
  } else if (file_output_format_version == HDF5::OUTPUT_VERSION_FORMAT) {
    count[0] = static_cast<hsize_t>(range.e - range.s + 1);
    const int ndim = shape.size();
    if (where == MetadataFlag(Metadata::Cell)) {
      for (int i = 0; i < ndim; i++) {
        count[1 + i] = shape[ndim - i - 1];
      }
      count[ndim + 1] = bsize[2];
      count[ndim + 2] = bsize[1];
      count[ndim + 3] = bsize[0];
      total_dim = 3 + ndim + 1;
    } else if (where == MetadataFlag(Metadata::None)) {
      for (int i = 0; i < ndim; i++) {
        count[1 + i] = shape[ndim - i - 1];
      }
      total_dim = ndim + 1;
    } else {
      PARTHENON_THROW("Only Cell and None locations supported!");
    }
  } else {
    PARTHENON_THROW("Unknown output format version in restart file.")
  }

  hsize_t total_count = 1;
  for (int i = 0; i < total_dim; ++i) {
    total_count *= count[i];
  }

  PARTHENON_REQUIRE_THROWS(dataVec.size() >= total_count,
                           "Buffer (size " + std::to_string(dataVec.size()) +
                               ") is too small for dataset " + name + " (size " +
                               std::to_string(total_count) + ")");
  PARTHENON_HDF5_CHECK(
      H5Sselect_hyperslab(hdl.dataspace, H5S_SELECT_SET, offset, NULL, count, NULL));

  const H5S memspace = H5S::FromHIDCheck(H5Screate_simple(total_dim, count, NULL));
  PARTHENON_HDF5_CHECK(
      H5Sselect_hyperslab(hdl.dataspace, H5S_SELECT_SET, offset, NULL, count, NULL));

  // Read data from file
  PARTHENON_HDF5_CHECK(H5Dread(hdl.dataset, hdl.type, memspace, hdl.dataspace,
                               H5P_DEFAULT, dataVec.data()));
#endif // ENABLE_HDF5
}

} // namespace parthenon
