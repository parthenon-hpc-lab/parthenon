//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2024 The Parthenon collaboration
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
#include "outputs/output_utils.hpp"
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

  has_ghost = GetAttr<int>("Info", "IncludesGhost");
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
    auto hdl_dealloc = OpenDataset<int>("SparseDeallocCount");

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
    info.dealloc_count.resize(hdl_dealloc.count);
    PARTHENON_HDF5_CHECK(H5Dread(hdl_dealloc.dataset, hdl_dealloc.type,
                                 H5S::FromHIDCheck(H5Screate_simple(
                                     hdl_dealloc.rank, hdl_dealloc.dims.data(), NULL)),
                                 hdl_dealloc.dataspace, H5P_DEFAULT,
                                 static_cast<void *>(info.dealloc_count.data())));
  }

  return info;
#endif // ENABLE_HDF5
}

RestartReaderHDF5::MeshInfo RestartReaderHDF5::GetMeshInfo() const {
#ifndef ENABLE_HDF5
  PARTHENON_FAIL("Restart functionality is not available because HDF5 is disabled");
#else
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

  auto status =
      PARTHENON_HDF5_CHECK(H5Lexists(fh_, "Blocks/derefinement_count", H5P_DEFAULT));
  if (status > 0) {
    mesh_info.derefinement_count = ReadDataset<int>("/Blocks/derefinement_count");
  } else {
    // File does not contain this dataset, so must be older. Set to default value of zero
    if (Globals::my_rank == 0 && (GetAttr<int>("Info", "Multilevel") != 0))
      PARTHENON_WARN("Restarting from an HDF5 file that doesn't contain "
                     "/Blocks/derefinement_count. \n"
                     "  If you are running with AMR, this may cause restarts to not be "
                     "bitwise exact \n"
                     "  with simulations that are run without restarting.");
    mesh_info.derefinement_count = std::vector<int>(mesh_info.nbtotal, 0);
  }
  return mesh_info;
#endif
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
                                   const OutputUtils::VarInfo &info,
                                   std::vector<Real> &dataVec,
                                   int file_output_format_version) const {
#ifndef ENABLE_HDF5
  PARTHENON_FAIL("Restart functionality is not available because HDF5 is disabled");
#else  // HDF5 enabled
  auto hdl = OpenDataset<Real>(name);

  constexpr int VNDIM = OutputUtils::VarInfo::VNDIM;

  /** Select hyperslab in dataset **/
  int total_dim = 0;
  hsize_t offset[VNDIM], count[VNDIM];
  std::fill(offset + 1, offset + VNDIM, 0);
  std::fill(count + 1, count + VNDIM, 1);

  offset[0] = static_cast<hsize_t>(range.s);
  count[0] = static_cast<hsize_t>(range.e - range.s + 1);
  const IndexDomain domain = has_ghost != 0 ? IndexDomain::entire : IndexDomain::interior;

  // Currently supports versions 3 and 4.
  if (file_output_format_version >= HDF5::OUTPUT_VERSION_FORMAT - 1) {
    total_dim = info.FillShape<hsize_t>(domain, &(count[1])) + 1;
  } else {
    std::stringstream msg;
    msg << "File format version " << file_output_format_version << " not supported. "
        << "Current format is " << HDF5::OUTPUT_VERSION_FORMAT << std::endl;
    PARTHENON_THROW(msg)
  }

  hsize_t total_count = 1;
  for (int i = 0; i < total_dim; ++i) {
    total_count *= count[i];
  }

  PARTHENON_REQUIRE_THROWS(dataVec.size() >= total_count,
                           "Buffer (size " + std::to_string(dataVec.size()) +
                               ") is too small for dataset " + name + " (size " +
                               std::to_string(total_count) + ")");

  const H5S memspace = H5S::FromHIDCheck(H5Screate_simple(total_dim, count, NULL));
  PARTHENON_HDF5_CHECK(
      H5Sselect_hyperslab(hdl.dataspace, H5S_SELECT_SET, offset, NULL, count, NULL));

  // Read data from file
  PARTHENON_HDF5_CHECK(H5Dread(hdl.dataset, hdl.type, memspace, hdl.dataspace,
                               H5P_DEFAULT, dataVec.data()));
#endif // ENABLE_HDF5
}

} // namespace parthenon
