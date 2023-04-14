//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2022 The Parthenon collaboration
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

#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "outputs/outputs.hpp"
#ifdef ENABLE_HDF5
#include "outputs/parthenon_hdf5.hpp"
#endif
#include "outputs/restart.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

//----------------------------------------------------------------------------------------
//! \fn void RestartReader::RestartReader(const std::string filename)
//  \brief Opens the restart file and stores appropriate file handle in fh_
RestartReader::RestartReader(const char *filename) : filename_(filename) {
#ifndef ENABLE_HDF5
  std::stringstream msg;
  msg << "### FATAL ERROR in Restart (Reader) constructor" << std::endl
      << "Executable not configured for HDF5 outputs, but HDF5 file format "
      << "is required for restarts" << std::endl;
  PARTHENON_FAIL(msg);
#else  // HDF5 enabled
  // Open the HDF file in read only mode
  fh_ = H5F::FromHIDCheck(H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT));

  hasGhost = GetAttr<int>("Info", "IncludesGhost");
#endif // ENABLE_HDF5
}

int RestartReader::GetOutputFormatVersion() const {
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

RestartReader::SparseInfo RestartReader::GetSparseInfo() const {
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

  // Gets the counts and offsets for MPI ranks for the meshblocks set
  // by the indexrange. Returns the total count on this rank.
std::size_t RestartReader::GetSwarmCounts(const std::string &swarm,
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

} // namespace parthenon
