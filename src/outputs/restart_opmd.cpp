//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2024 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file restart_opmd.cpp
//  \brief Restarts a simulation from an OpenPMD output with ADIOS2 backend

#include <memory>
#include <numeric>
#include <string>

#include "interface/params.hpp"
#include "openPMD/Iteration.hpp"
#include "openPMD/Series.hpp"
#include "outputs/restart.hpp"
#include "outputs/restart_opmd.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

//----------------------------------------------------------------------------------------
//! \fn void RestartReader::RestartReader(const std::string filename)
//  \brief Opens the restart file and stores appropriate file handle in fh_
RestartReaderOPMD::RestartReaderOPMD(const char *filename) : filename_(filename) {
  auto series = openPMD::Series(filename, openPMD::Access::READ_ONLY, MPI_COMM_WORLD);
  PARTHENON_REQUIRE_THROWS(
      series.iterations.size() == 1,
      "Parthenon restarts should only contain one iteration/timestep.");
  unsigned long idx;
  for (const auto &i : series.iterations) {
    idx = i.first;
  }
  it = std::make_unique<openPMD::Iteration>(series.iterations[idx]);
}

int RestartReaderOPMD::GetOutputFormatVersion() const {
  // TODO(pgrete) move info to shared header and introduce constexpr var
  if (it->containsAttribute("OutputFormatVersion")) {
    return it->getAttribute("OutputFormatVersion").get<int>();
  } else {
    return -1;
  }
}

RestartReaderOPMD::SparseInfo RestartReaderOPMD::GetSparseInfo() const {}

RestartReaderOPMD::MeshInfo RestartReaderOPMD::GetMeshInfo() const {
  RestartReaderOPMD::MeshInfo mesh_info;
  mesh_info.nbnew = it->getAttribute("NBNew").get<int>();
  mesh_info.nbdel = it->getAttribute("NBDel").get<int>();
  mesh_info.nbtotal = it->getAttribute("NumMeshBlocks").get<int>();
  mesh_info.root_level = it->getAttribute("RootLevel").get<int>();

  mesh_info.bound_cond =
      it->getAttribute("BoundaryConditions").get<std::vector<std::string>>();

  mesh_info.block_size = it->getAttribute("MeshBlockSize").get<std::vector<int>>();
  mesh_info.includes_ghost = it->getAttribute("IncludesGhost").get<int>();
  mesh_info.n_ghost = it->getAttribute("NGhost").get<int>();

  mesh_info.grid_dim = it->getAttribute("RootGridDomain").get<std::vector<Real>>();
  // TODO(pgrete) need impl
  // mesh_info.lx123 = ReadDataset<int64_t>("/Blocks/loc.lx123");
  // mesh_info.level_gid_lid_cnghost_gflag =
  //     ReadDataset<int>("/Blocks/loc.level-gid-lid-cnghost-gflag");

  return mesh_info;
}

RestartReaderOPMD::TimeInfo RestartReaderOPMD::GetTimeInfo() const {
  RestartReaderOPMD::TimeInfo time_info;

  time_info.time = it->time<Real>();
  time_info.dt = it->dt<Real>();
  time_info.ncycle = it->getAttribute("NCycle").get<int>();

  return time_info;
}
// Gets the counts and offsets for MPI ranks for the meshblocks set
// by the indexrange. Returns the total count on this rank.
std::size_t RestartReaderOPMD::GetSwarmCounts(const std::string &swarm,
                                              const IndexRange &range,
                                              std::vector<std::size_t> &counts,
                                              std::vector<std::size_t> &offsets) {}

void RestartReaderOPMD::ReadParams(const std::string &name, Params &p) {}
void RestartReaderOPMD::ReadBlocks(const std::string &name, IndexRange range,
                                   std::vector<Real> &dataVec,
                                   const std::vector<size_t> &bsize,
                                   int file_output_format_version, MetadataFlag where,
                                   const std::vector<int> &shape) const {}

} // namespace parthenon
