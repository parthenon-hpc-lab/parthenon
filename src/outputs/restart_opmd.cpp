//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2024 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file restart_opmd.cpp
//  \brief Restarts a simulation from an OpenPMD output with ADIOS2 backend

#include <algorithm>
#include <cstddef>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "basic_types.hpp"
#include "interface/params.hpp"
#include "openPMD/Iteration.hpp"
#include "openPMD/Series.hpp"
#include "outputs/parthenon_opmd.hpp"
#include "outputs/restart.hpp"
#include "outputs/restart_opmd.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

//----------------------------------------------------------------------------------------
//! \fn void RestartReader::RestartReader(const std::string filename)
//  \brief Opens the restart file and stores appropriate file handle in fh_
RestartReaderOPMD::RestartReaderOPMD(const char *filename)
    : filename_(filename), series(filename, openPMD::Access::READ_ONLY, MPI_COMM_WORLD) {
  PARTHENON_REQUIRE_THROWS(
      series.iterations.size() == 1,
      "Parthenon restarts should only contain one iteration/timestep.");
  std::uint64_t idx;
  for (const auto &i : series.iterations) {
    idx = i.first;
  }
  it = std::make_unique<openPMD::Iteration>(series.iterations[idx]);
  // Explicitly open (important for parallel execution)
  it->open();
}

int RestartReaderOPMD::GetOutputFormatVersion() const {
  // TODO(pgrete) move info to shared header and introduce constexpr var
  if (it->containsAttribute("OutputFormatVersion")) {
    return it->getAttribute("OutputFormatVersion").get<int>();
  } else {
    return -1;
  }
}

RestartReaderOPMD::SparseInfo RestartReaderOPMD::GetSparseInfo() const {
  // TODO(pgrete) needs impl
  return {};
}

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
  mesh_info.lx123 = it->getAttribute("loc.lx123").get<std::vector<int64_t>>();
  mesh_info.level_gid_lid_cnghost_gflag =
      it->getAttribute("loc.level-gid-lid-cnghost-gflag").get<std::vector<int>>();
  mesh_info.derefinement_count =
      it->getAttribute("derefinement_count").get<std::vector<int>>();

  return mesh_info;
}

SimTime RestartReaderOPMD::GetTimeInfo() const {
  SimTime time_info{};

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
                                              std::vector<std::size_t> &offsets) {
  // datasets
  auto counts_dset =
      it->particles[swarm].getAttribute("counts").get<std::vector<size_t>>();
  auto offsets_dset =
      it->particles[swarm].getAttribute("offsets").get<std::vector<size_t>>();

  // Read data for requested blocks in range
  counts.resize(range.e - range.s + 1);
  offsets.resize(range.e - range.s + 1);

  std::copy(counts_dset.begin() + range.s, counts_dset.begin() + range.e + 1,
            counts.begin());
  std::copy(offsets_dset.begin() + range.s, offsets_dset.begin() + range.e + 1,
            offsets.begin());

  // Compute total count rank
  std::size_t total_count_on_rank = std::accumulate(counts.begin(), counts.end(), 0);
  return total_count_on_rank;
}

template <typename T>
void RestartReaderOPMD::ReadAllParamsOfType(const std::string &pkg_name, Params &params) {
  for (const auto &key : params.GetKeys()) {
    const auto type = params.GetType(key);
    auto mutability = params.GetMutability(key);
    if (type == std::type_index(typeid(T)) && mutability == Params::Mutability::Restart) {
      auto val = it->getAttribute("Params/" + pkg_name + "/" + key).get<T>();
      params.Update(key, val);
    }
  }
}

template <typename... Ts>
void RestartReaderOPMD::ReadAllParamsOfMultipleTypes(const std::string &pkg_name,
                                                     Params &p) {
  ([&] { ReadAllParamsOfType<Ts>(pkg_name, p); }(), ...);
}

template <typename T>
void RestartReaderOPMD::ReadAllParams(const std::string &pkg_name, Params &p) {
  ReadAllParamsOfMultipleTypes<T, std::vector<T>>(pkg_name, p);
  // TODO(pgrete) check why this doens't work, i.e., which type is causing problems
  // ReadAllParamsOfMultipleTypes<PARTHENON_ATTR_VALID_VEC_TYPES(T)>(pkg, it);
}
void RestartReaderOPMD::ReadParams(const std::string &pkg_name, Params &p) {
  ReadAllParams<int32_t>(pkg_name, p);
  ReadAllParams<int64_t>(pkg_name, p);
  ReadAllParams<uint32_t>(pkg_name, p);
  ReadAllParams<uint64_t>(pkg_name, p);
  ReadAllParams<float>(pkg_name, p);
  ReadAllParams<double>(pkg_name, p);
  ReadAllParams<std::string>(pkg_name, p);
  ReadAllParamsOfType<bool>(pkg_name, p);
}

void RestartReaderOPMD::ReadBlocks(const std::string &var_name, IndexRange block_range,
                                   const OutputUtils::VarInfo &vinfo,
                                   std::vector<Real> &data_vec,
                                   int file_output_format_version, Mesh *pm) const {
  int64_t comp_offset = 0; // offset data_vector to store component data
  for (auto &pmb : pm->block_list) {
    // TODO(pgrete) check if we should skip the suffix for level 0
    const auto level = pmb->loc.level() - pm->GetRootLevel();

    int comp_idx = 0; // used in label for non-vector variables
    const auto &Nt = vinfo.GetDim(6);
    const auto &Nu = vinfo.GetDim(5);
    const auto &Nv = vinfo.GetDim(4);
    // loop over all components
    for (int t = 0; t < Nt; ++t) {
      for (int u = 0; u < Nu; ++u) {
        for (int v = 0; v < Nv; ++v) {
          // Get the correct record
          const auto [record_name, comp_name] =
              OpenPMDUtils::GetMeshRecordAndComponentNames(vinfo, comp_idx, level);

          PARTHENON_REQUIRE_THROWS(it->meshes.contains(record_name),
                                   "Missing mesh record '" + record_name +
                                       "' in restart file.");
          auto mesh_record = it->meshes[record_name];
          PARTHENON_REQUIRE_THROWS(mesh_record.contains(comp_name),
                                   "Missing component'" + comp_name +
                                       "' in mesh record '" + record_name +
                                       "' of restart file.");
          auto mesh_comp = mesh_record[comp_name];

          const auto [chunk_offset, chunk_extent] =
              OpenPMDUtils::GetChunkOffsetAndExtent(pm, pmb);
          mesh_comp.loadChunkRaw(&data_vec[comp_offset], chunk_offset, chunk_extent);
          // TODO(pgrete) check if output utils machinery can be used for non-cell
          // centered fields, which might not be that straightforward as a global mesh
          // is stored rather than individual blocks.
          comp_offset += pmb->block_size.nx(X1DIR) * pmb->block_size.nx(X2DIR) *
                         pmb->block_size.nx(X3DIR);
          comp_idx += 1;
        }
      }
    } // loop over components
  }   // loop over blocks

  // Now actually read the registered chunks form disk
  it->seriesFlush();
}

} // namespace parthenon
