//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2024 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file restart_opmd.cpp
//  \brief Restarts a simulation from an OpenPMD output with ADIOS2 backend

#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "basic_types.hpp"
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
RestartReaderOPMD::RestartReaderOPMD(const char *filename)
    : filename_(filename), series(filename, openPMD::Access::READ_ONLY, MPI_COMM_WORLD) {
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
  // TODO(pgrete) needs impl
  return 0;
}

void RestartReaderOPMD::ReadParams(const std::string &name, Params &p) {
#if 0
  // views and vecs of scalar types
  ReadFromHDF5AllParamsOfTypeOrVec<bool>(prefix, group);
  ReadFromHDF5AllParamsOfTypeOrVec<int32_t>(prefix, group);
  ReadFromHDF5AllParamsOfTypeOrVec<int64_t>(prefix, group);
  ReadFromHDF5AllParamsOfTypeOrVec<uint32_t>(prefix, group);
  ReadFromHDF5AllParamsOfTypeOrVec<uint64_t>(prefix, group);
  ReadFromHDF5AllParamsOfTypeOrVec<float>(prefix, group);
  ReadFromHDF5AllParamsOfTypeOrVec<double>(prefix, group);

  // strings
  ReadFromHDF5AllParamsOfType<std::string>(prefix, group);
  ReadFromHDF5AllParamsOfType<std::vector<std::string>>(prefix, group);

  template <typename ...Ts>
void call_my_func(my_list<Ts...> )
{
    (myFunc<Ts>(), ...);
}
#endif
}
void RestartReaderOPMD::ReadBlocks(const std::string &var_name, IndexRange block_range,
                                   const OutputUtils::VarInfo &vinfo,
                                   std::vector<Real> &data_vec,
                                   int file_output_format_version, Mesh *pm) const {
  for (auto &pmb : pm->block_list) {
    // TODO(pgrete) check if we should skip the suffix for level 0
    const auto level = pmb->loc.level() - pm->GetRootLevel();
    const std::string &mesh_record_name = var_name + "_lvl" + std::to_string(level);

    PARTHENON_REQUIRE_THROWS(it->meshes.contains(mesh_record_name),
                             "Missing mesh record '" + mesh_record_name +
                                 "' in restart file.");
    auto mesh_record = it->meshes[mesh_record_name];

    int64_t comp_offset = 0; // offset data_vector to store component data
    int idx_component = 0;   // used in label for non-vector variables
    const bool is_scalar =
        vinfo.GetDim(4) == 1 && vinfo.GetDim(5) == 1 && vinfo.GetDim(6) == 1;
    const auto &Nt = vinfo.GetDim(6);
    const auto &Nu = vinfo.GetDim(5);
    const auto &Nv = vinfo.GetDim(4);
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
          PARTHENON_REQUIRE_THROWS(mesh_record.contains(comp_name),
                                   "Missing component'" + comp_name +
                                       "' in mesh record '" + mesh_record_name +
                                       "' of restart file.");
          auto mesh_comp = mesh_record[comp_name];

          openPMD::Offset chunk_offset = {
              pmb->loc.lx3() * static_cast<uint64_t>(pmb->block_size.nx(X3DIR)),
              pmb->loc.lx2() * static_cast<uint64_t>(pmb->block_size.nx(X2DIR)),
              pmb->loc.lx1() * static_cast<uint64_t>(pmb->block_size.nx(X1DIR))};
          openPMD::Extent chunk_extent = {
              static_cast<uint64_t>(pmb->block_size.nx(X3DIR)),
              static_cast<uint64_t>(pmb->block_size.nx(X2DIR)),
              static_cast<uint64_t>(pmb->block_size.nx(X1DIR))};
          mesh_comp.loadChunkRaw(&data_vec[comp_offset], chunk_offset, chunk_extent);
          // TODO(pgrete) check if output utils machinery can be used for non-cell
          // centered fields, which might not be that straightforward as a global mesh is
          // stored rather than individual blocks.
          comp_offset += pmb->block_size.nx(X1DIR) * pmb->block_size.nx(X2DIR) *
                         pmb->block_size.nx(X3DIR);
          idx_component += 1;
        }
      }
    } // loop over components
  }   // loop over blocks

  // Now actually read the registered chunks form disk
  it->seriesFlush();
}

} // namespace parthenon
