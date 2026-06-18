//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2024 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2026. Triad National Security, LLC. All rights reserved.
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

// This file was made in part with generative AI.

#ifndef OUTPUTS_RESTART_HPP_
#define OUTPUTS_RESTART_HPP_
//! \file io_wrapper.hpp
//  \brief defines a set of small wrapper functions for MPI versus Serial Output.

#include <cinttypes>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "interface/metadata.hpp"
#include "mesh/domain.hpp"
#include "outputs/output_utils.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

class Mesh;
class Param;

// If this number changes, the logic for reading previously written restart files in
// mesh.cpp needs to be adjusted.
constexpr int NumIDsAndFlags{5};

class RestartReader {
 public:
  RestartReader() = default;
  virtual ~RestartReader() = default;

  struct SparseInfo {
    // labels of sparse fields (full label, i.e. base name and sparse id)
    std::vector<std::string> labels;

    // allocation status of sparse fields (2D array outer dimension: block, inner
    // dimension: sparse field)
    // can't use std::vector here because std::vector<hbool_t> is the same as
    // std::vector<bool> and it doesn't have .data() member
    std::unique_ptr<bool[]> allocated;

    std::vector<int> dealloc_count;

    int num_blocks = 0;
    int num_sparse = 0;

    bool IsAllocated(int block, int sparse_field_idx) const {
      PARTHENON_REQUIRE_THROWS(allocated != nullptr,
                               "Tried to get allocation status but no data present");
      PARTHENON_REQUIRE_THROWS((block >= 0) && (block < num_blocks),
                               "Invalid block index in SparseInfo::IsAllocated");
      PARTHENON_REQUIRE_THROWS((sparse_field_idx >= 0) && (sparse_field_idx < num_sparse),
                               "Invalid sparse field index in SparseInfo::IsAllocated");

      return allocated[block * num_sparse + sparse_field_idx];
    }

    int DeallocCount(int block, int sparse_field_idx) const {
      PARTHENON_REQUIRE_THROWS(allocated != nullptr,
                               "Tried to get allocation status but no data present");
      PARTHENON_REQUIRE_THROWS((block >= 0) && (block < num_blocks),
                               "Invalid block index in SparseInfo:: DeallocCount");
      PARTHENON_REQUIRE_THROWS((sparse_field_idx >= 0) && (sparse_field_idx < num_sparse),
                               "Invalid sparse field index in SparseInfo:: DeallocCount");

      return dealloc_count[block * num_sparse + sparse_field_idx];
    }
  };

  [[nodiscard]] virtual SparseInfo GetSparseInfo() const = 0;

  struct MeshInfo {
    int nbnew, nbdel, nbtotal, root_level, includes_ghost, n_ghost;
    std::vector<int> block_size;
    std::vector<Real> grid_dim;
    std::vector<int64_t> lx123;
    std::vector<int> level_gid_lid_cnghost_gflag; // what's this?!
    std::vector<int> derefinement_count;
  };
  [[nodiscard]] virtual MeshInfo GetMeshInfo() const = 0;

  [[nodiscard]] virtual SimTime GetTimeInfo() const = 0;

  [[nodiscard]] virtual std::string GetInputString() const = 0;

  // Return output format version number. Return -1 if not existent.
  [[nodiscard]] virtual int GetOutputFormatVersion() const = 0;

  // Gets data for all blocks on current rank.
  // Assumes blocks are contiguous
  // fills internal data for given pointer
  virtual void ReadBlocks(const std::string &name, IndexRange range,
                          const OutputUtils::VarInfo &info, std::vector<Real> &dataVec,
                          Mesh *pmesh) const = 0;

  //  The PackOrUnpack logic requires knowledge of how data is stored and being read into
  //  the buffer. For HDF5 data is padded if needed (i.e., a face centered field has tims
  //  nx#+1 in all dimensions) or OpenPMD it's not (i.e., a face centered field has dims
  //  nx1+1, nx2, nx3 in case of the F1 field).
  [[nodiscard]] virtual bool BlockdataIsPadded() const = 0;

  // Gets the data from a swarm var on current rank. Assumes all
  // blocks are contiguous. Fills dataVec based on shape from swarmvar
  // metadata.
  virtual void ReadSwarmVar(const std::string &swarmname, const std::string &varname,
                            const std::size_t count, const std::size_t offset,
                            const Metadata &m, std::vector<Real> &dataVec) = 0;
  virtual void ReadSwarmVar(const std::string &swarmname, const std::string &varname,
                            const std::size_t count, const std::size_t offset,
                            const Metadata &m, std::vector<std::uint64_t> &dataVec) = 0;
  virtual void ReadSwarmVar(const std::string &swarmname, const std::string &varname,
                            const std::size_t count, const std::size_t offset,
                            const Metadata &m, std::vector<int> &dataVec) = 0;

  // Gets the counts and offsets for MPI ranks for the meshblocks set
  // by the indexrange. Returns the total count on this rank.
  [[nodiscard]] virtual std::size_t GetSwarmCounts(const std::string &swarm,
                                                   const IndexRange &range,
                                                   std::vector<std::size_t> &counts,
                                                   std::vector<std::size_t> &offsets) = 0;

  virtual void ReadParams(const std::string &name, Params &p) = 0;

  enum class DataType { Field, Swarm, SwarmVar };
  [[nodiscard]] virtual bool
  VariableExists(const std::string &name, const DataType data_type,
                 const std::string swarmvarname = "") const = 0;

  // closes out the restart file
  // perhaps belongs in a destructor?
  void Close();

  [[nodiscard]] virtual int HasGhost() const = 0;

  // Backwards compatibility: map new swarm position names to old names
  static std::string GetBackwardsCompatibleSwarmVarName(const std::string &varname) {
    if (varname == "swarm.x1") {
      return "swarm.x";
    } else if (varname == "swarm.x2") {
      return "swarm.y";
    } else if (varname == "swarm.x3") {
      return "swarm.z";
    }
    return varname;
  }

  // High-level template function to read all swarm variables of a given type from restart
  // file and distribute them to blocks
  template <typename T>
  void ReadSwarmVars(const std::shared_ptr<Swarm> &pswarm,
                     const std::vector<std::shared_ptr<MeshBlock>> &block_list,
                     const std::size_t count_on_rank, const std::size_t offset);
};

// Include full definitions needed for template implementation
#include "interface/swarm.hpp"
#include "mesh/meshblock.hpp"
#include "utils/string_utils.hpp"

// Template implementation for ReadSwarmVars
// This needs to be in the header since it's a template function
template <typename T>
void RestartReader::ReadSwarmVars(
    const std::shared_ptr<Swarm> &pswarm,
    const std::vector<std::shared_ptr<MeshBlock>> &block_list,
    const std::size_t count_on_rank, const std::size_t offset) {
  const std::string &swarmname = pswarm->label();
  std::vector<T> dataVec;
  for (const auto &var : pswarm->GetVariableVector<T>()) {
    const std::string &varname = var->label();
    const auto &m = var->metadata();
    auto arrdims = m.GetArrayDims(pswarm->GetBlockPointer(), false);

    auto var_missing_on_disk =
        !VariableExists(swarmname, DataType::SwarmVar, varname);

    // Backwards compatibility: try old position names if new ones missing
    std::string varname_to_read = varname;
    if (var_missing_on_disk) {
      varname_to_read = GetBackwardsCompatibleSwarmVarName(varname);
      // Check if the old name exists
      if (varname_to_read != varname) {
        var_missing_on_disk =
            !VariableExists(swarmname, DataType::SwarmVar, varname_to_read);
        if (!var_missing_on_disk && Globals::my_rank == 0) {
          std::cout << "SwarmVar: " << varname
                    << " using backwards-compatible name: " << varname_to_read << "\n";
        }
      }
    }

    if (Globals::my_rank == 0 && var_missing_on_disk) {
      std::cout << "SwarmVar: " << varname << " missing on disk\n";
    } else if (Globals::my_rank == 0 && varname_to_read == varname) {
      std::cout << "SwarmVar: " << varname << "\n";
    }

    if (var_missing_on_disk) {
      // TODO(JMM/PG) Add failed load list of "fail/needs fix" list
      continue;
    }

    try {
      ReadSwarmVar(swarmname, varname_to_read, count_on_rank, offset, m, dataVec);
    } catch (std::exception &ex) {
      // Variable does exist but could not be read. So we definitely want to fail here.
      PARTHENON_THROW(StringPrintf("[%d] WARNING: Failed to read Swarm %s Variable %s "
                                   "from restart file:\n%s",
                                   Globals::my_rank, swarmname.c_str(), varname.c_str(),
                                   ex.what()));
    }

    // Only safe because swarm starts completely defragged.
    // Note ordering here: block is second-inner-most loop.
    // If output format changes, this needs to change too.
    std::size_t ivec = 0;
    for (int n6 = 0; n6 < arrdims[5]; ++n6) {
      for (int n5 = 0; n5 < arrdims[4]; ++n5) {
        for (int n4 = 0; n4 < arrdims[3]; ++n4) {
          for (int n3 = 0; n3 < arrdims[2]; ++n3) {
            for (int n2 = 0; n2 < arrdims[1]; ++n2) {
              for (auto &pmb : block_list) {
                // 1 deep copy per tensor component per swarmvar per
                // block, unfortunately. But only at initialization.
                auto swarm_container = pmb->meshblock_data.Get()->GetSwarmData();
                auto pswarm_blk = swarm_container->Get(swarmname);
                auto v = Kokkos::subview(pswarm_blk->Get<T>(varname).data, n6, n5, n4, n3,
                                         n2, Kokkos::ALL());
                auto v_h = Kokkos::create_mirror_view(v);
                for (int n1 = 0; n1 < pswarm_blk->GetNumActive(); ++n1) {
                  v_h(n1) = dataVec[ivec++];
                }
                Kokkos::deep_copy(v, v_h);
              }
            }
          }
        }
      }
    }
  }
}

} // namespace parthenon
#endif // OUTPUTS_RESTART_HPP_
