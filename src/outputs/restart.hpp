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
  };

  [[nodiscard]] virtual SparseInfo GetSparseInfo() const = 0;

  struct MeshInfo {
    int nbnew, nbdel, nbtotal, root_level, includes_ghost, n_ghost;
    std::vector<std::string> bound_cond;
    std::vector<int> block_size;
    std::vector<Real> grid_dim;
    std::vector<int64_t> lx123;
    std::vector<int> level_gid_lid_cnghost_gflag; // what's this?!
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
                          int file_output_format_version) const = 0;

  // Gets the data from a swarm var on current rank. Assumes all
  // blocks are contiguous. Fills dataVec based on shape from swarmvar
  // metadata.
  virtual void ReadSwarmVar(const std::string &swarmname, const std::string &varname,
                            const std::size_t count, const std::size_t offset,
                            const Metadata &m, std::vector<Real> &dataVec) = 0;
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

  // closes out the restart file
  // perhaps belongs in a destructor?
  void Close();

  // Does file have ghost cells?
  int hasGhost;
};

} // namespace parthenon
#endif // OUTPUTS_RESTART_HPP_
