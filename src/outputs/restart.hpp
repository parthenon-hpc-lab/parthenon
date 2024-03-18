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

#include "mesh/domain.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

class Mesh;
class Param;

class RestartReader {
 public:
  RestartReader();
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

  SparseInfo GetSparseInfo() const;

  // Return output format version number. Return -1 if not existent.
  int GetOutputFormatVersion() const;

 public:
  // Gets data for all blocks on current rank.
  // Assumes blocks are contiguous
  // fills internal data for given pointer
  template <typename T>
  void ReadBlocks(const std::string &name, IndexRange range, std::vector<T> &dataVec,
                  const std::vector<size_t> &bsize, int file_output_format_version,
                  MetadataFlag where, const std::vector<int> &shape = {}) const;

  // Gets the data from a swarm var on current rank. Assumes all
  // blocks are contiguous. Fills dataVec based on shape from swarmvar
  // metadata.
  template <typename T>
  void ReadSwarmVar(const std::string &swarmname, const std::string &varname,
                    const std::size_t count, const std::size_t offset, const Metadata &m,
                    std::vector<T> &dataVec);

  // Reads an array dataset from file as a 1D vector.
  template <typename T>
  std::vector<T> ReadDataset(const std::string &name) const;

  template <typename T>
  std::vector<T> GetAttrVec(const std::string &location, const std::string &name) const;

  template <typename T>
  T GetAttr(const std::string &location, const std::string &name) const;

  // Gets the counts and offsets for MPI ranks for the meshblocks set
  // by the indexrange. Returns the total count on this rank.
  std::size_t GetSwarmCounts(const std::string &swarm, const IndexRange &range,
                             std::vector<std::size_t> &counts,
                             std::vector<std::size_t> &offsets);

  void ReadParams(const std::string &name, Params &p);

  // closes out the restart file
  // perhaps belongs in a destructor?
  void Close();

  // Does file have ghost cells?
  int hasGhost;
};

} // namespace parthenon
#endif // OUTPUTS_RESTART_HDF5_HPP_
