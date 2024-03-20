//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2024 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
#ifndef OUTPUTS_RESTART_OPMD_HPP_
#define OUTPUTS_RESTART_OPMD_HPP_
//! \file restart_opmd.hpp
//  \brief Provides support for restarting from OpenPMD output

#include <memory>
#include <string>
#include <vector>

#include "openPMD/Iteration.hpp"
#include "outputs/restart.hpp"

#include "mesh/domain.hpp"

namespace parthenon {

class Mesh;
class Param;

class RestartReaderOPMD : public RestartReader {
 public:
  explicit RestartReaderOPMD(const char *filename);

  [[nodiscard]] SparseInfo GetSparseInfo() const override;

  [[nodiscard]] MeshInfo GetMeshInfo() const override;

  [[nodiscard]] TimeInfo GetTimeInfo() const override;

  [[nodiscard]] std::string GetInputString() const override {
    return it->getAttribute("InputFile").get<std::string>();
  };

  // Return output format version number. Return -1 if not existent.
  [[nodiscard]] int GetOutputFormatVersion() const override;

 public:
  // Gets data for all blocks on current rank.
  // Assumes blocks are contiguous
  // fills internal data for given pointer
  void ReadBlocks(const std::string &name, IndexRange range, std::vector<Real> &dataVec,
                  const std::vector<size_t> &bsize, int file_output_format_version,
                  MetadataFlag where, const std::vector<int> &shape = {}) const override;

  void ReadSwarmVar(const std::string &swarmname, const std::string &varname,
                    const std::size_t count, const std::size_t offset, const Metadata &m,
                    std::vector<Real> &dataVec) override{};
  void ReadSwarmVar(const std::string &swarmname, const std::string &varname,
                    const std::size_t count, const std::size_t offset, const Metadata &m,
                    std::vector<int> &dataVec) override{};

  // Gets the counts and offsets for MPI ranks for the meshblocks set
  // by the indexrange. Returns the total count on this rank.
  [[nodiscard]] std::size_t GetSwarmCounts(const std::string &swarm,
                                           const IndexRange &range,
                                           std::vector<std::size_t> &counts,
                                           std::vector<std::size_t> &offsets) override;

  void ReadParams(const std::string &name, Params &p) override;

  // closes out the restart file
  // perhaps belongs in a destructor?
  void Close();

  // Does file have ghost cells?
  int hasGhost;

 private:
  const std::string filename_;
  std::unique_ptr<openPMD::Iteration> it;
};

} // namespace parthenon
#endif // OUTPUTS_RESTART_OPMD_HPP_
