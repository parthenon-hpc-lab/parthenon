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

#include "basic_types.hpp"
#include "interface/swarm_default_names.hpp"
#include "openPMD/Iteration.hpp"
#include "openPMD/Series.hpp"
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

  [[nodiscard]] SimTime GetTimeInfo() const override;

  [[nodiscard]] std::string GetInputString() const override {
    return it->getAttribute("InputFile").get<std::string>();
  };

  // Return output format version number. Return -1 if not existent.
  [[nodiscard]] int GetOutputFormatVersion() const override;

  // Current not supported
  [[nodiscard]] int HasGhost() const override { return 0; };

 public:
  // Gets data for all blocks on current rank.
  // Assumes blocks are contiguous
  // fills internal data for given pointer
  void ReadBlocks(const std::string &name, IndexRange range,
                  const OutputUtils::VarInfo &info, std::vector<Real> &dataVec,
                  int file_output_format_version, Mesh *pmesh) const override;

  // Gets the data from a swarm var on current rank. Assumes all
  // blocks are contiguous. Fills dataVec based on shape from swarmvar
  // metadata.
  template <typename T>
  void ReadSwarmVar(const std::string &swarmname, const std::string &varname,
                    const std::size_t count, const std::size_t offset, const Metadata &m,
                    std::vector<T> &data_vec) {
    openPMD::ParticleSpecies swm = it->particles[swarmname];

    const auto &shape = m.Shape();
    const int rank = shape.size();
    std::size_t nvar = 1;
    for (int i = 0; i < rank; ++i) {
      nvar *= shape[rank - 1 - i];
    }
    std::size_t total_count = nvar * count;
    if (data_vec.size() < total_count) { // greedy re-alloc
      data_vec.resize(total_count);
    }

    std::string particle_record;
    std::string particle_record_component;
    for (auto n = 0; n < nvar; n++) {
      if (varname == swarm_position::x::name()) {
        particle_record = "position";
        particle_record_component = "x";
      } else if (varname == swarm_position::y::name()) {
        particle_record = "position";
        particle_record_component = "y";
      } else if (varname == swarm_position::z::name()) {
        particle_record = "position";
        particle_record_component = "z";
      } else {
        particle_record = varname;
        particle_record_component =
            rank == 0 ? openPMD::MeshRecordComponent::SCALAR : std::to_string(n);
      }

      openPMD::RecordComponent rc = swm[particle_record][particle_record_component];
      rc.loadChunkRaw(&data_vec[n * count], {offset}, {count});
    }

    // Now actually read the registered chunks form disk
    it->seriesFlush();
  }

  void ReadSwarmVar(const std::string &swarmname, const std::string &varname,
                    const std::size_t count, const std::size_t offset, const Metadata &m,
                    std::vector<Real> &dataVec) override {
    ReadSwarmVar<>(swarmname, varname, count, offset, m, dataVec);
  };
  void ReadSwarmVar(const std::string &swarmname, const std::string &varname,
                    const std::size_t count, const std::size_t offset, const Metadata &m,
                    std::vector<int> &dataVec) override {
    ReadSwarmVar<>(swarmname, varname, count, offset, m, dataVec);
  };

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

 private:
  const std::string filename_;

  openPMD::Series series;
  // Iteration is a pointer because it cannot be default constructed (it depends on the
  // Series).
  std::unique_ptr<openPMD::Iteration> it;

  template <typename T>
  void ReadAllParamsOfType(const std::string &pkg_name, Params &params);
  template <typename... Ts>
  void ReadAllParamsOfMultipleTypes(const std::string &pkg_name, Params &p);
  template <typename T>
  void ReadAllParams(const std::string &pkg_name, Params &p);
};

} // namespace parthenon
#endif // OUTPUTS_RESTART_OPMD_HPP_
