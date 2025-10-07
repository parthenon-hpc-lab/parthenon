//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2024-2025 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
#ifndef OUTPUTS_RESTART_OPMD_HPP_
#define OUTPUTS_RESTART_OPMD_HPP_
//! \file restart_opmd.hpp
//  \brief Provides support for restarting from OpenPMD output

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// OpenPMD headers
#include <openPMD/openPMD.hpp>

#include "basic_types.hpp"
#include "outputs/parthenon_opmd.hpp"
#include "outputs/restart.hpp"
#include "pack/swarm_default_names.hpp"

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
                  Mesh *pmesh) const override;

  //  The PackOrUnpack logic requires knowledge of how data is stored and being read into
  //  the buffer. OpenPMD is dense (i.e., a face centered field has dims
  //  nx1+1, nx2, nx3 in case of the F1 field).
  [[nodiscard]] bool BlockdataIsPadded() const override { return false; };

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
    std::size_t ncomp = 1;
    for (int i = 0; i < rank; ++i) {
      ncomp *= shape[rank - 1 - i];
    }
    std::size_t total_count = ncomp * count;
    if (data_vec.size() < total_count) { // greedy re-alloc
      data_vec.resize(total_count);
    }

    for (auto n = 0; n < ncomp; n++) {
      auto [particle_record, particle_record_component] =
          OpenPMDUtils::GetParticleRecordAndComponentNames(varname, rank, n);
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
  void ReadSwarmVar(const std::string &swarmname, const std::string &varname,
                    const std::size_t count, const std::size_t offset, const Metadata &m,
                    std::vector<uint64_t> &dataVec) override {
    ReadSwarmVar<>(swarmname, varname, count, offset, m, dataVec);
  };

  // Gets the counts and offsets for MPI ranks for the meshblocks set
  // by the indexrange. Returns the total count on this rank.
  [[nodiscard]] std::size_t GetSwarmCounts(const std::string &swarm,
                                           const IndexRange &range,
                                           std::vector<std::size_t> &counts,
                                           std::vector<std::size_t> &offsets) override;

  void ReadParams(const std::string &name, Params &p) override;

  template <typename T>
  void RestoreViewAttribute(const std::string &full_path, T &view) {
    auto rank_and_dims =
        it->getAttribute(full_path + ".rankdims").get<std::vector<size_t>>();
    // Resize view.
    typename T::array_layout layout;
    for (int d = 0; d < rank_and_dims[0]; ++d) {
      layout.dimension[d] = rank_and_dims[1 + d];
    }
    // Cannot use Kokkos::resize here as it's ambiguous at this point.
    // Also, resize() interally also just create a new view.
    view = T(Kokkos::view_alloc(Kokkos::WithoutInitializing, view.label()), layout);
    auto view_h = Kokkos::create_mirror_view(HostMemSpace(), view);

    using base_t = typename std::remove_pointer<decltype(view_h.data())>::type;
    auto flat_data = it->getAttribute(full_path).get<std::vector<base_t>>();
    for (auto i = 0; i < view_h.size(); i++) {
      view_h.data()[i] = flat_data[i];
    }
    Kokkos::deep_copy(view, view_h);
  }
  [[nodiscard]] bool VariableExists(const std::string &name, const DataType data_type,
                                    const std::string swarmvarname = "") const override {
    if (data_type == DataType::Field) {
      // Given that MeshRecord labels also carry information about the topological element
      // and level, we just check for the prefix (this silently assumes that if one
      // matching record is found, then the variable exists on all levels/for all
      // components). Might cause issue for edge cases (and or variable combinations that
      // contain the `_` separator), but this should not be an issue as the error message
      // in the OpenPMD restart reader is clear (about the variable) when reading fails
      // later.
      for (auto [label, mesh] : it->meshes) {
        if (label.compare(0, name.length() + 1, name + "_") == 0) {
          return true;
        }
      }
    } else if (data_type == DataType::Swarm) {
      return it->particles.contains(name);
    } else if (data_type == DataType::SwarmVar) {
      // rank = 0, and component index = 0 because we just care about the record name
      auto [particle_record, particle_record_component] =
          OpenPMDUtils::GetParticleRecordAndComponentNames(swarmvarname, 0, 0);
      return it->particles[name].contains(particle_record);
    }
    return false;
  }
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
  void ReadAllParamsOfType(const std::string &prefix, Params &params);
  template <typename... Ts>
  void ReadAllParamsOfMultipleTypes(const std::string &prefix, Params &p);
  template <typename T>
  void ReadAllParams(const std::string &pkg_name, Params &p);
};

} // namespace parthenon
#endif // OUTPUTS_RESTART_OPMD_HPP_
