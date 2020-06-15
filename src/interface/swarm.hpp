//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
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
#ifndef INTERFACE_SWARM_HPP_
#define INTERFACE_SWARM_HPP_

///
/// A swarm contains all particles of a particular species
/// Date: August 21, 2019

#include <array>
#include <list>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "athena.hpp"
#include "parthenon_arrays.hpp"
#include "variable.hpp"
#include "metadata.hpp"
#include "bvals/cc/bvals_cc.hpp"

namespace parthenon {
class MeshBlock;

enum class PARTICLE_STATUS {
  UNALLOCATED, ALIVE, DEAD
};

class Swarm {
  public:
    MeshBlock *pmy_block = nullptr; // ptr to MeshBlock

    Swarm(const std::string label, const Metadata &metadata,
          const int nmax_pool_in = 3) :
          label_(label),
          m_(metadata),
          nmax_pool_(nmax_pool_in),
          mpiStatus(true) {
      Add("x", Metadata({Metadata::Real}));
      Add("y", Metadata({Metadata::Real}));
      Add("z", Metadata({Metadata::Real}));
      Add("mask", Metadata({Metadata::Integer}));
      auto &mask = GetInteger("mask");
      for (int n = 0; n < nmax_pool_; n++) {
        mask(n) = 0;
        free_indices_.push_back(n);
      }
    }

  ///< Make a new Swarm based on an existing one
  std::shared_ptr<Swarm> AllocateCopy(const bool allocComms = false,
                                      MeshBlock *pmb = nullptr);

  ///< Add variable to swarm
  void Add(const std::string label, const Metadata &metadata);

  ///< Add multiple variables with common metadata to swarm
  void Add(const std::vector<std::string> labelVector, const Metadata &metadata);

  ///< Remote a variable from swarm
  void Remove(const std::string label);

  ParticleVariable<Real> &GetReal(const std::string label) {
    return *(realMap_.at(label));
  }

  ParticleVariable<int> &GetInteger(const std::string label) {
    return *(intMap_.at(label));
  }

  ///< Assign label for swarm
  void setLabel(const std::string label) { label_ = label; }

  ///< retrieve label for swarm
  std::string label() const { return label_; }

  ///< retrieve metadata for swarm
  const Metadata metadata() const { return m_; }

  /// Assign info for swarm
  void setInfo(const std::string info) { info_ = info; }

  /// return information string
  std::string info() { return info_; }

  /// Expand pool size geometrically as necessary
  void increasePoolMax() {
    setPoolMax(2*nmax_pool_);
  }

  /// Set max pool size
  void setPoolMax(const int nmax_pool);

  bool IsSet(const MetadataFlag bit) const { return m_.IsSet(bit); }

  int get_nmax_active() {
    return nmax_active_;
  }

  bool mpiStatus;

  int AddEmptyParticle() {
    if (free_indices_.size() == 0) {
      increasePoolMax();
    }

    auto free_index_iter = free_indices_.begin();
    int free_index = *free_index_iter;
    free_indices_.erase(free_index_iter);

    ParticleVariable<int> &mask = GetInteger("mask");
    ParticleVariable<Real> &x = GetReal("x");
    ParticleVariable<Real> &y = GetReal("y");
    ParticleVariable<Real> &z = GetReal("z");

    mask(free_index) = 1;
    nmax_active_ = std::max<int>(nmax_active_, free_index);

    x(free_index) = 0.;
    y(free_index) = 0.;
    z(free_index) = 0.;

    return free_index;
  }

  std::vector<int> AddEmptyParticles(int num_to_add) {
    while (free_indices_.size() < num_to_add) {
      increasePoolMax();
    }

    std::vector<int> indices(num_to_add);

    auto free_index = free_indices_.begin();

    ParticleVariable<int> &mask = GetInteger("mask");
    ParticleVariable<Real> &x = GetReal("x");
    ParticleVariable<Real> &y = GetReal("y");
    ParticleVariable<Real> &z = GetReal("z");

    for (int n = 0; n < num_to_add; n++) {
      indices[n] = *free_index;
      mask(*free_index) = 1;
      nmax_active_ = std::max<int>(nmax_active_, *free_index);

      x(*free_index) = 0.;
      y(*free_index) = 0.;
      z(*free_index) = 0.;

      free_index = free_indices_.erase(free_index);
    }

    return indices;
  }

  std::vector<int> AddUniformParticles(int num_to_add) {
    while (free_indices_.size() < num_to_add) {
      increasePoolMax();
    }

    return std::vector<int>();
  }

  void Defrag() {
    // TODO(BRR) Put a fast algorithm here to defrag memory
  }

 private:
  int nmax_pool_;
  int nmax_active_ = 0;
  Metadata m_;
  std::string label_;
  std::string info_;
  std::shared_ptr<ParArrayND<PARTICLE_STATUS>> pstatus_;
  ParticleVariableVector<int> intVector_;
  ParticleVariableVector<Real> realVector_;
  ParticleVariableVector<std::string> stringVector_;

  MapToParticle<int> intMap_;
  MapToParticle<Real> realMap_;
  MapToParticle<std::string> stringMap_;

  std::list<int> free_indices_;
};

using SP_Swarm = std::shared_ptr<Swarm>;
using SwarmVector = std::vector<SP_Swarm>;
using SwarmMap = std::map<std::string, SP_Swarm>;

} // namespace parthenon

#endif // INTERFACE_VARIABLE_HPP_
