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
    Swarm(const std::string label, const Metadata &metadata,
          const int nmax_pool_in = 3) :
          label_(label),
          m_(metadata),
          nmax_pool_(nmax_pool_in),
          mpiStatus(true) {
      printf("CONSTRUCTING SWARM: %s\n", label.c_str());
      Add("x", Metadata({Metadata::Real}));
      Add("y", Metadata({Metadata::Real}));
      Add("z", Metadata({Metadata::Real}));
      Add("test2", Metadata({Metadata::Real}));
      Add("mask", Metadata({Metadata::Integer}));
      auto &mask = GetInteger("mask");
      printf("Initializing pool:\n");
      for (int n = 0; n < nmax_pool_; n++) {
        mask(n) = 0;
        printf("  n = %i\n", n);
        free_indices_.push_back(n);
      }
      printf("Swarm \"%s\" constructed\n", label.c_str());
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
    for (auto key : realMap_) {
      printf("key: %s\n", key.first.c_str());
    }
    printf("realVector_.size() = %i\n", realVector_.size());
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

  /// Set max pool size
  void setPoolMax(const int nmax_pool) {
    printf("SET POOL MAX\n");
    if (nmax_pool < nmax_pool_) {
      printf("Must increase pool size!\n");
      exit(-1);
    }
    int n_new_begin = nmax_pool_;
    int n_new = nmax_pool - nmax_pool_;

    for (int n = 0; n < n_new; n++) {
      free_indices_.push_back(n + n_new_begin);
    }

    //mask.resize

    // Resize and copy data
    for (int n = 0; n < intVector_.size(); n++) {
      auto oldvar = intVector_[n];
      auto newvar = std::make_shared<ParticleVariable<int>>(oldvar->label(),
                                                            nmax_pool,
                                                            oldvar->metadata());
      for (int m = 0; m < nmax_pool_; m++) {
        (*oldvar)(m) = (*newvar)(m);
      }
      intVector_[n] = newvar;
      intMap_[oldvar->label()] = newvar;
    }


    // TODO(BRR) require that nmax_pool > nmax_pool_?
    // TODO(BRR) resize arrays and copy data
    nmax_pool_ = nmax_pool;
  }

  bool IsSet(const MetadataFlag bit) const { return m_.IsSet(bit); }

  void printrealvars() {
    for (auto var : realMap_) {
      printf("swarm var: %s\n", var.first.c_str());
    }
  }

  int get_nmax_active() {
    return nmax_active_;
  }

  bool mpiStatus;

  int AddEmptyParticle() {
    printf("AddParticle! swarm name: %s\n", label_.c_str());
    // Check that particle fits, if not double size of pool via
    // setPoolMax(2*_nmax_pool);
    // TODO silly example here
    nmax_active_ = 1;
    return 0;
  }

  std::vector<int> AddEmptyParticles(int num_to_add) {
    return std::vector<int>();
  }

  std::vector<int> AddUniformParticles(int num_to_add) {
    return std::vector<int>();
  }

  // TODO(BRR) add a bunch of particles at once?

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
