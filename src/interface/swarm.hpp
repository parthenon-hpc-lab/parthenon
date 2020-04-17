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

enum class PARTICLE_TYPE {
  INT, REAL, STRING
};

enum class PARTICLE_STATUS {
  UNALLOCATED, ALIVE, DEAD
};

class Swarm {
  public:
    Swarm(const std::string label, const Metadata &metadata,
          const int nmax_pool_in = 1000) :
          _label(label),
          _m(metadata),
          _nmax_pool(nmax_pool_in),
          mpiStatus(true) {
      Add("x", Metadata({Metadata::Real}));
      Add("y", Metadata({Metadata::Real}));
      Add("z", Metadata({Metadata::Real}));
      Add("mask", Metadata({Metadata::Integer}));
      auto &mask = GetInteger("mask");
      for (int n = 0; n < _nmax_pool; n++) {
        mask(n) = 0;
      }
    }

  ///< Add variable to swarm
  void Add(const std::string label, const Metadata &metadata);

  ///< Add multiple variables with common metadata to swarm
  void Add(const std::vector<std::string> labelVector, const Metadata &metadata);

  ///< Remote a variable from swarm
  void Remove(const std::string label);

  ParticleVariable<Real> &GetReal(const std::string label) {
    return *(_realMap.at(label));
  }

  ParticleVariable<int> &GetInteger(const std::string label) {
    return *(_intMap.at(label));
  }

  ///< Assign label for swarm
  void setLabel(const std::string label) { _label = label; }

  ///< retrieve label for swarm
  std::string label() const { return _label; }

  ///< retrieve metadata for swarm
  const Metadata metadata() const { return _m; }

  /// Assign info for swarm
  void setInfo(const std::string info) { _info = info; }

  /// return information string
  std::string info() { return _info; }

  /// Set max pool size
  void setPoolMax(const int nmax_pool) {
    _nmax_pool = nmax_pool;
    // TODO(BRR) resize arrays and copy data
  }

  int get_nmax_active() {
    return _nmax_active;
  }

  bool mpiStatus;

  void AddParticle() {
    // Check that particle fits, if not double size of pool via
    // setPoolMax(2*_nmax_pool);
  }

  void Defrag() {
    // TODO(BRR) Put a fast algorithm here to defrag memory
  }

 private:
  int _nmax_pool;
  int _nmax_active = 0;
  Metadata _m;
  std::string _label;
  std::string _info;
  std::vector<std::string> _labelArray;
  std::vector<PARTICLE_TYPE> _typeArray;
  std::shared_ptr<ParArrayND<PARTICLE_STATUS>> _pstatus;
  ParticleVariableVector<int> _intVector;
  ParticleVariableVector<Real> _realVector;
  ParticleVariableVector<std::string> _stringVector;

  MapToParticle<int> _intMap;
  MapToParticle<Real> _realMap;
  MapToParticle<std::string> _stringMap;
};

using SP_Swarm = std::shared_ptr<Swarm>;
using SwarmVector = std::vector<SP_Swarm>;
using SwarmMap = std::map<std::string, SP_Swarm>;

} // namespace parthenon

#endif // INTERFACE_VARIABLE_HPP_
