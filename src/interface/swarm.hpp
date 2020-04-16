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
#include "particle_metadata.hpp"
#include "swarm_metadata.hpp"
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
    Swarm(const std::string label, const SwarmMetadata &smetadata,
      const int nmax_pool_in = 1000) :
      _label(label),
      _sm(smetadata),
      _nmax_pool(nmax_pool_in),
      mpiStatus(true) {}

  ///< Assign label for swarm
  void setLabel(const std::string label) { _label = label; }

  ///< retrieve label for swarm
  std::string label() const { return _label; }

  ///< retrieve metadata for swarm
  const SwarmMetadata swarmmetadata() const { return _sm; }

  /// Assign info for swarm
  void setInfo(const std::string info) { _info = info; }

  /// return information string
  std::string info() { return _info; }

  /// Set max pool size
  void setPoolMax(const int nmax_pool) {
    _nmax_pool = nmax_pool;
    // TODO(BRR) resize arrays and copy data
  }

  bool mpiStatus;

  void Add(const std::string &label, const ParticleMetadata) {
    // TODO(BRR) fix this
    PARTICLE_TYPE type = PARTICLE_TYPE::REAL;
    // TODO(BRR) check that value isn't already enrolled?
    _labelArray.push_back(label);
    _typeArray.push_back(type);
    if (type == PARTICLE_TYPE::INT) {
      _intArray.push_back(std::make_shared<ParArrayND<int>>(label, _nmax_pool));
    } else if (type == PARTICLE_TYPE::REAL) {
      _realArray.push_back(std::make_shared<ParArrayND<Real>>(label, _nmax_pool));
    } else if (type == PARTICLE_TYPE::STRING) {
      _stringArray.push_back(std::make_shared<ParArrayND<std::string>>(label, _nmax_pool));
    } else {
      throw std::invalid_argument(std::string("\n") +
                                  std::to_string(static_cast<int>(type)) +
                                  std::string(" not a PARTICLE_TYPE in Add()\n") );

    }
  }

  void AddParticle() {
    // Check that particle fits, if not double size of pool via
    // setPoolMax(2*_nmax_pool);
  }

  void Defrag() {
    // TODO(BRR) Put a fast algorithm here to defrag memory
  }

 private:
  int _nmax_pool;
  int _nmax_occupied = 0;
  SwarmMetadata _sm;
  std::string _label;
  std::string _info;
  std::vector<std::string> _labelArray;
  std::vector<PARTICLE_TYPE> _typeArray;
  std::shared_ptr<ParArrayND<PARTICLE_STATUS>> _pstatus;
  std::vector<std::shared_ptr<ParArrayND<int>>> _intArray;
  std::vector<std::shared_ptr<ParArrayND<Real>>> _realArray;
  std::vector<std::shared_ptr<ParArrayND<std::string>>> _stringArray;
};

using SP_Swarm = std::shared_ptr<Swarm>;
using SwarmVector = std::vector<SP_Swarm>;
using SwarmMap = std::map<std::string, SP_Swarm>;

} // namespace parthenon

#endif // INTERFACE_VARIABLE_HPP_
