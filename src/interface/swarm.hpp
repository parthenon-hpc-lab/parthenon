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

#include "basic_types.hpp"
#include "bvals/cc/bvals_cc.hpp"
#include "metadata.hpp"
#include "parthenon_arrays.hpp"
#include "variable.hpp"
#include <array>
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace parthenon {
class MeshBlock;

enum class PARTICLE_STATUS { UNALLOCATED, ALIVE, DEAD };

class Swarm {
 public:
  MeshBlock *pmy_block = nullptr; // ptr to MeshBlock

  Swarm(const std::string label, const Metadata &metadata, const int nmax_pool_in = 3);

  /// Whether particle at index is active
  bool IsActive(int index) {
    PARTHENON_DEBUG_REQUIRE(index <= max_active_index_, "Requesting particle index outside of allocated data!");
    return mask_(index);
  }

  // TODO BRR This should really be const... mask_ is managed internally
  ///< Get mask array for active particles
  ParticleVariable<bool>& GetMask() { return mask_; }

  ParticleVariable<bool>& GetMarkedForRemoval() { return marked_for_removal_; }

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
  void increasePoolMax() { setPoolMax(2 * nmax_pool_); }

  /// Set max pool size
  void setPoolMax(const int nmax_pool);

  bool IsSet(const MetadataFlag bit) const { return m_.IsSet(bit); }

  /// Get the last index of active particles
  int get_max_active_index() { return max_active_index_; }

  int get_num_active() { return num_active_; }

  /// Get the quality of the data layout. 1 is perfectly organized, < 1
  /// indicates gaps in the list.
  Real get_packing_efficiency() {
    return num_active_ / ( max_active_index_ + 1);
  }

  bool mpiStatus;

  //int AddEmptyParticle();

  KOKKOS_INLINE_FUNCTION
  void MarkParticleForRemoval(int index) {
    marked_for_removal_(index) = true;
  }

  void RemoveMarkedParticles();

  //void RemoveParticle(int index);

  ParArrayND<bool> AddEmptyParticles(int num_to_add);

  std::vector<int> AddUniformParticles(int num_to_add);

  void Defrag();

 private:
  int nmax_pool_;
  int max_active_index_ = 0;
  int num_active_ = 0;
  Metadata m_;
  std::string label_;
  std::string info_;
  std::shared_ptr<ParArrayND<PARTICLE_STATUS>> pstatus_;
  ParticleVariableVector<int> intVector_;
  ParticleVariableVector<Real> realVector_;

  MapToParticle<int> intMap_;
  MapToParticle<Real> realMap_;

  std::list<int> free_indices_;
  ParticleVariable<bool> mask_;
  ParticleVariable<bool> marked_for_removal_;
};

using SP_Swarm = std::shared_ptr<Swarm>;
using SwarmVector = std::vector<SP_Swarm>;
using SwarmMap = std::map<std::string, SP_Swarm>;

} // namespace parthenon

#endif // INTERFACE_SWARM_HPP_
