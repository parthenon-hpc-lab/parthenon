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
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace parthenon {
class MeshBlock;

enum class PARTICLE_STATUS { UNALLOCATED, ALIVE, DEAD };

class SwarmDeviceContext {
 public:
  KOKKOS_FUNCTION
  bool IsActive(int n) const { return mask_(n); }

  KOKKOS_FUNCTION
  void MarkParticleForRemoval(int n) const { marked_for_removal_(n) = true; }

  KOKKOS_FUNCTION
  bool IsMarkedForRemoval(const int n) const { return marked_for_removal_(n); }

 private:
  ParArrayND<bool> marked_for_removal_;
  ParArrayND<bool> mask_;
  friend class Swarm;
};

class Swarm {
 public:
  Swarm(const std::string &label, const Metadata &metadata, const int nmax_pool_in = 3);

  /// Returns shared pointer to a block
  std::shared_ptr<MeshBlock> GetBlockPointer() {
    if (pmy_block.expired()) {
      PARTHENON_THROW("Invalid pointer to MeshBlock!");
    }
    return pmy_block.lock();
  }

  SwarmDeviceContext GetDeviceContext() const {
    SwarmDeviceContext context;
    context.marked_for_removal_ = marked_for_removal_.data;
    context.mask_ = mask_.data;
    return context;
  }

  // Set the pointer to the mesh block for this swarm
  void SetBlockPointer(std::weak_ptr<MeshBlock> pmb) { pmy_block = pmb; }

  /// Make a new Swarm based on an existing one
  std::shared_ptr<Swarm> AllocateCopy(const bool allocComms = false,
                                      MeshBlock *pmb = nullptr);

  /// Add variable to swarm
  void Add(const std::string &label, const Metadata &metadata);

  /// Add multiple variables with common metadata to swarm
  void Add(const std::vector<std::string> &labelVector, const Metadata &metadata);

  /// Remote a variable from swarm
  void Remove(const std::string &label);

  /// Get real particle variable
  ParticleVariable<Real> &GetReal(const std::string &label) {
    return *(realMap_.at(label));
  }

  /// Get integer particle variable
  ParticleVariable<int> &GetInteger(const std::string &label) {
    return *(intMap_.at(label));
  }

  /// Assign label for swarm
  void setLabel(const std::string &label) { label_ = label; }

  /// retrieve label for swarm
  std::string label() const { return label_; }

  /// retrieve metadata for swarm
  const Metadata metadata() const { return m_; }

  /// Assign info for swarm
  void setInfo(const std::string &info) { info_ = info; }

  /// return information string
  std::string info() const { return info_; }

  /// Expand pool size geometrically as necessary
  void increasePoolMax() { setPoolMax(2 * nmax_pool_); }

  /// Set max pool size
  void setPoolMax(const int nmax_pool);

  /// Check whether metadata bit is set
  bool IsSet(const MetadataFlag bit) const { return m_.IsSet(bit); }

  /// Get the last index of active particles
  int get_max_active_index() const { return max_active_index_; }

  /// Get number of active particles
  int get_num_active() const { return num_active_; }

  /// Get the quality of the data layout. 1 is perfectly organized, < 1
  /// indicates gaps in the list.
  Real get_packing_efficiency() const { return num_active_ / (max_active_index_ + 1); }

  bool mpiStatus;

  /// Remove particles marked for removal and update internal indexing
  void RemoveMarkedParticles();

  /// Open up memory for new empty particles, return a mask to these particles
  ParArrayND<bool> AddEmptyParticles(const int num_to_add);

  /// Defragment the list by moving active particles so they are contiguous in
  /// memory
  void Defrag();

 private:
  int debug = 0;
  std::weak_ptr<MeshBlock> pmy_block;

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
using SwarmMap = std::unordered_map<std::string, SP_Swarm>;

} // namespace parthenon

#endif // INTERFACE_SWARM_HPP_
