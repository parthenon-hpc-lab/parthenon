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
#ifndef INTERFACE_STATE_DESCRIPTOR_HPP_
#define INTERFACE_STATE_DESCRIPTOR_HPP_

#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "basic_types.hpp"
#include "interface/metadata.hpp"
#include "interface/params.hpp"
#include "interface/swarm.hpp"
#include "refinement/amr_criteria.hpp"

namespace parthenon {

// Forward declarations
template <typename T>
class MeshBlockData;
template <typename T>
class MeshData;

/// The state metadata descriptor class.
///
/// Each State descriptor has a label, associated parameters, and
/// metadata for all fields within that state.
class StateDescriptor {
 public:
  // copy constructor throw error
  StateDescriptor(const StateDescriptor &s) = delete;

  // Preferred constructor
  explicit StateDescriptor(std::string label) : label_(label) {
    PostFillDerivedBlock = nullptr;
    PostFillDerivedMesh = nullptr;
    PreFillDerivedBlock = nullptr;
    PreFillDerivedMesh = nullptr;
    FillDerivedBlock = nullptr;
    FillDerivedMesh = nullptr;
    EstimateTimestepBlock = nullptr;
    EstimateTimestepMesh = nullptr;
    CheckRefinementBlock = nullptr;
  }

  template <typename T>
  void AddParam(const std::string &key, T value) {
    params_.Add<T>(key, value);
  }

  template <typename T>
  const T &Param(const std::string &key) {
    return params_.Get<T>(key);
  }

  // Set (if not set) and get simultaneously.
  // infers type correctly.
  template <typename T>
  const T &Param(const std::string &key, T value) {
    params_.Get(key, value);
  }

  Params &AllParams() { return params_; }
  // retrieve label
  const std::string &label() const { return label_; }

  bool AddSwarm(const std::string &swarm_name, const Metadata &m) {
    if (swarmMetadataMap_.count(swarm_name) > 0) {
      throw std::invalid_argument("Swarm " + swarm_name + " already exists!");
    }
    swarmMetadataMap_[swarm_name] = m;

    return true;
  }

  bool AddSwarmValue(const std::string &value_name, const std::string &swarm_name,
                     const Metadata &m);

  // field addition / retrieval routines
  // add a field with associated metadata
  bool AddField(const std::string &field_name, const Metadata &m);

  // retrieve number of fields
  int size() const { return metadataMap_.size(); }

  // Ensure all required bits are present
  // projective and can be called multiple times with no harm
  void ValidateMetadata();

  // retrieve all field names
  std::vector<std::string> Fields() {
    std::vector<std::string> names;
    names.reserve(metadataMap_.size());
    for (auto &x : metadataMap_) {
      names.push_back(x.first);
    }
    return names;
  }

  // retrieve all swarm names
  std::vector<std::string> Swarms() {
    std::vector<std::string> names;
    names.reserve(swarmMetadataMap_.size());
    for (auto &x : swarmMetadataMap_) {
      names.push_back(x.first);
    }
    return names;
  }

  const Dictionary<Metadata> &AllFields() const { return metadataMap_; }
  const Dictionary<std::unordered_map<int, Metadata>> &AllSparseFields() const {
    return sparseMetadataMap_;
  }
  const Dictionary<Metadata> &AllSwarms() const { return swarmMetadataMap_; }
  const Dictionary<Metadata> &AllSwarmValues(const std::string &swarm_name) const {
    return swarmValueMetadataMap_.at(swarm_name);
  }
  bool FieldPresent(const std::string &field_name) const {
    return metadataMap_.count(field_name) > 0;
  }
  bool SparsePresent(const std::string &field_name) const {
    return sparseMetadataMap_.count(field_name) > 0;
  }
  bool SparsePresent(const std::string &field_name, int i) const {
    if (sparseMetadataMap_.count(field_name) > 0) {
      return sparseMetadataMap_.at(field_name).count(i) > 0;
    }
    return false;
  }
  bool SwarmPresent(const std::string &swarm_name) const {
    return swarmMetadataMap_.count(swarm_name) > 0;
  }
  bool SwarmValuePresent(const std::string &value_name,
                         const std::string &swarm_name) const {
    if (!SwarmPresent(swarm_name)) return false;
    return swarmValueMetadataMap_.at(swarm_name).count(value_name) > 0;
  }

  // retrieve metadata for a specific field
  Metadata &FieldMetadata(const std::string &field_name) {
    return metadataMap_[field_name];
  }

  Metadata &FieldMetadata(const std::string &field_name, int i) {
    return sparseMetadataMap_[field_name][i];
  }

  // retrieve metadata for a specific swarm
  Metadata &SwarmMetadata(const std::string &swarm_name) {
    return swarmMetadataMap_[swarm_name];
  }

  // get all metadata for this physics
  const Dictionary<Metadata> &AllMetadata() { return metadataMap_; }

  bool FlagsPresent(std::vector<MetadataFlag> const &flags, bool matchAny = false);

  void PreFillDerived(MeshBlockData<Real> *rc) const {
    if (PreFillDerivedBlock != nullptr) PreFillDerivedBlock(rc);
  }
  void PreFillDerived(MeshData<Real> *rc) const {
    if (PreFillDerivedMesh != nullptr) PreFillDerivedMesh(rc);
  }
  void PostFillDerived(MeshBlockData<Real> *rc) const {
    if (PostFillDerivedBlock != nullptr) PostFillDerivedBlock(rc);
  }
  void PostFillDerived(MeshData<Real> *rc) const {
    if (PostFillDerivedMesh != nullptr) PostFillDerivedMesh(rc);
  }
  void FillDerived(MeshBlockData<Real> *rc) const {
    if (FillDerivedBlock != nullptr) FillDerivedBlock(rc);
  }
  void FillDerived(MeshData<Real> *rc) const {
    if (FillDerivedMesh != nullptr) FillDerivedMesh(rc);
  }

  Real EstimateTimestep(MeshBlockData<Real> *rc) const {
    if (EstimateTimestepBlock != nullptr) return EstimateTimestepBlock(rc);
    return std::numeric_limits<Real>::max();
  }
  Real EstimateTimestep(MeshData<Real> *rc) const {
    if (EstimateTimestepMesh != nullptr) return EstimateTimestepMesh(rc);
    return std::numeric_limits<Real>::max();
  }

  AmrTag CheckRefinement(MeshBlockData<Real> *rc) const {
    if (CheckRefinementBlock != nullptr) return CheckRefinementBlock(rc);
    return AmrTag::derefine;
  }

  std::vector<std::shared_ptr<AMRCriteria>> amr_criteria;
  void (*PreFillDerivedBlock)(MeshBlockData<Real> *rc);
  void (*PreFillDerivedMesh)(MeshData<Real> *rc);
  void (*PostFillDerivedBlock)(MeshBlockData<Real> *rc);
  void (*PostFillDerivedMesh)(MeshData<Real> *rc);
  void (*FillDerivedBlock)(MeshBlockData<Real> *rc);
  void (*FillDerivedMesh)(MeshData<Real> *rc);
  Real (*EstimateTimestepBlock)(MeshBlockData<Real> *rc);
  Real (*EstimateTimestepMesh)(MeshData<Real> *rc);
  AmrTag (*CheckRefinementBlock)(MeshBlockData<Real> *rc);

  friend std::ostream &operator<<(std::ostream &os, const StateDescriptor &sd);

 private:
  template <typename F>
  void MetadataLoop_(F func) {
    for (auto &pair : metadataMap_) {
      func(pair.second);
    }
    for (auto &p1 : sparseMetadataMap_) {
      for (auto &p2 : p1.second) {
        func(p2.second);
      }
    }
    for (auto &pair : swarmMetadataMap_) {
      func(pair.second);
    }
  }

  Params params_;
  const std::string label_;

  Dictionary<Metadata> metadataMap_;
  Dictionary<std::unordered_map<int, Metadata>> sparseMetadataMap_;
  Dictionary<Metadata> swarmMetadataMap_;
  Dictionary<Dictionary<Metadata>> swarmValueMetadataMap_;
};

using Packages_t = Dictionary<std::shared_ptr<StateDescriptor>>;

std::shared_ptr<StateDescriptor> ResolvePackages(Packages_t &packages);

} // namespace parthenon

#endif // INTERFACE_STATE_DESCRIPTOR_HPP_
