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
#include "utils/error_checking.hpp"

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
  explicit StateDescriptor(std::string const &label) : label_(label) {}

  template <typename T>
  void AddParam(const std::string &key, T value) {
    params_.Add<T>(key, value);
  }

  template <typename T>
  void UpdateParam(const std::string &key, T value) {
    params_.Update<T>(key, value);
  }

  template <typename T>
  const T &Param(const std::string &key) const {
    return params_.Get<T>(key);
  }

  // Set (if not set) and get simultaneously.
  // infers type correctly.
  template <typename T>
  const T &Param(const std::string &key, T value) const {
    return params_.Get(key, value);
  }

  const std::type_index &ParamType(const std::string &key) const {
    return params_.GetType(key);
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

  void PreStepDiagnostics(SimTime const &simtime, MeshData<Real> *rc) const {
    if (PreStepDiagnosticsMesh != nullptr) PreStepDiagnosticsMesh(simtime, rc);
  }
  void PostStepDiagnostics(SimTime const &simtime, MeshData<Real> *rc) const {
    if (PostStepDiagnosticsMesh != nullptr) PostStepDiagnosticsMesh(simtime, rc);
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

  std::function<void(MeshBlockData<Real> *rc)> PreFillDerivedBlock = nullptr;
  std::function<void(MeshData<Real> *rc)> PreFillDerivedMesh = nullptr;
  std::function<void(MeshBlockData<Real> *rc)> PostFillDerivedBlock = nullptr;
  std::function<void(MeshData<Real> *rc)> PostFillDerivedMesh = nullptr;
  std::function<void(MeshBlockData<Real> *rc)> FillDerivedBlock = nullptr;
  std::function<void(MeshData<Real> *rc)> FillDerivedMesh = nullptr;

  std::function<void(SimTime const &simtime, MeshData<Real> *rc)> PreStepDiagnosticsMesh =
      nullptr;
  std::function<void(SimTime const &simtime, MeshData<Real> *rc)>
      PostStepDiagnosticsMesh = nullptr;

  std::function<Real(MeshBlockData<Real> *rc)> EstimateTimestepBlock = nullptr;
  std::function<Real(MeshData<Real> *rc)> EstimateTimestepMesh = nullptr;

  std::function<AmrTag(MeshBlockData<Real> *rc)> CheckRefinementBlock = nullptr;

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

class Packages_t {
 public:
  Packages_t() = default;
  void Add(const std::shared_ptr<StateDescriptor> &package) {
    const auto &name = package->label();
    PARTHENON_REQUIRE_THROWS(packages_.count(name) == 0,
                             "Package name " + name + " must be unique.");
    packages_[name] = package;
    return;
  }
  std::shared_ptr<StateDescriptor> const &Get(const std::string &name) {
    return packages_.at(name);
  }
  const Dictionary<std::shared_ptr<StateDescriptor>> &AllPackages() const {
    return packages_;
  }

 private:
  Dictionary<std::shared_ptr<StateDescriptor>> packages_;
};

std::shared_ptr<StateDescriptor> ResolvePackages(Packages_t &packages);

} // namespace parthenon

#endif // INTERFACE_STATE_DESCRIPTOR_HPP_
