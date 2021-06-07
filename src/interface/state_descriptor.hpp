//========================================================================================
// (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
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
#include "interface/sparse_pool.hpp"
#include "interface/swarm.hpp"
#include "interface/variable.hpp"
#include "refinement/amr_criteria.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

// Forward declarations
template <typename T>
class MeshBlockData;
template <typename T>
class MeshData;

class StateDescriptor; // forward declaration

class Packages_t {
 public:
  Packages_t() = default;
  void Add(const std::shared_ptr<StateDescriptor> &package);

  std::shared_ptr<StateDescriptor> const &Get(const std::string &name) {
    return packages_.at(name);
  }

  const Dictionary<std::shared_ptr<StateDescriptor>> &AllPackages() const {
    return packages_;
  }

 private:
  Dictionary<std::shared_ptr<StateDescriptor>> packages_;
};

/// VarID uniquely identifies a variable by its base name and sparse id
struct VarID {
  std::string base_name;
  int sparse_id;

  explicit VarID(const std::string base_name, int sparse_id = InvalidSparseID)
      : base_name(base_name), sparse_id(sparse_id) {}

  std::string label() const { return MakeVarLabel(base_name, sparse_id); }

  bool operator==(const VarID &other) const {
    return (base_name == other.base_name) && (sparse_id == other.sparse_id);
  }
};

struct VarIDHasher {
  auto operator()(const VarID &vid) const {
    return std::hash<std::string>{}(vid.label());
  }
};

/// The state metadata descriptor class.
///
/// Each State descriptor has a label, associated parameters, and
/// metadata for all fields within that state.
class StateDescriptor {
  friend class DenseFieldProvider;

 public:
  // copy constructor throw error
  StateDescriptor(const StateDescriptor &s) = delete;

  // Preferred constructor
  explicit StateDescriptor(std::string const &label) : label_(label) {}

  static std::shared_ptr<StateDescriptor>
  CreateResolvedStateDescriptor(Packages_t &packages);

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
 private:
  // internal function to add dense/sparse fields. Private because outside classes must
  // use the public interface below
  bool AddFieldImpl(const VarID &vid, const Metadata &m);

  // add a sparse pool
  bool AddSparsePoolImpl(const SparsePool &pool);

 public:
  bool AddField(const std::string &field_name, const Metadata &m) {
    if (m.IsSet(Metadata::Sparse)) {
      PARTHENON_THROW("Tried to add a sparse field with AddField (or deprecated "
                      "AddField), use AddSparseFields instead");
    }

    return AddFieldImpl(VarID(field_name), m);
  }

  // add sparse pool, all arguments will be forwarded to the SparsePool constructor, so
  // one can pass in a reference to a SparsePool or arguments that match one of the
  // SparsePool constructors
  template <typename... Args>
  bool AddSparsePool(const Args &... args) {
    return AddSparsePoolImpl(SparsePool(args...));
  }

  // retrieve number of fields
  int size() const { return metadataMap_.size(); }

  // retrieve all field names
  std::vector<std::string> Fields() {
    std::vector<std::string> names;
    names.reserve(metadataMap_.size());
    for (auto &x : metadataMap_) {
      names.push_back(x.first.label());
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

  const auto &AllFields() const { return metadataMap_; }
  const auto &AllSparsePools() const { return sparsePoolMap_; }
  const auto &AllSwarms() const { return swarmMetadataMap_; }
  const auto &AllSwarmValues(const std::string &swarm_name) const {
    return swarmValueMetadataMap_.at(swarm_name);
  }
  bool FieldPresent(const std::string &base_name, int sparse_id = InvalidSparseID) const {
    return metadataMap_.count(VarID(base_name, sparse_id)) > 0;
  }
  bool SparseBaseNamePresent(const std::string &base_name) const {
    return sparsePoolMap_.count(base_name) > 0;
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
  const Metadata &FieldMetadata(const std::string &base_name,
                                int sparse_id = InvalidSparseID) const {
    static const Metadata empty_metadata;

    const auto itr = metadataMap_.find(VarID(base_name, sparse_id));
    if (itr == metadataMap_.end()) {
      return empty_metadata;
    } else {
      return itr->second;
    }
  }

  const auto &GetSparsePool(const std::string &base_name) const {
    static const SparsePool empty_pool("EmptySparsePool", Metadata({Metadata::Sparse}));

    const auto itr = sparsePoolMap_.find(base_name);
    if (itr == sparsePoolMap_.end()) {
      return empty_pool;
    } else {
      return itr->second;
    }
  }

  // retrieve metadata for a specific swarm
  Metadata &SwarmMetadata(const std::string &swarm_name) {
    // TODO(JL) Do we want to add a default metadata for a non-existent swarm_name?
    return swarmMetadataMap_[swarm_name];
  }

  // JL: Disabling this for now because it's probably incomplete
  // // get all metadata for this physics
  // const Dictionary<Metadata> &AllMetadata() {
  //   // TODO (JL): What about swarmMetadataMap_ and swarmValueMetadataMap_
  //   return metadataMap_;
  // }

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
  Params params_;
  const std::string label_;

  // for each variable label (full label for sparse variables) hold metadata
  std::unordered_map<VarID, Metadata, VarIDHasher> metadataMap_;

  // for each sparse base name hold its sparse pool
  Dictionary<SparsePool> sparsePoolMap_;

  Dictionary<Metadata> swarmMetadataMap_;
  Dictionary<Dictionary<Metadata>> swarmValueMetadataMap_;
};

inline std::shared_ptr<StateDescriptor> ResolvePackages(Packages_t &packages) {
  return StateDescriptor::CreateResolvedStateDescriptor(packages);
}

} // namespace parthenon

#endif // INTERFACE_STATE_DESCRIPTOR_HPP_
