//========================================================================================
// (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
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
#include <unordered_set>
#include <utility>
#include <vector>

#include "basic_types.hpp"
#include "interface/metadata.hpp"
#include "interface/packages.hpp"
#include "interface/params.hpp"
#include "interface/sparse_pool.hpp"
#include "interface/swarm.hpp"
#include "interface/variable.hpp"
#include "mesh/refinement_in_one.hpp"
#include "refinement/amr_criteria.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

// Forward declarations
template <typename T>
class MeshBlockData;
template <typename T>
class MeshData;

/// We uniquely identify a variable by it's full label, i.e. base name plus sparse ID.
/// However, sometimes we also need to be able to separate the base name from the sparse
/// ID. Instead of relying on the fact that they are separated by a "_", we store them
/// separately in VarID struct. This way we know that a dense variable "foo_3" does not
/// have a sparse ID and a sparse field "foo_3" has base name "foo" and sparse ID 3,
/// however, the two VarIDs representing them are still considered equal, so that we find
/// such duplicates
/// TODO(JMM): Using VarID machinery for prolongation/restriction
/// implies that all vars in a sparse pool have the same custom
/// prolongation/restriction operators.
struct VarID {
  std::string base_name;
  int sparse_id;

  explicit VarID(const std::string base_name, int sparse_id = InvalidSparseID)
      : base_name(base_name), sparse_id(sparse_id) {}

  std::string label() const { return MakeVarLabel(base_name, sparse_id); }

  bool operator==(const VarID &other) const { return (label() == other.label()); }
};

struct VarIDHasher {
  auto operator()(const VarID &vid) const {
    return std::hash<std::string>{}(vid.label());
  }
};

/// A little container class owning refinement function properties
/// needed for the state descriptor.
/// Note using VarID here implies that custom prolongation/restriction
/// is identical for all sparse vars in a pool.
/// Note ignores sparse id, so all sparse ids of a
/// given sparse name have the same prolongation/restriction
/// operations
/// TODO(JMM): Should this cache be a static member field of
/// RefinementFunctions_t? That would mean we could avoid
/// StateDescriptor entirely.
struct RefinementFunctionMaps {
  void Register(const Metadata &m) {
    if (m.IsRefined()) {
      const auto &funcs = m.GetRefinementFunctions();
      bool in_map = (funcs_to_ids.count(funcs) > 0);
      if (!in_map) {
        funcs_to_ids[funcs] = next_refinement_id_++;
      }
    }
  }

  std::size_t size() const noexcept { return next_refinement_id_; }
  // A unique enumeration of refinement functions starting from zero.
  // This is used for caching which prolongation/restriction operator
  // matches which BndInfo struct in the buffer packing caches.
  // the other relevant information is in metadata, so this is all we
  // need.
  std::unordered_map<refinement::RefinementFunctions_t, std::size_t,
                     refinement::RefinementFunctionsHasher>
      funcs_to_ids;

 private:
  std::size_t next_refinement_id_ = 0;
};

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

  static std::shared_ptr<StateDescriptor>
  CreateResolvedStateDescriptor(Packages_t &packages);

  template <typename T>
  void AddParam(const std::string &key, T value, bool is_mutable = false) {
    params_.Add<T>(key, value, is_mutable);
  }

  template <typename T>
  void UpdateParam(const std::string &key, T value) {
    params_.Update<T>(key, value);
  }

  template <typename T>
  const T &Param(const std::string &key) const {
    return params_.Get<T>(key);
  }

  template <typename T>
  T *MutableParam(const std::string &key) const {
    return params_.GetMutable<T>(key);
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

  Params &AllParams() noexcept { return params_; }

  // retrieve label
  const std::string &label() const noexcept { return label_; }

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
  bool AddFieldImpl(const VarID &vid, const Metadata &m, const VarID &control_vid);

  // add a sparse pool
  bool AddSparsePoolImpl(const SparsePool &pool);

 public:
  bool AddField(const std::string &field_name, const Metadata &m,
                const std::string &controlling_field = "") {
    if (m.IsSet(Metadata::Sparse)) {
      PARTHENON_THROW(
          "Tried to add a sparse field with AddField, use AddSparsePool instead");
    }
    VarID controller = VarID(controlling_field);
    if (controlling_field == "") controller = VarID(field_name);
    return AddFieldImpl(VarID(field_name), m, controller);
  }

  // add sparse pool, all arguments will be forwarded to the SparsePool constructor, so
  // one can pass in a reference to a SparsePool or arguments that match one of the
  // SparsePool constructors
  template <typename... Args>
  bool AddSparsePool(Args &&...args) {
    return AddSparsePoolImpl(SparsePool(std::forward<Args>(args)...));
  }

  // retrieve number of fields
  int size() const noexcept { return metadataMap_.size(); }

  // retrieve all field names
  std::vector<std::string> Fields() noexcept {
    std::vector<std::string> names;
    names.reserve(metadataMap_.size());
    for (auto &x : metadataMap_) {
      names.push_back(x.first.label());
    }
    return names;
  }

  // retrieve all swarm names
  std::vector<std::string> Swarms() noexcept {
    std::vector<std::string> names;
    names.reserve(swarmMetadataMap_.size());
    for (auto &x : swarmMetadataMap_) {
      names.push_back(x.first);
    }
    return names;
  }

  const auto &AllFields() const noexcept { return metadataMap_; }
  const auto &AllSparsePools() const noexcept { return sparsePoolMap_; }
  const auto &AllSwarms() const noexcept { return swarmMetadataMap_; }
  const auto &AllSwarmValues(const std::string &swarm_name) const noexcept {
    return swarmValueMetadataMap_.at(swarm_name);
  }
  std::size_t
  RefinementFuncID(const refinement::RefinementFunctions_t &funcs) const noexcept {
    return refinementFuncMaps_.funcs_to_ids.at(funcs);
  }
  std::size_t RefinementFuncID(const Metadata &m) const noexcept {
    return RefinementFuncID(m.GetRefinementFunctions());
  }
  std::size_t NumRefinementFuncs() const noexcept { return refinementFuncMaps_.size(); }
  const auto &RefinementFncsToIDs() const noexcept {
    return refinementFuncMaps_.funcs_to_ids;
  }
  bool FieldPresent(const std::string &base_name,
                    int sparse_id = InvalidSparseID) const noexcept {
    return metadataMap_.count(VarID(base_name, sparse_id)) > 0;
  }
  bool FieldPresent(const VarID &var_id) const noexcept {
    return metadataMap_.count(var_id) > 0;
  }
  bool SparseBaseNamePresent(const std::string &base_name) const noexcept {
    return sparsePoolMap_.count(base_name) > 0;
  }
  bool SwarmPresent(const std::string &swarm_name) const noexcept {
    return swarmMetadataMap_.count(swarm_name) > 0;
  }
  bool SwarmValuePresent(const std::string &value_name,
                         const std::string &swarm_name) const noexcept {
    if (!SwarmPresent(swarm_name)) return false;
    return swarmValueMetadataMap_.at(swarm_name).count(value_name) > 0;
  }

  std::string GetFieldController(const std::string &field_name) {
    VarID field_id(field_name);
    auto controller = allocControllerReverseMap_.find(field_id);
    PARTHENON_REQUIRE(controller != allocControllerReverseMap_.end(),
                      "Asking for controlling field that is not in this package (" +
                          field_name + ")");
    return controller->second.label();
  }

  bool ControlVariablesSet() { return (allocControllerMap_.size() > 0); }

  const std::vector<std::string> &GetControlledVariables(const std::string &field_name) {
    auto iter = allocControllerMap_.find(field_name);
    if (iter == allocControllerMap_.end()) return nullControl_;
    return iter->second;
  }

  std::vector<std::string> GetControlVariables() {
    std::vector<std::string> vars;
    for (auto &pair : allocControllerMap_) {
      vars.push_back(pair.first);
    }
    return vars;
  }

  // retrieve metadata for a specific field
  const Metadata &FieldMetadata(const std::string &base_name,
                                int sparse_id = InvalidSparseID) const {
    const auto itr = metadataMap_.find(VarID(base_name, sparse_id));
    PARTHENON_REQUIRE_THROWS(itr != metadataMap_.end(),
                             "FieldMetadata: Non-existent field: " +
                                 MakeVarLabel(base_name, sparse_id));
    return itr->second;
  }

  const auto &GetSparsePool(const std::string &base_name) const noexcept {
    const auto itr = sparsePoolMap_.find(base_name);
    PARTHENON_REQUIRE_THROWS(itr != sparsePoolMap_.end(),
                             "GetSparsePool: Non-existent sparse pool: " + base_name);
    return itr->second;
  }

  // retrieve metadata for a specific swarm
  Metadata &SwarmMetadata(const std::string &swarm_name) noexcept {
    // TODO(JL) Do we want to add a default metadata for a non-existent swarm_name?
    return swarmMetadataMap_[swarm_name];
  }

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

  void InitNewlyAllocatedVars(MeshData<Real> *rc) const {
    if (InitNewlyAllocatedVarsMesh != nullptr) return InitNewlyAllocatedVarsMesh(rc);
  }

  void InitNewlyAllocatedVars(MeshBlockData<Real> *rc) const {
    if (InitNewlyAllocatedVarsBlock != nullptr) return InitNewlyAllocatedVarsBlock(rc);
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

  std::function<void(MeshData<Real> *rc)> InitNewlyAllocatedVarsMesh = nullptr;
  std::function<void(MeshBlockData<Real> *rc)> InitNewlyAllocatedVarsBlock = nullptr;

  friend std::ostream &operator<<(std::ostream &os, const StateDescriptor &sd);

 private:
  void InvertControllerMap();

  Params params_;
  const std::string label_;

  // for each variable label (full label for sparse variables) hold metadata
  std::unordered_map<VarID, Metadata, VarIDHasher> metadataMap_;
  std::unordered_map<VarID, VarID, VarIDHasher> allocControllerReverseMap_;
  std::unordered_map<std::string, std::vector<std::string>> allocControllerMap_;
  const std::vector<std::string> nullControl_{};

  // for each sparse base name hold its sparse pool
  Dictionary<SparsePool> sparsePoolMap_;

  Dictionary<Metadata> swarmMetadataMap_;
  Dictionary<Dictionary<Metadata>> swarmValueMetadataMap_;

  RefinementFunctionMaps refinementFuncMaps_;
};

inline std::shared_ptr<StateDescriptor> ResolvePackages(Packages_t &packages) {
  return StateDescriptor::CreateResolvedStateDescriptor(packages);
}

} // namespace parthenon

#endif // INTERFACE_STATE_DESCRIPTOR_HPP_
