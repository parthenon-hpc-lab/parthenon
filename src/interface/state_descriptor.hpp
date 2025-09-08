//========================================================================================
// (C) (or copyright) 2020-2025. Triad National Security, LLC. All rights reserved.
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
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "basic_types.hpp"
#include "bvals/boundary_conditions.hpp"
#include "interface/metadata.hpp"
#include "interface/params.hpp"
#include "interface/sparse_pool.hpp"
#include "interface/var_id.hpp"
#include "outputs/output_parameters.hpp"
#include "parameter_input.hpp"
#include "prolong_restrict/prolong_restrict.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

// Forward declarations
template <typename T>
class MeshBlockData;
template <typename T>
class MeshData;
class AMRCriteria;
class Packages_t;
class PackDescriptorCacheBase;

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
/// TODO(JMM): The IDs here are not the same as the variable unique
/// IDs but they maybe could be? We should consider unifying that.
struct RefinementFunctionMaps {
  void Register(const Metadata &m, std::string varname);
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
  explicit StateDescriptor(std::string const &label) : label_(label) {
    if (Metadata::FlagNameExists(label)) {
      AddParam("PackageMetadataFlag_", Metadata::GetUserFlag(label));
    } else {
      AddParam("PackageMetadataFlag_", Metadata::AddUserFlag(label));
    }
  }

  // Virtual destructor for subclassing
  virtual ~StateDescriptor() = default;

  static std::shared_ptr<StateDescriptor>
  CreateResolvedStateDescriptor(Packages_t &packages);

  MetadataFlag GetMetadataFlag() {
    return params_.Get<MetadataFlag>("PackageMetadataFlag_");
  }

  template <typename T>
  void AddParam(const std::string &key, T value, Params::Mutability mutability) {
    params_.Add<T>(key, value, mutability);
  }

  template <typename T>
  void AddParam(const std::string &key, T value, bool is_mutable = false) {
    params_.Add<T>(key, value, is_mutable);
  }

  // JMM: The additional arguments here are to support additional
  // docstring arguments for ParameterInput down the line
  // I wish I could figure out how to do this with fewer overloads...
  template <typename T, typename... Args>
  void AddParamFromInput(Params::Mutability mutability, const std::string &block,
                         const std::string &name, ParameterInput *pinput,
                         Args &&...args) {
    T val = pinput->template Get<T>(block, name, std::forward<Args>(args)...);
    params_.Add(name, val, mutability);
  }
  template <typename T, typename... Args>
  void AddParamFromInput(Params::Mutability mutability, const std::string &block,
                         const std::string &name, const T &default_val,
                         ParameterInput *pinput, Args &&...args) {
    T val = pinput->template GetOrAdd<T>(block, name, default_val,
                                         std::forward<Args>(args)...);
    params_.Add(name, val, mutability);
  }
  template <typename T, typename... Args>
  void AddParamFromInput(const std::string &block, const std::string &name,
                         ParameterInput *pinput, Args &&...args) {
    AddParamFromInput(Params::Mutability::Immutable, block, name, pinput,
                      std::forward<Args>(args)...);
  }
  template <typename T, typename... Args>
  void AddParamFromInput(const std::string &block, const std::string &name,
                         const T &default_val, ParameterInput *pinput, Args &&...args) {
    AddParamFromInput(Params::Mutability::Immutable, block, name, default_val, pinput,
                      std::forward<Args>(args)...);
  }
  template <typename T, typename... Args>
  void AddParamFromInput(Params::Mutability mutability, const std::string &name,
                         ParameterInput *pinput, Args &&...args) {
    AddParamFromInput(mutability, label(), name, pinput, std::forward<Args>(args)...);
  }
  template <typename T, typename... Args>
  void AddParamFromInput(Params::Mutability mutability, const std::string &name,
                         const T &default_val, ParameterInput *pinput, Args &&...args) {
    AddParamFromInput(mutability, label(), name, default_val, pinput,
                      std::forward<Args>(args)...);
  }
  template <typename T, typename... Args>
  void AddParamFromInput(const std::string &name, ParameterInput *pinput,
                         Args &&...args) {
    AddParamFromInput(Params::Mutability::Immutable, label(), name, pinput,
                      std::forward<Args>(args)...);
  }
  template <typename T, typename... Args>
  void AddParamFromInput(const std::string &name, const T &default_val,
                         ParameterInput *pinput, Args &&...args) {
    AddParamFromInput(Params::Mutability::Immutable, label(), name, default_val, pinput,
                      std::forward<Args>(args)...);
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

  const std::type_index ParamType(const std::string &key) const {
    return params_.GetType(key);
  }

  Params &AllParams() noexcept { return params_; }

  // retrieve label
  const std::string &label() const noexcept { return label_; }

  bool AddSwarm(const std::string &swarm_name, const Metadata &m_in) {
    PARTHENON_REQUIRE(
        swarm_name != "swarm",
        "A swarm may not be named \"swarm\", as this may cause name collisions.");
    PARTHENON_REQUIRE(swarm_name.find_first_of("\n\t ") == std::string::npos,
                      "A swarm name may not contain whitespace");
    if (swarmMetadataMap_.count(swarm_name) > 0) {
      throw std::invalid_argument("Swarm " + swarm_name + " already exists!");
    }
    Metadata m = m_in; // so we can modify it
    if (!m.IsSet(GetMetadataFlag())) m.Set(GetMetadataFlag());
    swarmMetadataMap_[swarm_name] = m;

    return true;
  }
  template <typename T>
  bool AddSwarm(const Metadata &m) {
    return AddSwarm(T::name(), m);
  }

  bool AddSwarmValue(const std::string &value_name, const std::string &swarm_name,
                     const Metadata &m);
  template <typename T, typename V>
  bool AddSwarmValue(const Metadata &m) {
    return AddSwarmValue(T::name(), V::name(), m);
  }

  // field addition / retrieval routines
  bool AddField(const std::string &field_name, const Metadata &m_in,
                const std::string &controlling_field = "");
  template <typename T>
  bool AddField(const Metadata &m, const std::string &controlling_field = "") {
    return AddField(T::name(), m, controlling_field);
  }

  // add sparse pool, all arguments will be forwarded to the SparsePool constructor, so
  // one can pass in a reference to a SparsePool or arguments that match one of the
  // SparsePool constructors
  template <typename... Args>
  bool AddSparsePool(Args &&...args) {
    return AddSparsePoolImpl_(SparsePool(std::forward<Args>(args)...));
  }
  template <typename... Args>
  bool AddSparsePool(const std::string &base_name, const Metadata &m_in, Args &&...args) {
    Metadata m = m_in; // so we can modify it
    if (!m.IsSet(GetMetadataFlag())) m.Set(GetMetadataFlag());
    return AddSparsePoolImpl_(SparsePool(base_name, m, std::forward<Args>(args)...));
  }
  template <typename T, typename... Args>
  bool AddSparsePool(const Metadata &m_in, Args &&...args) {
    return AddSparsePool(T::name(), m_in, std::forward<Args>(args)...);
  }

  // retrieve number of fields
  int size() const noexcept { return metadataMap_.size(); }

  // retrieve all field names
  std::vector<std::string> Fields() noexcept;

  // retrieve all swarm names
  std::vector<std::string> Swarms() noexcept;

  const auto GetFieldVarID(const VarID &id) const {
    PARTHENON_REQUIRE_THROWS(
        metadataMap_.count(id),
        "Asking for a variable that is not in this StateDescriptor.");
    return id;
  }

  const auto &GetFieldVarID(const std::string &label) const {
    return labelToVidMap_.at(label);
  }
  const auto &GetFieldMetadata(const std::string &label) const {
    return metadataMap_.at(labelToVidMap_.at(label));
  }
  const auto &GetFieldMetadata(const VarID &id) const { return metadataMap_.at(id); }
  const auto &AllFields() const noexcept { return metadataMap_; }
  const auto &AllSparsePools() const noexcept { return sparsePoolMap_; }
  const auto &AllSwarms() const noexcept { return swarmMetadataMap_; }
  const auto &AllSwarmValues(const std::string &swarm_name) noexcept {
    // JMM: It's ok for this to be empty. Swarms with no values
    // automatically have x, y, z.
    return swarmValueMetadataMap_[swarm_name];
  }
  std::vector<std::string> GetVariableNames(const std::vector<std::string> &req_names,
                                            const Metadata::FlagCollection &flags,
                                            const std::vector<int> &sparse_ids);
  std::vector<std::string> GetVariableNames(const std::vector<std::string> &req_names,
                                            const std::vector<int> &sparse_ids);
  std::vector<std::string> GetVariableNames(const Metadata::FlagCollection &flags,
                                            const std::vector<int> &sparse_ids);
  std::vector<std::string> GetVariableNames(const std::vector<std::string> &req_names);
  std::vector<std::string> GetVariableNames(const Metadata::FlagCollection &flags);

  int GetPackDimension(const std::vector<std::string> &req_names,
                       const Metadata::FlagCollection &flags,
                       const std::vector<int> &sparse_ids);
  int GetPackDimension(const std::vector<std::string> &req_names,
                       const std::vector<int> &sparse_ids);
  int GetPackDimension(const Metadata::FlagCollection &flags,
                       const std::vector<int> &sparse_ids);
  int GetPackDimension(const std::vector<std::string> &req_names);
  int GetPackDimension(const Metadata::FlagCollection &flags);

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
                         const std::string &swarm_name) const noexcept;

  std::string GetFieldController(const std::string &field_name);
  bool ControlVariablesSet() { return (allocControllerMap_.size() > 0); }
  const std::vector<std::string> &GetControlledVariables(const std::string &field_name);
  std::vector<std::string> GetControlVariables();

  // retrieve metadata for a specific field
  const Metadata &FieldMetadata(const std::string &base_name,
                                int sparse_id = InvalidSparseID) const;
  // retrieve metadata for a specific swarm
  Metadata &SwarmMetadata(const std::string &swarm_name) noexcept {
    return swarmMetadataMap_[swarm_name];
  }
  const auto &GetSparsePool(const std::string &base_name) const noexcept {
    const auto itr = sparsePoolMap_.find(base_name);
    PARTHENON_REQUIRE_THROWS(itr != sparsePoolMap_.end(),
                             "GetSparsePool: Non-existent sparse pool: " + base_name);
    return itr->second;
  }

  bool FlagsPresent(std::vector<MetadataFlag> const &flags, bool matchAny = false);

  void PreCommFillDerived(MeshBlockData<Real> *rc) const {
    if (PreCommFillDerivedBlock != nullptr) PreCommFillDerivedBlock(rc);
  }
  void PreCommFillDerived(MeshData<Real> *rc) const {
    if (PreCommFillDerivedMesh != nullptr) PreCommFillDerivedMesh(rc);
  }
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

  void CheckRefinement(MeshData<Real> *mc, ParArray1D<AmrTag> &delta_level) const {
    if (CheckRefinementMesh != nullptr) CheckRefinementMesh(mc, delta_level);
  }

  void InitNewlyAllocatedVars(MeshData<Real> *rc) const {
    if (InitNewlyAllocatedVarsMesh != nullptr) return InitNewlyAllocatedVarsMesh(rc);
  }

  void InitNewlyAllocatedVars(MeshBlockData<Real> *rc) const {
    if (InitNewlyAllocatedVarsBlock != nullptr) return InitNewlyAllocatedVarsBlock(rc);
  }

  void UserWorkBeforeLoop(Mesh *pmesh, ParameterInput *pin, SimTime &tm) const {
    if (UserWorkBeforeLoopMesh != nullptr) return UserWorkBeforeLoopMesh(pmesh, pin, tm);
  }

  void UserWorkBeforeOutput(Mesh *pmesh, ParameterInput *pin, SimTime &tm) const {
    if (UserWorkBeforeOutputMesh != nullptr)
      return UserWorkBeforeOutputMesh(pmesh, pin, tm);
  }

  void UserWorkBeforeRestartOutput(Mesh *pmesh, ParameterInput *pin, SimTime &tm,
                                   OutputParameters *pparams) const {
    if (UserWorkBeforeRestartOutputMesh != nullptr)
      return UserWorkBeforeRestartOutputMesh(pmesh, pin, tm, pparams);
  }

  std::vector<std::shared_ptr<AMRCriteria>> amr_criteria;

  std::function<void(MeshBlockData<Real> *rc)> PreCommFillDerivedBlock = nullptr;
  std::function<void(MeshData<Real> *rc)> PreCommFillDerivedMesh = nullptr;
  std::function<void(MeshBlockData<Real> *rc)> PreFillDerivedBlock = nullptr;
  std::function<void(MeshData<Real> *rc)> PreFillDerivedMesh = nullptr;
  std::function<void(MeshBlockData<Real> *rc)> PostFillDerivedBlock = nullptr;
  std::function<void(MeshData<Real> *rc)> PostFillDerivedMesh = nullptr;
  std::function<void(MeshBlockData<Real> *rc)> FillDerivedBlock = nullptr;
  std::function<void(MeshData<Real> *rc)> FillDerivedMesh = nullptr;
  std::function<void(Mesh *, ParameterInput *, SimTime &)> UserWorkBeforeLoopMesh =
      nullptr;
  std::function<void(Mesh *, ParameterInput *, SimTime &)> UserWorkBeforeOutputMesh =
      nullptr;
  std::function<void(Mesh *, ParameterInput *, SimTime &, OutputParameters *)>
      UserWorkBeforeRestartOutputMesh = nullptr;

  std::function<void(SimTime const &simtime, MeshData<Real> *rc)> PreStepDiagnosticsMesh =
      nullptr;
  std::function<void(SimTime const &simtime, MeshData<Real> *rc)>
      PostStepDiagnosticsMesh = nullptr;

  std::function<void(Mesh *, ParameterInput *, MeshData<Real> *)> PostInitializationMesh =
      nullptr;
  std::function<void(MeshBlock *, ParameterInput *)> PostInitializationBlock = nullptr;

  std::function<Real(MeshBlockData<Real> *rc)> EstimateTimestepBlock = nullptr;
  std::function<Real(MeshData<Real> *rc)> EstimateTimestepMesh = nullptr;

  std::function<AmrTag(MeshBlockData<Real> *rc)> CheckRefinementBlock = nullptr;
  std::function<void(MeshData<Real> *rc, ParArray1D<AmrTag> &delta_level)>
      CheckRefinementMesh = nullptr;

  std::function<void(MeshData<Real> *rc)> InitNewlyAllocatedVarsMesh = nullptr;
  std::function<void(MeshBlockData<Real> *rc)> InitNewlyAllocatedVarsBlock = nullptr;

  friend std::ostream &operator<<(std::ostream &os, const StateDescriptor &sd);
  std::array<std::vector<BValFunc>, BOUNDARY_NFACES> UserBoundaryFunctions;
  std::array<std::vector<SBValFunc>, BOUNDARY_NFACES> UserSwarmBoundaryFunctions;

  // Caches for PackDescriptors associated with this StateDescriptor
  std::unordered_map<std::string, std::shared_ptr<PackDescriptorCacheBase>>
      pack_desc_cache_map;

 protected:
  // internal function to add dense/sparse fields. Private because outside classes must
  // use the public interface below
  bool AddFieldImpl_(const VarID &vid, const Metadata &m, const VarID &control_vid);

  // add a sparse pool
  bool AddSparsePoolImpl_(const SparsePool &pool);

  void InvertControllerMap();

  Params params_;
  const std::string label_;

  // for each variable label (full label for sparse variables) hold metadata
  std::unordered_map<std::string, VarID> labelToVidMap_;
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
