//========================================================================================
// (C) (or copyright) 2020-2024. Triad National Security, LLC. All rights reserved.
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
#ifndef INTERFACE_MESHBLOCK_DATA_HPP_
#define INTERFACE_MESHBLOCK_DATA_HPP_

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "basic_types.hpp"
#include "interface/data_collection.hpp"
#include "interface/sparse_pack_base.hpp"
#include "interface/swarm.hpp"
#include "interface/swarm_container.hpp"
#include "interface/swarm_pack_base.hpp"
#include "interface/variable.hpp"
#include "interface/variable_pack.hpp"
#include "mesh/domain.hpp"
#include "utils/error_checking.hpp"
#include "utils/unique_id.hpp"

namespace parthenon {

/// Interface to underlying infrastructure for data declaration and access.
///
/// The MeshBlockData class is a container for the variables that make up
/// the simulation.  At this point it is expected that this includes
/// both simulation parameters and state variables, but that could
/// change in the future.
///
/// The MeshBlockData class will provide the following methods:
///

class MeshBlock;
template <typename T>
class MeshData;
class StateDescriptor;

template <typename T>
class MeshBlockData {
 public:
  // So that `MeshData` can access private packing functions that have the Cache key
  friend class MeshData<T>;
  // So that `MeshBlock` can call AllocateSparse and DeallocateSparse
  friend class MeshBlock;

  //-----------------
  // Public Methods
  //-----------------
  /// Constructor
  MeshBlockData<T>() = default;
  explicit MeshBlockData<T>(const std::string &name) : stage_name_(name) {}

  // Constructors for getting sub-containers
  // the variables returned are all shallow copies of the src container.
  MeshBlockData<T>(const MeshBlockData<T> &src, const std::vector<std::string> &names,
                   const std::vector<int> &sparse_ids = {});
  MeshBlockData<T>(const MeshBlockData<T> &src, const std::vector<MetadataFlag> &flags,
                   const std::vector<int> &sparse_ids = {});

  std::shared_ptr<MeshBlock> GetBlockSharedPointer() const {
    if (pmy_block.expired()) {
      PARTHENON_THROW("Invalid pointer to MeshBlock!");
    }
    return pmy_block.lock();
  }
  MeshBlock *GetBlockPointer() const { return GetBlockSharedPointer().get(); }
  MeshBlock *GetParentPointer() const { return GetBlockPointer(); }
  void SetAllowedDt(const Real dt) const { GetBlockPointer()->SetAllowedDt(dt); }
  Mesh *GetMeshPointer() const { return GetBlockPointer()->pmy_mesh; }

  template <class... Ts>
  IndexRange GetBoundsI(Ts &&...args) const {
    return GetBlockPointer()->cellbounds.GetBoundsI(std::forward<Ts>(args)...);
  }
  template <class... Ts>
  IndexRange GetBoundsJ(Ts &&...args) const {
    return GetBlockPointer()->cellbounds.GetBoundsJ(std::forward<Ts>(args)...);
  }
  template <class... Ts>
  IndexRange GetBoundsK(Ts &&...args) const {
    return GetBlockPointer()->cellbounds.GetBoundsK(std::forward<Ts>(args)...);
  }

  /// Set the pointer to the mesh block for this container
  void SetBlockPointer(std::weak_ptr<MeshBlock> pmb) { pmy_block = pmb.lock(); }
  void SetBlockPointer(const std::shared_ptr<MeshBlockData<T>> &other) {
    SetBlockPointer(other.get());
  }
  void SetBlockPointer(const MeshBlockData<T> &other) {
    pmy_block = other.GetBlockSharedPointer();
  }
  void SetBlockPointer(const MeshBlockData<T> *other) {
    pmy_block = other->GetBlockSharedPointer();
  }

  void Initialize(const std::shared_ptr<StateDescriptor> resolved_packages,
                  const std::shared_ptr<MeshBlock> pmb);

  /// Create copy of MeshBlockData, possibly with a subset of named fields,
  /// and possibly shallow.  Note when shallow=false, new storage is allocated
  /// for non-OneCopy vars, but the data from src is not actually deep copied
  template <typename ID_t>
  void Initialize(const MeshBlockData<T> *src, const std::vector<ID_t> &vars,
                  const bool shallow_copy) {
    PARTHENON_DEBUG_REQUIRE(src != nullptr, "Source data must be non-null.");
    SetBlockPointer(src);
    resolved_packages_ = src->resolved_packages_;
    is_shallow_ = shallow_copy;

    // clear all variables, maps, and pack caches
    varVector_.clear();
    varMap_.clear();
    varUidMap_.clear();
    flagsToVars_.clear();
    varPackMap_.clear();
    coarseVarPackMap_.clear();
    varFluxPackMap_.clear();

    auto add_var = [=](auto var) {
      if (shallow_copy || var->IsSet(Metadata::OneCopy)) {
        Add(var);
      } else {
        Add(var->AllocateCopy(pmy_block));
      }
    };

    // special case when the list of vars is empty, copy everything
    if (vars.empty()) {
      for (auto v : src->GetVariableVector()) {
        add_var(v);
      }
    } else {
      for (const auto &v : vars) {
        auto var = src->GetVarPtr(v);
        add_var(var);
        // Add the associated flux as well if not explicitly
        // asked for
        if (var->IsSet(Metadata::WithFluxes)) {
          auto flx_name = var->metadata().GetFluxName();
          bool found = false;
          for (const auto &v2 : vars) {
            if (src->GetVarPtr(v2)->label() == flx_name) found = true;
          }
          if (!found) add_var(src->GetVarPtr(flx_name));
        }
      }
    }
  }

  //
  // Queries related to Variable objects
  //
  bool HasVariable(const std::string &label) const noexcept {
    return varMap_.count(label) > 0;
  }

  const VariableVector<T> &GetVariableVector() const noexcept { return varVector_; }

  const MapToVars<T> &GetVariableMap() const noexcept { return varMap_; }

  std::shared_ptr<Variable<T>> GetVarPtr(const std::string &label) const {
    PARTHENON_REQUIRE_THROWS(varMap_.count(label),
                             "Couldn't find variable '" + label + "'");
    return varMap_.at(label);
  }
  std::shared_ptr<Variable<T>> GetVarPtr(const Uid_t &uid) const {
    PARTHENON_REQUIRE_THROWS(varUidMap_.count(uid),
                             "Variable ID " + std::to_string(uid) + " not found!");
    return varUidMap_.at(uid);
  }

  const auto &GetUidMap() const { return varUidMap_; }

  Variable<T> &Get(const std::string &base_name, int sparse_id = InvalidSparseID) const {
    return *GetVarPtr(MakeVarLabel(base_name, sparse_id));
  }
  Variable<T> &Get(const Uid_t &uid) const { return *(varUidMap_.at(uid)); }

  Uid_t UniqueID(const std::string &label) noexcept {
    auto it = varMap_.find(label);
    if (it == varMap_.end()) return INVALID_UID;
    return (it->second)->GetUniqueID();
  }

#ifdef ENABLE_SPARSE
  inline bool IsAllocated(std::string const &label) const noexcept {
    auto it = varMap_.find(label);
    if (it == varMap_.end()) {
      return false;
    }
    return it->second->IsAllocated();
  }

  inline bool IsAllocated(std::string const &base_name, int sparse_id) const noexcept {
    return IsAllocated(MakeVarLabel(base_name, sparse_id));
  }

#else
  constexpr inline bool IsAllocated(std::string const & /*label*/) const noexcept {
    return true;
  }

  constexpr inline bool IsAllocated(std::string const & /*base_name*/,
                                    int /*sparse_id*/) const noexcept {
    return true;
  }
#endif

  std::vector<bool> AllocationStatus(const std::string &label) const noexcept {
    return std::vector<bool>({IsAllocated(label)});
  }

  using VarList = VarListWithKeys<T>;

  /// Get list of variables and labels by names (either a full variable name or sparse
  /// base name), optionally selecting only given sparse ids
  VarList GetVariablesByName(const std::vector<std::string> &names,
                             const std::vector<int> &sparse_ids = {},
                             const FluxRequest flux = FluxRequest::NoFlux);

  /// Get list of variables and UIDs by metadata flags (must match all flags if
  /// match_all is true, otherwise must only match at least one), optionally selecting
  /// only given sparse ids
  VarList GetVariablesByFlag(const Metadata::FlagCollection &flags,
                             const std::vector<int> &sparse_ids = {},
                             const FluxRequest flux = FluxRequest::NoFlux);

  // Get list of variables specified by unique identifiers
  VarList GetVariablesByUid(const std::vector<Uid_t> &uids,
                            const FluxRequest flux = FluxRequest::NoFlux);

  /// Get list of all variables and labels, optionally selecting only given sparse ids
  VarList GetAllVariables(const std::vector<int> &sparse_ids = {},
                          const FluxRequest flux = FluxRequest::NoFlux) {
    return GetVariablesByFlag(Metadata::FlagCollection(), sparse_ids, flux);
  }

  std::vector<Uid_t> GetVariableUIDs(const std::vector<std::string> &names,
                                     const std::vector<int> &sparse_ids = {},
                                     const FluxRequest flux = FluxRequest::NoFlux) {
    return GetVariablesByName(names, sparse_ids, flux).unique_ids();
  }
  std::vector<Uid_t> GetVariableUIDs(const Metadata::FlagCollection &flags,
                                     const std::vector<int> &sparse_ids = {},
                                     const FluxRequest flux = FluxRequest::NoFlux) {
    return GetVariablesByFlag(flags, sparse_ids, flux).unique_ids();
  }
  std::vector<Uid_t> GetVariableUIDs(const std::vector<int> &sparse_ids = {},
                                     const FluxRequest flux = FluxRequest::NoFlux) {
    return GetAllVariables(sparse_ids, flux).unique_ids();
  }

  // Queries related to swarm data
  std::shared_ptr<SwarmContainer> &GetSwarmData(int n = 0) {
    PARTHENON_REQUIRE(stage_name_ == "base",
                      "Swarm data must be accessed through base register!");
    PARTHENON_REQUIRE(n == 0, "MeshBlockData::GetSwarmData requires n==0");
    return swarm_data;
  }
  std::vector<std::shared_ptr<Swarm>> GetAllSwarms() {
    return this->GetSwarmData()->GetSwarmVector();
  }
  std::shared_ptr<Swarm> GetSwarm(const std::string &name) {
    auto swarm_map = this->GetSwarmData()->GetSwarmMap();
    auto it = swarm_map.find(name);
    PARTHENON_REQUIRE(it != swarm_map.end(), "Couldn't find swarm '" + name + "'");
    return it->second;
  }

  /// Queries related to variable packs
  /// For all of these functions, vmap and key are optional output parameters, they will
  /// be set if not null.
  /// sparse_ids is an optional set of sparse ids to be included, all dense variables are
  /// always included (if they match name or flags), but sparse variables are only
  /// included if sparse_ids is not empty and contains the sparse id of the sparse
  /// variable

  SparsePackCache &GetSparsePackCache() { return sparse_pack_cache_; }

  template <typename TYPE>
  SwarmPackCache<TYPE> &GetSwarmPackCache() {
    if constexpr (std::is_same<TYPE, int>::value) {
      return swarm_pack_int_cache_;
    } else if constexpr (std::is_same<TYPE, Real>::value) {
      return swarm_pack_real_cache_;
    }
    PARTHENON_THROW("SwarmPacks only compatible with int and Real types");
  }

  void ClearSwarmCaches() {
    if (swarm_pack_real_cache_.size() > 0) swarm_pack_real_cache_.clear();
    if (swarm_pack_int_cache_.size() > 0) swarm_pack_int_cache_.clear();
  }

  /// Pack variables and fluxes by separate variables and fluxes names
  const VariableFluxPack<T> &
  PackVariablesAndFluxes(const std::vector<std::string> &var_names,
                         const std::vector<std::string> &flx_names,
                         const std::vector<int> &sparse_ids, PackIndexMap &map) {
    return PackVariablesAndFluxesImpl(var_names, flx_names, sparse_ids, &map, nullptr);
  }
  const VariableFluxPack<T> &
  PackVariablesAndFluxes(const std::vector<std::string> &var_names,
                         const std::vector<std::string> &flx_names,
                         const std::vector<int> &sparse_ids) {
    return PackVariablesAndFluxesImpl(var_names, flx_names, sparse_ids, nullptr, nullptr);
  }
  const VariableFluxPack<T> &
  PackVariablesAndFluxes(const std::vector<std::string> &var_names,
                         const std::vector<std::string> &flx_names, PackIndexMap &map) {
    return PackVariablesAndFluxesImpl(var_names, flx_names, {}, &map, nullptr);
  }
  const VariableFluxPack<T> &
  PackVariablesAndFluxes(const std::vector<std::string> &var_names,
                         const std::vector<std::string> &flx_names) {
    return PackVariablesAndFluxesImpl(var_names, flx_names, {}, nullptr, nullptr);
  }

  /// Pack variables and fluxes by same variables and fluxes names
  const VariableFluxPack<T> &PackVariablesAndFluxes(const std::vector<std::string> &names,
                                                    const std::vector<int> &sparse_ids,
                                                    PackIndexMap &map) {
    return PackVariablesAndFluxesImpl(names, names, sparse_ids, &map, nullptr);
  }
  const VariableFluxPack<T> &PackVariablesAndFluxes(const std::vector<std::string> &names,
                                                    const std::vector<int> &sparse_ids) {
    return PackVariablesAndFluxesImpl(names, names, sparse_ids, nullptr, nullptr);
  }
  const VariableFluxPack<T> &PackVariablesAndFluxes(const std::vector<std::string> &names,
                                                    PackIndexMap &map) {
    return PackVariablesAndFluxesImpl(names, names, {}, &map, nullptr);
  }
  const VariableFluxPack<T> &
  PackVariablesAndFluxes(const std::vector<std::string> &names) {
    return PackVariablesAndFluxesImpl(names, names, {}, nullptr, nullptr);
  }

  /// Pack variables and fluxes by Metadata flags
  template <class... ARGS>
  const VariableFluxPack<T> &PackVariablesAndFluxes(std::vector<MetadataFlag> flags,
                                                    ARGS &&...args) {
    return PackVariablesAndFluxes(Metadata::FlagCollection(flags),
                                  std::forward<ARGS>(args)...);
  }
  const VariableFluxPack<T> &PackVariablesAndFluxes(const Metadata::FlagCollection &flags,
                                                    const std::vector<int> &sparse_ids,
                                                    PackIndexMap &map) {
    return PackVariablesAndFluxesImpl(flags, sparse_ids, &map, nullptr);
  }
  const VariableFluxPack<T> &PackVariablesAndFluxes(const Metadata::FlagCollection &flags,
                                                    const std::vector<int> &sparse_ids) {
    return PackVariablesAndFluxesImpl(flags, sparse_ids, nullptr, nullptr);
  }
  const VariableFluxPack<T> &PackVariablesAndFluxes(const Metadata::FlagCollection &flags,
                                                    PackIndexMap &map) {
    return PackVariablesAndFluxesImpl(flags, {}, &map, nullptr);
  }
  const VariableFluxPack<T> &
  PackVariablesAndFluxes(const Metadata::FlagCollection &flags) {
    return PackVariablesAndFluxesImpl(flags, {}, nullptr, nullptr);
  }

  /// Pack all variables and fluxes
  const VariableFluxPack<T> &PackVariablesAndFluxes(const std::vector<int> &sparse_ids,
                                                    PackIndexMap &map) {
    return PackVariablesAndFluxesImpl(sparse_ids, &map, nullptr);
  }
  const VariableFluxPack<T> &PackVariablesAndFluxes(const std::vector<int> &sparse_ids) {
    return PackVariablesAndFluxesImpl(sparse_ids, nullptr, nullptr);
  }
  const VariableFluxPack<T> &PackVariablesAndFluxes(PackIndexMap &map) {
    return PackVariablesAndFluxesImpl({}, &map, nullptr);
  }
  const VariableFluxPack<T> &PackVariablesAndFluxes() {
    return PackVariablesAndFluxesImpl({}, nullptr, nullptr);
  }

  /// Pack variables by name
  const VariablePack<T> &PackVariables(const std::vector<std::string> &names,
                                       const std::vector<int> &sparse_ids,
                                       PackIndexMap &map, bool coarse = false) {
    return PackVariablesImpl(names, sparse_ids, coarse, &map, nullptr);
  }
  const VariablePack<T> &PackVariables(const std::vector<std::string> &names,
                                       const std::vector<int> &sparse_ids,
                                       bool coarse = false) {
    return PackVariablesImpl(names, sparse_ids, coarse, nullptr, nullptr);
  }
  const VariablePack<T> &PackVariables(const std::vector<std::string> &names,
                                       PackIndexMap &map, bool coarse = false) {
    return PackVariablesImpl(names, {}, coarse, &map, nullptr);
  }
  const VariablePack<T> &PackVariables(const std::vector<std::string> &names,
                                       bool coarse = false) {
    return PackVariablesImpl(names, {}, coarse, nullptr, nullptr);
  }

  /// Pack variables by Metadata flags
  template <class... ARGS>
  const VariablePack<T> &PackVariables(const std::vector<MetadataFlag> &flags,
                                       ARGS &&...args) {
    return PackVariables(Metadata::FlagCollection(flags), std::forward<ARGS>(args)...);
  }
  const VariablePack<T> &PackVariables(const Metadata::FlagCollection &flags,
                                       const std::vector<int> &sparse_ids,
                                       PackIndexMap &map, bool coarse = false) {
    return PackVariablesImpl(flags, sparse_ids, coarse, &map, nullptr);
  }
  const VariablePack<T> &PackVariables(const Metadata::FlagCollection &flags,
                                       const std::vector<int> &sparse_ids,
                                       bool coarse = false) {
    return PackVariablesImpl(flags, sparse_ids, coarse, nullptr, nullptr);
  }
  const VariablePack<T> &PackVariables(const Metadata::FlagCollection &flags,
                                       PackIndexMap &map, bool coarse = false) {
    return PackVariablesImpl(flags, {}, coarse, &map, nullptr);
  }
  const VariablePack<T> &PackVariables(const Metadata::FlagCollection &flags,
                                       bool coarse = false) {
    return PackVariablesImpl(flags, {}, coarse, nullptr, nullptr);
  }

  /// Pack all variables
  const VariablePack<T> &PackVariables(const std::vector<int> &sparse_ids,
                                       PackIndexMap &map, bool coarse = false) {
    return PackVariablesImpl(sparse_ids, coarse, &map, nullptr);
  }
  const VariablePack<T> &PackVariables(const std::vector<int> &sparse_ids,
                                       bool coarse = false) {
    return PackVariablesImpl(sparse_ids, coarse, nullptr, nullptr);
  }
  const VariablePack<T> &PackVariables(PackIndexMap &map, bool coarse = false) {
    return PackVariablesImpl({}, coarse, &map, nullptr);
  }
  const VariablePack<T> &PackVariables(bool coarse = false) {
    return PackVariablesImpl({}, coarse, nullptr, nullptr);
  }

  /// Remove a variable from the container or throw exception if not
  /// found.
  /// @param label the name of the variable to be deleted
  void Remove(const std::string &label);

  /// Print list of labels in container
  void Print();

  // return number of stored arrays
  int Size() noexcept { return varVector_.size(); }

  bool operator==(const MeshBlockData<T> &cmp) {
    // do some kind of check of equality
    // do the two containers contain the same named fields?
    std::vector<std::string> my_keys;
    std::vector<std::string> cmp_keys;
    for (auto &v : varMap_) {
      my_keys.push_back(v.first);
    }
    for (auto &v : cmp.GetVariableMap()) {
      cmp_keys.push_back(v.first);
    }
    return (my_keys == cmp_keys);
  }

  bool Contains(const std::string &name) const noexcept { return varMap_.count(name); }
  bool Contains(const Uid_t &uid) const noexcept { return varUidMap_.count(uid); }
  template <typename ID_t>
  bool Contains(const std::vector<ID_t> &vars) const noexcept {
    return std::all_of(vars.begin(), vars.end(),
                       [this](const auto &v) { return this->Contains(v); });
  }
  template <typename ID_t>
  bool ContainsExactly(const std::vector<ID_t> &vars) const noexcept {
    // JMM: Assumes vars contains no duplicates. But that would have
    // been caught elsewhere because `MeshBlockData::Add` would have failed.
    return Contains(vars) && (vars.size() == varVector_.size());
  }

  void SetAllVariablesToInitialized() {
    std::for_each(varVector_.begin(), varVector_.end(),
                  [](auto &sp_var) { sp_var->data.initialized = true; });
  }

  bool AllVariablesInitialized() {
    bool all_initialized = true;
    std::for_each(varVector_.begin(), varVector_.end(), [&](auto &sp_var) {
      all_initialized = all_initialized && sp_var->data.initialized;
    });
    return all_initialized;
  }

  bool IsShallow() const { return is_shallow_; }

 private:
  void AddField(const std::string &base_name, const Metadata &metadata,
                int sparse_id = InvalidSparseID);

  void Add(std::shared_ptr<Variable<T>> var) noexcept {
    if (varUidMap_.count(var->GetUniqueID())) {
      PARTHENON_THROW("Tried to add variable " + var->label() + " twice!");
    }
    varVector_.push_back(var);
    varMap_[var->label()] = var;
    varUidMap_[var->GetUniqueID()] = var;
    for (const auto &flag : var->metadata().Flags()) {
      flagsToVars_[flag].insert(var);
    }
  }

  std::shared_ptr<Variable<T>> AllocateSparse(std::string const &label,
                                              bool flag_uninitialized = false) {
    if (!HasVariable(label)) {
      PARTHENON_THROW("Tried to allocate sparse variable '" + label +
                      "', but no such sparse variable exists");
    }

    auto var = GetVarPtr(label);
    PARTHENON_REQUIRE_THROWS(var->IsSparse(),
                             "Tried to allocate non-sparse variable " + label);

    var->Allocate(pmy_block, flag_uninitialized);

    return var;
  }

  std::shared_ptr<Variable<T>> AllocSparseID(std::string const &base_name,
                                             const int sparse_id) {
    return AllocateSparse(MakeVarLabel(base_name, sparse_id));
  }

  void DeallocateSparse(std::string const &label) {
    PARTHENON_REQUIRE_THROWS(HasVariable(label),
                             "Tried to deallocate sparse variable '" + label +
                                 "', but no such sparse variable exists");

    auto var = GetVarPtr(label);
    // PARTHENON_REQUIRE_THROWS(var->IsSparse(),
    //                         "Tried to deallocate non-sparse variable " + label);

    if (var->IsAllocated()) {
      std::int64_t bytes = var->Deallocate();
      auto pmb = GetBlockPointer();
      pmb->LogMemUsage(-bytes);
    }
  }

  std::weak_ptr<MeshBlock> pmy_block;
  std::shared_ptr<StateDescriptor> resolved_packages_;
  bool is_shallow_ = false;
  const std::string stage_name_;

  VariableVector<T> varVector_; ///< the saved variable array
  std::map<Uid_t, std::shared_ptr<Variable<T>>> varUidMap_;

  MapToVars<T> varMap_;
  MetadataFlagToVariableMap<T> flagsToVars_;

  // variable packing
  MapToVariablePack<T> varPackMap_;
  MapToVariablePack<T> coarseVarPackMap_; // cache for varpacks over coarse arrays
  MapToVariableFluxPack<T> varFluxPackMap_;
  SparsePackCache sparse_pack_cache_;
  SwarmPackCache<int> swarm_pack_int_cache_;
  SwarmPackCache<Real> swarm_pack_real_cache_;

  // swarm data
  std::shared_ptr<SwarmContainer> swarm_data = std::make_shared<SwarmContainer>();

  // These functions have private scope and are visible only to MeshData
  const VariableFluxPack<T> &
  PackVariablesAndFluxes(const std::vector<std::string> &var_names,
                         const std::vector<std::string> &flx_names,
                         const std::vector<int> &sparse_ids, PackIndexMap &map,
                         vpack_types::UidVecPair &key) {
    return PackVariablesAndFluxesImpl(var_names, flx_names, sparse_ids, &map, &key);
  }
  const VariableFluxPack<T> &
  PackVariablesAndFluxes(const std::vector<std::string> &var_names,
                         const std::vector<std::string> &flx_names,
                         const std::vector<int> &sparse_ids,
                         vpack_types::UidVecPair &key) {
    return PackVariablesAndFluxesImpl(var_names, flx_names, sparse_ids, nullptr, &key);
  }
  const VariableFluxPack<T> &
  PackVariablesAndFluxes(const std::vector<std::string> &var_names,
                         const std::vector<std::string> &flx_names, PackIndexMap &map,
                         vpack_types::UidVecPair &key) {
    return PackVariablesAndFluxesImpl(var_names, flx_names, {}, &map, &key);
  }
  const VariableFluxPack<T> &
  PackVariablesAndFluxes(const std::vector<std::string> &var_names,
                         const std::vector<std::string> &flx_names,
                         vpack_types::UidVecPair &key) {
    return PackVariablesAndFluxesImpl(var_names, flx_names, {}, nullptr, &key);
  }
  const VariableFluxPack<T> &PackVariablesAndFluxes(const std::vector<std::string> &names,
                                                    const std::vector<int> &sparse_ids,
                                                    PackIndexMap &map,
                                                    vpack_types::UidVecPair &key) {
    return PackVariablesAndFluxesImpl(names, names, sparse_ids, &map, &key);
  }
  const VariableFluxPack<T> &PackVariablesAndFluxes(const std::vector<std::string> &names,
                                                    const std::vector<int> &sparse_ids,
                                                    vpack_types::UidVecPair &key) {
    return PackVariablesAndFluxesImpl(names, names, sparse_ids, nullptr, &key);
  }
  const VariableFluxPack<T> &PackVariablesAndFluxes(const std::vector<std::string> &names,
                                                    PackIndexMap &map,
                                                    vpack_types::UidVecPair &key) {
    return PackVariablesAndFluxesImpl(names, names, {}, &map, &key);
  }
  const VariableFluxPack<T> &PackVariablesAndFluxes(const std::vector<std::string> &names,
                                                    vpack_types::UidVecPair &key) {
    return PackVariablesAndFluxesImpl(names, names, {}, nullptr, &key);
  }
  const VariableFluxPack<T> &PackVariablesAndFluxes(const Metadata::FlagCollection &flags,
                                                    const std::vector<int> &sparse_ids,
                                                    PackIndexMap &map,
                                                    vpack_types::UidVecPair &key) {
    return PackVariablesAndFluxesImpl(flags, sparse_ids, &map, &key);
  }
  const VariableFluxPack<T> &PackVariablesAndFluxes(const Metadata::FlagCollection &flags,
                                                    const std::vector<int> &sparse_ids,
                                                    vpack_types::UidVecPair &key) {
    return PackVariablesAndFluxesImpl(flags, sparse_ids, nullptr, &key);
  }
  const VariableFluxPack<T> &PackVariablesAndFluxes(const Metadata::FlagCollection &flags,
                                                    PackIndexMap &map,
                                                    vpack_types::UidVecPair &key) {
    return PackVariablesAndFluxesImpl(flags, {}, &map, &key);
  }
  const VariableFluxPack<T> &PackVariablesAndFluxes(const Metadata::FlagCollection &flags,
                                                    vpack_types::UidVecPair &key) {
    return PackVariablesAndFluxesImpl(flags, {}, nullptr, &key);
  }
  const VariableFluxPack<T> &PackVariablesAndFluxes(const std::vector<int> &sparse_ids,
                                                    PackIndexMap &map,
                                                    vpack_types::UidVecPair &key) {
    return PackVariablesAndFluxesImpl(sparse_ids, &map, &key);
  }
  const VariableFluxPack<T> &PackVariablesAndFluxes(const std::vector<int> &sparse_ids,
                                                    vpack_types::UidVecPair &key) {
    return PackVariablesAndFluxesImpl(sparse_ids, nullptr, &key);
  }
  const VariableFluxPack<T> &PackVariablesAndFluxes(PackIndexMap &map,
                                                    vpack_types::UidVecPair &key) {
    return PackVariablesAndFluxesImpl({}, &map, &key);
  }
  const VariableFluxPack<T> &PackVariablesAndFluxes(vpack_types::UidVecPair &key) {
    return PackVariablesAndFluxesImpl({}, nullptr, &key);
  }
  const VariablePack<T> &PackVariables(const std::vector<std::string> &names,
                                       const std::vector<int> &sparse_ids,
                                       PackIndexMap &map, vpack_types::VPackKey_t &key,
                                       bool coarse = false) {
    return PackVariablesImpl(names, sparse_ids, coarse, &map, &key);
  }
  const VariablePack<T> &PackVariables(const std::vector<std::string> &names,
                                       const std::vector<int> &sparse_ids,
                                       vpack_types::VPackKey_t &key,
                                       bool coarse = false) {
    return PackVariablesImpl(names, sparse_ids, coarse, nullptr, &key);
  }
  const VariablePack<T> &PackVariables(const std::vector<std::string> &names,
                                       PackIndexMap &map, vpack_types::VPackKey_t &key,
                                       bool coarse = false) {
    return PackVariablesImpl(names, {}, coarse, &map, &key);
  }
  const VariablePack<T> &PackVariables(const std::vector<std::string> &names,
                                       vpack_types::VPackKey_t &key,
                                       bool coarse = false) {
    return PackVariablesImpl(names, {}, coarse, nullptr, &key);
  }
  const VariablePack<T> &PackVariables(const Metadata::FlagCollection &flags,
                                       const std::vector<int> &sparse_ids,
                                       PackIndexMap &map, vpack_types::VPackKey_t &key,
                                       bool coarse = false) {
    return PackVariablesImpl(flags, sparse_ids, coarse, &map, &key);
  }
  const VariablePack<T> &PackVariables(const Metadata::FlagCollection &flags,
                                       const std::vector<int> &sparse_ids,
                                       vpack_types::VPackKey_t &key,
                                       bool coarse = false) {
    return PackVariablesImpl(flags, sparse_ids, coarse, nullptr, &key);
  }
  const VariablePack<T> &PackVariables(const Metadata::FlagCollection &flags,
                                       PackIndexMap &map, vpack_types::VPackKey_t &key,
                                       bool coarse = false) {
    return PackVariablesImpl(flags, {}, coarse, &map, &key);
  }
  const VariablePack<T> &PackVariables(const Metadata::FlagCollection &flags,
                                       vpack_types::VPackKey_t &key,
                                       bool coarse = false) {
    return PackVariablesImpl(flags, {}, coarse, nullptr, &key);
  }
  const VariablePack<T> &PackVariables(const std::vector<int> &sparse_ids,
                                       PackIndexMap &map, vpack_types::VPackKey_t &key,
                                       bool coarse = false) {
    return PackVariablesImpl(sparse_ids, coarse, &map, &key);
  }
  const VariablePack<T> &PackVariables(const std::vector<int> &sparse_ids,
                                       vpack_types::VPackKey_t &key,
                                       bool coarse = false) {
    return PackVariablesImpl(sparse_ids, coarse, nullptr, &key);
  }
  const VariablePack<T> &PackVariables(PackIndexMap &map, vpack_types::VPackKey_t &key,
                                       bool coarse = false) {
    return PackVariablesImpl({}, coarse, &map, &key);
  }
  // we have to disable this overload because it overshadows packing by name without any
  // output parameters
  // const VariablePack<T> &PackVariables(vpack_types::VPackKey_t &key,
  //                                      bool coarse = false) {
  //   return PackVariablesImpl({}, coarse, nullptr, &key);
  // }

  // These helper functions are private scope because they assume that
  // the names include the components of sparse variables.
  const VariableFluxPack<T> &PackListedVariablesAndFluxes(const VarList &var_list,
                                                          const VarList &flux_list,
                                                          PackIndexMap *map,
                                                          vpack_types::UidVecPair *key);
  const VariableFluxPack<T> &
  PackVariablesAndFluxesImpl(const std::vector<std::string> &var_names,
                             const std::vector<std::string> &flx_names,
                             const std::vector<int> &sparse_ids, PackIndexMap *map,
                             vpack_types::UidVecPair *key);

  const VariableFluxPack<T> &
  PackVariablesAndFluxesImpl(const Metadata::FlagCollection &flags,
                             const std::vector<int> &sparse_ids, PackIndexMap *map,
                             vpack_types::UidVecPair *key);

  const VariableFluxPack<T> &
  PackVariablesAndFluxesImpl(const std::vector<int> &sparse_ids, PackIndexMap *map,
                             vpack_types::UidVecPair *key);

  const VariablePack<T> &PackListedVariables(const VarList &var_list, bool coarse,
                                             PackIndexMap *map,
                                             vpack_types::VPackKey_t *key);

  const VariablePack<T> &PackVariablesImpl(const std::vector<std::string> &names,
                                           const std::vector<int> &sparse_ids,
                                           bool coarse, PackIndexMap *map,
                                           vpack_types::VPackKey_t *key);

  const VariablePack<T> &PackVariablesImpl(const Metadata::FlagCollection &flags,
                                           const std::vector<int> &sparse_ids,
                                           bool coarse, PackIndexMap *map,
                                           vpack_types::VPackKey_t *key);

  const VariablePack<T> &PackVariablesImpl(const std::vector<int> &sparse_ids,
                                           bool coarse, PackIndexMap *map,
                                           vpack_types::VPackKey_t *key);
};

template <typename T, typename... Args>
std::vector<Uid_t> UidIntersection(MeshBlockData<T> *mbd1, MeshBlockData<T> *mbd2,
                                   Args &&...args) {
  auto u1 = mbd1->GetVariableUIDs(std::forward<Args>(args)...);
  auto u2 = mbd2->GetVariableUIDs(std::forward<Args>(args)...);
  return UidIntersection(u1, u2);
}

} // namespace parthenon

#endif // INTERFACE_MESHBLOCK_DATA_HPP_
