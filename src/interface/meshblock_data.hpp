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
#ifndef INTERFACE_MESHBLOCK_DATA_HPP_
#define INTERFACE_MESHBLOCK_DATA_HPP_

#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "interface/data_collection.hpp"
#include "interface/variable.hpp"
#include "interface/variable_pack.hpp"
#include "mesh/domain.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

/// Interface to underlying infrastructure for data declaration and access.
/// Date: August 22, 2019
///
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
  // So that `MeshData` can access private packing functions
  // that have the Cache key
  friend class MeshData<T>;

  //-----------------
  // Public Methods
  //-----------------
  /// Constructor
  MeshBlockData<T>() = default;

  /// Copies variables from src, optionally only copying names given and/or variables
  /// matching any of the flags given. If sparse_ids is not empty, only sparse fields with
  /// listed sparse ids will be copied, dense fields will always be copied. If both names
  /// and flags are provided, only variables that show up in names AND have metadata in
  /// FLAGS are copied. If shallow_copy is true, no copies of variables will be allocated
  /// regardless whether they are flagged as OneCopy or not
  void CopyFrom(const MeshBlockData<T> &src, bool shallow_copy,
                const std::vector<std::string> &names = {},
                const std::vector<MetadataFlag> &flags = {},
                const std::vector<int> &sparse_ids = {});

  // Constructors for getting sub-containers
  // the variables returned are all shallow copies of the src container.
  MeshBlockData<T>(const MeshBlockData<T> &src, const std::vector<std::string> &names,
                   const std::vector<int> &sparse_ids = {});
  MeshBlockData<T>(const MeshBlockData<T> &src, const std::vector<MetadataFlag> &flags,
                   const std::vector<int> &sparse_ids = {});

  /// Returns shared pointer to a block
  std::shared_ptr<MeshBlock> GetBlockPointer() const {
    if (pmy_block.expired()) {
      PARTHENON_THROW("Invalid pointer to MeshBlock!");
    }
    return pmy_block.lock();
  }
  auto GetParentPointer() const { return GetBlockPointer(); }
  void SetAllowedDt(const Real dt) const { GetBlockPointer()->SetAllowedDt(dt); }

  IndexRange GetBoundsI(const IndexDomain &domain) {
    return GetBlockPointer()->cellbounds.GetBoundsI(domain);
  }
  IndexRange GetBoundsJ(const IndexDomain &domain) {
    return GetBlockPointer()->cellbounds.GetBoundsJ(domain);
  }
  IndexRange GetBoundsK(const IndexDomain &domain) {
    return GetBlockPointer()->cellbounds.GetBoundsK(domain);
  }

  /// Create non-shallow copy of MeshBlockData, but only include named variables
  void Copy(const std::shared_ptr<MeshBlockData<T>> &src,
            const std::vector<std::string> &names) {
    CopyFrom(*src, false, names);
  }
  /// Create non-shallow copy of MeshBlockData
  void Copy(const std::shared_ptr<MeshBlockData<T>> &src) { CopyFrom(*src, false); }

  /// Get a container containing only dense fields and the sparse fields with a sparse id
  /// from the given list of sparse ids.
  ///
  /// @param sparse_ids The list of sparse ids to include
  /// @return New container with slices from all variables
  std::shared_ptr<MeshBlockData<T>> SparseSlice(const std::vector<int> &sparse_ids) const;

  /// As above but for just one sparse id
  std::shared_ptr<MeshBlockData<T>> SparseSlice(int sparse_id) const {
    return SparseSlice({sparse_id});
  }

  ///
  /// Set the pointer to the mesh block for this container
  void SetBlockPointer(std::weak_ptr<MeshBlock> pmb) { pmy_block = pmb; }
  void SetBlockPointer(const std::shared_ptr<MeshBlockData<T>> &other) {
    SetBlockPointer(*other);
  }
  void SetBlockPointer(const MeshBlockData<T> &other) { pmy_block = other.pmy_block; }

  void Initialize(const std::shared_ptr<StateDescriptor> resolved_packages,
                  const std::shared_ptr<MeshBlock> pmb);

  //
  // Queries related to CellVariable objects
  //
  bool HasCellVariable(const std::string &label) const noexcept {
    return varMap_.count(label) > 0;
  }

  const CellVariableVector<T> &GetCellVariableVector() const noexcept {
    return varVector_;
  }

  const MapToCellVars<T> &GetCellVariableMap() const noexcept { return varMap_; }

  std::shared_ptr<CellVariable<T>> GetCellVarPtr(const std::string &label) const {
    auto it = varMap_.find(label);
    PARTHENON_REQUIRE_THROWS(it != varMap_.end(),
                             "Couldn't find variable '" + label + "'");
    return it->second;
  }

  CellVariable<T> &Get(const std::string &base_name,
                       int sparse_id = InvalidSparseID) const {
    return *GetCellVarPtr(MakeVarLabel(base_name, sparse_id));
  }
  CellVariable<T> &Get(const int index) const { return *(varVector_[index]); }

  int Index(const std::string &label) noexcept {
    for (int i = 0; i < (varVector_).size(); i++) {
      if (!varVector_[i]->label().compare(label)) return i;
    }
    return -1;
  }

  std::shared_ptr<CellVariable<T>> AllocateSparse(std::string const &label) {
    if (!HasCellVariable(label)) {
      PARTHENON_THROW("Tried to allocate sparse variable '" + label +
                      "', but no such sparse variable exists");
    }

    auto var = GetCellVarPtr(label);
    PARTHENON_REQUIRE_THROWS(var->IsSparse(),
                             "Tried to allocate non-sparse variable " + label);

    var->Allocate(pmy_block);

    return var;
  }

  std::shared_ptr<CellVariable<T>> AllocSparseID(std::string const &base_name,
                                                 const int sparse_id) {
    return AllocateSparse(MakeVarLabel(base_name, sparse_id));
  }

  void DeallocateSparse(std::string const &label) {
    PARTHENON_REQUIRE_THROWS(HasCellVariable(label),
                             "Tried to deallocate sparse variable '" + label +
                                 "', but no such sparse variable exists");

    auto var = GetCellVarPtr(label);
    PARTHENON_REQUIRE_THROWS(var->IsSparse(),
                             "Tried to deallocate non-sparse variable " + label);

    if (var->IsAllocated()) {
      var->Deallocate();
    }
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

  //
  // Queries related to FaceVariable objects
  //
  const FaceVector<T> &GetFaceVector() const noexcept { return faceVector_; }
  const MapToFace<T> &GetFaceMap() const noexcept { return faceMap_; }
  // DO NOT make this a const reference. Passing in C-style string literals
  // cuases it to misbehave.
  FaceVariable<T> &GetFace(std::string label) {
    auto it = faceMap_.find(label);
    if (it == faceMap_.end()) {
      PARTHENON_THROW(std::string("\n") + std::string(label) +
                      std::string(" array not found in Get() Face\n"));
    }
    return *(it->second);
  }

  ParArrayND<Real> &GetFace(std::string &label, int dir) {
    return GetFace(label).Get(dir);
  }

  ///
  /// Get an edge variable from the container
  /// @param label the name of the variable
  /// @return the CellVariable<T> if found or throw exception
  ///
  EdgeVariable<T> *GetEdge(std::string label) {
    // for (auto v : edgeVector_) {
    //   if (! v->label().compare(label)) return v;
    // }
    PARTHENON_THROW(std::string("\n") + std::string(label) +
                    std::string(" array not found in Get() Edge\n"));
  }

  using VarLabelList = VarListWithLabels<T>;

  /// Get list of variables and labels by names (either a full variable name or sparse
  /// base name), optionally selecting only given sparse ids
  VarLabelList GetVariablesByName(const std::vector<std::string> &names,
                                  const std::vector<int> &sparse_ids = {});

  /// Get list of variables and labels by metadata flags (must match all flags if
  /// match_all is true, otherwise must only match at least one), optionally selecting
  /// only given sparse ids
  VarLabelList GetVariablesByFlag(const std::vector<MetadataFlag> &flags, bool match_all,
                                  const std::vector<int> &sparse_ids = {});

  /// Get list of all variables and labels, optionally selecting only given sparse ids
  VarLabelList GetAllVariables(const std::vector<int> &sparse_ids = {}) {
    return GetVariablesByFlag({}, false, sparse_ids);
  }

  /// Queries related to variable packs
  /// For all of these functions, vmap and key are optional output parameters, they will
  /// be set if not null.
  /// sparse_ids is an optional set of sparse ids to be included, all dense variables are
  /// always included (if they match name or flags), but sparse variables are only
  /// included if sparse_ids is not empty and contains the sparse id of the sparse
  /// variable

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
  const VariableFluxPack<T> &
  PackVariablesAndFluxes(const std::vector<MetadataFlag> &flags,
                         const std::vector<int> &sparse_ids, PackIndexMap &map) {
    return PackVariablesAndFluxesImpl(flags, sparse_ids, &map, nullptr);
  }
  const VariableFluxPack<T> &
  PackVariablesAndFluxes(const std::vector<MetadataFlag> &flags,
                         const std::vector<int> &sparse_ids) {
    return PackVariablesAndFluxesImpl(flags, sparse_ids, nullptr, nullptr);
  }
  const VariableFluxPack<T> &
  PackVariablesAndFluxes(const std::vector<MetadataFlag> &flags, PackIndexMap &map) {
    return PackVariablesAndFluxesImpl(flags, {}, &map, nullptr);
  }
  const VariableFluxPack<T> &
  PackVariablesAndFluxes(const std::vector<MetadataFlag> &flags) {
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
  const VariablePack<T> &PackVariables(const std::vector<MetadataFlag> &flags,
                                       const std::vector<int> &sparse_ids,
                                       PackIndexMap &map, bool coarse = false) {
    return PackVariablesImpl(flags, sparse_ids, coarse, &map, nullptr);
  }
  const VariablePack<T> &PackVariables(const std::vector<MetadataFlag> &flags,
                                       const std::vector<int> &sparse_ids,
                                       bool coarse = false) {
    return PackVariablesImpl(flags, sparse_ids, coarse, nullptr, nullptr);
  }
  const VariablePack<T> &PackVariables(const std::vector<MetadataFlag> &flags,
                                       PackIndexMap &map, bool coarse = false) {
    return PackVariablesImpl(flags, {}, coarse, &map, nullptr);
  }
  const VariablePack<T> &PackVariables(const std::vector<MetadataFlag> &flags,
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

  // Communication routines
  void SetLocalNeighborAllocated();
  void ResetBoundaryCellVariables();
  void SetupPersistentMPI();
  TaskStatus ReceiveBoundaryBuffers();
  TaskStatus StartReceiving(BoundaryCommSubset phase);
  TaskStatus ClearBoundary(BoundaryCommSubset phase);
  TaskStatus SendFluxCorrection();
  TaskStatus ReceiveFluxCorrection();

  // physical boundary routines
  void ProlongateBoundaries();

  bool operator==(const MeshBlockData<T> &cmp) {
    // do some kind of check of equality
    // do the two containers contain the same named fields?
    std::vector<std::string> my_keys;
    std::vector<std::string> cmp_keys;
    for (auto &v : varMap_) {
      my_keys.push_back(v.first);
    }
    for (auto &v : faceMap_) {
      my_keys.push_back(v.first);
    }
    for (auto &v : cmp.GetCellVariableMap()) {
      cmp_keys.push_back(v.first);
    }
    for (auto &v : cmp.GetFaceMap()) {
      cmp_keys.push_back(v.first);
    }
    return (my_keys == cmp_keys);
  }

  bool Contains(const std::string &name) const noexcept {
    if (varMap_.find(name) != varMap_.end()) return true;
    if (faceMap_.find(name) != faceMap_.end()) return true;
    return false;
  }
  bool Contains(const std::vector<std::string> &names) const noexcept {
    for (const auto &name : names) {
      if (!Contains(name)) return false;
    }
    return true;
  }

 private:
  void AddField(const std::string &base_name, const Metadata &metadata,
                int sparse_id = InvalidSparseID);

  void Add(std::shared_ptr<CellVariable<T>> var) noexcept {
    varVector_.push_back(var);
    varMap_[var->label()] = var;
  }

  void Add(std::shared_ptr<FaceVariable<T>> var) noexcept {
    faceVector_.push_back(var);
    faceMap_[var->label()] = var;
  }

  std::weak_ptr<MeshBlock> pmy_block;
  std::shared_ptr<StateDescriptor> resolved_packages_;

  CellVariableVector<T> varVector_; ///< the saved variable array
  FaceVector<T> faceVector_;        ///< the saved face arrays

  MapToCellVars<T> varMap_;
  MapToFace<T> faceMap_;

  // variable packing
  MapToVariablePack<T> varPackMap_;
  MapToVariablePack<T> coarseVarPackMap_; // cache for varpacks over coarse arrays
  MapToVariableFluxPack<T> varFluxPackMap_;

  // These functions have private scope and are visible only to MeshData
  const VariableFluxPack<T> &
  PackVariablesAndFluxes(const std::vector<std::string> &var_names,
                         const std::vector<std::string> &flx_names,
                         const std::vector<int> &sparse_ids, PackIndexMap &map,
                         vpack_types::StringPair &key) {
    return PackVariablesAndFluxesImpl(var_names, flx_names, sparse_ids, &map, &key);
  }
  const VariableFluxPack<T> &
  PackVariablesAndFluxes(const std::vector<std::string> &var_names,
                         const std::vector<std::string> &flx_names,
                         const std::vector<int> &sparse_ids,
                         vpack_types::StringPair &key) {
    return PackVariablesAndFluxesImpl(var_names, flx_names, sparse_ids, nullptr, &key);
  }
  const VariableFluxPack<T> &
  PackVariablesAndFluxes(const std::vector<std::string> &var_names,
                         const std::vector<std::string> &flx_names, PackIndexMap &map,
                         vpack_types::StringPair &key) {
    return PackVariablesAndFluxesImpl(var_names, flx_names, {}, &map, &key);
  }
  const VariableFluxPack<T> &
  PackVariablesAndFluxes(const std::vector<std::string> &var_names,
                         const std::vector<std::string> &flx_names,
                         vpack_types::StringPair &key) {
    return PackVariablesAndFluxesImpl(var_names, flx_names, {}, nullptr, &key);
  }
  const VariableFluxPack<T> &PackVariablesAndFluxes(const std::vector<std::string> &names,
                                                    const std::vector<int> &sparse_ids,
                                                    PackIndexMap &map,
                                                    vpack_types::StringPair &key) {
    return PackVariablesAndFluxesImpl(names, names, sparse_ids, &map, &key);
  }
  const VariableFluxPack<T> &PackVariablesAndFluxes(const std::vector<std::string> &names,
                                                    const std::vector<int> &sparse_ids,
                                                    vpack_types::StringPair &key) {
    return PackVariablesAndFluxesImpl(names, names, sparse_ids, nullptr, &key);
  }
  const VariableFluxPack<T> &PackVariablesAndFluxes(const std::vector<std::string> &names,
                                                    PackIndexMap &map,
                                                    vpack_types::StringPair &key) {
    return PackVariablesAndFluxesImpl(names, names, {}, &map, &key);
  }
  const VariableFluxPack<T> &PackVariablesAndFluxes(const std::vector<std::string> &names,
                                                    vpack_types::StringPair &key) {
    return PackVariablesAndFluxesImpl(names, names, {}, nullptr, &key);
  }
  const VariableFluxPack<T> &
  PackVariablesAndFluxes(const std::vector<MetadataFlag> &flags,
                         const std::vector<int> &sparse_ids, PackIndexMap &map,
                         vpack_types::StringPair &key) {
    return PackVariablesAndFluxesImpl(flags, sparse_ids, &map, &key);
  }
  const VariableFluxPack<T> &
  PackVariablesAndFluxes(const std::vector<MetadataFlag> &flags,
                         const std::vector<int> &sparse_ids,
                         vpack_types::StringPair &key) {
    return PackVariablesAndFluxesImpl(flags, sparse_ids, nullptr, &key);
  }
  const VariableFluxPack<T> &
  PackVariablesAndFluxes(const std::vector<MetadataFlag> &flags, PackIndexMap &map,
                         vpack_types::StringPair &key) {
    return PackVariablesAndFluxesImpl(flags, {}, &map, &key);
  }
  const VariableFluxPack<T> &
  PackVariablesAndFluxes(const std::vector<MetadataFlag> &flags,
                         vpack_types::StringPair &key) {
    return PackVariablesAndFluxesImpl(flags, {}, nullptr, &key);
  }
  const VariableFluxPack<T> &PackVariablesAndFluxes(const std::vector<int> &sparse_ids,
                                                    PackIndexMap &map,
                                                    vpack_types::StringPair &key) {
    return PackVariablesAndFluxesImpl(sparse_ids, &map, &key);
  }
  const VariableFluxPack<T> &PackVariablesAndFluxes(const std::vector<int> &sparse_ids,
                                                    vpack_types::StringPair &key) {
    return PackVariablesAndFluxesImpl(sparse_ids, nullptr, &key);
  }
  const VariableFluxPack<T> &PackVariablesAndFluxes(PackIndexMap &map,
                                                    vpack_types::StringPair &key) {
    return PackVariablesAndFluxesImpl({}, &map, &key);
  }
  const VariableFluxPack<T> &PackVariablesAndFluxes(vpack_types::StringPair &key) {
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
  const VariablePack<T> &PackVariables(const std::vector<MetadataFlag> &flags,
                                       const std::vector<int> &sparse_ids,
                                       PackIndexMap &map, vpack_types::VPackKey_t &key,
                                       bool coarse = false) {
    return PackVariablesImpl(flags, sparse_ids, coarse, &map, &key);
  }
  const VariablePack<T> &PackVariables(const std::vector<MetadataFlag> &flags,
                                       const std::vector<int> &sparse_ids,
                                       vpack_types::VPackKey_t &key,
                                       bool coarse = false) {
    return PackVariablesImpl(flags, sparse_ids, coarse, nullptr, &key);
  }
  const VariablePack<T> &PackVariables(const std::vector<MetadataFlag> &flags,
                                       PackIndexMap &map, vpack_types::VPackKey_t &key,
                                       bool coarse = false) {
    return PackVariablesImpl(flags, {}, coarse, &map, &key);
  }
  const VariablePack<T> &PackVariables(const std::vector<MetadataFlag> &flags,
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
  const VariableFluxPack<T> &PackListedVariablesAndFluxes(const VarLabelList &var_list,
                                                          const VarLabelList &flux_list,
                                                          PackIndexMap *map,
                                                          vpack_types::StringPair *key);
  const VariableFluxPack<T> &
  PackVariablesAndFluxesImpl(const std::vector<std::string> &var_names,
                             const std::vector<std::string> &flx_names,
                             const std::vector<int> &sparse_ids, PackIndexMap *map,
                             vpack_types::StringPair *key);

  const VariableFluxPack<T> &
  PackVariablesAndFluxesImpl(const std::vector<MetadataFlag> &flags,
                             const std::vector<int> &sparse_ids, PackIndexMap *map,
                             vpack_types::StringPair *key);

  const VariableFluxPack<T> &
  PackVariablesAndFluxesImpl(const std::vector<int> &sparse_ids, PackIndexMap *map,
                             vpack_types::StringPair *key);

  const VariablePack<T> &PackListedVariables(const VarLabelList &var_list, bool coarse,
                                             PackIndexMap *map,
                                             std::vector<std::string> *key);

  const VariablePack<T> &PackVariablesImpl(const std::vector<std::string> &names,
                                           const std::vector<int> &sparse_ids,
                                           bool coarse, PackIndexMap *map,
                                           std::vector<std::string> *key);

  const VariablePack<T> &PackVariablesImpl(const std::vector<MetadataFlag> &flags,
                                           const std::vector<int> &sparse_ids,
                                           bool coarse, PackIndexMap *map,
                                           std::vector<std::string> *key);

  const VariablePack<T> &PackVariablesImpl(const std::vector<int> &sparse_ids,
                                           bool coarse, PackIndexMap *map,
                                           std::vector<std::string> *key);
};

} // namespace parthenon

#endif // INTERFACE_MESHBLOCK_DATA_HPP_
