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
#include "interface/sparse_variable.hpp"
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
class MeshBlockData {
 public:
  //-----------------
  // Public Methods
  //-----------------
  /// Constructor
  MeshBlockData<T>() = default;

  // Constructors for getting sub-containers
  // the variables returned are all shallow copies of the src container.
  // Optionally extract only some of the sparse ids of src variable.
  MeshBlockData<T>(const MeshBlockData<T> &src, const std::vector<std::string> &names,
                   const std::vector<int> &sparse_ids = {});
  MeshBlockData<T>(const MeshBlockData<T> &src, const std::vector<MetadataFlag> &flags);

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

  void Copy(const std::shared_ptr<MeshBlockData<T>> &src,
            const std::vector<std::string> &names) {
    SetBlockPointer(src);
    for (const auto &name : names) {
      bool found = false;
      const auto &vMap = src->GetCellVariableMap();
      auto vit = vMap.find(name);
      if (vit != vMap.end()) {
        auto &v = vit->second;
        if (v->IsSet(Metadata::OneCopy)) {
          Add(v);
        } else {
          Add(v->AllocateCopy());
        }
        found = true;
      }
      const auto &sMap = src->GetSparseMap();
      auto sit = sMap.find(name);
      if (sit != sMap.end()) {
        auto &v = sit->second;
        if (v->IsSet(Metadata::OneCopy)) {
          Add(v);
        } else {
          Add(v->AllocateCopy());
        }
        found = true;
      }
      if (!found) {
        PARTHENON_THROW("MeshBlockData::Copy did not find " + name);
      }
    }
  }

  void Copy(const std::shared_ptr<MeshBlockData<T>> &src) {
    SetBlockPointer(src);
    for (auto v : src->GetCellVariableVector()) {
      if (v->IsSet(Metadata::OneCopy)) {
        // just copy the (shared) pointer
        Add(v);
      } else {
        // allocate new storage
        Add(v->AllocateCopy());
      }
    }

    for (auto v : src->GetFaceVector()) {
      if (v->IsSet(Metadata::OneCopy)) {
        Add(v);
      } else {
        throw std::runtime_error("Non-oneCopy face variables are not yet supported");
      }
    }

    for (auto v : src->GetSparseVector()) {
      if (v->IsSet(Metadata::OneCopy)) {
        // copy the shared pointer
        Add(v);
      } else {
        Add(v->AllocateCopy());
      }
    }
  }

  /// We can initialize a container with slices from a different
  /// container.  For variables that have the sparse tag, this will
  /// return the sparse slice.  All other variables are added as
  /// is. This call returns a new container.
  ///
  /// @param sparse_id The sparse id
  /// @return New container with slices from all variables
  std::shared_ptr<MeshBlockData<T>> SparseSlice(int sparse_id);

  ///
  /// Set the pointer to the mesh block for this container
  void SetBlockPointer(std::weak_ptr<MeshBlock> pmb) { pmy_block = pmb; }
  void SetBlockPointer(const std::shared_ptr<MeshBlockData<T>> &other) {
    pmy_block = other->GetBlockPointer();
  }

  ///
  /// Allocate and add a variable<T> to the container
  ///
  /// This function will eventually look at the metadata flags to
  /// identify the size of the first dimension based on the
  /// topological location.
  ///
  /// @param label the name of the variable
  /// @param metadata the metadata associated with the variable
  /// @param dims the size of each element
  ///
  /// TODO(JMM): DO NOT make these strings const reference.
  /// passing in C-style string literals misbehaves
  void Add(const std::string &label, const Metadata &metadata,
           const std::vector<int> &dims);

  ///
  /// Allocate and add a variable<T> to the container
  ///
  /// This function will eventually look at the metadata flags to
  /// identify the size of the first dimension based on the
  /// topological location.
  ///
  /// @param labelVector the array of names of variables
  /// @param metadata the metadata associated with the variable
  /// @param dims the size of each element
  ///
  void Add(const std::vector<std::string> &labelVector, const Metadata &metadata,
           const std::vector<int> &dims);

  ///
  /// Allocate and add a variable<T> to the container
  ///
  /// This function will eventually look at the metadata flags to
  /// identify the size of the first dimension based on the
  /// topological location.  Dimensions will be taken from the metadata.
  ///
  /// @param label the name of the variable
  /// @param metadata the metadata associated with the variable
  ///
  void Add(const std::string &label, const Metadata &metadata);

  ///
  /// Allocate and add a variable<T> to the container
  ///
  /// This function will eventually look at the metadata flags to
  /// identify the size of the first dimension based on the
  /// topological location.  Dimensions will be taken from the metadata.
  ///
  /// @param labelVector the array of names of variables
  /// @param metadata the metadata associated with the variable
  ///
  void Add(const std::vector<std::string> &labelVector, const Metadata &metadata);

  void Add(std::shared_ptr<CellVariable<T>> var) {
    varVector_.push_back(var);
    varMap_[var->label()] = var;
  }
  void Add(std::shared_ptr<FaceVariable<T>> var) {
    faceVector_.push_back(var);
    faceMap_[var->label()] = var;
  }
  void Add(std::shared_ptr<SparseVariable<T>> var) {
    sparseVector_.push_back(var);
    sparseMap_[var->label()] = var;
  }

  //
  // Queries related to CellVariable objects
  //
  const CellVariableVector<T> &GetCellVariableVector() const { return varVector_; }
  const MapToCellVars<T> &GetCellVariableMap() const { return varMap_; }
  CellVariable<T> &Get(const std::string &label) const {
    auto it = varMap_.find(label);
    if (it == varMap_.end()) {
      PARTHENON_THROW(std::string("\n") + std::string(label) +
                      std::string(" array not found in Get()\n"));
    }
    return *(it->second);
  }

  CellVariable<T> &Get(const int index) const { return *(varVector_[index]); }

  int Index(const std::string &label) {
    for (int i = 0; i < (varVector_).size(); i++) {
      if (!varVector_[i]->label().compare(label)) return i;
    }
    return -1;
  }

  //
  // Queries related to SparseVariable objects
  //
  const SparseVector<T> &GetSparseVector() const { return sparseVector_; }
  const MapToSparse<T> &GetSparseMap() const { return sparseMap_; }
  SparseVariable<T> &GetSparseVariable(const std::string &label) {
    auto it = sparseMap_.find(label);
    if (it == sparseMap_.end()) {
      PARTHENON_THROW("sparseMap_ does not have " + label);
    }
    return *(it->second);
  }

  SparseMap<T> &GetSparseMap(const std::string &label) {
    return GetSparseVariable(label).GetMap();
  }

  CellVariableVector<T> &GetSparseVector(const std::string &label) {
    return GetSparseVariable(label).GetVector();
  }

  std::shared_ptr<CellVariable<T>> &Get(const std::string &label, const int sparse_id) {
    return GetSparseVariable(label).Get(sparse_id);
  }

  std::vector<int> &GetSparseIndexMap(const std::string &label) {
    return GetSparseVariable(label).GetIndexMap();
  }

  //
  // Queries related to FaceVariable objects
  //
  const FaceVector<T> &GetFaceVector() const { return faceVector_; }
  const MapToFace<T> &GetFaceMap() const { return faceMap_; }
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

  /// Gets an array of real variables from container.
  /// @param names is the variables we want
  /// @param indexCount a map of names to std::pair<index,count> for each name
  /// @param sparse_ids if specified is list of sparse ids we are interested in.  Note
  ///        that non-sparse variables specified are aliased in as is.
  int GetCellVariables(const std::vector<std::string> &names,
                       std::vector<CellVariable<T>> &vRet,
                       std::map<std::string, std::pair<int, int>> &indexCount,
                       const std::vector<int> &sparse_ids = {});

  /// Queries related to variable packs
  VariableFluxPack<T> PackVariablesAndFluxes(const std::vector<std::string> &var_names,
                                             const std::vector<std::string> &flx_names,
                                             PackIndexMap &vmap);
  VariableFluxPack<T> PackVariablesAndFluxes(const std::vector<std::string> &var_names,
                                             const std::vector<std::string> &flx_names);
  VariableFluxPack<T> PackVariablesAndFluxes(const std::vector<MetadataFlag> &flags,
                                             PackIndexMap &vmap);
  VariableFluxPack<T> PackVariablesAndFluxes(const std::vector<MetadataFlag> &flags);
  VariableFluxPack<T> PackVariablesAndFluxes(const std::vector<std::string> &var_names,
                                             const std::vector<std::string> &flx_names,
                                             PackIndexMap &vmap,
                                             vpack_types::StringPair &key);
  VariableFluxPack<T> PackVariablesAndFluxes(const std::vector<std::string> &var_names,
                                             const std::vector<std::string> &flx_names,
                                             vpack_types::StringPair &key);
  VariableFluxPack<T> PackVariablesAndFluxes(const std::vector<MetadataFlag> &flags,
                                             PackIndexMap &vmap,
                                             vpack_types::StringPair &key);
  VariableFluxPack<T> PackVariablesAndFluxes(const std::vector<MetadataFlag> &flags,
                                             vpack_types::StringPair &key);
  VariablePack<T> PackVariables(const std::vector<std::string> &names,
                                const std::vector<int> &sparse_ids, PackIndexMap &vmap,
                                const bool coarse = false);
  VariablePack<T> PackVariables(const std::vector<std::string> &names,
                                const std::vector<int> &sparse_ids,
                                const bool coarse = false);
  VariablePack<T> PackVariables(const std::vector<std::string> &names, PackIndexMap &vmap,
                                const bool coarse = false);
  VariablePack<T> PackVariables(const std::vector<std::string> &names,
                                const bool coarse = false);
  VariablePack<T> PackVariables(const std::vector<MetadataFlag> &flags,
                                PackIndexMap &vmap, const bool coarse = false);
  VariablePack<T> PackVariables(const std::vector<MetadataFlag> &flags,
                                const bool coarse = false);
  VariablePack<T> PackVariables(PackIndexMap &vmap, const bool coarse = false);
  // no coarse flag because you need to be able to specify non one-copy variables
  // and also because the C++ compiler automatically typecasts initializer lists
  // to bool if you let it.
  VariablePack<T> PackVariables();
  VariablePack<T> PackVariables(const std::vector<std::string> &names,
                                const std::vector<int> &sparse_ids, PackIndexMap &vmap,
                                std::vector<std::string> &key, const bool coarse = false);
  VariablePack<T> PackVariables(const std::vector<std::string> &names,
                                const std::vector<int> &sparse_ids,
                                std::vector<std::string> &key, const bool coarse = false);
  VariablePack<T> PackVariables(const std::vector<std::string> &names, PackIndexMap &vmap,
                                std::vector<std::string> &key, const bool coarse = false);
  VariablePack<T> PackVariables(const std::vector<std::string> &names,
                                std::vector<std::string> &key, const bool coarse = false);
  VariablePack<T> PackVariables(const std::vector<MetadataFlag> &flags,
                                PackIndexMap &vmap, std::vector<std::string> &key,
                                const bool coarse = false);
  VariablePack<T> PackVariables(const std::vector<MetadataFlag> &flags,
                                std::vector<std::string> &key, const bool coarse = false);
  VariablePack<T> PackVariables(PackIndexMap &vmap, std::vector<std::string> &key,
                                const bool coarse = false);

  /// Remove a variable from the container or throw exception if not
  /// found.
  /// @param label the name of the variable to be deleted
  void Remove(const std::string &label);

  /// Print list of labels in container
  void Print();

  // return number of stored arrays
  int Size() { return varVector_.size(); }

  // Communication routines
  void ResetBoundaryCellVariables();
  void SetupPersistentMPI();
  TaskStatus SetBoundaries();
  TaskStatus SendBoundaryBuffers();
  TaskStatus ReceiveAndSetBoundariesWithWait();
  TaskStatus ReceiveBoundaryBuffers();
  TaskStatus StartReceiving(BoundaryCommSubset phase);
  TaskStatus ClearBoundary(BoundaryCommSubset phase);
  TaskStatus SendFluxCorrection();
  TaskStatus ReceiveFluxCorrection();

  // physical boundary routines
  void RestrictBoundaries();
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
    for (auto &v : sparseMap_) {
      my_keys.push_back(v.first);
    }
    for (auto &v : cmp.GetCellVariableMap()) {
      cmp_keys.push_back(v.first);
    }
    for (auto &v : cmp.GetFaceMap()) {
      cmp_keys.push_back(v.first);
    }
    for (auto &v : cmp.GetSparseMap()) {
      cmp_keys.push_back(v.first);
    }
    return (my_keys == cmp_keys);
  }

  bool Contains(const std::string &name) const {
    if (varMap_.find(name) != varMap_.end()) return true;
    if (sparseMap_.find(name) != sparseMap_.end()) return true;
    if (faceMap_.find(name) != faceMap_.end()) return true;
    return false;
  }
  bool Contains(const std::vector<std::string> &names) const {
    for (const auto &name : names) {
      if (!Contains(name)) return false;
    }
    return true;
  }

 private:
  int debug = 0;
  std::weak_ptr<MeshBlock> pmy_block;

  CellVariableVector<T> varVector_; ///< the saved variable array
  FaceVector<T> faceVector_;        ///< the saved face arrays
  SparseVector<T> sparseVector_;

  MapToCellVars<T> varMap_;
  MapToFace<T> faceMap_;
  MapToSparse<T> sparseMap_;

  // variable packing
  MapToVariablePack<T> varPackMap_;
  MapToVariablePack<T> coarseVarPackMap_; // cache for varpacks over coarse arrays
  MapToVariableFluxPack<T> varFluxPackMap_;

  void calcArrDims_(std::array<int, 6> &arrDims, const std::vector<int> &dims,
                    const Metadata &metadata);

  // helper functions for VariablePack
  vpack_types::VarList<T> MakeList_(const std::vector<std::string> &names,
                                    std::vector<std::string> &names_out,
                                    const std::vector<int> &sparse_ids = {});
  vpack_types::VarList<T> MakeList_(const std::vector<MetadataFlag> &flags,
                                    std::vector<std::string> &labels);
  vpack_types::VarList<T> MakeList_(std::vector<std::string> &names);

  // These helper functions are private scope because they assume that
  // the names include the components of sparse variables.
  VariableFluxPack<T>
  PackVariablesAndFluxesHelper_(const std::vector<std::string> &var_names,
                                const std::vector<std::string> &flx_names,
                                const vpack_types::VarList<T> &vars,
                                const vpack_types::VarList<T> &fvars, PackIndexMap &vmap,
                                vpack_types::StringPair &key);
  VariablePack<T> PackVariablesHelper_(const std::vector<std::string> &names,
                                       const vpack_types::VarList<T> &vars,
                                       PackIndexMap &vmap, const bool coarse = false);
};

using MeshBlockDataCollection = DataCollection<MeshBlockData<Real>>;

} // namespace parthenon

#endif // INTERFACE_MESHBLOCK_DATA_HPP_
