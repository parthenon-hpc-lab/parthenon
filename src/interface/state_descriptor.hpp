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
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "interface/metadata.hpp"
#include "interface/params.hpp"
#include "refinement/amr_criteria.hpp"

namespace parthenon {

// Forward declarations
template <typename T>
class MeshBlockData;
template <typename T>
class VariablePack;
template <typename T>
class VariableFluxPack;
template <typename T>
class MeshBlockPack;
template <typename T>
using MeshBlockVarPack = MeshBlockPack<VariablePack<T>>;
template <typename T>
using MeshBlockVarFluxPack = MeshBlockPack<VariableFluxPack<T>>;
template <typename T>
using VarPackingFunc = std::function<std::vector<MeshBlockVarPack<T>>(Mesh *)>;
template <typename T>
using FluxPackingFunc = std::function<std::vector<MeshBlockVarFluxPack<T>>(Mesh *)>;

enum class DerivedOwnership { shared, unique };

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
    FillDerived = nullptr;
    EstimateTimestep = nullptr;
    CheckRefinement = nullptr;
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
  const std::string &label() { return label_; }

  // field addition / retrieval routines
  // add a field with associated metadata
  bool AddField(const std::string &field_name, Metadata &m,
                DerivedOwnership owner = DerivedOwnership::unique) {
    if (m.IsSet(Metadata::Sparse)) {
      auto miter = sparseMetadataMap_.find(field_name);
      if (miter != sparseMetadataMap_.end()) {
        miter->second.push_back(m);
      } else {
        sparseMetadataMap_[field_name] = {m};
      }
    } else {
      const std::string &assoc = m.getAssociated();
      if (!assoc.length()) m.Associate(field_name);
      auto miter = metadataMap_.find(field_name);
      if (miter != metadataMap_.end()) { // this field has already been added
        Metadata &mprev = miter->second;
        if (owner == DerivedOwnership::unique) {
          throw std::invalid_argument(
              "Field " + field_name +
              " add with DerivedOwnership::unique already exists");
        }
        if (mprev != m) {
          throw std::invalid_argument("Field " + field_name +
                                      " already exists with different metadata");
        }
        return false;
      } else {
        metadataMap_[field_name] = m;
        m.Associate("");
      }
    }
    return true;
  }

  void AddMeshBlockPack(const std::string &pack_name, const VarPackingFunc<Real> &func) {
    realVarPackerMap_[pack_name] = func;
  }
  void AddMeshBlockPack(const std::string &pack_name, const FluxPackingFunc<Real> &func) {
    realFluxPackerMap_[pack_name] = func;
  }

  // retrieve number of fields
  int size() const { return metadataMap_.size(); }

  // retrieve all field names
  std::vector<std::string> Fields() {
    std::vector<std::string> names;
    names.reserve(metadataMap_.size());
    for (auto &x : metadataMap_) {
      names.push_back(x.first);
    }
    return names;
  }

  std::map<std::string, Metadata> &AllFields() { return metadataMap_; }
  std::map<std::string, std::vector<Metadata>> &AllSparseFields() {
    return sparseMetadataMap_;
  }

  // retrieve metadata for a specific field
  Metadata &FieldMetadata(const std::string &field_name) {
    return metadataMap_[field_name];
  }

  // get all metadata for this physics
  const std::map<std::string, Metadata> &AllMetadata() { return metadataMap_; }

  // Get all MeshBlockPacker functions
  const std::map<std::string, VarPackingFunc<Real>> &AllMeshBlockVarPackers() {
    return realVarPackerMap_;
  }
  const std::map<std::string, FluxPackingFunc<Real>> &AllMeshBlockFluxPackers() {
    return realFluxPackerMap_;
  }

  bool FlagsPresent(std::vector<MetadataFlag> const &flags, bool matchAny = false) {
    for (auto &pair : metadataMap_) {
      auto &metadata = pair.second;
      if (metadata.FlagsSet(flags, matchAny)) return true;
    }
    for (auto &pair : sparseMetadataMap_) {
      auto &sparsevec = pair.second;
      for (auto &metadata : sparsevec) {
        if (metadata.FlagsSet(flags, matchAny)) return true;
      }
    }
    return false;
  }

  std::vector<std::shared_ptr<AMRCriteria>> amr_criteria;
  void (*FillDerived)(std::shared_ptr<MeshBlockData<Real>> &rc);
  Real (*EstimateTimestep)(std::shared_ptr<MeshBlockData<Real>> &rc);
  AmrTag (*CheckRefinement)(std::shared_ptr<MeshBlockData<Real>> &rc);

 private:
  Params params_;
  const std::string label_;
  std::map<std::string, Metadata> metadataMap_;
  std::map<std::string, std::vector<Metadata>> sparseMetadataMap_;
  std::map<std::string, VarPackingFunc<Real>> realVarPackerMap_;
  std::map<std::string, FluxPackingFunc<Real>> realFluxPackerMap_;
};

using Packages_t = std::map<std::string, std::shared_ptr<StateDescriptor>>;

} // namespace parthenon

#endif // INTERFACE_STATE_DESCRIPTOR_HPP_
