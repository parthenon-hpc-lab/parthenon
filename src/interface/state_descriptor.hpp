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

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "interface/container.hpp"
#include "interface/metadata.hpp"
#include "interface/params.hpp"
#include "refinement/amr_criteria.hpp"

namespace parthenon {

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
  T &Param(const std::string &key) {
    return params_.Get<T>(key);
  }

  // Set (if not set) and get simultaneously.
  // infers type correctly.
  template <typename T>
  T &Param(const std::string &key, T value) {
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

  std::vector<std::shared_ptr<AMRCriteria>> amr_criteria;
  void (*FillDerived)(std::shared_ptr<Container<Real>> &rc);
  Real (*EstimateTimestep)(std::shared_ptr<Container<Real>> &rc);
  AmrTag (*CheckRefinement)(std::shared_ptr<Container<Real>> &rc);

 private:
  Params params_;
  const std::string label_;
  std::map<std::string, Metadata> metadataMap_;
  std::map<std::string, std::vector<Metadata>> sparseMetadataMap_;
};

using Packages_t = std::map<std::string, std::shared_ptr<StateDescriptor>>;

} // namespace parthenon

#endif // INTERFACE_STATE_DESCRIPTOR_HPP_
