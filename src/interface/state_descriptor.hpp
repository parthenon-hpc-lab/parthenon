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
#include "interface/swarm.hpp"
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
  explicit StateDescriptor(std::string label) : _label(label) {
    FillDerived = nullptr;
    EstimateTimestep = nullptr;
    CheckRefinement = nullptr;
  }

  template <typename T>
  void AddParam(const std::string &key, T &value) {
    _params.Add<T>(key, value);
  }

  template <typename T>
  const T &Param(const std::string &key) {
    return _params.Get<T>(key);
  }

  Params &AllParams() { return _params; }
  // retrieve label
  const std::string &label() { return _label; }

  // field addition / retrieval routines
  // add a field with associated metadata
  bool AddField(const std::string &field_name, Metadata &m,
                DerivedOwnership owner = DerivedOwnership::unique) {
    if (m.IsSet(Metadata::Sparse)) {
      auto miter = _sparseMetadataMap.find(field_name);
      if (miter != _sparseMetadataMap.end()) {
        miter->second.push_back(m);
      } else {
        _sparseMetadataMap[field_name] = {m};
      }
    } else {
      const std::string &assoc = m.getAssociated();
      if (!assoc.length()) m.Associate(field_name);
      auto miter = _metadataMap.find(field_name);
      if (miter != _metadataMap.end()) { // this field has already been added
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
        _metadataMap[field_name] = m;
        m.Associate("");
      }
    }
    return true;
  }

  // retrieve number of fields
  int size() const { return _metadataMap.size(); }

  // retrieve all field names
  std::vector<std::string> Fields() {
    std::vector<std::string> names;
    names.reserve(_metadataMap.size());
    for (auto &x : _metadataMap) {
      names.push_back(x.first);
    }
    return names;
  }

  const std::map<std::string, Metadata> &AllFields() { return _metadataMap; }
  const std::map<std::string, std::vector<Metadata>> &AllSparseFields() {
    return _sparseMetadataMap;
  }

  // retrieve metadata for a specific field
  Metadata &FieldMetadata(const std::string &field_name) {
    return _metadataMap[field_name];
  }

  // get all metadata for this physics
  const std::map<std::string, Metadata> &AllMetadata() { return _metadataMap; }

  std::vector<std::shared_ptr<AMRCriteria>> amr_criteria;
  void (*FillDerived)(Container<Real> &rc);
  Real (*EstimateTimestep)(Container<Real> &rc);
  AmrTag (*CheckRefinement)(Container<Real> &rc);

 private:
  Params _params;
  const std::string _label;
  std::map<std::string, Metadata> _metadataMap;
  std::map<std::string, std::vector<Metadata>> _sparseMetadataMap;
};

using Packages_t = std::map<std::string, std::shared_ptr<StateDescriptor>>;

} // namespace parthenon

#endif // INTERFACE_STATE_DESCRIPTOR_HPP_
