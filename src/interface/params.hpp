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
#ifndef INTERFACE_PARAMS_HPP_
#define INTERFACE_PARAMS_HPP_

#include <any>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_set>
#include <utility>
#include <vector>

#include "utils/error_checking.hpp"

#ifdef ENABLE_HDF5
#include "outputs/parthenon_hdf5_types.hpp"
#endif

namespace parthenon {

/// Defines a class that can be used to hold parameters
/// of any kind
class Params {
 public:
  // Immutable is default. Mutable is it can be updated at runtime.
  // Restart is a subset of mutable. Param not only can be updated at
  // runtime, but should be read from the restart file upon restart.
  enum class Mutability : int { Immutable = 0, Mutable = 1, Restart = 2 };

  Params() {}

  // can't copy because we have a map of unique_ptr
  Params(const Params &p) = delete;

  /// Adds object based on type of the value
  ///
  /// Throws an error if the key is already in use
  template <typename T>
  void Add(const std::string &key, T value, Mutability mutability) {
    PARTHENON_REQUIRE_THROWS(!(hasKey(key)), "Key " + key + " already exists");
    myParams_[key] = std::make_unique<std::any>(value);
    myMutable_[key] = mutability;
  }
  template <typename T>
  void Add(const std::string &key, T value, bool is_mutable = false) {
    Add(key, value, static_cast<Mutability>(is_mutable));
  }

  /// Updates existing object
  /// Throws an error if the key is not already in use
  template <typename T>
  void Update(const std::string &key, T value) {
    PARTHENON_REQUIRE_THROWS((hasKey(key)), "Key " + key + "missing.");
    // immutable casts to false all others cast to true
    PARTHENON_REQUIRE_THROWS(IsMutable(key),
                             "Parameter " + key + " must be marked as mutable");
    PARTHENON_REQUIRE_THROWS(std::type_index(myParams_.at(key)->type()) ==
                                 std::type_index(typeid(T)),
                             "Parameter " + key + " must have the relevant type");
    myParams_[key] = std::make_unique<std::any>(value);
  }

  void reset() {
    myParams_.clear();
    myMutable_.clear();
  }

  template <typename T>
  const T &Get(const std::string &key) const {
    auto &pparam = myParams_.at(key);
    PARTHENON_REQUIRE_THROWS(std::type_index(pparam->type()) ==
                                 std::type_index(typeid(T)),
                             "Parameter " + key + " must have the relevant type");
    return *std::any_cast<T>(pparam.get());
  }

  // Returning a pointer feels safer than returning a non-const reference.
  // Memory is managed by params so we don't want reference counting.
  // But we also don't want the reference completely re-assigned.
  // This also avoids extraneous copies.
  template <typename T>
  T *GetMutable(const std::string &key) const {
    // immutable casts to false all others cast to true
    PARTHENON_REQUIRE_THROWS(static_cast<bool>(myMutable_.at(key)),
                             "Parameter " + key + " must be marked as mutable");
    auto &pparam = myParams_.at(key);
    PARTHENON_REQUIRE_THROWS(std::type_index(pparam->type()) ==
                                 std::type_index(typeid(T)),
                             "Parameter " + key + " must have the relevant type");
    return std::any_cast<T>(pparam.get());
  }

  bool hasKey(const std::string &key) const {
    return (myParams_.find(key) != myParams_.end());
  }

  // Overload Get to return value if available,
  // otherwise add default value to params and return it.
  template <typename T>
  const T &Get(const std::string &key, T default_value) {
    if (!hasKey(key)) {
      Add(key, default_value);
    }
    return Get<T>(key);
  }

  const std::type_index GetType(const std::string &key) const {
    return myParams_.at(key)->type();
  }

  std::vector<std::string> GetKeys() const {
    std::vector<std::string> keys;
    for (auto &x : myParams_) {
      keys.push_back(x.first);
    }
    return keys;
  }

  auto GetMutability(const std::string &key) const { return myMutable_.at(key); }
  bool IsMutable(const std::string &key) const {
    return static_cast<bool>(myMutable_.at(key));
  }

  // void Params::
  void list() {
    std::cout << std::endl << "Items are:" << std::endl;
    for (auto &[k, v] : myParams_) {
      std::cout << "   " << k << ":" << v.get() << ":" << v->type().name() << std::endl;
    }
    std::cout << std::endl;
  }

#ifdef ENABLE_HDF5

 public:
  void WriteAllToHDF5(const std::string &prefix, const HDF5::H5G &group) const;
  void ReadFromRestart(const std::string &prefix, const HDF5::H5G &group);

 private:
  // these can go in implementation file, since the only relevant
  // instantiations are in that same implementation file.
  // Pattern here is the following:
  //
  // - {ReadTo, WriteFrom}HDF5AllParamsOfType<T>(prefix, group)
  //   works on any single type supported by parthenon_hdf5.
  //   see outputs/parthenon_hdf5.hpp for more details.
  //   scalars, std::vector<T>, and several view/ParArray types are supported.
  //
  // - {ReadTo, WriteFrom}HDF5AllParamsOfMultipleTypes<Ts...>(prefix, group)
  //   loops through all types in the variatic list and calls
  //   the above scalar function
  //
  // - {ReadTo, WriteFrom}HDF5AllParamsOfTypeOrVec<T>(prefix, group)
  //   calls the above functions on a single scalar type, as well as
  //   vectors and views of said scalar type.
  //
  // - The public functions call {ReadTo, WriteFrom}HDF5AllParamsOfTypeOrVec<T>
  //   on a set of relevant types.
  template <typename T>
  void WriteToHDF5AllParamsOfType(const std::string &prefix,
                                  const HDF5::H5G &group) const;

  template <typename... Ts>
  void WriteToHDF5AllParamsOfMultipleTypes(const std::string &prefix,
                                           const HDF5::H5G &group) const;

  template <typename T>
  void WriteToHDF5AllParamsOfTypeOrVec(const std::string &prefix,
                                       const HDF5::H5G &group) const;

  template <typename T>
  void ReadFromHDF5AllParamsOfType(const std::string &prefix, const HDF5::H5G &group);

  template <typename... Ts>
  void ReadFromHDF5AllParamsOfMultipleTypes(const std::string &prefix,
                                            const HDF5::H5G &group);

  template <typename T>
  void ReadFromHDF5AllParamsOfTypeOrVec(const std::string &prefix,
                                        const HDF5::H5G &group);

#endif // ifdef ENABLE_HDF5

 private:
  std::map<std::string, std::unique_ptr<std::any>> myParams_;
  std::map<std::string, Mutability> myMutable_;
};

} // namespace parthenon

#endif // INTERFACE_PARAMS_HPP_
