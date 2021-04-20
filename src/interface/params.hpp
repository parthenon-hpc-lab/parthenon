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
#ifndef INTERFACE_PARAMS_HPP_
#define INTERFACE_PARAMS_HPP_

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

#include "utils/error_checking.hpp"

namespace parthenon {

/// Defines a class that can be used to hold parameters
/// of any kind
class Params {
 public:
  Params() {}

  // can't copy because we have a map of unique_ptr
  Params(const Params &p) = delete;

  /// Adds object based on type of the value
  ///
  /// Throws an error if the key is already in use
  template <typename T>
  void Add(const std::string &key, T value) {
    PARTHENON_REQUIRE_THROWS(!(hasKey(key)), "Key " + key + " already exists");
    myParams_[key] = std::unique_ptr<Params::base_t>(new object_t<T>(value));
    myTypes_.emplace(make_pair(key, std::type_index(typeid(value))));
  }

  /// Updates existing object
  /// Throws an error if the key is not already in use
  template <typename T>
  void Update(const std::string &key, T value) {
    PARTHENON_REQUIRE_THROWS((hasKey(key)), "Key " + key + "missing.");
    PARTHENON_REQUIRE_THROWS(myTypes_.at(key) == std::type_index(typeid(T)),
                             "WRONG TYPE FOR KEY '" + key + "'");
    myParams_[key] = std::unique_ptr<Params::base_t>(new object_t<T>(value));
  }

  void reset() {
    myParams_.clear();
    myTypes_.clear();
  }

  template <typename T>
  const T &Get(const std::string &key) const {
    auto const it = myParams_.find(key);
    PARTHENON_REQUIRE_THROWS(it != myParams_.end(), "Key " + key + " doesn't exist");
    PARTHENON_REQUIRE_THROWS(myTypes_.at(key) == std::type_index(typeid(T)),
                             "WRONG TYPE FOR KEY '" + key + "'");
    auto typed_ptr = dynamic_cast<Params::object_t<T> *>((it->second).get());
    return *typed_ptr->pValue;
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

  const std::type_index &GetType(const std::string &key) const {
    auto const it = myTypes_.find(key);
    PARTHENON_REQUIRE_THROWS(it != myTypes_.end(), "Key " + key + " doesn't exist");
    return it->second;
  }

  std::vector<std::string> GetKeys() const {
    std::vector<std::string> keys;
    for (auto &x : myParams_) {
      keys.push_back(x.first);
    }
    return keys;
  }

  // void Params::
  void list() {
    std::cout << std::endl << "Items are:" << std::endl;
    for (auto &x : myParams_) {
      std::cout << "   " << x.first << ":" << x.second.get() << ":" << x.second->address()
                << ":" << myTypes_.at(x.first).name() << std::endl;
    }
    std::cout << std::endl;
  }

 private:
  // private first so that I can use the structs defined here
  struct base_t {
    virtual ~base_t() = default; // for whatever reason I need a virtual destructor
    virtual const void *address() { return nullptr; } // for listing and debugging
  };

  template <typename T>
  struct object_t : base_t {
    std::unique_ptr<T> pValue;
    explicit object_t(T val) : pValue(std::make_unique<T>(val)) {}
    ~object_t() = default;
    const void *address() { return reinterpret_cast<void *>(pValue.get()); }
  };

  std::map<std::string, std::unique_ptr<Params::base_t>> myParams_;
  std::map<std::string, std::type_index> myTypes_;
};

} // namespace parthenon

#endif // INTERFACE_PARAMS_HPP_
