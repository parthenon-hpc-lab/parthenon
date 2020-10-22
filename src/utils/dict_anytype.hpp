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
#ifndef UTILS_DICT_ANYTYPE_
#define UTILS_DICT_ANYTYPE_

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <vector>

#include "utils/error_checking.hpp"

namespace parthenon {

// A map from "std::string" to any type
class DictAnyType {
 public:
  DictAnyType() {}

  // can't copy because we have a map of unique_ptr
  DictAnyType(const DictAnyType &p) = delete;

  /// Adds object based on type of the value
  ///
  /// Throws an error if the key is already in use
  template <typename T>
  void Add(const std::string &key, T value) {
    PARTHENON_REQUIRE_THROWS(!(hasKey(key)), "Key " + key + "already exists");
    myDict_[key] = std::unique_ptr<DictAnyType::base_t>(new object_t<T>(value));
    myTypes_[key] = std::string(typeid(value).name());
  }

  void reset() {
    myDict_.clear();
    myTypes_.clear();
  }

  template <typename T>
  T &Get(const std::string &key) {
    auto it = myDict_.find(key);
    PARTHENON_REQUIRE_THROWS(it != myDict_.end(), "Key " + key + " doesn't exist");
    PARTHENON_REQUIRE_THROWS(!(myTypes_[key].compare(std::string(typeid(T).name()))),
                             "WRONG TYPE FOR KEY '" + key + "'");
    auto typed_ptr = dynamic_cast<DictAnyType::object_t<T> *>((it->second).get());
    return *typed_ptr->pValue;
  }

  bool hasKey(const std::string &key) const {
    return (myDict_.find(key) != myDict_.end());
  }

  // Overload Get to return value if available,
  // otherwise add default value to params and return it.
  template <typename T>
  T &Get(const std::string &key, T default_value) {
    if (!hasKey(key)) {
      Add(key, default_value);
    }
    return Get<T>(key);
  }

  // void DictAnyType::
  void list() {
    std::cout << std::endl << "Items are:" << std::endl;
    for (auto &x : myDict_) {
      std::cout << "   " << x.first << ":" << x.second.get() << ":" << x.second->address()
                << ":" << myTypes_[x.first] << std::endl;
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

  std::map<std::string, std::unique_ptr<DictAnyType::base_t>> myDict_;
  std::map<std::string, std::string> myTypes_;
};

} // namespace parthenon

#endif // UTILS_DICT_ANYTYPE_
