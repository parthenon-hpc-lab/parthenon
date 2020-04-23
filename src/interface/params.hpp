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

#ifndef DEBUG_
#define DEBUG_ 0
#endif

#include "utils/error_checking.hpp"

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#if (DEBUG_ > 0)
#include <typeindex>
#include <typeinfo>
#endif

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
    if (hasKey(key)) {
      throw std::invalid_argument("Key value pair already exists, cannot add key.");
    }
    myParams_[key] = std::unique_ptr<Params::base_t>(new object_t<T>(value));
    myTypes_[key] = std::string(typeid(value).name());
  }

  void reset() {
    myParams_.clear();
    myTypes_.clear();
  }

  template <typename T>
  const T &Get(const std::string key) {
    keyCheck(key, false);
    typeCheck<T>(key, false);
    auto typed_ptr = dynamic_cast<Params::object_t<T> *>(myParams_[key].get());
    if (typed_ptr == nullptr)
      throw std::invalid_argument("Cannot cast Params[" + key + "] to requested type");
    return *typed_ptr->pValue;
  }

  bool hasKey(const std::string key) const {
    return (myParams_.find(key) != myParams_.end());
  }

  // void Params::
  void list() {
    std::cout << std::endl << "Items are:" << std::endl;
    for (auto &x : myParams_) {
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

  template <typename T>
  void typeCheck(const std::string key, bool die) {
    // check on return type
    if (myTypes_[key].compare(std::string(typeid(T).name()))) {
      std::cout << "WRONG TYPE FOR KEY '" << key << "'" << std::endl;
      PARTHENON_REQUIRE(!die, "Exit request on wrong type for key " + key);
    }
  }

  void keyCheck(const std::string key, bool die) {
#if (DEBUG_ > 0)
    if (!hasKey(key)) {
      // key alread exists, replace
      std::cout << std::endl;
      std::cout << "----------------------------------------------" << std::endl;
      std::cout << "    WARNING: key '" << key << "' not found." << std::endl;
      std::cout << "             Swift death will follow..." << std::endl;
      std::cout << "----------------------------------------------" << std::endl;
      std::cout << std::endl;

      if (die) {
        throw std::invalid_argument("Key " + key + " doesn't exist");
      }
    }
#endif
  }

  std::map<std::string, std::unique_ptr<Params::base_t>> myParams_;
  std::map<std::string, std::string> myTypes_;
};

} // namespace parthenon

#endif // INTERFACE_PARAMS_HPP_
