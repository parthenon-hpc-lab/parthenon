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
#ifndef INTERFACE_DATA_COLLECTION_HPP_
#define INTERFACE_DATA_COLLECTION_HPP_

#include <map>
#include <memory>
#include <string>

namespace parthenon {

template <typename T>
class DataCollection {
 public:
  DataCollection() {
    containers_["base"] = std::make_shared<T>(); // always add "base" container
  }

  std::shared_ptr<T> Add(const std::string &label, const std::shared_ptr<T> &src);
  std::shared_ptr<T> Add(const std::string &label) {
    // error check for duplicate names
    auto it = containers_.find(label);
    if (it != containers_.end()) {
      return it->second;
    }
    containers_[label] = std::make_shared<T>();
    return containers_[label];
  }

  std::shared_ptr<T> &Get() { return containers_["base"]; }
  std::shared_ptr<T> &Get(const std::string &label) {
    auto it = containers_.find(label);
    if (it == containers_.end()) {
      throw std::runtime_error("Container " + label + " does not exist in collection.");
    }
    return it->second;
  }

  void PurgeNonBase() {
    auto c = containers_.begin();
    while (c != containers_.end()) {
      if (c->first != "base") {
        c = containers_.erase(c);
      } else {
        ++c;
      }
    }
  }

 private:
  std::map<std::string, std::shared_ptr<T>> containers_;
};

} // namespace parthenon

#endif // INTERFACE_DATA_COLLECTION_HPP_
