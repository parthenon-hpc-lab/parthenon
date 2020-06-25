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
#ifndef INTERFACE_CONTAINER_COLLECTION_HPP_
#define INTERFACE_CONTAINER_COLLECTION_HPP_

#include <algorithm>
#include <map>
#include <memory>
#include <string>

#include "interface/container.hpp"
#include "interface/metadata.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

template <typename T>
class ContainerCollection {
 public:
  ContainerCollection() {
    // always add "base" container
    containers_["base"] = std::make_shared<Container<T>>();
  }

  void Add(const std::string &label, Container<T> &src);

  Container<T> &Get() { return *containers_["base"]; }
  Container<T> &Get(const std::string &label);

  void Swap(const std::string &label1, const std::string &label2);

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

  void Print() {
    for (auto &c : containers_) {
      std::cout << "Container " << c.first << " has:" << std::endl;
      c.second->Print();
      std::cout << std::endl;
    }
  }

 private:

  auto FindIterator(const std::string &label) {
    auto it = containers_.find(label);
    if (it == containers_.end()) {
      std::string message = ("Container " + label
                             + " does not exist in collection.");
      PARTHENON_FAIL(message.c_str());
    }
    return it;
  }

  std::map<std::string, std::shared_ptr<Container<T>>> containers_;

};

} // namespace parthenon

#endif // INTERFACE_CONTAINER_COLLECTION_HPP_
