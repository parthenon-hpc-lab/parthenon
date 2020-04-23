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

#include <map>
#include <memory>
#include <string>

#include "interface/container.hpp"
#include "interface/swarm_container.hpp"
#include "interface/metadata.hpp"

namespace parthenon {

template <typename T>
class ContainerCollection {
 public:
  ContainerCollection() {
    // Always add "base" containers
    containers_["base"] = std::make_shared<Container<T>>();
    swarmContainers_["base"] = std::make_shared<SwarmContainer>();
  }

  void Add(const std::string &label, Container<T> &src);

  void Add(const std::string &label, SwarmContainer &src);

  Container<T> &Get() { return *containers_["base"]; }
  Container<T> &Get(const std::string &label) {
    auto it = containers_.find(label);
    if (it == containers_.end()) {
      throw std::runtime_error("Container " + label + " does not exist in collection.");
    }
    return *(it->second);
  }

  SwarmContainer &GetSwarmContainer() { return *swarmContainers_["base"]; }
  SwarmContainer &GetSwarmContainer(const std::string &label) {
    auto it = swarmContainers_.find(label);
    if (it == swarmContainers_.end()) {
      throw std::runtime_error("SwarmContainer " + label + " does not exist in collection.");
    }
    return *(it->second);
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
    auto sc = swarmContainers_.begin();
    while (sc != swarmContainers_.end()) {
      if (sc->first != "base") {
        sc = swarmContainers_.erase(sc);
      } else {
        ++sc;
      }
    }
  }

  void Print() {
    for (auto &c : containers_) {
      std::cout << "Container " << c.first << " has:" << std::endl;
      c.second->Print();
      std::cout << std::endl;
    }
    for (auto &sc : swarmContainers_) {
      std::cout << "SwarmContainer " << sc.first << " has:" << std::endl;
      sc.second->Print();
      std::cout << std::endl;
    }
  }

 private:
  std::map<std::string, std::shared_ptr<Container<T>>> containers_;
  std::map<std::string, std::shared_ptr<SwarmContainer>> swarmContainers_;
};

} // namespace parthenon

#endif // INTERFACE_CONTAINER_COLLECTION_HPP_
