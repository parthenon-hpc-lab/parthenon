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
#include <stdexcept>
#include <string>

#include "interface/swarm_container.hpp"

namespace parthenon {
class Mesh;

template <typename T>
class DataCollection {
 public:
<<<<<<< HEAD:src/interface/data_collection.hpp
  DataCollection() {
    containers_["base"] = std::make_shared<T>(); // always add "base" container
    swarmContainers_["base"] = std::make_shared<SwarmContainer>();
    pmy_mesh_ = nullptr;
  }

  void SetMeshPointer(Mesh *pmesh) { pmy_mesh_ = pmesh; }

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
  void Add(const std::string &label, const std::shared_ptr<SwarmContainer> &src);

  std::shared_ptr<T> &Get() { return containers_["base"]; }
  std::shared_ptr<T> &Get(const std::string &label) {
    auto it = containers_.find(label);
    if (it == containers_.end()) {
      throw std::runtime_error("Container " + label + " does not exist in collection.");
    }
    return it->second;
  }

  std::shared_ptr<T> &GetOrAdd(const std::string &mbd_label, const int &partition_id);
  std::shared_ptr<SwarmContainer> &GetSwarmContainer() {
    return swarmContainers_["base"];
  }
  std::shared_ptr<SwarmContainer> &GetSwarmContainer(const std::string &label) {
    auto it = swarmContainers_.find(label);
    if (it == swarmContainers_.end()) {
      throw std::runtime_error("SwarmContainer " + label +
                               " does not exist in collection.");
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
    auto sc = swarmContainers_.begin();
    while (sc != swarmContainers_.end()) {
      if (sc->first != "base") {
        sc = swarmContainers_.erase(sc);
      } else {
        ++sc;
      }
    }
  }

 private:
  Mesh *pmy_mesh_;
  std::map<std::string, std::shared_ptr<T>> containers_;
  std::map<std::string, std::shared_ptr<SwarmContainer>> swarmContainers_;
};

} // namespace parthenon

#endif // INTERFACE_DATA_COLLECTION_HPP_
