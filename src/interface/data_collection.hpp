//========================================================================================
// (C) (or copyright) 2020-2024. Triad National Security, LLC. All rights reserved.
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
#include <vector>

#include "basic_types.hpp"
#include "globals.hpp"
#include "utils/concepts_lite.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {
class Mesh;
class MeshBlock;
struct BlockListPartition;
template <class T>
class MeshData;
template <class T>
class MeshBlockData;
/// The DataCollection class is an abstract container that contains at least a
/// "base" container of some type (e.g., of MeshData or MeshBlockData) plus
/// additional containers identified by string labels.
/// Current usage includes (but is not limited to) storing MeshBlockData for different
/// stages in multi-stage drivers or the corresponding MeshBlockPacks in a
/// DataCollection of MeshData.
///
/// T must implement:
///   bool Contains(std::vector<std::string>)
///   Initialize(T*, std::vector<std::string>, bool)
/// TODO: implement a concept
template <typename T>
class DataCollection {
 public:
  DataCollection() {
    containers_["base"] = std::make_shared<T>("base"); // always add "base" container
    pmy_mesh_ = nullptr;
  }

  void SetMeshPointer(Mesh *pmesh) { pmy_mesh_ = pmesh; }

  template <class SRC_t, typename ID_t>
  std::shared_ptr<T> &Add(const std::string &name, const std::shared_ptr<SRC_t> &src,
                          const std::vector<ID_t> &fields, const bool shallow) {
    auto key = GetKey(name, src);
    auto it = containers_.find(key);
    if (it != containers_.end()) {
      if (fields.size() && !(it->second)->CreatedFrom(fields)) {
        PARTHENON_THROW(key + " already exists in collection but fields do not match.");
      }
      return it->second;
    }

    auto c = std::make_shared<T>(name);
    c->Initialize(src, fields, shallow);

    containers_[key] = c;
    return containers_[key];
  }

  template <class SRC_t, typename ID_t = std::string>
  std::shared_ptr<T> &Add(const std::string &label, const std::shared_ptr<SRC_t> &src,
                          const std::vector<ID_t> &fields = {}) {
    return Add(label, src, fields, false);
  }

  template <class SRC_t, typename ID_t = std::string>
  std::shared_ptr<T> &AddShallow(const std::string &label,
                                 const std::shared_ptr<SRC_t> &src,
                                 const std::vector<ID_t> &fields = {}) {
    return Add(label, src, fields, true);
  }

  auto &Stages() { return containers_; }
  const auto &Stages() const { return containers_; }
  
  template <class SRC_t>
  const std::shared_ptr<T> &Get(const std::string &name, const std::shared_ptr<SRC_t> &src) const {
    const auto key = GetKey(name, src); 
    const auto it = containers_.find(key);
    if (it == containers_.end()) {
      throw std::runtime_error("Container " + key + " does not exist in collection.");
    }
    return it->second;
  }
  
  template <class SRC_t>
  std::shared_ptr<T> &Get(const std::string &name, const std::shared_ptr<SRC_t> &src) {
    const auto key = GetKey(name, src); 
    const auto it = containers_.find(key);
    if (it == containers_.end()) {
      throw std::runtime_error("Container " + key + " does not exist in collection.");
    }
    return it->second;
  }

  std::shared_ptr<T> &Get(const std::string &name = "base"); 
  const std::shared_ptr<T> &Get(const std::string &name = "base") const; 

  void Set(const std::string &name, std::shared_ptr<T> &d) { containers_[name] = d; }

  // Legacy methods that are specific to MeshData
  std::shared_ptr<T> &GetOrAdd(const std::string &mbd_label, const int &partition_id);
  std::shared_ptr<T> &GetOrAdd(int gmg_level, const std::string &mbd_label,
                               const int &partition_id);

  void clear() {
    containers_.clear();
  }

 private:
  std::string GetKey(const std::string &stage_label,
                     const std::shared_ptr<BlockListPartition> &in) const;
  std::string GetKey(const std::string &stage_label,
                     const std::shared_ptr<MeshData<Real>> &in) const;
  template <class U>
  std::string GetKey(const std::string &stage_label, const std::shared_ptr<U> &in) const {
    return stage_label;
  }

  Mesh *pmy_mesh_;
  std::map<std::string, std::shared_ptr<T>> containers_;
};

} // namespace parthenon

#endif // INTERFACE_DATA_COLLECTION_HPP_
