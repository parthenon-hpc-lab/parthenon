//========================================================================================
// (C) (or copyright) 2020-2023. Triad National Security, LLC. All rights reserved.
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
#include "utils/concepts_lite.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {
class Mesh;
class MeshBlock;
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
    if constexpr (!((std::is_same_v<SRC_t, MeshBlock> &&
                     std::is_same_v<T, MeshBlockData<Real>>) ||
                    std::is_same_v<SRC_t, T>)) {
      // SRC_t and T are incompatible
      static_assert(always_false<SRC_t>);
    }

    auto key = GetKey(name, src);
    auto it = containers_.find(key);
    if (it != containers_.end()) {
      if (fields.size() && !(it->second)->Contains(fields)) {
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
  std::shared_ptr<T> &Add(const std::string &label);

  auto &Stages() { return containers_; }
  const auto &Stages() const { return containers_; }

  std::shared_ptr<T> &Get() { return containers_.at("base"); }
  const std::shared_ptr<T> &Get() const { return containers_.at("base"); }
  std::shared_ptr<T> &Get(const std::string &label) {
    auto it = containers_.find(label);
    if (it == containers_.end()) {
      throw std::runtime_error("Container " + label + " does not exist in collection.");
    }
    return it->second;
  }

  void Set(const std::string &name, std::shared_ptr<T> &d) { containers_[name] = d; }

  std::shared_ptr<T> &GetOrAdd(const std::string &mbd_label, const int &partition_id);
  std::shared_ptr<T> &GetOrAdd(int gmg_level, const std::string &mbd_label,
                               const int &partition_id);

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
  template <class U>
  std::string GetKey(const std::string &stage_label, const std::shared_ptr<U> &in) {
    if constexpr (std::is_same_v<U, MeshData<Real>>) {
      std::string key = stage_label + "_part-" + std::to_string(in->partition);
      if (in->grid.type == GridType::two_level_composite)
        key = key + "_gmg-" + std::to_string(in->grid.logical_level);
      return key;
    } else {
      return stage_label;
    }
  }
  std::string GetKey(const std::string &stage_label, std::optional<int> partition_id,
                     std::optional<int> gmg_level) {
    std::string key = stage_label;
    if (partition_id) key = key + "_part-" + std::to_string(*partition_id);
    if (gmg_level) key = key + "_gmg-" + std::to_string(*gmg_level);
    return key;
  }

  std::shared_ptr<T> &GetOrAdd_impl(const std::string &mbd_label, const int &partition_id,
                                    const std::optional<int> gmg_level);

  Mesh *pmy_mesh_;
  std::map<std::string, std::shared_ptr<T>> containers_;
};

} // namespace parthenon

#endif // INTERFACE_DATA_COLLECTION_HPP_
