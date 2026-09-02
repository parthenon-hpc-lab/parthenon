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
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "basic_types.hpp"
#include "globals.hpp"
#include "interface/variable.hpp"
#include "utils/concepts_lite.hpp"
#include "utils/error_checking.hpp"
#include "utils/unique_id.hpp"

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

  template <class SRC_t, typename ID_t = std::string>
  std::shared_ptr<T> &Add(const std::string &label, const std::shared_ptr<SRC_t> &src,
                          const std::vector<ID_t> &fields = {}) {
    return AddImpl(label, src, fields, false);
  }

  template <class SRC_t, typename ID_t>
  std::shared_ptr<T> &Add(const std::string &label, const std::shared_ptr<SRC_t> &src,
                          const std::vector<ID_t> &fields, const bool shallow) {
    return AddImpl(label, src, fields, shallow);
  }

  template <class SRC_t, typename ID_t = std::string>
  std::shared_ptr<T> &AddShallow(const std::string &label,
                                 const std::shared_ptr<SRC_t> &src,
                                 const std::vector<ID_t> &fields = {}) {
    return AddImpl(label, src, fields, true);
  }

  template <class SRC_t, typename ID_t = Uid_t>
  std::shared_ptr<T> &AddFromSet(const std::string &label,
                                 const std::shared_ptr<SRC_t> &src,
                                 const std::set<ID_t> &fields) {
    return AddImpl(label, src, fields, false);
  }

  template <class SRC_t, typename ID_t = Uid_t>
  std::shared_ptr<T> &AddShallowFromSet(const std::string &label,
                                        const std::shared_ptr<SRC_t> &src,
                                        const std::set<ID_t> &fields) {
    return AddImpl(label, src, fields, true);
  }

  auto &Stages() { return containers_; }
  const auto &Stages() const { return containers_; }

  template <class SRC_t>
  const std::shared_ptr<T> &Get(const std::string &name,
                                const std::shared_ptr<SRC_t> &src) const {
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

  // The field list (as a canonical variable-uid set) that the named container was created
  // from. Every container sharing a base name is created from the same list (see the
  // warning in Add). If the name has never been added, returns a static empty set.
  const std::set<Uid_t> &GetCreationFields(const std::string &name) const {
    static const std::set<Uid_t> empty;
    const auto nit = name_creation_fields_.find(name);
    return nit == name_creation_fields_.end() ? empty : nit->second;
  }

  void Set(const std::string &name, std::shared_ptr<T> &d) { containers_[name] = d; }

  // Legacy methods that are specific to MeshData
  std::shared_ptr<T> &GetOrAdd(const std::string &mbd_label, const int &partition_id);
  std::shared_ptr<T> &GetOrAdd(int gmg_level, const std::string &mbd_label,
                               const int &partition_id);

  void clear() { containers_.clear(); }

 private:
  template <class SRC_t, class Fields_t>
  std::shared_ptr<T> &AddImpl(const std::string &name, const std::shared_ptr<SRC_t> &src,
                              const Fields_t &fields, const bool shallow) {
    auto key = GetKey(name, src);
    auto it = containers_.find(key);
    if (it != containers_.end()) {
      // Existing container. An explicit field list must match what the container was
      // actually created from (checked against the container itself, which also catches
      // containers built by hand or through a different DataCollection); an empty list
      // means "all fields"/"don't check" and always passes.
      if (fields.size() && !(it->second)->CreatedFrom(fields))
        PARTHENON_THROW(key + " already exists in collection but fields do not match.");
      return it->second;
    }

    using ID_t = typename Fields_t::value_type;
    auto to_uid = [](const ID_t &f) -> Uid_t {
      if constexpr (std::is_same_v<ID_t, std::string>)
        return Variable<Real>::GetUniqueID(f);
      else
        return f;
    };

    // Track the field list (as a canonical uid set) each container name is created from,
    // so the DataCollection is the single source of truth for it (see GetCreationFields).
    // Containers sharing a base name but built from different sources get distinct keys,
    // so the per-key CreatedFrom check above cannot compare them; this does. All
    // instances of a name must be created from the same list -- fail if not.
    std::set<Uid_t> created;
    for (const auto &f : fields)
      created.insert(to_uid(f));
    auto nit = name_creation_fields_.find(name);
    if (nit == name_creation_fields_.end()) {
      name_creation_fields_[name] = created;
    } else if (nit->second != created) {
      PARTHENON_THROW(
          "Container \"" + name +
          "\" is being created from different field lists on different sources. All "
          "instances sharing a name must be created from the same field list.");
    }

    std::vector<Uid_t> uids(created.begin(), created.end());
    auto c = std::make_shared<T>(name);
    c->Initialize(src, uids, shallow);
    containers_[key] = c;
    return containers_[key];
  }

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
  std::map<std::string, std::set<Uid_t>> name_creation_fields_;
};

} // namespace parthenon

#endif // INTERFACE_DATA_COLLECTION_HPP_
