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
class StateDescriptor;
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
  DataCollection() { pmy_mesh_ = nullptr; }

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

  // Overload used when the source (e.g. a bare MeshBlock) does not carry a
  // resolved_packages of its own, so the field set is defined by an explicitly
  // supplied StateDescriptor.
  template <class SRC_t, typename ID_t = std::string>
  std::shared_ptr<T> &Add(const std::string &label,
                          const std::shared_ptr<StateDescriptor> &resolved_packages,
                          const std::shared_ptr<SRC_t> &src,
                          const std::vector<ID_t> &fields = {}) {
    return AddImpl(label, src, fields, false, resolved_packages);
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
  const std::vector<Uid_t> &GetCreationFields(const std::string &name) const {
    static const std::vector<Uid_t> empty;
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
  std::shared_ptr<T> &
  AddImpl(const std::string &name, const std::shared_ptr<SRC_t> &src,
          const Fields_t &fields, const bool shallow,
          const std::shared_ptr<StateDescriptor> &resolved_packages = nullptr) {
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

    auto same_fields = [](const std::vector<Uid_t> &a,
                          const std::vector<Uid_t> &b) {
         return a.size() == b.size() && std::is_permutation(a.begin(), a.end(), b.begin());
    };
   
    // Track the field list (as a canonical uid set) each container base name is created
    // from, so every container with a given base name contains the same set of fields.
    // Containers sharing a base name but built from different sources get distinct
    // internal names, so the check above cannot compare them; this does. All instances of
    // a name must be created from the same list.
    //
    // Three possibilities in order of precedence:
    //   1. fields is not empty, so we explicitly only include those fields in the container 
    //      and check that set against name_creation_fields_ if the base name exists, otherwise
    //      store the field set in name_creation_fields_ since this is the first container 
    //      created with that base name. [Should we be checking that this is a subset of the
    //      parent container? Yes, probably.]
    //   2. fields is empty but src has a base name set (which means it is a MeshBlockData or MeshData), 
    //      then we inherit the field set from base. 
    //   3. fields is empty and src has no base name (which means it is a MeshBlock or BlockListPartition), 
    //      then we store the empty field list which implies all variables are included.
    std::vector<Uid_t> field_uids;
    for (const auto &f : fields) field_uids.push_back(to_uid(f));
    if constexpr (requires { src->StageName(); }) {
      if (field_uids.empty()) {
        field_uids = name_creation_fields_.at(src->StageName());
      }
    }

    auto nit = name_creation_fields_.find(name);
    if (nit == name_creation_fields_.end()) {
      name_creation_fields_[name] = field_uids;
    } else if (!same_fields(field_uids, nit->second)) {
      PARTHENON_THROW(
          "Container \"" + name +
          "\" is being created from different field lists on different sources. All "
          "instances sharing a name must be created from the same field list.");
    }

    std::shared_ptr<T> c;
    if constexpr (std::is_constructible_v<T, const std::string &,
                                          const std::shared_ptr<StateDescriptor> &,
                                          const std::shared_ptr<SRC_t> &,
                                          const std::vector<Uid_t> &, const bool>) {
      if (resolved_packages) {
        c = std::make_shared<T>(name, resolved_packages, src, field_uids, shallow);
      }
    }
    if (!c) c = std::make_shared<T>(name, src, field_uids, shallow);
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
  std::map<std::string, std::vector<Uid_t>> name_creation_fields_;
};

} // namespace parthenon

#endif // INTERFACE_DATA_COLLECTION_HPP_
